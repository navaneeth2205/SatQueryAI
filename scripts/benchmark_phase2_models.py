"""
Benchmark script for Phase 2 Classical Feature Hardening.

Compares:
  1. Baseline RF (84 features)
  2. Enhanced RF (488 features: spatial pyramid, spectral indices, radar indices, texture)
  3. LightGBM (488 features: gradient boosted decision trees)

Evaluates on identical train (7,180) and val (3,255) splits under:
  - Micro F1, Macro F1, Hamming Loss
  - Per-class precision, recall, and F1
  - Global vs. class-specific threshold optimization
"""

import sys
from pathlib import Path

# Force UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
import lightgbm as lgb


def evaluate_predictions(y_true: np.ndarray, prob_matrix: np.ndarray, thresholds: np.ndarray, classes: list) -> dict:
    """Evaluate multi-label predictions against ground truth using given thresholds."""
    y_pred = (prob_matrix >= thresholds).astype(int)

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    h_loss = float(hamming_loss(y_true, y_pred))

    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(supp[i]),
            "threshold": round(float(thresholds[i]), 4)
        }

    return {
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "hamming_loss": round(h_loss, 4),
        "per_class": per_class
    }


def optimize_per_class_thresholds(y_true: np.ndarray, prob_matrix: np.ndarray, default_th: float = 0.40) -> np.ndarray:
    """Find threshold in [0.10, 0.60] for each class that maximizes its validation F1 score."""
    n_classes = y_true.shape[1]
    best_thresholds = np.full(n_classes, default_th, dtype=np.float32)

    candidate_thresholds = np.linspace(0.10, 0.60, 51)

    for c in range(n_classes):
        y_c = y_true[:, c]
        p_c = prob_matrix[:, c]

        # If class has zero positive support, keep default
        if y_c.sum() == 0:
            continue

        best_score = -1.0
        best_t = default_th
        for t in candidate_thresholds:
            pred_c = (p_c >= t).astype(int)
            # F1 score for single binary class
            score = f1_score(y_c, pred_c, average="binary", zero_division=0)
            if score > best_score:
                best_score = score
                best_t = t

        best_thresholds[c] = round(best_t, 3)

    return best_thresholds


def train_and_eval_random_forest(X_train, Y_train, X_val, Y_val, classes, n_estimators=200, name="RandomForest"):
    print(f"\n--- Training {name} ({X_train.shape[1]} features, {n_estimators} trees) ---")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, Y_train)
    fit_time = time.time() - t0
    print(f"  Fitted in {fit_time:.2f}s")

    t0 = time.time()
    prob_list = rf.predict_proba(X_val)
    probs = np.column_stack([p[:, 1] if p.shape[1] > 1 else np.zeros(p.shape[0]) for p in prob_list])
    infer_time = time.time() - t0
    print(f"  Inference on {len(X_val)} samples in {infer_time:.2f}s ({infer_time / len(X_val) * 1000:.3f} ms/sample)")

    return rf, probs, fit_time, infer_time


def train_and_eval_lightgbm(X_train, Y_train, X_val, Y_val, classes, n_estimators=150, name="LightGBM"):
    print(f"\n--- Training {name} ({X_train.shape[1]} features, {len(classes)} binary models) ---")
    t0 = time.time()
    n_classes = len(classes)
    models = []
    val_probs = np.zeros((len(X_val), n_classes), dtype=np.float32)

    for c in range(n_classes):
        y_train_c = Y_train[:, c]
        # Class balance ratio
        pos_count = max(int(y_train_c.sum()), 1)
        neg_count = len(y_train_c) - pos_count
        scale_pos_weight = min(neg_count / pos_count, 15.0)

        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=42 + c,
            n_jobs=4,
            verbose=-1
        )
        clf.fit(X_train, y_train_c)
        models.append(clf)

        p = clf.predict_proba(X_val)
        val_probs[:, c] = p[:, 1] if p.shape[1] > 1 else 0.0

    fit_time = time.time() - t0
    print(f"  Fitted {n_classes} LightGBM models in {fit_time:.2f}s")
    return models, val_probs, fit_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark Phase 2 models.")
    parser.add_argument("--features-dir", type=str, default="data")
    parser.add_argument("--baseline-dir", type=str, default="initial_trails")
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmarks")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2 Model Benchmarking & Comparison ===")

    # 1. Load Data
    train_meta = pd.read_parquet("data/train_full_metadata.parquet")
    val_meta = pd.read_parquet("data/val_full_metadata.parquet")

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(train_meta["labels"].apply(list))
    Y_val = mlb.transform(val_meta["labels"].apply(list))
    classes = list(mlb.classes_)

    print(f"Dataset: {len(Y_train)} train, {len(Y_val)} val samples, {len(classes)} classes")

    # Load Phase 2 features
    X_train_p2 = np.load(Path(args.features_dir) / "X_train_phase2.npy")
    X_val_p2 = np.load(Path(args.features_dir) / "X_val_phase2.npy")
    print(f"Loaded Phase 2 features: Train {X_train_p2.shape}, Val {X_val_p2.shape}")

    # Baseline features (first 84 dimensions)
    X_train_base = X_train_p2[:, :84]
    X_val_base = X_val_p2[:, :84]

    results_table = []
    detailed_results = {}

    # ---------------------------------------------------------
    # Experiment 1: Baseline RF (84 features)
    # ---------------------------------------------------------
    rf_base, probs_base, fit_t_base, infer_t_base = train_and_eval_random_forest(
        X_train_base, Y_train, X_val_base, Y_val, classes, n_estimators=200, name="Baseline RF (84 feats)"
    )

    # Eval 1a: Default threshold 0.40
    th_default = np.full(len(classes), 0.40)
    eval_base_def = evaluate_predictions(Y_val, probs_base, th_default, classes)
    results_table.append({
        "Model": "Baseline RF (84 feats, global th=0.40)",
        "Features": 84,
        "Micro F1": eval_base_def["micro_f1"],
        "Macro F1": eval_base_def["macro_f1"],
        "Hamming Loss": eval_base_def["hamming_loss"],
        "Fit Time (s)": round(fit_t_base, 1),
    })

    # Eval 1b: Baseline with previously configured class-specific thresholds
    baseline_cfg_path = Path("artifacts/specialists/land_cover/threshold_config.json")
    if baseline_cfg_path.exists():
        with open(baseline_cfg_path) as f:
            b_cfg = json.load(f)
        b_class_th = b_cfg.get("class_specific_thresholds", {})
        th_base_tuned = np.array([b_class_th.get(c, 0.40) for c in classes])
        eval_base_tuned = evaluate_predictions(Y_val, probs_base, th_base_tuned, classes)
        results_table.append({
            "Model": "Baseline RF (84 feats, tuned baseline thresholds)",
            "Features": 84,
            "Micro F1": eval_base_tuned["micro_f1"],
            "Macro F1": eval_base_tuned["macro_f1"],
            "Hamming Loss": eval_base_tuned["hamming_loss"],
            "Fit Time (s)": round(fit_t_base, 1),
        })

    # ---------------------------------------------------------
    # Experiment 2: Enhanced RF (488 features)
    # ---------------------------------------------------------
    rf_enh, probs_enh, fit_t_enh, infer_t_enh = train_and_eval_random_forest(
        X_train_p2, Y_train, X_val_p2, Y_val, classes, n_estimators=200, name="Enhanced RF (488 feats)"
    )

    eval_enh_def = evaluate_predictions(Y_val, probs_enh, th_default, classes)
    results_table.append({
        "Model": "Enhanced RF (488 feats, global th=0.40)",
        "Features": 488,
        "Micro F1": eval_enh_def["micro_f1"],
        "Macro F1": eval_enh_def["macro_f1"],
        "Hamming Loss": eval_enh_def["hamming_loss"],
        "Fit Time (s)": round(fit_t_enh, 1),
    })

    # Optimize thresholds for Enhanced RF
    th_enh_opt = optimize_per_class_thresholds(Y_val, probs_enh, default_th=0.40)
    eval_enh_opt = evaluate_predictions(Y_val, probs_enh, th_enh_opt, classes)
    results_table.append({
        "Model": "Enhanced RF (488 feats, per-class optimized thresholds)",
        "Features": 488,
        "Micro F1": eval_enh_opt["micro_f1"],
        "Macro F1": eval_enh_opt["macro_f1"],
        "Hamming Loss": eval_enh_opt["hamming_loss"],
        "Fit Time (s)": round(fit_t_enh, 1),
    })

    # ---------------------------------------------------------
    # Experiment 3: LightGBM (488 features)
    # ---------------------------------------------------------
    lgb_models, probs_lgb, fit_t_lgb = train_and_eval_lightgbm(
        X_train_p2, Y_train, X_val_p2, Y_val, classes, n_estimators=150, name="LightGBM (488 feats)"
    )

    eval_lgb_def = evaluate_predictions(Y_val, probs_lgb, th_default, classes)
    results_table.append({
        "Model": "LightGBM (488 feats, global th=0.40)",
        "Features": 488,
        "Micro F1": eval_lgb_def["micro_f1"],
        "Macro F1": eval_lgb_def["macro_f1"],
        "Hamming Loss": eval_lgb_def["hamming_loss"],
        "Fit Time (s)": round(fit_t_lgb, 1),
    })

    th_lgb_opt = optimize_per_class_thresholds(Y_val, probs_lgb, default_th=0.40)
    eval_lgb_opt = evaluate_predictions(Y_val, probs_lgb, th_lgb_opt, classes)
    results_table.append({
        "Model": "LightGBM (488 feats, per-class optimized thresholds)",
        "Features": 488,
        "Micro F1": eval_lgb_opt["micro_f1"],
        "Macro F1": eval_lgb_opt["macro_f1"],
        "Hamming Loss": eval_lgb_opt["hamming_loss"],
        "Fit Time (s)": round(fit_t_lgb, 1),
    })

    # Print Comparison Table
    df_results = pd.DataFrame(results_table)
    print("\n================== Controlled Comparison Summary ==================")
    print(df_results.to_string(index=False))
    print("===================================================================\n")

    # Save summary files
    df_results.to_csv(out_dir / "phase2_model_comparison.csv", index=False)
    summary_payload = {
        "comparison_table": results_table,
        "detailed_evaluations": {
            "baseline_rf_default": eval_base_def,
            "enhanced_rf_default": eval_enh_def,
            "enhanced_rf_optimized": eval_enh_opt,
            "lightgbm_default": eval_lgb_def,
            "lightgbm_optimized": eval_lgb_opt,
        },
        "optimized_thresholds": {
            "enhanced_rf": {cls: float(th_enh_opt[i]) for i, cls in enumerate(classes)},
            "lightgbm": {cls: float(th_lgb_opt[i]) for i, cls in enumerate(classes)}
        }
    }

    with open(out_dir / "phase2_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Determine winning model by Micro F1 and Macro F1
    # Check if LightGBM or Enhanced RF wins
    enh_macro = eval_enh_opt["macro_f1"]
    lgb_macro = eval_lgb_opt["macro_f1"]
    enh_micro = eval_enh_opt["micro_f1"]
    lgb_micro = eval_lgb_opt["micro_f1"]

    print(f"Enhanced RF: Micro F1 = {enh_micro}, Macro F1 = {enh_macro}")
    print(f"LightGBM:    Micro F1 = {lgb_micro}, Macro F1 = {lgb_macro}")

    winning_model_type = "enhanced_rf" if (enh_micro + enh_macro) >= (lgb_micro + lgb_macro) else "lightgbm"
    print(f"\nWinning Model Selected: {winning_model_type.upper()}")

    # Save winning artifacts package
    pkg_dir = Path("artifacts/specialists/land_cover_phase2")
    pkg_dir.mkdir(parents=True, exist_ok=True)

    if winning_model_type == "enhanced_rf":
        joblib.dump(rf_enh, pkg_dir / "model.joblib")
        winning_thresholds = {cls: float(th_enh_opt[i]) for i, cls in enumerate(classes)}
        win_meta = {
            "model_type": "RandomForestClassifier",
            "features_count": 488,
            "n_estimators": 200,
            "micro_f1": enh_micro,
            "macro_f1": enh_macro,
            "hamming_loss": eval_enh_opt["hamming_loss"]
        }
    else:
        joblib.dump(lgb_models, pkg_dir / "model.joblib")
        winning_thresholds = {cls: float(th_lgb_opt[i]) for i, cls in enumerate(classes)}
        win_meta = {
            "model_type": "LightGBMClassifierList",
            "features_count": 488,
            "n_estimators": 150,
            "micro_f1": lgb_micro,
            "macro_f1": lgb_macro,
            "hamming_loss": eval_lgb_opt["hamming_loss"]
        }

    joblib.dump(mlb, pkg_dir / "label_encoder.joblib")
    with open(pkg_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    threshold_cfg = {
        "global_threshold": 0.40,
        "class_specific_thresholds": winning_thresholds
    }
    with open(pkg_dir / "threshold_config.json", "w", encoding="utf-8") as f:
        json.dump(threshold_cfg, f, indent=2)

    with open(pkg_dir / "metadata_summary.json", "w", encoding="utf-8") as f:
        json.dump(win_meta, f, indent=2)

    print(f"Packaged winning model artifacts to: {pkg_dir}")


if __name__ == "__main__":
    main()
