"""
Rigorous 2x2 Controlled Model Comparison and Held-Out Test Evaluation.

Protocol:
  1. Identical Data Splits:
     - Train: 7,180 samples
     - Validation: 3,255 samples
     - Untouched Held-Out Test: 3,248 samples
  2. Model Grid:
     - Random Forest (84 features, unweighted baseline)
     - Random Forest (84 features, class_weight='balanced')
     - Random Forest (488 features)
     - LightGBM (84 features)
     - LightGBM (488 features)
  3. Threshold Tuning:
     - Optimized strictly on Validation Set predictions.
  4. Final Evaluation:
     - Evaluated ONCE on Untouched Held-Out Test Set with frozen models & thresholds.
     - Reports Micro F1, Macro F1, Hamming Loss, and per-class metrics.
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


def evaluate_split(y_true: np.ndarray, prob_matrix: np.ndarray, thresholds: np.ndarray, classes: list) -> dict:
    """Evaluate multi-label predictions against ground truth using given thresholds."""
    y_pred = (prob_matrix >= thresholds).astype(int)

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    h_loss = float(hamming_loss(y_true, y_pred))

    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    pred_positives = y_pred.sum(axis=0)

    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(supp[i]),
            "predicted_positives": int(pred_positives[i]),
            "threshold": round(float(thresholds[i]), 4)
        }

    return {
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "hamming_loss": round(h_loss, 4),
        "per_class": per_class
    }


def optimize_thresholds_on_val(y_val: np.ndarray, prob_val: np.ndarray, default_th: float = 0.40) -> np.ndarray:
    """Find per-class threshold in [0.10, 0.60] maximizing validation F1."""
    n_classes = y_val.shape[1]
    best_thresholds = np.full(n_classes, default_th, dtype=np.float32)
    candidate_thresholds = np.linspace(0.10, 0.60, 51)

    for c in range(n_classes):
        y_c = y_val[:, c]
        p_c = prob_val[:, c]
        if y_c.sum() == 0:
            continue

        best_score = -1.0
        best_t = default_th
        for t in candidate_thresholds:
            pred_c = (p_c >= t).astype(int)
            score = f1_score(y_c, pred_c, average="binary", zero_division=0)
            if score > best_score:
                best_score = score
                best_t = t

        best_thresholds[c] = round(best_t, 3)

    return best_thresholds


def train_rf(X_train, Y_train, class_weight=None, n_estimators=200):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, Y_train)
    return rf


def predict_rf(rf, X):
    prob_list = rf.predict_proba(X)
    return np.column_stack([p[:, 1] if p.shape[1] > 1 else np.zeros(p.shape[0]) for p in prob_list])


def train_lightgbm(X_train, Y_train, n_classes, n_estimators=150):
    models = []
    for c in range(n_classes):
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42 + c,
            n_jobs=4,
            verbose=-1
        )
        clf.fit(X_train, Y_train[:, c])
        models.append(clf)
    return models


def predict_lightgbm(models, X):
    n_samples = len(X)
    n_classes = len(models)
    probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    for c, clf in enumerate(models):
        p = clf.predict_proba(X)
        probs[:, c] = p[:, 1] if p.shape[1] > 1 else 0.0
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmarks")
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print("  SatQuery AI: Rigorous 2x2 Controlled Benchmark & Test Evaluation   ")
    print("=====================================================================")

    # 1. Load Data Splits
    train_meta = pd.read_parquet("data/train_full_metadata.parquet")
    val_meta = pd.read_parquet("data/val_full_metadata.parquet")
    test_meta = pd.read_parquet("data/test_full_metadata.parquet")

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(train_meta["labels"].apply(list))
    Y_val = mlb.transform(val_meta["labels"].apply(list))
    Y_test = mlb.transform(test_meta["labels"].apply(list))
    classes = list(mlb.classes_)
    n_classes = len(classes)

    print(f"Data Splits: Train = {len(Y_train)}, Val = {len(Y_val)}, Test = {len(Y_test)}")
    print(f"Classes ({n_classes}): {classes}")

    # Load Features
    X_train_488 = np.load(feat_dir / "X_train_phase2.npy")
    X_val_488 = np.load(feat_dir / "X_val_phase2.npy")
    X_test_488 = np.load(feat_dir / "X_test_phase2.npy")

    X_train_84 = X_train_488[:, :84]
    X_val_84 = X_val_488[:, :84]
    X_test_84 = X_test_488[:, :84]

    print(f"Feature Shapes: Train {X_train_488.shape}, Val {X_val_488.shape}, Test {X_test_488.shape}\n")

    th_default = np.full(n_classes, 0.40)

    # Dictionary of models to train
    experiments = [
        ("RF (84 feats, unweighted)", "rf", X_train_84, X_val_84, X_test_84, 84, None),
        ("RF (84 feats, balanced)", "rf", X_train_84, X_val_84, X_test_84, 84, "balanced"),
        ("RF (488 feats, unweighted)", "rf", X_train_488, X_val_488, X_test_488, 488, None),
        ("LightGBM (84 feats)", "lgb", X_train_84, X_val_84, X_test_84, 84, None),
        ("LightGBM (488 feats)", "lgb", X_train_488, X_val_488, X_test_488, 488, None),
    ]

    val_records = []
    test_records = []
    detailed_results = {}

    for name, mtype, X_tr, X_v, X_te, feats_cnt, cw in experiments:
        print(f"--> Training {name}...")
        t0 = time.time()
        if mtype == "rf":
            model = train_rf(X_tr, Y_train, class_weight=cw)
            prob_val = predict_rf(model, X_v)
            prob_test = predict_rf(model, X_te)
        else:
            model = train_lightgbm(X_tr, Y_train, n_classes)
            prob_val = predict_lightgbm(model, X_v)
            prob_test = predict_lightgbm(model, X_te)
        train_time = round(time.time() - t0, 1)

        # 1. Validation at global threshold 0.40
        eval_val_def = evaluate_split(Y_val, prob_val, th_default, classes)
        val_records.append({
            "Model": name,
            "Features": feats_cnt,
            "Threshold": "Global (0.40)",
            "Val Micro F1": eval_val_def["micro_f1"],
            "Val Macro F1": eval_val_def["macro_f1"],
            "Val Hamming": eval_val_def["hamming_loss"],
            "Train Time (s)": train_time
        })

        # 2. Tune thresholds strictly on Validation Set
        tuned_thresholds = optimize_thresholds_on_val(Y_val, prob_val, default_th=0.40)
        eval_val_tuned = evaluate_split(Y_val, prob_val, tuned_thresholds, classes)
        val_records.append({
            "Model": name,
            "Features": feats_cnt,
            "Threshold": "Val-Tuned Per-Class",
            "Val Micro F1": eval_val_tuned["micro_f1"],
            "Val Macro F1": eval_val_tuned["macro_f1"],
            "Val Hamming": eval_val_tuned["hamming_loss"],
            "Train Time (s)": train_time
        })

        # 3. Independent Evaluation on Untouched Held-Out Test Set
        eval_test_def = evaluate_split(Y_test, prob_test, th_default, classes)
        eval_test_tuned = evaluate_split(Y_test, prob_test, tuned_thresholds, classes)

        test_records.append({
            "Model": name,
            "Features": feats_cnt,
            "Threshold": "Global (0.40)",
            "Test Micro F1": eval_test_def["micro_f1"],
            "Test Macro F1": eval_test_def["macro_f1"],
            "Test Hamming": eval_test_def["hamming_loss"],
        })
        test_records.append({
            "Model": name,
            "Features": feats_cnt,
            "Threshold": "Frozen Val-Tuned",
            "Test Micro F1": eval_test_tuned["micro_f1"],
            "Test Macro F1": eval_test_tuned["macro_f1"],
            "Test Hamming": eval_test_tuned["hamming_loss"],
        })

        detailed_results[name] = {
            "validation_global_0.40": eval_val_def,
            "validation_tuned": eval_val_tuned,
            "test_global_0.40": eval_test_def,
            "test_frozen_tuned": eval_test_tuned,
            "tuned_thresholds": {cls: float(tuned_thresholds[i]) for i, cls in enumerate(classes)}
        }

    # Present Results
    df_val = pd.DataFrame(val_records)
    df_test = pd.DataFrame(test_records)

    print("\n========================= 1. VALIDATION SET COMPARISON (N=3,255) =========================")
    print(df_val.to_string(index=False))

    print("\n==================== 2. UNTOUCHED HELD-OUT TEST SET EVALUATION (N=3,248) ====================")
    print(df_test.to_string(index=False))
    print("=============================================================================================\n")

    # Save to disk
    df_val.to_csv(out_dir / "controlled_grid_val_results.csv", index=False)
    df_test.to_csv(out_dir / "controlled_grid_test_results.csv", index=False)

    payload = {
        "val_results": val_records,
        "test_results": test_records,
        "detailed": detailed_results,
        "data_split_counts": {
            "train": len(Y_train),
            "validation": len(Y_val),
            "test": len(Y_test)
        }
    }
    with open(out_dir / "controlled_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Results saved to {out_dir / 'controlled_comparison_results.json'}")


if __name__ == "__main__":
    main()
