"""
Script to reproduce and reconcile Multimodal Land Cover classification results
on both Validation (N=3,255) and Test (N=3,248) splits of BigEarthNet-14k.

Computes:
- Default threshold (0.50) vs. Validation-Tuned Thresholds
- Baseline RF (84 features) vs. Phase 2 LightGBM (488 features)
- Macro F1 across all 16 predefined classes vs. 14 test-present classes (support > 0)
- Per-class confusion statistics: TP, FP, FN, TN, Support, Precision, Recall, F1, Threshold
"""

import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate_split(split_name: str = "test"):
    meta_path = REPO_ROOT / "data" / f"{split_name}_full_metadata.parquet"
    feats_path = REPO_ROOT / "data" / f"X_{split_name}_phase2.npy"

    meta_df = pd.read_parquet(meta_path)
    X = np.load(feats_path)

    # Load Phase 2 LightGBM model and encoders
    p2_dir = REPO_ROOT / "artifacts" / "specialists" / "land_cover_phase2"
    p2_models = joblib.load(p2_dir / "model.joblib")
    p2_mlb = joblib.load(p2_dir / "label_encoder.joblib")

    with open(p2_dir / "threshold_config.json", "r", encoding="utf-8") as f:
        thresh_cfg = json.load(f)

    classes = list(p2_mlb.classes_)
    class_thresh_map = thresh_cfg.get("class_specific_thresholds", {})
    tuned_thresholds = np.array([class_thresh_map.get(c, 0.40) for c in classes])
    default_thresholds = np.full(len(classes), 0.50)

    y_true = p2_mlb.transform(meta_df["labels"].values)
    present_indices = [i for i in range(len(classes)) if np.sum(y_true[:, i]) > 0]

    # Predict LightGBM probabilities
    prob_matrix = np.zeros((len(X), len(classes)), dtype=np.float32)
    for i, clf in enumerate(p2_models):
        p = clf.predict_proba(X)
        prob_matrix[:, i] = p[:, 1] if p.shape[1] > 1 else 0.0

    # Also evaluate Baseline RF (84 features) if available
    rf_dir = REPO_ROOT / "artifacts" / "specialists" / "land_cover"
    rf_metrics_default = None
    rf_metrics_tuned = None
    if (rf_dir / "model.joblib").exists():
        rf_models = joblib.load(rf_dir / "model.joblib")
        # Global stats are the first 84 features
        X_rf = X[:, :84]
        rf_prob_matrix = np.zeros((len(X_rf), len(classes)), dtype=np.float32)
        if isinstance(rf_models, list):
            for i, clf in enumerate(rf_models):
                p = clf.predict_proba(X_rf)
                rf_prob_matrix[:, i] = p[:, 1] if p.shape[1] > 1 else 0.0
        else:
            p_list = rf_models.predict_proba(X_rf)
            for i, p in enumerate(p_list):
                rf_prob_matrix[:, i] = p[:, 1] if p.shape[1] > 1 else 0.0

        with open(rf_dir / "threshold_config.json", "r", encoding="utf-8") as f:
            rf_thresh_cfg = json.load(f)
        rf_thresh_map = rf_thresh_cfg.get("class_specific_thresholds", {})
        rf_tuned_thresholds = np.array([rf_thresh_map.get(c, rf_thresh_cfg.get("global_threshold", 0.4)) for c in classes])

        # RF Default
        y_pred_rf_def = (rf_prob_matrix >= default_thresholds).astype(int)
        _, _, rf_f1_def_per_class, _ = precision_recall_fscore_support(y_true, y_pred_rf_def, average=None, zero_division=0)
        rf_metrics_default = {
            "micro_f1": round(f1_score(y_true, y_pred_rf_def, average="micro", zero_division=0) * 100, 2),
            "macro_f1_all_16": round(f1_score(y_true, y_pred_rf_def, average="macro", zero_division=0) * 100, 2),
            "macro_f1_present_classes": round(float(np.mean(rf_f1_def_per_class[present_indices])) * 100, 2),
            "hamming_loss": round(hamming_loss(y_true, y_pred_rf_def), 4),
            "subset_acc": round(np.mean(np.all(y_true == y_pred_rf_def, axis=1)) * 100, 2)
        }

        # RF Tuned
        y_pred_rf_tuned = (rf_prob_matrix >= rf_tuned_thresholds).astype(int)
        _, _, rf_f1_tuned_per_class, _ = precision_recall_fscore_support(y_true, y_pred_rf_tuned, average=None, zero_division=0)
        rf_metrics_tuned = {
            "micro_f1": round(f1_score(y_true, y_pred_rf_tuned, average="micro", zero_division=0) * 100, 2),
            "macro_f1_all_16": round(f1_score(y_true, y_pred_rf_tuned, average="macro", zero_division=0) * 100, 2),
            "macro_f1_present_classes": round(float(np.mean(rf_f1_tuned_per_class[present_indices])) * 100, 2),
            "hamming_loss": round(hamming_loss(y_true, y_pred_rf_tuned), 4),
            "subset_acc": round(np.mean(np.all(y_true == y_pred_rf_tuned, axis=1)) * 100, 2)
        }

    # Evaluate LightGBM Default
    y_pred_lgb_def = (prob_matrix >= default_thresholds).astype(int)
    prec_def, rec_def, f1_def, supp_def = precision_recall_fscore_support(y_true, y_pred_lgb_def, average=None, zero_division=0)
    lgb_metrics_default = {
        "micro_f1": round(f1_score(y_true, y_pred_lgb_def, average="micro", zero_division=0) * 100, 2),
        "macro_f1_all_16": round(f1_score(y_true, y_pred_lgb_def, average="macro", zero_division=0) * 100, 2),
        "macro_f1_present_classes": round(float(np.mean(f1_def[present_indices])) * 100, 2),
        "hamming_loss": round(hamming_loss(y_true, y_pred_lgb_def), 4),
        "subset_acc": round(np.mean(np.all(y_true == y_pred_lgb_def, axis=1)) * 100, 2)
    }

    per_class_default_records = []
    for i, cls_name in enumerate(classes):
        tp = int(np.sum((y_pred_lgb_def[:, i] == 1) & (y_true[:, i] == 1)))
        fp = int(np.sum((y_pred_lgb_def[:, i] == 1) & (y_true[:, i] == 0)))
        fn = int(np.sum((y_pred_lgb_def[:, i] == 0) & (y_true[:, i] == 1)))
        tn = int(np.sum((y_pred_lgb_def[:, i] == 0) & (y_true[:, i] == 0)))
        per_class_default_records.append({
            "class_name": cls_name,
            "support": int(supp_def[i]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(float(prec_def[i]), 4),
            "recall": round(float(rec_def[i]), 4),
            "f1": round(float(f1_def[i]), 4),
            "threshold": 0.50
        })

    # Evaluate LightGBM Tuned
    y_pred_lgb_tuned = (prob_matrix >= tuned_thresholds).astype(int)
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred_lgb_tuned, average=None, zero_division=0)
    lgb_metrics_tuned = {
        "micro_f1": round(f1_score(y_true, y_pred_lgb_tuned, average="micro", zero_division=0) * 100, 2),
        "macro_f1_all_16": round(f1_score(y_true, y_pred_lgb_tuned, average="macro", zero_division=0) * 100, 2),
        "macro_f1_present_classes": round(float(np.mean(f1[present_indices])) * 100, 2),
        "hamming_loss": round(hamming_loss(y_true, y_pred_lgb_tuned), 4),
        "subset_acc": round(np.mean(np.all(y_true == y_pred_lgb_tuned, axis=1)) * 100, 2)
    }

    per_class_tuned_records = []
    for i, cls_name in enumerate(classes):
        tp = int(np.sum((y_pred_lgb_tuned[:, i] == 1) & (y_true[:, i] == 1)))
        fp = int(np.sum((y_pred_lgb_tuned[:, i] == 1) & (y_true[:, i] == 0)))
        fn = int(np.sum((y_pred_lgb_tuned[:, i] == 0) & (y_true[:, i] == 1)))
        tn = int(np.sum((y_pred_lgb_tuned[:, i] == 0) & (y_true[:, i] == 0)))

        # Explicitly document mathematical status for classes with 0 support
        note = "supported"
        if supp[i] == 0:
            if fp > 0:
                note = f"0 support in {split_name}; FP={fp} -> precision=0.0, recall=0/0 (indeterminate, imputed 0.0)"
            else:
                note = f"0 support in {split_name}; no predictions -> precision=0/0, recall=0/0 (imputed 0.0)"

        per_class_tuned_records.append({
            "class_name": cls_name,
            "support": int(supp[i]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "tuned_threshold": round(float(tuned_thresholds[i]), 4),
            "evaluation_note": note
        })

    return {
        "split": split_name,
        "sample_count": len(X),
        "num_classes_total": len(classes),
        "num_classes_with_support": len(present_indices),
        "rf_metrics_default": rf_metrics_default,
        "rf_metrics_tuned": rf_metrics_tuned,
        "lgb_metrics_default": lgb_metrics_default,
        "lgb_metrics_tuned": lgb_metrics_tuned,
        "per_class_lgb_default": per_class_default_records,
        "per_class_lgb_tuned": per_class_tuned_records
    }


def main():
    print("=" * 80)
    print("   BIGEARTHNET-14K LAND COVER SPECIALIST BENCHMARK RECONCILIATION")
    print("=" * 80)

    val_res = evaluate_split("val")
    test_res = evaluate_split("test")

    print("\n--- 1. VALIDATION SPLIT (N=3,255) ---")
    print("LightGBM (488 feats, default tau=0.50)  :", val_res["lgb_metrics_default"])
    print("LightGBM (488 feats, tuned tau)         :", val_res["lgb_metrics_tuned"])

    print("\n--- 2. HELD-OUT TEST SPLIT (N=3,248) ---")
    print("LightGBM (488 feats, default tau=0.50)  :", test_res["lgb_metrics_default"])
    print("LightGBM (488 feats, tuned tau)         :", test_res["lgb_metrics_tuned"])
    print("RF Baseline (84 feats, default tau=0.50):", test_res["rf_metrics_default"])

    print("\n--- 3. PER-CLASS CONFUSION & PERFORMANCE (TEST SPLIT, TUNED THRESHOLDS) ---")
    df_per_class = pd.DataFrame(test_res["per_class_lgb_tuned"])
    cols = ["class_name", "support", "tp", "fp", "fn", "precision", "recall", "f1", "tuned_threshold"]
    print(df_per_class[cols].to_string(index=False))

    # Save summary artifact
    out_json = REPO_ROOT / "artifacts" / "benchmarks" / "land_cover_reconciliation_audit.json"
    out_csv = REPO_ROOT / "artifacts" / "benchmarks" / "land_cover_per_class_test_metrics.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"val": val_res, "test": test_res}, f, indent=2)
    df_per_class.to_csv(out_csv, index=False)
    print(f"\nSaved audit JSON to: {out_json.relative_to(REPO_ROOT)}")
    print(f"Saved per-class CSV to: {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
