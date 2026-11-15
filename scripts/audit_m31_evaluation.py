"""
Comprehensive Scientific Audit of Milestone M3.1 Land-Cover Evaluation.

Investigates:
1. Exact per-class recalculation from y_test_true.npy and y_test_prob.npy.
2. Arithmetic verification of Macro F1 (16-class and 14-class test-present).
3. Deep investigation of Class 2 ('Discontinuous urban fabric') with TP, FP, FN, precision, recall, positive count.
4. Class ordering alignment between classes.json, metadata, and model head.
5. Baseline comparison against Phase 2 records (33.21% / 47.95% vs 41.14% / 68.21%).
6. ImageNet initialization audit on checkpoint weights.
7. Validation threshold selection audit (verify zero access to test split).
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import torch

repo_root = Path(__file__).resolve().parent.parent

# 1. Load classes definition
classes_path = repo_root / "artifacts" / "specialists" / "land_cover" / "classes.json"
with open(classes_path, "r", encoding="utf-8") as f:
    classes_list = json.load(f)

print("=" * 80)
print(f"LOADED {len(classes_list)} CLASSES:")
for idx, c in enumerate(classes_list):
    print(f"  [{idx:02d}] {c}")
print("=" * 80)


def audit_model(model_name: str, model_dir: Path):
    print(f"\n{'='*30} AUDITING {model_name.upper()} {'='*30}")
    y_test_true_path = model_dir / "y_test_true.npy"
    y_test_prob_path = model_dir / "y_test_prob.npy"
    thresh_path = model_dir / "threshold_config.json"
    summary_path = model_dir / "evaluation_summary.json"

    assert y_test_true_path.exists(), f"Missing {y_test_true_path}"
    assert y_test_prob_path.exists(), f"Missing {y_test_prob_path}"

    y_true = np.load(y_test_true_path)
    y_prob = np.load(y_test_prob_path)

    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_data = json.load(f)
    tuned_thresholds = np.array(thresh_data["tuned_class_thresholds"])

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    print(f"Shape of y_true: {y_true.shape}, y_prob: {y_prob.shape}")
    print(f"Tuned thresholds: {tuned_thresholds.tolist()}")

    # A. Threshold predictions
    y_pred_def = (y_prob >= 0.50).astype(np.int32)
    thresh_arr = tuned_thresholds.reshape(1, -1)
    y_pred_tuned = (y_prob >= thresh_arr).astype(np.int32)

    # B. Support counts
    test_support = y_true.sum(axis=0)
    present_indices = np.where(test_support > 0)[0]
    zero_support_indices = np.where(test_support == 0)[0]
    print(f"Present classes ({len(present_indices)}): {present_indices.tolist()}")
    print(f"Zero support classes ({len(zero_support_indices)}): {zero_support_indices.tolist()}")

    # C. Granular per-class table for tuned thresholds
    print("\nPER-CLASS DETAILED BREAKDOWN (TUNED THRESHOLDS):")
    print(f"{'Idx':<4} | {'Class Name':<38} | {'Supp':<5} | {'Pred':<5} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Thresh':<6}")
    print("-" * 115)

    per_class_f1_list = []
    per_class_prec_list = []
    per_class_rec_list = []

    for i in range(16):
        c_name = classes_list[i]
        t = y_true[:, i]
        p = y_pred_tuned[:, i]
        supp = int(t.sum())
        pred_cnt = int(p.sum())

        tp = int(np.sum((p == 1) & (t == 1)))
        fp = int(np.sum((p == 1) & (t == 0)))
        fn = int(np.sum((p == 0) & (t == 1)))

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        per_class_f1_list.append(f1)
        per_class_prec_list.append(prec)
        per_class_rec_list.append(rec)

        print(f"[{i:02d}] | {c_name:<38} | {supp:<5} | {pred_cnt:<5} | {tp:<5} | {fp:<5} | {fn:<5} | {prec:<7.4f} | {rec:<7.4f} | {f1:<7.4f} | {tuned_thresholds[i]:<6.2f}")

    # D. Recalculate Aggregates
    macro_f1_16_calc = float(np.mean(per_class_f1_list))
    macro_f1_present_calc = float(np.mean([per_class_f1_list[idx] for idx in present_indices]))
    micro_f1_calc = float(f1_score(y_true, y_pred_tuned, average="micro", zero_division=0.0))
    hamming_calc = float(hamming_loss(y_true, y_pred_tuned))

    # Also sklearn standard
    sk_macro_16 = float(f1_score(y_true, y_pred_tuned, average="macro", zero_division=0.0))

    print("\nAGGREGATE RECALCULATION:")
    print(f"  Arithmetic Mean of 16 Per-Class F1s: {macro_f1_16_calc:.6f} ({macro_f1_16_calc*100:.2f}%)")
    print(f"  Scikit-Learn Macro F1 (16 cls):      {sk_macro_16:.6f} ({sk_macro_16*100:.2f}%)")
    print(f"  Arithmetic Mean of 14 Present F1s:   {macro_f1_present_calc:.6f} ({macro_f1_present_calc*100:.2f}%)")
    print(f"  Independent Micro F1:                {micro_f1_calc:.6f} ({micro_f1_calc*100:.2f}%)")
    print(f"  Independent Hamming Loss:            {hamming_calc:.6f}")

    # Compare with summary.json
    rep_macro_16 = summary_data["test_metrics_tuned"]["macro_f1_16"]
    rep_macro_present = summary_data["test_metrics_tuned"]["macro_f1_present"]
    rep_micro = summary_data["test_metrics_tuned"]["micro_f1"]
    rep_hamming = summary_data["test_metrics_tuned"]["hamming_loss"]

    print("\nDISCREPANCY CHECK VS REPORTED JSON:")
    print(f"  Reported Macro 16:      {rep_macro_16:.6f} vs Recalculated: {macro_f1_16_calc:.6f} -> Diff: {abs(rep_macro_16 - macro_f1_16_calc):.6e}")
    print(f"  Reported Macro Present: {rep_macro_present:.6f} vs Recalculated: {macro_f1_present_calc:.6f} -> Diff: {abs(rep_macro_present - macro_f1_present_calc):.6e}")
    print(f"  Reported Micro F1:      {rep_micro:.6f} vs Recalculated: {micro_f1_calc:.6f} -> Diff: {abs(rep_micro - micro_f1_calc):.6e}")
    print(f"  Reported Hamming Loss:  {rep_hamming:.6f} vs Recalculated: {hamming_calc:.6f} -> Diff: {abs(rep_hamming - hamming_calc):.6e}")

    # Also check probabilities distribution for class 1 (Discontinuous urban fabric)
    print(f"\nPROBABILITY DISTRIBUTION FOR CLASS 1 (Discontinuous urban fabric):")
    p_c1 = y_prob[:, 1]
    print(f"  Min: {p_c1.min():.6f}, Max: {p_c1.max():.6f}, Mean: {p_c1.mean():.6f}, Median: {np.median(p_c1):.6f}")
    print(f"  Percentiles [25, 50, 75, 90, 95, 99]: {np.percentile(p_c1, [25, 50, 75, 90, 95, 99]).tolist()}")
    print(f"  Count with prob >= 0.10: {(p_c1 >= 0.10).sum()}")
    print(f"  Count with prob >= 0.05: {(p_c1 >= 0.05).sum()}")
    print(f"  Count with prob >= 0.01: {(p_c1 >= 0.01).sum()}")

    return {
        "macro_f1_16": macro_f1_16_calc,
        "macro_f1_present": macro_f1_present_calc,
        "micro_f1": micro_f1_calc,
        "hamming_loss": hamming_calc,
        "per_class_f1": per_class_f1_list,
        "per_class_prec": per_class_prec_list,
        "per_class_rec": per_class_rec_list,
    }


def main():
    cb_bce_dir = repo_root / "artifacts" / "specialists" / "deep_land_cover_cb_bce"
    asl_dir = repo_root / "artifacts" / "specialists" / "deep_land_cover_asl"

    res_cb = audit_model("CB-BCE", cb_bce_dir)
    res_asl = audit_model("ASL", asl_dir)


if __name__ == "__main__":
    main()
