"""
Validation Threshold Sweep & Held-Out Test Evaluation for Change Detection.
Step 3 & 4 of Milestone 3.

Evaluates the TemporalDifferenceBaseline:
1. Sweeps decision threshold tau on the 4 validation scenes to maximize F1 against ground-truth change masks.
2. Freezes tau*.
3. Evaluates on the 10 held-out test scenes, reporting Precision, Recall, F1, and IoU.
4. Serializes threshold configuration and comprehensive evaluation summary.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
import sys
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from satquery.specialists.change_detection import TemporalDifferenceBaseline
from satquery.utils.temporal import validate_temporal_pair

DATA_DIR = REPO_ROOT / "data" / "oscd"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "specialists" / "change_detection"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_scene_data(scene_dir: Path):
    """Load t1, t2, and ground-truth mask for an OSCD scene."""
    t1_path = scene_dir / "t1.png"
    t2_path = scene_dir / "t2.png"
    mask_path = scene_dir / "cm.png"

    with Image.open(t1_path) as im:
        t1 = np.array(im).astype(np.float32)
    with Image.open(t2_path) as im:
        t2 = np.array(im).astype(np.float32)
    with Image.open(mask_path) as im:
        mask = (np.array(im) > 0).astype(np.uint8)

    val = validate_temporal_pair(t1, t2)
    return val, mask


def evaluate_cva_threshold(val_scenes, threshold: float):
    """Compute total TP, FP, FN, TN across scenes for a candidate threshold."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    baseline = TemporalDifferenceBaseline(decision_threshold=threshold)

    for val, gt_mask in val_scenes:
        _, pred_mask, _ = baseline.compute_change_map(val.t1_array, val.t2_array, val.valid_mask)
        gt_valid = gt_mask[val.valid_mask]
        pred_valid = pred_mask[val.valid_mask]

        tp = int(np.sum((pred_valid == 1) & (gt_valid == 1)))
        fp = int(np.sum((pred_valid == 1) & (gt_valid == 0)))
        fn = int(np.sum((pred_valid == 0) & (gt_valid == 1)))
        tn = int(np.sum((pred_valid == 0) & (gt_valid == 0)))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) > 0 else 0.0

    return {
        "threshold": round(threshold, 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn
    }


def run_pipeline():
    print("=" * 60)
    print("[Step 3 & 4] THRESHOLD SELECTION & HELD-OUT TEST EVALUATION")
    print("=" * 60)

    # 1. Load validation scenes
    val_dir = DATA_DIR / "val"
    val_scene_names = sorted([p.name for p in val_dir.iterdir() if p.is_dir()])
    print(f"\nLoaded {len(val_scene_names)} validation scenes: {val_scene_names}")

    val_scenes = []
    for name in val_scene_names:
        val, mask = load_scene_data(val_dir / name)
        val_scenes.append((val, mask))

    # 2. Sweep threshold on validation scenes
    print("\nRunning threshold sweep on validation data (objective: maximize F1)...")
    threshold_candidates = np.arange(0.06, 0.40, 0.02)
    sweep_results = []
    best_f1 = -1.0
    best_res = None

    for tau in threshold_candidates:
        res = evaluate_cva_threshold(val_scenes, float(tau))
        sweep_results.append(res)
        print(f"  tau={res['threshold']:5.2f} -> Prec={res['precision']:.4f}, Rec={res['recall']:.4f}, F1={res['f1']:.4f}, IoU={res['iou']:.4f}")
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_res = res

    optimal_threshold = float(best_res["threshold"])
    print("\n" + "-" * 50)
    print(f"Optimal Validation Threshold: tau* = {optimal_threshold:.2f}")
    print(f"  Validation Precision: {best_res['precision']:.4f}")
    print(f"  Validation Recall   : {best_res['recall']:.4f}")
    print(f"  Validation F1       : {best_res['f1']:.4f}")
    print(f"  Validation IoU      : {best_res['iou']:.4f}")
    print("-" * 50)

    # 3. Freeze optimal threshold and evaluate on held-out test scenes
    test_dir = DATA_DIR / "test"
    test_scene_names = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])
    print(f"\nEvaluating frozen threshold tau*={optimal_threshold:.2f} on {len(test_scene_names)} held-out test scenes...")

    frozen_baseline = TemporalDifferenceBaseline(decision_threshold=optimal_threshold)
    scene_breakdown = []
    total_test_tp = 0
    total_test_fp = 0
    total_test_fn = 0
    total_test_tn = 0

    for name in test_scene_names:
        val, gt_mask = load_scene_data(test_dir / name)
        _, pred_mask, mean_deltas = frozen_baseline.compute_change_map(val.t1_array, val.t2_array, val.valid_mask)
        gt_valid = gt_mask[val.valid_mask]
        pred_valid = pred_mask[val.valid_mask]

        tp = int(np.sum((pred_valid == 1) & (gt_valid == 1)))
        fp = int(np.sum((pred_valid == 1) & (gt_valid == 0)))
        fn = int(np.sum((pred_valid == 0) & (gt_valid == 1)))
        tn = int(np.sum((pred_valid == 0) & (gt_valid == 0)))

        total_test_tp += tp
        total_test_fp += fp
        total_test_fn += fn
        total_test_tn += tn

        sc_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        sc_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        sc_f1 = float(2 * sc_prec * sc_rec / (sc_prec + sc_rec)) if (sc_prec + sc_rec) > 0 else 0.0
        sc_iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

        sc_meta = {
            "scene_name": name,
            "width": val.width,
            "height": val.height,
            "valid_pixels": val.valid_pixels,
            "ground_truth_change_pct": round(float(np.sum(gt_valid) / val.valid_pixels * 100), 2),
            "predicted_change_pct": round(float(np.sum(pred_valid) / val.valid_pixels * 100), 2),
            "precision": round(sc_prec, 4),
            "recall": round(sc_rec, 4),
            "f1": round(sc_f1, 4),
            "iou": round(sc_iou, 4),
            "mean_channel_deltas": mean_deltas
        }
        scene_breakdown.append(sc_meta)
        print(f"  [TEST] {name:15s}: Prec={sc_prec:.4f}, Rec={sc_rec:.4f}, F1={sc_f1:.4f}, IoU={sc_iou:.4f} (GT={sc_meta['ground_truth_change_pct']}%, Pred={sc_meta['predicted_change_pct']}%)")

    test_prec = float(total_test_tp / (total_test_tp + total_test_fp)) if (total_test_tp + total_test_fp) > 0 else 0.0
    test_rec = float(total_test_tp / (total_test_tp + total_test_fn)) if (total_test_tp + total_test_fn) > 0 else 0.0
    test_f1 = float(2 * test_prec * test_rec / (test_prec + test_rec)) if (test_prec + test_rec) > 0 else 0.0
    test_iou = float(total_test_tp / (total_test_tp + total_test_fp + total_test_fn)) if (total_test_tp + total_test_fp + total_test_fn) > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"HELD-OUT TEST AGGREGATE RESULTS (Frozen tau*={optimal_threshold:.2f}):")
    print(f"  Aggregate Precision : {test_prec:.4f}")
    print(f"  Aggregate Recall    : {test_rec:.4f}")
    print(f"  Aggregate F1-Score  : {test_f1:.4f}")
    print(f"  Aggregate IoU       : {test_iou:.4f}")
    print("=" * 60)

    # 4. Save threshold configuration
    thresh_cfg = {
        "decision_threshold": optimal_threshold,
        "min_component_area": 9,
        "selection_objective": "Maximized pixel-level F1 on validation split",
        "validation_scenes": val_scene_names,
        "validation_metrics": best_res,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    with open(ARTIFACTS_DIR / "threshold_config.json", "w", encoding="utf-8") as f:
        json.dump(thresh_cfg, f, indent=2)

    # 5. Save evaluation summary
    eval_summary = {
        "model_name": "TemporalDifferenceBaseline",
        "model_family": "temporal_difference_baseline",
        "benchmark_dataset": "OSCD (Onera Satellite Change Detection) RGB",
        "frozen_decision_threshold": optimal_threshold,
        "min_component_area_px": 9,
        "overall_test_metrics": {
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1": round(test_f1, 4),
            "iou": round(test_iou, 4),
            "total_test_pixels": sum(s["valid_pixels"] for s in scene_breakdown),
            "total_test_tp": total_test_tp,
            "total_test_fp": total_test_fp,
            "total_test_fn": total_test_fn,
            "total_test_tn": total_test_tn
        },
        "test_scenes_count": len(scene_breakdown),
        "scene_breakdown": scene_breakdown,
        "dataset_provenance": {
            "source": "OSCD (Daudt et al., 2018; blanchon/OSCD_RGB on HuggingFace)",
            "split_method": "Scene/site-level separation (zero spatial leakage)",
            "validation_scenes": val_scene_names,
            "test_scenes": test_scene_names
        },
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(ARTIFACTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nSaved threshold config to: {ARTIFACTS_DIR / 'threshold_config.json'}")
    print(f"Saved evaluation summary to: {ARTIFACTS_DIR / 'evaluation_summary.json'}")


if __name__ == "__main__":
    run_pipeline()
