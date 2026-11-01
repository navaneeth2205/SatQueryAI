"""
Comprehensive OSCD Change Detection Evaluator & Multi-Regime Reconciliation Script.

Evaluates and reconciles:
1. Regime A: Validation-Tuned Thresholds (Historical Benchmark from experiment_change_detection.py)
   - Standard CVA: tau=0.28, min_area=9
   - Radiometric CVA: tau=0.22, min_area=9
   - Gaussian-Smoothed CVA: tau=0.20, sigma=1.2, min_area=9
   - Dual-Threshold Hysteresis CVA: tau_high=0.24, tau_low=0.14, sigma=1.0, min_area=9

2. Regime B: Controlled Fixed-Threshold Ablation (Ablation Benchmark)
   - Standard CVA: tau=0.28, min_area=9
   - Radiometric CVA: tau=0.28, min_area=9
   - Gaussian-Smoothed CVA: tau=0.26, sigma=1.2, min_area=9
   - Dual-Threshold Hysteresis CVA: tau_high=0.28, tau_low=0.14, min_area=9 (raw rho)

Outputs per-scene metrics across all 10 held-out test scenes and pooled metrics for both regimes.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "oscd"


def load_scene(scene_dir: Path):
    with Image.open(scene_dir / "t1.png") as im:
        t1 = np.array(im).astype(np.float32) / 255.0
    with Image.open(scene_dir / "t2.png") as im:
        t2 = np.array(im).astype(np.float32) / 255.0
    with Image.open(scene_dir / "cm.png") as im:
        mask = (np.array(im) > 0).astype(np.uint8)
    return t1, t2, mask


def filter_min_area(binary: np.ndarray, min_area: int = 9):
    if min_area <= 1 or not np.any(binary):
        return binary.astype(np.uint8)
    labeled, num = scipy.ndimage.label(binary)
    if num == 0:
        return binary.astype(np.uint8)
    sizes = scipy.ndimage.sum(binary, labeled, range(1, num + 1))
    keep = np.zeros(num + 1, dtype=bool)
    keep[1:] = sizes >= min_area
    return keep[labeled].astype(np.uint8)


# --- METHOD DEFINITIONS ---

def method_standard_cva(t1, t2, tau=0.28, min_area=9):
    diff = np.abs(t2 - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    binary = (rho >= tau)
    return filter_min_area(binary, min_area)


def method_radiometric_cva(t1, t2, tau=0.22, min_area=9):
    t2_adj = np.empty_like(t2)
    for c in range(3):
        m1, s1 = np.mean(t1[..., c]), np.std(t1[..., c]) + 1e-6
        m2, s2 = np.mean(t2[..., c]), np.std(t2[..., c]) + 1e-6
        t2_adj[..., c] = np.clip((t2[..., c] - m2) * (s1 / s2) + m1, 0.0, 1.0)
    diff = np.abs(t2_adj - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    binary = (rho >= tau)
    return filter_min_area(binary, min_area)


def method_smoothed_cva(t1, t2, tau=0.20, sigma=1.2, min_area=9):
    diff = np.abs(t2 - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    rho_smooth = scipy.ndimage.gaussian_filter(rho, sigma=sigma)
    binary = (rho_smooth >= tau)
    return filter_min_area(binary, min_area)


def method_hysteresis_cva(t1, t2, tau_high=0.24, tau_low=0.14, sigma=1.0, min_area=9):
    diff = np.abs(t2 - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    if sigma > 0:
        rho = scipy.ndimage.gaussian_filter(rho, sigma=sigma)

    seeds = (rho >= tau_high)
    candidates = (rho >= tau_low)

    if not np.any(seeds):
        return np.zeros_like(seeds, dtype=np.uint8)

    labeled, num = scipy.ndimage.label(candidates)
    if num == 0:
        return np.zeros_like(seeds, dtype=np.uint8)

    seed_labels = np.unique(labeled[seeds])
    seed_labels = seed_labels[seed_labels > 0]

    sizes = scipy.ndimage.sum(candidates, labeled, range(1, num + 1))
    valid_labels = set(l for l in seed_labels if sizes[l - 1] >= min_area)

    keep = np.zeros(num + 1, dtype=bool)
    for l in valid_labels:
        keep[l] = True

    binary = keep[labeled]
    return binary.astype(np.uint8)


def evaluate_regime(scenes, methods_dict, regime_name):
    scene_records = []
    pooled_records = []

    for method_name, fn in methods_dict.items():
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

        for s in scenes:
            t1, t2, gt = s["t1"], s["t2"], s["gt"]
            pred = fn(t1, t2)

            tp = int(np.sum((pred == 1) & (gt == 1)))
            fp = int(np.sum((pred == 1) & (gt == 0)))
            fn_cnt = int(np.sum((pred == 0) & (gt == 1)))
            tn = int(np.sum((pred == 0) & (gt == 0)))

            total_tp += tp
            total_fp += fp
            total_fn += fn_cnt
            total_tn += tn

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn_cnt) if (tp + fn_cnt) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            iou = tp / (tp + fp + fn_cnt) if (tp + fp + fn_cnt) > 0 else 0.0

            scene_records.append({
                "regime": regime_name,
                "method": method_name,
                "scene": s["name"],
                "gt_change_pixels": int(np.sum(gt)),
                "gt_change_pct": round(float(np.mean(gt)) * 100, 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "iou": round(iou, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn_cnt,
                "tn": tn
            })

        pooled_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        pooled_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        pooled_f1 = 2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec) if (pooled_prec + pooled_rec) > 0 else 0.0
        pooled_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        pooled_records.append({
            "regime": regime_name,
            "method": method_name,
            "pooled_precision": round(pooled_prec, 4),
            "pooled_recall": round(pooled_rec, 4),
            "pooled_f1": round(pooled_f1, 4),
            "pooled_iou": round(pooled_iou, 4),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_tn": total_tn
        })

    return pd.DataFrame(scene_records), pd.DataFrame(pooled_records)


def main():
    print("=" * 85)
    print("   OSCD CHANGE DETECTION: MULTI-REGIME RECONCILIATION & PER-SCENE EVALUATION")
    print("=" * 85)

    test_dirs = sorted([p for p in (DATA_DIR / "test").iterdir() if p.is_dir()])
    print(f"Loaded {len(test_dirs)} held-out test scenes: {[p.name for p in test_dirs]}")

    scenes = []
    for p in test_dirs:
        t1, t2, gt = load_scene(p)
        scenes.append({"name": p.name, "t1": t1, "t2": t2, "gt": gt})

    # Regime A: Historical Validation-Tuned Thresholds
    regime_a_methods = {
        "Standard CVA": lambda t1, t2: method_standard_cva(t1, t2, tau=0.28, min_area=9),
        "Radiometrically Aligned CVA": lambda t1, t2: method_radiometric_cva(t1, t2, tau=0.22, min_area=9),
        "Gaussian-Smoothed CVA": lambda t1, t2: method_smoothed_cva(t1, t2, tau=0.20, sigma=1.2, min_area=9),
        "Dual-Threshold Hysteresis CVA": lambda t1, t2: method_hysteresis_cva(t1, t2, tau_high=0.24, tau_low=0.14, sigma=1.0, min_area=9)
    }

    # Regime B: Controlled Fixed-Threshold Ablation
    regime_b_methods = {
        "Standard CVA": lambda t1, t2: method_standard_cva(t1, t2, tau=0.28, min_area=9),
        "Radiometrically Aligned CVA": lambda t1, t2: method_radiometric_cva(t1, t2, tau=0.28, min_area=9),
        "Gaussian-Smoothed CVA": lambda t1, t2: method_smoothed_cva(t1, t2, tau=0.26, sigma=1.2, min_area=9),
        "Dual-Threshold Hysteresis CVA": lambda t1, t2: method_hysteresis_cva(t1, t2, tau_high=0.28, tau_low=0.14, sigma=0.0, min_area=9)
    }

    df_scene_a, df_pooled_a = evaluate_regime(scenes, regime_a_methods, "Validation-Tuned (Historical)")
    df_scene_b, df_pooled_b = evaluate_regime(scenes, regime_b_methods, "Fixed-Threshold Ablation")

    print("\n--- REGIME A: VALIDATION-TUNED THRESHOLDS (HISTORICAL BENCHMARK) ---")
    print(df_pooled_a[["method", "pooled_precision", "pooled_recall", "pooled_f1", "pooled_iou", "total_tp", "total_fn", "total_fp"]].to_string(index=False))

    print("\n--- REGIME B: CONTROLLED FIXED-THRESHOLD ABLATION ---")
    print(df_pooled_b[["method", "pooled_precision", "pooled_recall", "pooled_f1", "pooled_iou", "total_tp", "total_fn", "total_fp"]].to_string(index=False))

    # Save outputs
    out_dir = REPO_ROOT / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_combined_pooled = pd.concat([df_pooled_a, df_pooled_b], ignore_index=True)
    df_combined_scene = pd.concat([df_scene_a, df_scene_b], ignore_index=True)

    df_combined_pooled.to_csv(out_dir / "oscd_methods_pooled_comparison.csv", index=False)
    df_combined_scene.to_csv(out_dir / "oscd_per_scene_evaluation_breakdown.csv", index=False)

    df_pooled_a.to_csv(out_dir / "oscd_historical_validation_tuned_pooled.csv", index=False)
    df_scene_a.to_csv(out_dir / "oscd_historical_validation_tuned_per_scene.csv", index=False)
    df_pooled_b.to_csv(out_dir / "oscd_fixed_threshold_ablation_pooled.csv", index=False)
    df_scene_b.to_csv(out_dir / "oscd_fixed_threshold_ablation_per_scene.csv", index=False)

    print(f"\nSaved combined pooled records to: {out_dir / 'oscd_methods_pooled_comparison.csv'}")
    print(f"Saved combined per-scene records to: {out_dir / 'oscd_per_scene_evaluation_breakdown.csv'}")


if __name__ == "__main__":
    main()
