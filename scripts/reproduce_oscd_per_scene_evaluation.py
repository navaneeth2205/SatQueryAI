"""
Per-scene evaluation and radiometric suppression audit on OSCD 10 held-out test scenes.
Evaluates:
- Standard CVA (Baseline)
- Radiometrically Aligned CVA
- Gaussian-Smoothed CVA
- Dual-Threshold Hysteresis CVA

Computes:
- Per-scene TP, FP, FN, TN, Precision, Recall, F1, and IoU for each scene
- Pooled aggregate metrics across all 10 scenes
- Radiometric alignment impact analysis: does mean/std matching suppress genuine change?
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "oscd"


def load_scene(path: Path):
    with Image.open(path / "t1.png") as im:
        t1 = np.array(im).astype(np.float32) / 255.0
    with Image.open(path / "t2.png") as im:
        t2 = np.array(im).astype(np.float32) / 255.0
    with Image.open(path / "cm.png") as im:
        mask = (np.array(im) > 0).astype(np.uint8)
    return t1, t2, mask


def calc_metrics(pred: np.ndarray, gt: np.ndarray):
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4)
    }


def run_cva(t1, t2, tau=0.28, min_area=9):
    diff = np.abs(t2 - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    binary = (rho >= tau)
    if min_area > 1 and np.any(binary):
        labeled, num = scipy.ndimage.label(binary)
        if num > 0:
            counts = np.bincount(labeled.ravel())
            remove = counts < min_area
            remove[0] = False
            binary = binary & (~remove[labeled])
    return binary.astype(np.uint8)


def run_aligned_cva(t1, t2, tau=0.28, min_area=9):
    t2_aligned = np.zeros_like(t2)
    for c in range(3):
        m1, s1 = np.mean(t1[..., c]), np.std(t1[..., c])
        m2, s2 = np.mean(t2[..., c]), np.std(t2[..., c])
        if s2 > 1e-6:
            t2_aligned[..., c] = (t2[..., c] - m2) * (s1 / s2) + m1
        else:
            t2_aligned[..., c] = t2[..., c]
    t2_aligned = np.clip(t2_aligned, 0.0, 1.0)
    return run_cva(t1, t2_aligned, tau=tau, min_area=min_area)


def run_gaussian_cva(t1, t2, tau=0.26, min_area=9, sigma=1.2):
    diff = np.abs(t2 - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    rho_smooth = scipy.ndimage.gaussian_filter(rho, sigma=sigma)
    binary = (rho_smooth >= tau)
    if min_area > 1 and np.any(binary):
        labeled, num = scipy.ndimage.label(binary)
        if num > 0:
            counts = np.bincount(labeled.ravel())
            remove = counts < min_area
            remove[0] = False
            binary = binary & (~remove[labeled])
    return binary.astype(np.uint8)


def run_hysteresis_cva(t1, t2, tau_high=0.28, tau_low=0.14, min_area=9):
    diff = np.abs(t2 - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    high_mask = (rho >= tau_high)
    low_mask = (rho >= tau_low)

    labeled_low, num_low = scipy.ndimage.label(low_mask)
    if num_low == 0 or not np.any(high_mask):
        return np.zeros_like(high_mask, dtype=np.uint8)

    labels_with_seeds = np.unique(labeled_low[high_mask])
    labels_with_seeds = labels_with_seeds[labels_with_seeds > 0]
    binary = np.isin(labeled_low, labels_with_seeds)

    if min_area > 1 and np.any(binary):
        labeled, num = scipy.ndimage.label(binary)
        if num > 0:
            counts = np.bincount(labeled.ravel())
            remove = counts < min_area
            remove[0] = False
            binary = binary & (~remove[labeled])
    return binary.astype(np.uint8)


def main():
    print("=" * 85)
    print("   OSCD CHANGE DETECTION: PER-SCENE EVALUATION & RADIOMETRIC ANALYSIS")
    print("=" * 85)

    test_dir = DATA_DIR / "test"
    test_scenes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    print(f"Loaded {len(test_scenes)} held-out test scenes: {test_scenes}")

    methods = [
        ("Standard CVA", run_cva),
        ("Radiometrically Aligned CVA", run_aligned_cva),
        ("Gaussian-Smoothed CVA", run_gaussian_cva),
        ("Dual-Threshold Hysteresis CVA", run_hysteresis_cva)
    ]

    all_scene_records = []
    pooled_records = []

    for method_name, method_fn in methods:
        pooled_tp = 0
        pooled_fp = 0
        pooled_fn = 0
        pooled_tn = 0

        for scene in test_scenes:
            t1, t2, gt = load_scene(test_dir / scene)
            pred = method_fn(t1, t2)
            m = calc_metrics(pred, gt)

            pooled_tp += m["tp"]
            pooled_fp += m["fp"]
            pooled_fn += m["fn"]
            pooled_tn += m["tn"]

            all_scene_records.append({
                "method": method_name,
                "scene": scene,
                "gt_change_pixels": int(np.sum(gt)),
                "gt_change_pct": round(float(np.mean(gt) * 100), 2),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "iou": m["iou"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"]
            })

        p_prec = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
        p_rec = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
        p_f1 = 2 * p_prec * p_rec / (p_prec + p_rec) if (p_prec + p_rec) > 0 else 0.0
        p_iou = pooled_tp / (pooled_tp + pooled_fp + pooled_fn) if (pooled_tp + pooled_fp + pooled_fn) > 0 else 0.0

        pooled_records.append({
            "method": method_name,
            "pooled_precision": round(p_prec, 4),
            "pooled_recall": round(p_rec, 4),
            "pooled_f1": round(p_f1, 4),
            "pooled_iou": round(p_iou, 4),
            "total_tp": pooled_tp,
            "total_fp": pooled_fp,
            "total_fn": pooled_fn,
            "total_tn": pooled_tn
        })

    df_scenes = pd.DataFrame(all_scene_records)
    df_pooled = pd.DataFrame(pooled_records)

    print("\n--- POOLED METRICS ACROSS ALL 10 SCENES ---")
    print(df_pooled[["method", "pooled_precision", "pooled_recall", "pooled_f1", "pooled_iou", "total_tp", "total_fn", "total_fp"]].to_string(index=False))

    print("\n--- PER-SCENE F1 BREAKDOWN ACROSS ALL 10 TEST SCENES ---")
    pivot_f1 = df_scenes.pivot(index="scene", columns="method", values="f1")
    print(pivot_f1.to_string())

    print("\n--- PER-SCENE IoU BREAKDOWN ACROSS ALL 10 TEST SCENES ---")
    pivot_iou = df_scenes.pivot(index="scene", columns="method", values="iou")
    print(pivot_iou.to_string())

    # Save CSV artifacts
    csv_scenes = REPO_ROOT / "artifacts" / "benchmarks" / "oscd_per_scene_evaluation_breakdown.csv"
    csv_pooled = REPO_ROOT / "artifacts" / "benchmarks" / "oscd_methods_pooled_comparison.csv"
    df_scenes.to_csv(csv_scenes, index=False)
    df_pooled.to_csv(csv_pooled, index=False)

    print(f"\nSaved per-scene records to: {csv_scenes.relative_to(REPO_ROOT)}")
    print(f"Saved pooled records to: {csv_pooled.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
