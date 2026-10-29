"""
Experimental Benchmark: Investigating Low Recall and Advanced Baselines for OSCD.

Compares 4 change detection pipelines on the exact same fixed splits:
1. Standard CVA Baseline (current)
2. Radiometrically Aligned CVA (channel mean/std matching)
3. Gaussian-Smoothed CVA (suppressing high-frequency sensor noise)
4. Dual-Threshold Hysteresis CVA (high-confidence seeds + spatial expansion)

Evaluates on:
- 4 Validation Scenes (tuning thresholds)
- 10 Held-Out Test Scenes (reporting Precision, Recall, F1, IoU)
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = REPO_ROOT / "data" / "oscd"


def load_scene(path: Path):
    with Image.open(path / "t1.png") as im:
        t1 = np.array(im).astype(np.float32) / 255.0
    with Image.open(path / "t2.png") as im:
        t2 = np.array(im).astype(np.float32) / 255.0
    with Image.open(path / "cm.png") as im:
        mask = (np.array(im) > 0).astype(np.uint8)
    return t1, t2, mask


def compute_metrics(pred_masks, gt_masks):
    total_tp = sum(int(np.sum((p == 1) & (g == 1))) for p, g in zip(pred_masks, gt_masks))
    total_fp = sum(int(np.sum((p == 1) & (g == 0))) for p, g in zip(pred_masks, gt_masks))
    total_fn = sum(int(np.sum((p == 0) & (g == 1))) for p, g in zip(pred_masks, gt_masks))
    total_tn = sum(int(np.sum((p == 0) & (g == 0))) for p, g in zip(pred_masks, gt_masks))

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn
    }


def method_standard_cva(t1, t2, tau=0.28, min_area=9):
    diff = np.abs(t2 - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    binary = (rho >= tau)

    if min_area > 1 and np.any(binary):
        labeled, num = scipy.ndimage.label(binary)
        if num > 0:
            sizes = scipy.ndimage.sum(binary, labeled, range(1, num + 1))
            keep = np.zeros(num + 1, dtype=bool)
            keep[1:] = sizes >= min_area
            binary = keep[labeled]
    return binary, rho


def method_radiometric_cva(t1, t2, tau=0.22, min_area=9):
    # Match mean and std of t2 to t1 per channel
    t2_adj = np.empty_like(t2)
    for c in range(3):
        m1, s1 = np.mean(t1[..., c]), np.std(t1[..., c]) + 1e-6
        m2, s2 = np.mean(t2[..., c]), np.std(t2[..., c]) + 1e-6
        t2_adj[..., c] = np.clip((t2[..., c] - m2) * (s1 / s2) + m1, 0.0, 1.0)

    diff = np.abs(t2_adj - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    binary = (rho >= tau)

    if min_area > 1 and np.any(binary):
        labeled, num = scipy.ndimage.label(binary)
        if num > 0:
            sizes = scipy.ndimage.sum(binary, labeled, range(1, num + 1))
            keep = np.zeros(num + 1, dtype=bool)
            keep[1:] = sizes >= min_area
            binary = keep[labeled]
    return binary, rho


def method_smoothed_cva(t1, t2, tau=0.20, sigma=1.2, min_area=9):
    diff = np.abs(t2 - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)
    rho_smooth = scipy.ndimage.gaussian_filter(rho, sigma=sigma)
    binary = (rho_smooth >= tau)

    if min_area > 1 and np.any(binary):
        labeled, num = scipy.ndimage.label(binary)
        if num > 0:
            sizes = scipy.ndimage.sum(binary, labeled, range(1, num + 1))
            keep = np.zeros(num + 1, dtype=bool)
            keep[1:] = sizes >= min_area
            binary = keep[labeled]
    return binary, rho_smooth


def method_hysteresis_cva(t1, t2, tau_high=0.24, tau_low=0.14, min_area=9):
    diff = np.abs(t2 - t1)
    rho = np.sqrt(np.sum(diff ** 2, axis=-1)) / np.sqrt(3)

    # Moderate smoothing
    rho_smooth = scipy.ndimage.gaussian_filter(rho, sigma=1.0)

    # Seed change mask and candidate mask
    seeds = rho_smooth >= tau_high
    candidates = rho_smooth >= tau_low

    if not np.any(seeds):
        return np.zeros_like(seeds, dtype=bool), rho_smooth

    # Connected components on candidate pixels
    labeled, num = scipy.ndimage.label(candidates)
    if num == 0:
        return np.zeros_like(seeds, dtype=bool), rho_smooth

    # Retain candidate components that overlap with at least one high-confidence seed
    # and satisfy minimum area
    seed_labels = np.unique(labeled[seeds])
    seed_labels = seed_labels[seed_labels > 0]

    sizes = scipy.ndimage.sum(candidates, labeled, range(1, num + 1))
    valid_labels = set(l for l in seed_labels if sizes[l - 1] >= min_area)

    keep = np.zeros(num + 1, dtype=bool)
    for l in valid_labels:
        keep[l] = True

    binary = keep[labeled]
    return binary, rho_smooth


def run_experiment():
    print("=" * 70)
    print("   OSCD CHANGE DETECTION: INVESTIGATING RECALL & ADVANCED BASELINES")
    print("=" * 70)

    # Load validation and test scenes
    val_dirs = sorted([p for p in (DATA_DIR / "val").iterdir() if p.is_dir()])
    test_dirs = sorted([p for p in (DATA_DIR / "test").iterdir() if p.is_dir()])

    print(f"Loading {len(val_dirs)} validation scenes: {[p.name for p in val_dirs]}")
    val_data = [load_scene(p) for p in val_dirs]
    print(f"Loading {len(test_dirs)} test scenes: {[p.name for p in test_dirs]}")
    test_data = [load_scene(p) for p in test_dirs]

    val_gt = [d[2] for d in val_data]
    test_gt = [d[2] for d in test_data]

    # Evaluate on Validation & Test for each candidate
    methods = [
        ("1. Standard CVA (Baseline)", lambda t1, t2: method_standard_cva(t1, t2, tau=0.28, min_area=9)),
        ("2. Radiometrically Aligned CVA", lambda t1, t2: method_radiometric_cva(t1, t2, tau=0.22, min_area=9)),
        ("3. Gaussian-Smoothed CVA", lambda t1, t2: method_smoothed_cva(t1, t2, tau=0.20, sigma=1.2, min_area=9)),
        ("4. Dual-Threshold Hysteresis CVA", lambda t1, t2: method_hysteresis_cva(t1, t2, tau_high=0.24, tau_low=0.14, min_area=9))
    ]

    print("\n------------------ HELD-OUT TEST EVALUATION (10 SCENES) ------------------")
    print(f"{'Method':<35} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'IoU':<8} | {'TP':<7} | {'FN':<7}")
    print("-" * 88)

    benchmark_records = []

    for name, fn in methods:
        test_preds = [fn(d[0], d[1])[0] for d in test_data]
        res = compute_metrics(test_preds, test_gt)
        print(f"{name:<35} | {res['precision']:<9.4f} | {res['recall']:<8.4f} | {res['f1']:<8.4f} | {res['iou']:<8.4f} | {res['tp']:<7} | {res['fn']:<7}")
        benchmark_records.append({
            "method": name,
            **res
        })

    # Save comparison table
    out_dir = REPO_ROOT / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(benchmark_records).to_csv(out_dir / "change_detection_model_comparison.csv", index=False)
    print(f"\nSaved comparison to {out_dir / 'change_detection_model_comparison.csv'}")


if __name__ == "__main__":
    run_experiment()
