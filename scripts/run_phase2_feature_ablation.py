"""
Controlled Feature-Family Ablation Study for Phase 2 LightGBM Specialist.

Ablations:
  1. Global moments only (84 features)
  2. Global moments + optical indices (84 + 35 = 119 features)
  3. Global moments + dual-polarization SAR descriptors (84 + 21 = 105 features)
  4. Global moments + gradient statistics (84 + 12 = 96 features)
  5. Global moments + spatial quadrants (84 + 336 = 420 features)
  6. Complete 488-feature model (All features)

Evaluates strictly on identical train (7,180) and val (3,255) partitions.
Reports validation Macro F1 (all 16), validation Macro F1 (15 present), Micro F1, and Hamming loss.
Preserves all frozen production artifacts.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support
import lightgbm as lgb

repo_root = Path(__file__).resolve().parent.parent

# Load data
X_train_full = np.load(repo_root / "data" / "X_train_phase2.npy")
X_val_full = np.load(repo_root / "data" / "X_val_phase2.npy")
y_val = np.load(repo_root / "data" / "y_val_ground_truth.npy")

train_meta = pd.read_parquet(repo_root / "data" / "train_full_metadata.parquet")
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_meta["labels"].apply(list))
classes = list(mlb.classes_)

# Feature index ranges:
# Global moments: [0:84]
# Spatial quadrants: [84:420]
# Optical indices: [420:455]
# SAR descriptors: [455:476]
# Gradient textures: [476:488]

idx_global = list(range(0, 84))
idx_quadrants = list(range(84, 420))
idx_optical = list(range(420, 455))
idx_sar = list(range(455, 476))
idx_texture = list(range(476, 488))

ablation_configs = {
    "1_global_moments_only": {
        "description": "Global moments only (Level 0)",
        "indices": idx_global,
        "n_features": len(idx_global)
    },
    "2_global_plus_optical_indices": {
        "description": "Global moments + Multispectral optical indices",
        "indices": idx_global + idx_optical,
        "n_features": len(idx_global + idx_optical)
    },
    "3_global_plus_sar_descriptors": {
        "description": "Global moments + Dual-polarization SAR descriptors",
        "indices": idx_global + idx_sar,
        "n_features": len(idx_global + idx_sar)
    },
    "4_global_plus_gradient_textures": {
        "description": "Global moments + Gradient texture statistics",
        "indices": idx_global + idx_texture,
        "n_features": len(idx_global + idx_texture)
    },
    "5_global_plus_spatial_quadrants": {
        "description": "Global moments + Level-1 spatial quadrants",
        "indices": idx_global + idx_quadrants,
        "n_features": len(idx_global + idx_quadrants)
    },
    "6_complete_488_features": {
        "description": "Complete 488-feature multimodal architecture",
        "indices": list(range(0, 488)),
        "n_features": 488
    }
}

ablation_results = {}
val_support = y_val.sum(axis=0)
present_mask = val_support > 0

print("=" * 80)
print(f"RUNNING CONTROLLED FEATURE-FAMILY ABLATION EXPERIMENTS (VAL N={len(y_val)})")
print("=" * 80)

for name, cfg in ablation_configs.items():
    feat_idx = cfg["indices"]
    X_tr = X_train_full[:, feat_idx]
    X_va = X_val_full[:, feat_idx]
    
    t0 = time.time()
    n_classes = len(classes)
    val_probs = np.zeros((len(X_va), n_classes), dtype=np.float32)
    
    for c in range(n_classes):
        y_tr_c = y_train[:, c]
        pos_count = max(int(y_tr_c.sum()), 1)
        neg_count = len(y_tr_c) - pos_count
        scale_pos_weight = min(neg_count / pos_count, 15.0)
        
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.08,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=42 + c,
            n_jobs=4,
            verbose=-1
        )
        clf.fit(X_tr, y_tr_c)
        p = clf.predict_proba(X_va)
        val_probs[:, c] = p[:, 1] if p.shape[1] > 1 else 0.0
        
    fit_duration = time.time() - t0
    
    # 1. Evaluate with default global threshold tau = 0.50
    y_pred_def = (val_probs >= 0.50).astype(int)
    micro_def = float(f1_score(y_val, y_pred_def, average="micro", zero_division=0))
    macro_def_all = float(f1_score(y_val, y_pred_def, average="macro", zero_division=0))
    p_arr, r_arr, f1_arr, _ = precision_recall_fscore_support(y_val, y_pred_def, average=None, zero_division=0)
    macro_def_present = float(np.mean(f1_arr[present_mask]))
    hl_def = float(hamming_loss(y_val, y_pred_def))
    
    # 2. Threshold optimization on validation set [0.10, 0.60]
    candidate_thresholds = np.linspace(0.10, 0.60, 51)
    best_thresholds = np.full(n_classes, 0.40, dtype=np.float32)
    for c in range(n_classes):
        if val_support[c] == 0:
            continue
        y_c = y_val[:, c]
        p_c = val_probs[:, c]
        best_f1 = -1.0
        best_t = 0.40
        for t in candidate_thresholds:
            s = f1_score(y_c, (p_c >= t).astype(int), average="binary", zero_division=0)
            if s > best_f1:
                best_f1 = s
                best_t = t
        best_thresholds[c] = round(best_t, 3)
        
    y_pred_tuned = (val_probs >= best_thresholds).astype(int)
    micro_tuned = float(f1_score(y_val, y_pred_tuned, average="micro", zero_division=0))
    macro_tuned_all = float(f1_score(y_val, y_pred_tuned, average="macro", zero_division=0))
    p_t_arr, r_t_arr, f1_t_arr, _ = precision_recall_fscore_support(y_val, y_pred_tuned, average=None, zero_division=0)
    macro_tuned_present = float(np.mean(f1_t_arr[present_mask]))
    hl_tuned = float(hamming_loss(y_val, y_pred_tuned))
    
    ablation_results[name] = {
        "description": cfg["description"],
        "n_features": cfg["n_features"],
        "fit_time_seconds": round(fit_duration, 2),
        "val_default_tau_0_50": {
            "micro_f1": round(micro_def * 100, 2),
            "macro_f1_all_16": round(macro_def_all * 100, 2),
            "macro_f1_present_15": round(macro_def_present * 100, 2),
            "hamming_loss": round(hl_def, 4)
        },
        "val_tuned_thresholds": {
            "micro_f1": round(micro_tuned * 100, 2),
            "macro_f1_all_16": round(macro_tuned_all * 100, 2),
            "macro_f1_present_15": round(macro_tuned_present * 100, 2),
            "hamming_loss": round(hl_tuned, 4)
        }
    }
    
    print(f"[{name}] {cfg['description']} ({cfg['n_features']} feats, {fit_duration:.1f}s):")
    print(f"   Default tau=0.50: Macro F1(all)={macro_def_all*100:.2f}%, Macro F1(present)={macro_def_present*100:.2f}%, Micro={micro_def*100:.2f}%, HL={hl_def:.4f}")
    print(f"   Tuned thresholds: Macro F1(all)={macro_tuned_all*100:.2f}%, Macro F1(present)={macro_tuned_present*100:.2f}%, Micro={micro_tuned*100:.2f}%, HL={hl_tuned:.4f}")

out_path = repo_root / "artifacts" / "specialists" / "land_cover_phase2" / "ablation_study.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(ablation_results, f, indent=2)

print(f"\nAblation study results successfully saved to: {out_path}")
