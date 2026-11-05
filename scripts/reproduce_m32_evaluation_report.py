"""
Milestone M3.2: Deep Bitemporal Change-Detection Specialist.
Generates comprehensive comparative audit artifacts and evaluation reports.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# 1. Load FC-Siam-diff and BIT results
fc_path = REPO_ROOT / "artifacts" / "specialists" / "deep_change_detection_fc_siam_diff" / "evaluation_summary.json"
bit_path = REPO_ROOT / "artifacts" / "specialists" / "deep_change_detection_bit" / "evaluation_summary.json"

with open(fc_path, "r", encoding="utf-8") as f:
    fc_data = json.load(f)

with open(bit_path, "r", encoding="utf-8") as f:
    bit_data = json.load(f)

# 2. Load Phase 2 Classical CVA baselines
cva_csv_path = REPO_ROOT / "artifacts" / "benchmarks" / "oscd_per_scene_evaluation_breakdown.csv"
cva_df = pd.read_csv(cva_csv_path)

cva_gaussian = cva_df[
    (cva_df["method"] == "Gaussian-Smoothed CVA") &
    (cva_df["regime"] == "Validation-Tuned (Historical)")
].set_index("scene")
cva_standard = cva_df[
    (cva_df["method"] == "Standard CVA") &
    (cva_df["regime"] == "Validation-Tuned (Historical)")
].set_index("scene")

# Build per-scene comparison table
test_scenes = sorted(fc_data["held_out_test_evaluation"]["validation_tuned_regime"]["per_scene"].keys())

fc_tuned_scenes = fc_data["held_out_test_evaluation"]["validation_tuned_regime"]["per_scene"]
fc_def_scenes = fc_data["held_out_test_evaluation"]["default_threshold_regime"]["per_scene"]

bit_tuned_scenes = bit_data["held_out_test_evaluation"]["validation_tuned_regime"]["per_scene"]
bit_def_scenes = bit_data["held_out_test_evaluation"]["default_threshold_regime"]["per_scene"]

rows = []
for s in test_scenes:
    fc_m = fc_tuned_scenes[s]
    bit_m = bit_tuned_scenes[s]
    cva_g_m = cva_gaussian.loc[s] if s in cva_gaussian.index else {}

    rows.append({
        "scene": s,
        "phase2_cva_gaussian_f1": round(float(cva_g_m.get("f1", 0.0)), 4),
        "phase2_cva_gaussian_prec": round(float(cva_g_m.get("precision", 0.0)), 4),
        "phase2_cva_gaussian_rec": round(float(cva_g_m.get("recall", 0.0)), 4),
        "fc_siam_diff_tuned_f1": fc_m["f1"],
        "fc_siam_diff_tuned_prec": fc_m["precision"],
        "fc_siam_diff_tuned_rec": fc_m["recall"],
        "fc_siam_diff_tuned_iou": fc_m["iou"],
        "bit_tuned_f1": bit_m["f1"],
        "bit_tuned_prec": bit_m["precision"],
        "bit_tuned_rec": bit_m["recall"],
        "bit_tuned_iou": bit_m["iou"],
    })

per_scene_df = pd.DataFrame(rows)
per_scene_csv = REPO_ROOT / "artifacts" / "specialists" / "m32_per_scene_evaluation.csv"
per_scene_df.to_csv(per_scene_csv, index=False)
print(f"Per-scene evaluation CSV saved to: {per_scene_csv}")

# Compile authoritative comparison manifest
manifest = {
    "milestone": "M3.2",
    "milestone_status": "IMPLEMENTATION_AND_EVALUATION_COMPLETE_CERTIFICATION_TARGET_NOT_MET",
    "task": "BITEMPORAL_CHANGE_DETECTION",
    "dataset": "OSCD (Onera Satellite Change Detection)",
    "protocol_freeze": "v3.0.0-freeze-rev21",
    "certification_audit": {
        "frozen_target_metric": "Macro F1 >= 0.50 across 10 held-out test scenes",
        "target_macro_f1": 0.50,
        "achieved_macro_f1": 0.3766,
        "certification_target_met": False,
        "certification_verdict": "FAILED_TARGET_UNMET",
        "scenes_passing_target": 5,
        "scenes_failing_target": 5,
        "passing_scenes": ["aguasclaras (0.5124)", "brasilia (0.7268)", "budapest (0.5462)", "nantes (0.5384)", "pisa (0.5809)"],
        "failing_scenes": ["austin (0.1479)", "beihai (0.0265)", "berlin (0.4671)", "campinas (0.1041)", "mumbai (0.1154)"],
        "threshold_integrity": "Certification threshold (0.50) preserved strictly intact; zero post-hoc relaxation."
    },
    "spatial_partitioning": {
        "dataset_origin": "Official OSCD Benchmark (Daudt et al., 2018) totaling 24 cities",
        "original_partition": "14 training cities, 10 held-out test cities",
        "operational_carveout": "The 14 training cities were partitioned into 10 training cities (for parameter updates) and 4 validation cities (for threshold tuning and checkpoint selection) to guarantee zero test leakage.",
        "train_scenes": 10,
        "val_scenes": 4,
        "held_out_test_scenes": 10,
        "train_cities": ["beirut", "berrida", "bordeaux", "chongqing", "dubai", "hong_kong", "lasvegas", "milano", "rennes", "saclay_w"],
        "val_cities": ["montpellier", "paris", "rio", "valencia"],
        "test_cities": ["aguasclaras", "austin", "beihai", "berlin", "brasilia", "budapest", "campinas", "mumbai", "nantes", "pisa"],
        "disjoint_check": "100% disjoint; no city occurs in multiple partitions.",
        "leakage_prevented": True,
        "tuning_scope": "Strictly 4 validation cities (montpellier, paris, rio, valencia)",
        "held_out_evaluation_rule": "Test split unblinded strictly once for final reporting"
    },
    "primary_comparison_validation_tuned": {
        "phase2_classical_gaussian_cva": {
            "model_family": "Normalized Change Vector Analysis (Gaussian Smoothed)",
            "decision_threshold": 0.28,
            "macro_f1": 0.2731,
            "pooled_f1": 0.3393,
            "pooled_precision": 0.3970,
            "pooled_recall": 0.2963,
            "pooled_iou": 0.2043
        },
        "m32_fc_siam_diff": {
            "model_family": "Fully Convolutional Siamese Difference (Daudt et al., 2018) [Best Evaluated Specialist]",
            "decision_threshold": fc_data["validation_tuning"]["optimal_threshold"],
            "val_macro_f1": fc_data["validation_tuning"]["val_macro_f1_tuned"],
            "training_duration_sec": fc_data["training_metadata"]["training_duration_sec"],
            "training_duration_min": fc_data["training_metadata"]["training_duration_minutes"],
            "best_epoch": fc_data["training_metadata"]["best_epoch"],
            **fc_data["held_out_test_evaluation"]["validation_tuned_regime"]["summary"]
        },
        "m32_bit": {
            "model_family": "Bitemporal Image Transformer (Chen et al., 2021) [Evaluated Underperforming Architecture]",
            "decision_threshold": bit_data["validation_tuning"]["optimal_threshold"],
            "val_macro_f1": bit_data["validation_tuning"]["val_macro_f1_tuned"],
            "training_duration_sec": bit_data["training_metadata"]["training_duration_sec"],
            "training_duration_min": bit_data["training_metadata"]["training_duration_minutes"],
            "best_epoch": bit_data["training_metadata"]["best_epoch"],
            **bit_data["held_out_test_evaluation"]["validation_tuned_regime"]["summary"]
        }
    },
    "secondary_comparison_default_threshold": {
        "m32_fc_siam_diff": {
            "threshold": 0.50,
            **fc_data["held_out_test_evaluation"]["default_threshold_regime"]["summary"]
        },
        "m32_bit": {
            "threshold": 0.50,
            **bit_data["held_out_test_evaluation"]["default_threshold_regime"]["summary"]
        }
    },
    "scientific_findings": {
        "champion_evaluated_model": "FC-Siam-diff",
        "macro_f1_gain_over_classical": round(fc_data["held_out_test_evaluation"]["validation_tuned_regime"]["summary"]["macro_f1"] - 0.2731, 4),
        "pooled_f1_gain_over_classical": round(fc_data["held_out_test_evaluation"]["validation_tuned_regime"]["summary"]["pooled_f1"] - 0.3393, 4),
        "scenes_exceeding_target_50": sum(1 for r in rows if r["fc_siam_diff_tuned_f1"] >= 0.50),
        "total_test_scenes": len(rows),
        "hypotheses_on_bit_underperformance": "Plausible hypothesis: In small-sample regimes (10 training scenes in OSCD), spatial tokenizer cross-attention overfits compact semantic token spaces, whereas convolutional difference skip connections in FC-Siam-diff provide appropriate local pixel differencing inductive bias. This remains a working hypothesis subject to further large-scale pretraining validation."
    }
}

manifest_path = REPO_ROOT / "artifacts" / "specialists" / "m32_change_detection_comparison.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Comparison manifest saved to: {manifest_path}")
