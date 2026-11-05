"""
Automated Test Suite for Milestone M3.2 Change Detection Specialist Evaluation.

Validates:
1. City-level split integrity (10 train, 4 val, 10 held-out test).
2. Threshold-regime consistency across validation-tuned and default fixed regimes.
3. FC-Siam-diff and BIT evaluation manifest validity.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_m32_manifest_structure_and_regimes():
    manifest_path = REPO_ROOT / "artifacts" / "specialists" / "m32_change_detection_comparison.json"
    assert manifest_path.exists(), f"Missing M3.2 comparison manifest: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Split partitioning verification
    sp = data["spatial_partitioning"]
    assert sp["train_scenes"] == 10
    assert sp["val_scenes"] == 4
    assert sp["held_out_test_scenes"] == 10
    assert sp["leakage_prevented"] is True

    # 2. Primary validation-tuned regime verification
    primary = data["primary_comparison_validation_tuned"]
    assert "m32_fc_siam_diff" in primary
    assert "m32_bit" in primary
    assert "phase2_classical_gaussian_cva" in primary

    fc = primary["m32_fc_siam_diff"]
    bit = primary["m32_bit"]

    assert 0.0 < fc["decision_threshold"] < 1.0
    assert 0.0 < bit["decision_threshold"] < 1.0
    assert 0.0 <= fc["macro_f1"] <= 1.0
    assert 0.0 <= fc["pooled_f1"] <= 1.0
    assert 0.0 <= bit["macro_f1"] <= 1.0
    assert 0.0 <= bit["pooled_f1"] <= 1.0

    # Specifically verify FC-Siam-diff performance
    assert fc["macro_f1"] == 0.3766
    assert fc["pooled_f1"] == 0.3818

    # 3. Secondary default-threshold regime verification
    secondary = data["secondary_comparison_default_threshold"]
    assert secondary["m32_fc_siam_diff"]["threshold"] == 0.50
    assert secondary["m32_bit"]["threshold"] == 0.50


def test_m32_per_scene_evaluation_file():
    csv_path = REPO_ROOT / "artifacts" / "specialists" / "m32_per_scene_evaluation.csv"
    assert csv_path.exists(), f"Missing M3.2 per-scene CSV: {csv_path}"

    import pandas as pd
    df = pd.read_csv(csv_path)
    assert len(df) == 10, f"Expected 10 test scenes, found {len(df)}"

    expected_scenes = {
        "aguasclaras", "austin", "beihai", "berlin", "brasilia",
        "budapest", "campinas", "mumbai", "nantes", "pisa"
    }
    assert set(df["scene"]) == expected_scenes
