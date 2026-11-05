"""
Automated Test Suite for Milestone M3.2 Change Detection Specialist Evaluation.

Validates:
1. City-level split integrity (10 train, 4 val, 10 held-out test).
2. Threshold-regime consistency across validation-tuned and default fixed regimes.
3. Certification status verification (Macro F1 = 0.3766 < 0.50 target; target unmet).
4. FC-Siam-diff and BIT evaluation manifest validity and arithmetic reproduction.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_m32_manifest_structure_and_regimes():
    manifest_path = REPO_ROOT / "artifacts" / "specialists" / "m32_change_detection_comparison.json"
    assert manifest_path.exists(), f"Missing M3.2 comparison manifest: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Certification Audit Verification
    cert = data["certification_audit"]
    assert cert["target_macro_f1"] == 0.50
    assert cert["achieved_macro_f1"] == 0.3766
    assert cert["certification_target_met"] is False
    assert cert["certification_verdict"] == "FAILED_TARGET_UNMET"
    assert cert["scenes_passing_target"] == 5
    assert cert["scenes_failing_target"] == 5

    # 2. Split Partitioning Verification
    sp = data["spatial_partitioning"]
    assert sp["train_scenes"] == 10
    assert sp["val_scenes"] == 4
    assert sp["held_out_test_scenes"] == 10
    assert sp["leakage_prevented"] is True

    train_set = set(sp["train_cities"])
    val_set = set(sp["val_cities"])
    test_set = set(sp["test_cities"])

    assert len(train_set) == 10
    assert len(val_set) == 4
    assert len(test_set) == 10
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # 3. Primary Validation-Tuned Regime Verification
    primary = data["primary_comparison_validation_tuned"]
    assert "m32_fc_siam_diff" in primary
    assert "m32_bit" in primary
    assert "phase2_classical_gaussian_cva" in primary

    fc = primary["m32_fc_siam_diff"]
    bit = primary["m32_bit"]

    assert 0.0 < fc["decision_threshold"] < 1.0
    assert 0.0 < bit["decision_threshold"] < 1.0
    assert fc["macro_f1"] == 0.3766
    assert fc["pooled_f1"] == 0.3818
    assert bit["macro_f1"] == 0.1295
    assert bit["pooled_f1"] == 0.1260

    # 4. Secondary Default-Threshold Regime Verification
    secondary = data["secondary_comparison_default_threshold"]
    assert secondary["m32_fc_siam_diff"]["threshold"] == 0.50
    assert secondary["m32_bit"]["threshold"] == 0.50


def test_m32_per_scene_evaluation_and_arithmetic_reproduction():
    csv_path = REPO_ROOT / "artifacts" / "specialists" / "m32_per_scene_evaluation.csv"
    assert csv_path.exists(), f"Missing M3.2 per-scene CSV: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 10, f"Expected 10 test scenes, found {len(df)}"

    expected_scenes = {
        "aguasclaras", "austin", "beihai", "berlin", "brasilia",
        "budapest", "campinas", "mumbai", "nantes", "pisa"
    }
    assert set(df["scene"]) == expected_scenes

    # Verify Macro F1 exact arithmetic mean
    calculated_fc_macro = round(float(df["fc_siam_diff_tuned_f1"].mean()), 4)
    assert calculated_fc_macro == 0.3766, f"Macro F1 mismatch: {calculated_fc_macro}"

    calculated_bit_macro = round(float(df["bit_tuned_f1"].mean()), 4)
    assert calculated_bit_macro == 0.1295, f"BIT Macro F1 mismatch: {calculated_bit_macro}"

    # Verify passing scenes (>= 0.50)
    passing = df[df["fc_siam_diff_tuned_f1"] >= 0.50]
    assert len(passing) == 5
    assert set(passing["scene"]) == {"aguasclaras", "brasilia", "budapest", "nantes", "pisa"}
