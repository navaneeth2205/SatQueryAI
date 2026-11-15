"""
Test suite ensuring threshold-regime consistency in Milestone M3.1 Land Cover evaluation.

Guarantees that:
1. Primary comparison scorecard strictly contains validation-tuned metrics for all models.
2. Secondary comparison scorecard strictly contains default-threshold metrics for all models.
3. No metrics are cross-contaminated between regimes.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_m31_scorecard_regime_consistency():
    json_path = REPO_ROOT / "artifacts" / "specialists" / "m31_model_comparison.json"
    assert json_path.exists(), f"Missing comparison JSON: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Primary Comparison Validation-Tuned Checks
    primary = data["primary_comparison_validation_tuned"]
    for model_key in [
        "phase2_classical_random_forest",
        "phase2_classical_lightgbm",
        "m31_dual_resnet34_cb_bce",
        "m31_dual_resnet34_asl",
    ]:
        assert model_key in primary, f"Missing {model_key} in primary comparison"
        m = primary[model_key]
        assert m["regime"] == "validation_tuned", f"Regime violation in primary comparison: {m['regime']}"
        assert 0.0 <= m["macro_f1_16_standard"] <= 1.0
        assert 0.0 <= m["macro_f1_14_present"] <= 1.0
        assert 0.0 <= m["micro_f1"] <= 1.0
        assert 0.0 <= m["hamming_loss"] <= 1.0

    # Specifically verify Random Forest tuned metrics (not default)
    rf_tuned = primary["phase2_classical_random_forest"]
    assert rf_tuned["macro_f1_14_present"] == 0.3886
    assert rf_tuned["micro_f1"] == 0.4481
    assert rf_tuned["hamming_loss"] == 0.3153

    # Specifically verify LightGBM tuned metrics
    lgb_tuned = primary["phase2_classical_lightgbm"]
    assert lgb_tuned["macro_f1_16_standard"] == 0.4739
    assert lgb_tuned["macro_f1_14_present"] == 0.5416
    assert lgb_tuned["micro_f1"] == 72.99 / 100.0 or abs(lgb_tuned["micro_f1"] - 0.7299) < 1e-3
    assert lgb_tuned["hamming_loss"] == 0.0976

    # 2. Secondary Comparison Default Threshold Checks
    secondary = data["secondary_comparison_default_threshold"]
    for model_key in [
        "phase2_classical_random_forest",
        "phase2_classical_lightgbm",
        "m31_dual_resnet34_cb_bce",
        "m31_dual_resnet34_asl",
    ]:
        assert model_key in secondary, f"Missing {model_key} in secondary comparison"
        m = secondary[model_key]
        assert m["regime"] == "default_threshold", f"Regime violation in secondary comparison: {m['regime']}"
        assert m["threshold"] == 0.50

    # Specifically verify Random Forest default metrics
    rf_default = secondary["phase2_classical_random_forest"]
    assert rf_default["macro_f1_14_present"] == 0.3795
    assert rf_default["micro_f1"] == 0.4795
    assert rf_default["hamming_loss"] == 0.2228

    # 3. Disjointness check
    assert rf_tuned["micro_f1"] != rf_default["micro_f1"]
    assert rf_tuned["hamming_loss"] != rf_default["hamming_loss"]
