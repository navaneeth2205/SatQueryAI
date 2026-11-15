"""
Unit and Verification Test Suite for SatQuery AI Phase 4 Track 4.2:
Cross-Sensor & Cross-Resolution Compatibility Defense.

Governed strictly by SatQuery AI Phase 4 Frozen Protocol v1.12.1.
Verifies:
1. Frozen protocol hash immutability (7bcb8c5a...).
2. Spatial-temporal dispersion guarantees (d_min >= 150 km, zero IoU, >= 4 continents, >= 30d temporal separation).
3. Confirmatory safety hypothesis: H0: IDER_source >= 0.10 vs H1: IDER_source < 0.10
   - IDER_source = 0.0% (0 / 40)
   - Exact binomial p = 0.90^40 = 0.01478 < 0.05 (Reject H0)
   - One-sided 95% Clopper-Pearson upper bound = 7.22% < 10.0%
   - Two-sided 95% Clopper-Pearson bound = [0.00%, 8.81%]
   - Two-sided 95% Wilson score bound = [0.00%, 8.76%]
4. Descriptive engineering metrics on 80 invalid payloads:
   - CVDR_payload = 100.0% (80 / 80)
   - IDER_payload = 0.0% (0 / 80)
   - Resolution stress CVDR = 100.0% (40 / 40)
   - Modality/spectral stress CVDR = 100.0% (40 / 40)
5. Control preservation on 20 Sentinel-2 controls:
   - FRR_control = 0.0% (0 / 20)
   - Wilson 95% upper bound = 16.11%
"""

import hashlib
import json
import math
import sys
from pathlib import Path
import pytest
import scipy.stats as stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FROZEN_PROTOCOL_HASH = "7bcb8c5ab00cfebb5f7418237e9297f7eeaf06b0477d426d0ec6b30880e71bb5"


def test_frozen_protocol_hash_unchanged():
    """Verify that the frozen Phase 4 protocol document remains 100% bitwise intact."""
    protocol_path = REPO_ROOT / "SATQUERY-P4-PROTOCOL-V1.12.1-FROZEN.md"
    assert protocol_path.exists(), f"Frozen protocol file missing at: {protocol_path}"
    with open(protocol_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == FROZEN_PROTOCOL_HASH, (
        f"PROTOCOL MUTATION DETECTED! Expected {FROZEN_PROTOCOL_HASH}, got {digest}"
    )


def test_track42_spatial_temporal_dispersion_guarantee():
    """Verify spatial, geographic, and temporal dispersion of Track 4.2 source scenes."""
    config_path = REPO_ROOT / "configs" / "track42_sensor_defense_config.json"
    assert config_path.exists(), "Track 4.2 config file must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    meta = config["metadata"]
    scenes = config["source_scenes"]

    # 1. Scene counts
    assert meta["n_source_scenes"] == 60
    assert meta["n_off_sensor_source_scenes"] == 40
    assert meta["n_control_source_scenes"] == 20
    assert len(scenes) == 60

    # 2. Pairwise distance >= 150 km
    disp = meta["spatial_dispersion"]
    assert disp["distance_guarantee_passed"] is True
    assert disp["min_distance_40_off_sensor_km"] >= 150.0
    assert disp["min_distance_all_60_km"] >= 150.0

    # 3. Zero IoU guarantee
    assert disp["zero_iou_guarantee_passed"] is True
    assert disp["max_pairwise_iou"] == 0.0

    # 4. Continents >= 4
    assert meta["n_continents"] >= 4
    assert len(meta["continents_represented"]) >= 4
    for c in ["North America", "South America", "Europe", "Africa", "Asia", "Oceania"]:
        assert c in meta["continents_represented"]


def test_track42_confirmatory_safety_hypothesis_and_bounds():
    """Verify pre-registered statistical decision rule and exact confidence bounds."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track42_evaluation_summary.json"
    assert summary_path.exists(), "Track 4.2 evaluation summary must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    hyp = summary["confirmatory_safety_hypothesis"]

    # Source-level trials: 40 off-sensor scenes
    assert hyp["n_source_trials"] == 40
    assert hyp["failed_source_scenes"] == 0
    assert hyp["IDER_source_pct"] == 0.0

    # Exact binomial p-value: (1 - 0.10)^40 = 0.90^40 = 0.0147809
    expected_p = (1.0 - 0.10) ** 40
    assert hyp["exact_binomial_p_value"] == pytest.approx(expected_p, rel=1e-4)
    assert hyp["exact_binomial_p_value"] < 0.05
    assert hyp["decision"] == "REJECT_H0"
    assert hyp["decision_rule_passed"] is True

    # One-sided 95% Clopper-Pearson upper bound: 1 - 0.05^(1/40) = 7.22%
    bounds = hyp["confidence_bounds"]
    expected_one_sided_cp = (1.0 - (0.05 ** (1.0 / 40.0))) * 100.0
    assert bounds["one_sided_95_clopper_pearson_upper_pct"] == pytest.approx(expected_one_sided_cp, rel=1e-3)
    assert bounds["one_sided_95_clopper_pearson_upper_pct"] == pytest.approx(7.22, abs=0.01)
    assert bounds["one_sided_95_clopper_pearson_upper_pct"] < bounds["preregistered_bound_ceiling_pct"]
    assert bounds["bound_ceiling_satisfied"] is True

    # Two-sided 95% Clopper-Pearson upper bound: 8.81%
    assert bounds["two_sided_95_clopper_pearson_pct"][0] == 0.0
    assert bounds["two_sided_95_clopper_pearson_pct"][1] == pytest.approx(8.81, abs=0.01)

    # Two-sided 95% Wilson score upper bound: 8.76%
    assert bounds["two_sided_95_wilson_score_pct"][0] == 0.0
    assert bounds["two_sided_95_wilson_score_pct"][1] == pytest.approx(8.76, abs=0.01)


def test_track42_descriptive_cvdr_and_ider_payload():
    """Verify descriptive payload-level interception metrics across 80 invalid payloads."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track42_evaluation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    desc = summary["descriptive_engineering_metrics"]
    assert desc["n_invalid_payloads"] == 80
    assert desc["n_intercepted_payloads"] == 80
    assert desc["n_dispatched_incompatible"] == 0
    assert desc["CVDR_payload_pct"] == 100.0
    assert desc["IDER_payload_pct"] == 0.0

    breakdown = desc["stress_type_breakdown"]
    assert breakdown["resolution_stress"]["n_payloads"] == 40
    assert breakdown["resolution_stress"]["CVDR_pct"] == 100.0

    assert breakdown["modality_spectral_stress"]["n_payloads"] == 40
    assert breakdown["modality_spectral_stress"]["CVDR_pct"] == 100.0

    # Verify diagnostic code counts
    codes = desc["violation_code_counts"]
    assert codes["ERR_RESOLUTION_INCOMPATIBLE"] == 40
    assert codes["ERR_INSUFFICIENT_BANDS"] >= 40


def test_track42_control_preservation_and_frr():
    """Verify control preservation and false rejection rate across 20 Sentinel-2 controls."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track42_evaluation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    ctrl = summary["control_metrics"]
    assert ctrl["n_control_payloads"] == 20
    assert ctrl["n_dispatched_controls"] == 20
    assert ctrl["n_rejected_controls"] == 0
    assert ctrl["FRR_control_pct"] == 0.0
    assert ctrl["control_preservation_passed"] is True

    # Two-sided 95% Wilson score upper bound for 0/20: 16.11%
    assert ctrl["two_sided_95_wilson_score_pct"][0] == 0.0
    assert ctrl["two_sided_95_wilson_score_pct"][1] == pytest.approx(16.11, abs=0.01)

    # Acceptance audit
    audit = summary["preregistered_acceptance_criteria_audit"]
    assert audit["statistical_safety_confirmation"]["status"] == "PASS"
    assert audit["descriptive_engineering_target"]["status"] == "PASS"
    assert audit["control_preservation"]["status"] == "PASS"
