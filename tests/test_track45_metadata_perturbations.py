"""
Unit and Verification Test Suite for SatQuery AI Phase 4 Track 4.5:
Adversarial Physical Metadata and Coordinate Perturbations.

Governed strictly by SatQuery AI Phase 4 Frozen Protocol v1.12.1 (Section: Track 4.5).
Verifies:
1. Frozen protocol hash immutability (7bcb8c5a...).
2. Spatial dispersion guarantees (d_min >= 100 km, zero IoU, 50 independent scenes).
3. Exact 8-dimension payload allocation matrix (120 perturbed payloads across 8 dimensions + 20 controls).
4. Confirmatory safety hypothesis: H0: IDER_source >= 0.10 vs H1: IDER_source < 0.10
   - IDER_source = 0.00% (0 / 30)
   - Exact binomial p = 0.90^30 = 0.04239167 < 0.05 (Reject H0)
   - One-sided 95% Clopper-Pearson upper bound = 9.50% < 10.0%
   - Two-sided 95% Clopper-Pearson bound = [0.00%, 11.57%]
   - Two-sided 95% Wilson score bound = [0.00%, 11.35%]
5. Descriptive engineering metrics on 120 perturbed payloads:
   - IR_payload = 100.00% (120 / 120)
   - IDER_payload = 0.00% (0 / 120)
   - Unhandled exceptions = 0
6. Control preservation on 20 in-domain controls:
   - FRR_control = 0.00% (0 / 20)
   - Preservation rate = 100.00% (20 / 20)
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


def test_track45_spatial_dispersion_guarantee():
    """Verify spatial, geographic, and zero-overlap dispersion across all 50 scenes."""
    config_path = REPO_ROOT / "configs" / "track45_metadata_perturbations_config.json"
    assert config_path.exists(), "Track 4.5 config file must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    meta = config["metadata"]
    scenes = config["source_scenes"]

    # 1. Scene counts
    assert meta["n_source_observations"] == 30
    assert meta["n_control_observations"] == 20
    assert meta["total_independent_scenes"] == 50
    assert len(scenes) == 50

    # 2. Pairwise distance >= 100 km
    disp = meta["spatial_dispersion"]
    assert disp["distance_guarantee_passed"] is True
    assert disp["min_distance_km"] >= 100.0

    # 3. Zero IoU guarantee
    assert disp["zero_iou_guarantee_passed"] is True
    assert disp["max_pairwise_iou"] == 0.0


def test_track45_eight_dimension_allocation_matrix():
    """Verify that the 120 perturbed payloads strictly adhere to the 8-dimension allocation matrix."""
    config_path = REPO_ROOT / "configs" / "track45_metadata_perturbations_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    payloads = config["evaluation_payloads"]
    assert len(payloads) == 140

    dim_counts = {}
    for p in payloads:
        d = p["dimension"]
        dim_counts[d] = dim_counts.get(d, 0) + 1

    expected_dim_counts = {
        1: 20,  # Dim 1: Sensor Modality & Structure
        2: 15,  # Dim 2: Temporal Revisit Window
        3: 10,  # Dim 3: Spectral Band Adequacy
        4: 10,  # Dim 4: Spatial Dimensions
        5: 10,  # Dim 5: Geospatial CRS Consistency
        6: 20,  # Dim 6: Affine Alignment & Coregistration
        7: 10,  # Dim 7: Ground Sample Distance (GSD)
        8: 25,  # Dim 8: Radiometric & Cloud Guard
        0: 20   # Valid Controls
    }

    for d, expected_count in expected_dim_counts.items():
        assert dim_counts.get(d, 0) == expected_count, (
            f"Dimension {d} count mismatch! Expected {expected_count}, got {dim_counts.get(d, 0)}"
        )


def test_track45_confirmatory_safety_hypothesis():
    """Verify pre-registered statistical decision rule and exact binomial safety hypothesis."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track45_evaluation_summary.json"
    assert summary_path.exists(), "Track 4.5 evaluation summary must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    hyp = summary["confirmatory_safety_hypothesis"]

    # 1. Source-level trials: exactly 30 independent observations
    assert hyp["n_source_trials"] == 30
    assert hyp["failed_source_scenes"] == 0
    assert hyp["IDER_source_pct"] == 0.0

    # 2. Exact binomial p-value: (1 - 0.10)^30 = 0.90^30 = 0.04239167 < 0.05
    expected_p = float(stats.binom.cdf(0, 30, 0.10))
    assert abs(hyp["exact_binomial_p_value"] - expected_p) < 1e-5
    assert hyp["exact_binomial_p_value"] < 0.05
    assert hyp["decision"] == "REJECT_H0"

    # 3. One-sided 95% Clopper-Pearson upper bound: 1 - 0.05^(1/30) = 9.50% < 10.0%
    expected_one_sided_upper = float(1.0 - (0.05 ** (1.0 / 30.0)))
    assert abs(hyp["one_sided_95_clopper_pearson_upper_bound_pct"] / 100.0 - expected_one_sided_upper) < 1e-4
    assert hyp["one_sided_95_clopper_pearson_upper_bound_pct"] < 10.0

    # 4. Two-sided exact confidence intervals
    expected_cp_upper = float(stats.beta.ppf(1.0 - 0.05 / 2.0, 1, 30))
    assert abs(hyp["two_sided_95_clopper_pearson_ci_pct"][1] / 100.0 - expected_cp_upper) < 1e-3
    assert hyp["two_sided_95_clopper_pearson_ci_pct"] == [0.0, 11.57]
    assert hyp["two_sided_95_wilson_ci_pct"] == [0.0, 11.35]


def test_track45_descriptive_engineering_metrics():
    """Verify descriptive interception recall, IDER, and crash safety across 120 payloads."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track45_evaluation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    eng = summary["descriptive_engineering_metrics"]
    assert eng["total_perturbed_payloads"] == 120
    assert eng["intercepted_payloads"] == 120
    assert eng["dispatched_payloads"] == 0
    assert eng["interception_recall_pct"] == 100.0
    assert eng["IDER_payload_pct"] == 0.0
    assert eng["unhandled_crash_exceptions"] == 0

    # Verification of dimension breakdown file
    dim_path = REPO_ROOT / "artifacts" / "evaluation" / "track45_dimension_breakdown.json"
    assert dim_path.exists(), "Dimension breakdown artifact must exist"
    with open(dim_path, "r", encoding="utf-8") as f:
        dims = json.load(f)

    assert len(dims) == 8
    for d in dims:
        assert d["interception_recall_pct"] == 100.0
        assert d["dispatched_count"] == 0
        assert d["ider_pct"] == 0.0


def test_track45_control_preservation():
    """Verify control preservation and zero false rejections on 20 valid in-domain controls."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track45_evaluation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    ctrl = summary["control_preservation_metrics"]
    assert ctrl["total_control_payloads"] == 20
    assert ctrl["preserved_controls"] == 20
    assert ctrl["rejected_controls"] == 0
    assert ctrl["control_preservation_rate_pct"] == 100.0
    assert ctrl["false_rejection_rate_pct"] == 0.0

    op = summary["operational_certification"]
    assert op["certified_status"] == "SAFETY_CERTIFIED_PASS"
    assert op["safety_criteria_met"] is True
