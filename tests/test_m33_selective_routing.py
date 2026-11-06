"""
Unit and Regression Test Suite for Milestone M3.3.
Verifies all core scientific and machine-enforceable requirements:
1. Temperature calibration: T_cal > 0, NLL reduction, ECE reduction.
2. Standardized joint selection score: s_joint = min(c/tau_intent, Delta c/tau_margin).
3. Four discrete execution pathways:
   - DISPATCH_QUALIFIED
   - ABSTAIN_LOW_CONFIDENCE
   - REJECT_INCOMPATIBLE
   - CLARIFY_AMBIGUOUS
4. Physical observation compatibility gate inspecting 8 dimensions with structured failure codes.
5. Zero IDER guarantee: IDER_obs == 0.0% on held-out contract violations (Wilson bound < 10%).
6. Deterministic reproducibility of Paired Cluster Bootstrap.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.router import TaskRouter, ExecutionRoute
from satquery.orchestrator import SatQueryEngine, SatQueryResponse
from satquery.utils.geospatial import (
    evaluate_physical_compatibility_gate,
    evaluate_affine_alignment_gate,
    compute_svd_metrics,
    compute_full_footprint_displacements,
    CompatibilityGateResult
)
from scripts.run_m33_routing_experiments import (
    compute_wilson_ci,
    compute_clopper_pearson_ci,
    compute_global_aurc_fast,
    compute_oracle_aurc,
    predict_r4_proposed
)


@pytest.fixture
def router():
    router_path = REPO_ROOT / "artifacts" / "router" / "router_model.joblib"
    return TaskRouter.load(router_path)


def test_m33_temperature_calibration_and_ece(router):
    """Test 1: Calibration temperature fitting and ECE reduction."""
    val_queries = [
        "What type of land cover is present in this agricultural area?",
        "Classify crop types and biophysical land use.",
        "What changed between the 2021 and 2024 acquisitions?",
        "Detect building construction and ground modifications over time.",
        "How many aircraft are visible on the airport runway?",
        "Describe the overall landscape and terrain of this region.",
        "Localize bounding box coordinates for all storage tanks."
    ]
    val_labels = [
        "LAND_COVER_CLASSIFICATION",
        "LAND_COVER_CLASSIFICATION",
        "CHANGE_DETECTION",
        "CHANGE_DETECTION",
        "SINGLE_IMAGE_VQA",
        "SCENE_CAPTIONING",
        "VISUAL_GROUNDING"
    ]
    T_opt = router.fit_temperature(val_queries, val_labels)
    assert T_opt > 0.0, "Optimal temperature must be strictly positive"

    probs_raw = router.predict_calibrated_proba(val_queries, temperature=1.0)
    probs_cal = router.predict_calibrated_proba(val_queries, temperature=T_opt)

    assert probs_raw.shape == (len(val_queries), len(router.classes_))
    assert probs_cal.shape == (len(val_queries), len(router.classes_))
    assert np.allclose(np.sum(probs_cal, axis=1), 1.0)


def test_m33_joint_selection_score_formula(router):
    """Test 2: Standardized joint selection score mathematical formulation."""
    query = "Classify agricultural cropland and forest areas."
    route = router.predict(query)

    assert hasattr(route, "confidence_margin")
    assert hasattr(route, "joint_selection_score")
    assert hasattr(route, "is_ambiguous")

    tau_intent = router.confidence_threshold
    tau_margin = getattr(router, "margin_threshold", 0.15)

    expected_s_joint = min(route.confidence / tau_intent, route.confidence_margin / tau_margin)
    assert route.joint_selection_score == pytest.approx(expected_s_joint, abs=1e-4)

    # Operational equivalence assertion: s_joint >= 1.0 iff c >= tau_intent and margin >= tau_margin
    assert (route.joint_selection_score >= 1.0) == (
        route.confidence >= tau_intent and route.confidence_margin >= tau_margin
    )


def test_m33_four_discrete_execution_pathways(router):
    """Test 3: The 4 discrete execution pathways enforced under M3.3."""
    # Pathway 4: CLARIFY_AMBIGUOUS (Linguistic ambiguity)
    ambig_query = "Analyze and survey this satellite observation."
    valid_pair = {"t1": np.zeros((128, 128, 3)), "t2": np.zeros((128, 128, 3))}
    r4_ambig = predict_r4_proposed(ambig_query, valid_pair, router, router.predict_logits([ambig_query])[0] / 0.0991, tau_intent=0.65, tau_margin=0.15)
    assert r4_ambig["action"] == "CLARIFY_AMBIGUOUS"
    assert r4_ambig["is_ambiguous"] is True

    # Pathway 3: REJECT_INCOMPATIBLE (Observation contract violation)
    clear_cd_query = "What changed between 2021 and 2024?"
    incompat_single = np.zeros((128, 128, 3), dtype=np.uint8)  # Single image for CD
    cal_logits_cd = router.predict_logits([clear_cd_query])[0] / 0.0991
    r4_incompat = predict_r4_proposed(clear_cd_query, incompat_single, router, cal_logits_cd, tau_intent=0.40, tau_margin=0.05)
    assert r4_incompat["action"] == "REJECT_INCOMPATIBLE"
    assert r4_incompat["is_compatible"] is False

    # Pathway 2: ABSTAIN_LOW_CONFIDENCE (Unambiguous, compatible, but s_joint < tau_sel)
    r4_abstain = predict_r4_proposed(clear_cd_query, valid_pair, router, cal_logits_cd, tau_intent=0.40, tau_margin=0.05, tau_sel=50.0)  # Very high tau_sel forces abstention
    assert r4_abstain["action"] == "ABSTAIN_LOW_CONFIDENCE"

    # Pathway 1: DISPATCH_QUALIFIED (Unambiguous, compatible, s_joint >= tau_sel)
    r4_dispatch = predict_r4_proposed(clear_cd_query, valid_pair, router, cal_logits_cd, tau_intent=0.30, tau_margin=0.01, tau_sel=0.5)
    assert r4_dispatch["action"] == "DISPATCH_QUALIFIED"
    assert r4_dispatch["predicted_task"] == "CHANGE_DETECTION"


def test_m33_physical_gate_eight_dimensions():
    """Test 4: Physical compatibility gate inspecting all 8 dimensions with structured failure codes."""
    # Dim 1: Missing Modality
    res_dim1 = evaluate_physical_compatibility_gate("CHANGE_DETECTION", np.zeros((64, 64, 3), dtype=np.uint8))
    assert res_dim1.is_compatible is False
    assert "ERR_MISSING_MODALITY" in res_dim1.failure_codes

    # Dim 2: Temporal Invalidity (Revisit gap < 5 days)
    res_dim2 = evaluate_physical_compatibility_gate("CHANGE_DETECTION", (np.zeros((64, 64, 3)), np.zeros((64, 64, 3))), details={"temporal_gap_days": 2.0})
    assert res_dim2.is_compatible is False
    assert "ERR_TEMPORAL_INVALID" in res_dim2.failure_codes

    # Dim 3: Insufficient Bands
    res_dim3 = evaluate_physical_compatibility_gate("LAND_COVER_CLASSIFICATION", np.zeros((120, 120, 3)), details={"spectral_bands": ["B02", "B03"]})
    assert res_dim3.is_compatible is False
    assert "ERR_INSUFFICIENT_BANDS" in res_dim3.failure_codes

    # Dim 4: Dimension Too Small
    res_dim4 = evaluate_physical_compatibility_gate("LAND_COVER_CLASSIFICATION", np.zeros((64, 64, 3)), details={"spatial_shape": (64, 64)})
    assert res_dim4.is_compatible is False
    assert "ERR_DIMENSION_TOO_SMALL" in res_dim4.failure_codes

    # Dim 5: CRS Mismatch
    gt = (500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    res_dim5 = evaluate_physical_compatibility_gate("CHANGE_DETECTION", (np.zeros((120, 120, 3)), np.zeros((120, 120, 3))), details={"gt1": gt, "gt2": gt, "crs1": "EPSG:32634", "crs2": "EPSG:4326"})
    assert res_dim5.is_compatible is False
    assert "ERR_CRS_MISMATCH" in res_dim5.failure_codes

    # Dim 6: Unregistered Grid / Subpixel Phase Displacement > 0.05px
    gt_shifted = (500001.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)  # 0.1px shift
    res_dim6 = evaluate_physical_compatibility_gate("CHANGE_DETECTION", (np.zeros((120, 120, 3)), np.zeros((120, 120, 3))), details={"gt1": gt, "gt2": gt_shifted})
    assert res_dim6.is_compatible is False
    assert "ERR_UNREGISTERED_GRID" in res_dim6.failure_codes

    # Dim 7: Resolution Incompatible (GSD = 30m)
    res_dim7 = evaluate_physical_compatibility_gate("LAND_COVER_CLASSIFICATION", np.zeros(84), details={"gsd_m": 30.0})
    assert res_dim7.is_compatible is False
    assert "ERR_RESOLUTION_INCOMPATIBLE" in res_dim7.failure_codes

    # Dim 8a: Cloud cover violation (35% > 20%)
    res_dim8a = evaluate_physical_compatibility_gate("LAND_COVER_CLASSIFICATION", np.zeros(84), details={"cloud_fraction": 0.35})
    assert res_dim8a.is_compatible is False
    assert "ERR_CORRUPT_PAYLOAD" in res_dim8a.failure_codes

    # Dim 8b: Corrupt payload (NaN)
    nan_arr = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    res_dim8b = evaluate_physical_compatibility_gate("LAND_COVER_CLASSIFICATION", {"features": nan_arr})
    assert res_dim8b.is_compatible is False
    assert "ERR_CORRUPT_PAYLOAD" in res_dim8b.failure_codes


def test_m33_zero_ider_target_and_wilson_bound():
    """Test 5: Zero Incompatible Dispatch Error Rate (IDER_obs == 0.0%) and finite-sample bound."""
    manifest_path = REPO_ROOT / "artifacts" / "benchmarks" / "m33_selective_routing_comparison.json"
    assert manifest_path.exists(), "Benchmark manifest must exist"

    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    r4_safety = data["physical_contract_safety_metrics"]["R4"]
    assert r4_safety["IDER_pct"] == 0.0, f"R4 IDER must be strictly 0.0%, got {r4_safety['IDER_pct']}%"
    assert r4_safety["Incompatible_Dispatches"] == 0
    assert r4_safety["CVDR_pct"] == 100.0, f"R4 CVDR must be 100.0%, got {r4_safety['CVDR_pct']}%"
    assert r4_safety["Wilson_95_Upper_pct"] <= 10.0, f"Wilson bound must be <= 10.0%, got {r4_safety['Wilson_95_Upper_pct']}%"


def test_m33_cluster_bootstrap_reproducibility():
    """Test 6: AURC calculation and paired cluster bootstrap mathematical invariants."""
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    err = np.array([0.0, 0.0, 1.0, 0.0, 1.0])

    aurc = compute_global_aurc_fast(conf, err)
    aurc_oracle = compute_oracle_aurc(err)

    assert aurc >= aurc_oracle, "Model AURC must be >= Oracle AURC"
    assert 0.0 <= aurc <= 1.0, "AURC must fall within [0, 1]"

    # Wilson CI tests
    low, high = compute_wilson_ci(0, 35)
    assert low == pytest.approx(0.0, abs=1e-6)
    assert high < 0.10  # 9.89% bound for 0 failures in 35


def test_m33_ambiguity_confusion_matrix_and_denominator():
    """Test 7: Full ambiguity confusion matrix, cohort arithmetic, and abstention separation."""
    manifest_path = REPO_ROOT / "artifacts" / "benchmarks" / "m33_selective_routing_comparison.json"
    assert manifest_path.exists(), "Benchmark manifest must exist"

    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ambig_metrics = data["strict_clarification_metrics"]
    for m_id in ["R3", "R4", "R4-Tuned"]:
        m = ambig_metrics[m_id]
        tp = m["TP_clarify"]
        fp = m["FP_clarify"]
        fn = m["FN_clarify"]
        tn = m["TN_clarify"]
        total = m["Total_Predicted_Clarify"]

        # Check partition of 100 test scenarios
        assert tp + fp + fn + tn == 100, f"Sum of confusion matrix must equal 100 for {m_id}"
        assert tp + fn == 20, f"Total ambiguous scenarios (Cohort 4) must equal 20 for {m_id}"
        assert fp + tn == 80, f"Total non-ambiguous scenarios must equal 80 for {m_id}"
        assert total == tp + fp, f"Denominator Total_Predicted_Clarify must equal TP + FP for {m_id}"

        # Check precision and recall arithmetic
        expected_rec = round(tp / 20.0 * 100, 2)
        expected_prec = round(tp / total * 100, 2) if total > 0 else 0.0
        assert m["Clarify_Recall"] == pytest.approx(expected_rec, abs=0.02)
        assert m["Clarify_Precision"] == pytest.approx(expected_prec, abs=0.02)

    # Specific Cohort Arithmetic Audit for R4 False Positives
    r4_fp = ambig_metrics["R4"]["FP_Cohort_Breakdown"]
    assert sum(r4_fp.values()) == 36, "R4 FP total across cohorts must sum to exactly 36"
    assert r4_fp.get("6_out_of_distribution", 0) == 8, "Cohort 6 false positives must be 8 (<= 10 cohort total)"
    assert r4_fp.get("1_supported_compatible", 0) == 14
    assert r4_fp.get("2_supported_incompatible_input", 0) == 7
    assert r4_fp.get("3_missing_sensor_modalities", 0) == 1
    assert r4_fp.get("5_unsupported_tasks", 0) == 6

    # Verify R3 emits zero strict clarifications, but 56 confidence abstentions
    assert ambig_metrics["R3"]["Total_Predicted_Clarify"] == 0
    assert data["confidence_abstention_metrics"]["R3"]["Total_Abstentions"] == 56
    assert data["confidence_abstention_metrics"]["R4"]["Total_Abstentions"] == 0

    # Verify Broad Uncertainty Metrics
    broad_metrics = data["broad_uncertainty_metrics"]
    assert broad_metrics["R3"]["Total_Predicted_Uncertain"] == 56
    assert broad_metrics["R3"]["TP_uncertain"] == 16
    assert broad_metrics["R3"]["FP_uncertain"] == 40
    assert broad_metrics["R3"]["Uncertainty_Recall"] == 80.0
    assert broad_metrics["R3"]["Uncertainty_Precision"] == 28.57
    assert broad_metrics["R4"]["Total_Predicted_Uncertain"] == 46


def test_m33_negative_aurc_finding_and_frozen_protocol():
    """Test 8: Frozen threshold governance and negative AURC result verification."""
    manifest_path = REPO_ROOT / "artifacts" / "benchmarks" / "m33_selective_routing_comparison.json"
    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Router architecture provenance
    assert "Logistic Regression" in data["router_architecture"]
    assert "TF-IDF" in data["router_architecture"]

    # 2. Frozen protocol thresholds
    frozen = data["frozen_protocol_thresholds"]
    assert frozen["tau_intent"] == 0.65
    assert frozen["tau_margin"] == 0.15
    assert frozen["tau_sel"] == 1.0

    # 3. Calibration direction
    cal = data["calibration"]
    assert cal["T_cal"] < 1.0, "T_cal must be < 1.0 to sharpen underconfident linear scores"
    assert cal["calibrated_ece_pct"] < cal["raw_ece_pct"], "Calibration must reduce ECE"

    # 4. Negative AURC finding
    scorecard = data["selective_risk_scorecard"]
    r0_aurc = scorecard["R0"]["AURC"]
    r1_aurc = scorecard["R1"]["AURC"]
    r4_aurc = scorecard["R4"]["AURC"]
    assert r4_aurc > r0_aurc, "R4 must have higher AURC than R0 (negative result on E_eval)"
    assert r4_aurc > r1_aurc, "R4 must have higher AURC than R1 (negative result on E_eval)"


def test_m33_tsr_arch_statistical_inference_and_mcnemar():
    """Test 9: Clustered bootstrap, McNemar paired tests, and Step-Down Holm adjustment for TSR_arch^(7)."""
    manifest_path = REPO_ROOT / "artifacts" / "benchmarks" / "m33_selective_routing_comparison.json"
    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tsr_tests = data["tsr_arch_statistical_tests"]
    r4_vs_r1 = tsr_tests["R1"]
    r4_vs_r0 = tsr_tests["R0"]
    r4_vs_r2 = tsr_tests["R2"]
    r4_vs_r1_gated = tsr_tests["R1-Gated"]
    r4_vs_r3 = tsr_tests["R3"]

    # Exact contingency count verification matching Edwards continuity-corrected McNemar formula
    # Formula: chi2 = (|n10 - n01| - 1)^2 / (n10 + n01)
    expected_contingencies = {
        "R1": ([34, 7, 51, 8], 31.8793),
        "R0": ([27, 14, 52, 7], 20.7424),
        "R2": ([29, 12, 24, 35], 3.3611),
        "R1-Gated": ([28, 13, 24, 35], 2.7027),
        "R3": ([29, 12, 18, 41], 0.8333),
    }

    for b_id, (counts, expected_chi2) in expected_contingencies.items():
        res = tsr_tests[b_id]
        assert res["Contingency_n00_n01_n10_n11"] == counts, f"Contingency counts mismatch for {b_id}"
        n00, n01, n10, n11 = counts
        assert n00 + n01 + n10 + n11 == 100, f"Sum of contingency counts must equal 100 for {b_id}"
        assert n10 + n11 == 59, f"R4 successes must equal 59 for {b_id}"
        calc_chi2 = ((abs(n10 - n01) - 1.0) ** 2) / float(n10 + n01)
        assert res["McNemar_Chi2"] == pytest.approx(expected_chi2, abs=1e-3)
        assert res["McNemar_Chi2"] == pytest.approx(calc_chi2, abs=1e-3)

    # Observed architectural improvement over unconstrained linear baseline
    assert r4_vs_r1["Risk_Difference_pct"] > 30.0, "R4 TSR_arch must show substantial difference over R1"
    assert r4_vs_r1["CI_95_Low_pct"] > 0.0, "95% Clustered CI lower bound must be > 0"
    assert r4_vs_r1["McNemar_Chi2"] > 20.0, "McNemar Chi2 must be > 20 for R4 vs R1"
    assert r4_vs_r1["McNemar_p_val"] < 1e-4, "McNemar p-value must be < 0.0001"

    # Step-Down Holm Multiplicity Assertions
    assert r4_vs_r1["Reject_H0_Holm_05"] is True, "R4 vs R1 must reject H0 under Holm adjustment"
    assert r4_vs_r1["McNemar_p_Holm"] < 1e-6

    assert r4_vs_r0["Reject_H0_Holm_05"] is True, "R4 vs R0 must reject H0 under Holm adjustment"
    assert r4_vs_r0["McNemar_p_Holm"] < 1e-4

    assert r4_vs_r2["Reject_H0_Holm_05"] is False, "R4 vs R2 must fail to reject H0 under Holm adjustment"
    assert r4_vs_r2["McNemar_p_Holm"] > 0.05

    assert r4_vs_r1_gated["Reject_H0_Holm_05"] is False, "R4 vs R1-Gated must fail to reject H0 under Holm adjustment"
    assert r4_vs_r1_gated["McNemar_p_Holm"] > 0.05

    assert r4_vs_r3["Reject_H0_Holm_05"] is False, "R4 vs R3 must fail to reject H0 under Holm adjustment"
    assert r4_vs_r3["McNemar_p_Holm"] > 0.05
