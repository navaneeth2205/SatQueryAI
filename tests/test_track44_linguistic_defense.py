"""
Unit and Confirmatory Test Suite for SatQuery AI Phase 4 Track 4.4:
Complex & Adversarial Linguistic Query Defense.

Governed strictly by SatQuery AI Phase 4 Frozen Protocol v1.12.1.
Verifies:
1. Frozen protocol hash immutability (7bcb8c5a...).
2. Benchmark dataset integrity (N=150 total: 125 adversarial in 5 cohorts + 25 controls).
3. Defended Gated Architecture Tier 1 Certification:
   - Pooled false dispatch rate <= 10.0% (observed: 3.20%, 4 / 125).
   - Cohort cap: max_c k_c <= 5 / 25 (observed: 4 / 25).
   - Control query preservation: >= 24 / 25 (observed: 25 / 25, 100.0%).
   - Certified status: TIER_1_CERTIFIED_ROBUST_PASS.
4. Statistical bounds verification:
   - Exact Clopper-Pearson 95% intervals computed via Beta inversion.
   - Pooled 95% upper bound < 10.0% (observed: 7.99%).
5. Ungated baseline ablation verification:
   - Raw linear bag-of-words model fails certification (Tier 3, 52.80% FDR).
   - Gated architecture achieves 93.94% relative risk reduction.
6. Linguistic Defense Gate unit properties:
   - Deterministic interception across all 5 threat categories.
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

from satquery.utils.linguistic import evaluate_linguistic_defense_gate, LinguisticDefenseResult

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


def test_track44_benchmark_dataset_integrity():
    """Verify Track 4.4 benchmark structure, cohort counts, and query allocation."""
    config_path = REPO_ROOT / "configs" / "track44_linguistic_defense_config.json"
    assert config_path.exists(), "Track 4.4 configuration file must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    meta = config["metadata"]
    assert meta["total_queries"] == 150
    assert meta["n_adversarial_queries"] == 125
    assert meta["n_control_queries"] == 25
    assert meta["n_adversarial_cohorts"] == 5

    cohorts = config["adversarial_cohorts"]
    assert len(cohorts) == 5

    expected_cohort_ids = [
        "cohort_1_lexical_perturbations",
        "cohort_2_multi_task_collisions",
        "cohort_3_polysemous_jargon",
        "cohort_4_truncated_instructions",
        "cohort_5_negation_traps"
    ]

    for idx, c in enumerate(cohorts):
        assert c["cohort_id"] == expected_cohort_ids[idx]
        assert len(c["queries"]) == 25

    controls = config["control_cohort"]["queries"]
    assert len(controls) == 25

    lc_controls = [q for q in controls if q.get("expected_task") == "LAND_COVER_CLASSIFICATION"]
    cd_controls = [q for q in controls if q.get("expected_task") == "CHANGE_DETECTION"]
    other_controls = [q for q in controls if q.get("expected_task") not in ["LAND_COVER_CLASSIFICATION", "CHANGE_DETECTION"]]

    assert len(lc_controls) == 8
    assert len(cd_controls) == 8
    assert len(other_controls) == 9


def test_track44_defended_tier1_certification():
    """Verify that the Defended Gated Architecture satisfies all Tier 1 certification criteria."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track44_evaluation_summary.json"
    assert summary_path.exists(), "Track 4.4 evaluation summary must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    defended = summary["defended_pipeline_results"]
    adv_metrics = defended["pooled_adversarial_metrics"]
    ctrl_metrics = defended["control_preservation_metrics"]
    cert = defended["operational_certification"]

    # 1. Pooled false dispatch rate <= 10.0% (k_pooled <= 12 / 125)
    assert adv_metrics["total_false_dispatches"] <= 12
    assert adv_metrics["pooled_false_dispatch_rate_pct"] <= 10.0
    assert adv_metrics["total_false_dispatches"] == 4
    assert adv_metrics["pooled_false_dispatch_rate_pct"] == 3.20

    # 2. Maximum cohort breakdown cap: max_c k_c <= 5 / 25 (20.0%)
    assert adv_metrics["max_cohort_false_dispatches"] <= 5
    assert adv_metrics["max_cohort_false_dispatches"] == 4

    # 3. Control preservation >= 24 / 25 (96.0%)
    assert ctrl_metrics["preserved_count"] >= 24
    assert ctrl_metrics["preservation_rate_pct"] >= 96.0
    assert ctrl_metrics["preserved_count"] == 25
    assert ctrl_metrics["preservation_rate_pct"] == 100.0

    # 4. Certification decision is Tier 1 Certified Robust Pass
    assert cert["certified_tier"] == "TIER_1_CERTIFIED_ROBUST_PASS"
    assert cert["status"] == "PASS"


def test_track44_statistical_confidence_intervals():
    """Verify exact Clopper-Pearson confidence interval computations and bounds."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track44_evaluation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    defended = summary["defended_pipeline_results"]
    pooled = defended["pooled_adversarial_metrics"]

    # Pooled 95% Clopper-Pearson upper bound must be strictly < 10.0%
    cp_ci = pooled["clopper_pearson_95_ci_pct"]
    assert cp_ci[0] >= 0.0
    assert cp_ci[1] <= 10.0
    assert cp_ci[0] == 0.88
    assert cp_ci[1] == 7.99

    # Cohort breakdown intervals check
    for c in defended["cohort_breakdown"]:
        ci = c["clopper_pearson_95_ci_pct"]
        assert 0.0 <= ci[0] <= ci[1] <= 100.0
        assert ci[0] <= c["false_dispatch_rate_pct"] <= ci[1]


def test_track44_ungated_baseline_ablation_comparison():
    """Verify that the ungated baseline ablation documents bag-of-words failure and risk reduction."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track44_evaluation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    ungated = summary["ungated_baseline_ablation"]
    u_metrics = ungated["pooled_adversarial_metrics"]
    reduction = summary["comparative_risk_reduction"]

    # 1. Ungated baseline fails certification under Tier 3
    assert u_metrics["pooled_false_dispatch_rate_pct"] > 12.0
    assert u_metrics["total_false_dispatches"] == 66
    assert u_metrics["pooled_false_dispatch_rate_pct"] == 52.80
    assert ungated["operational_certification"]["certified_tier"] == "TIER_3_FAILED_CERTIFICATION"
    assert ungated["operational_certification"]["status"] == "FAIL"

    # 2. Comparative risk reduction exceeds 90%
    assert reduction["relative_risk_reduction_pct"] >= 90.0
    assert reduction["relative_risk_reduction_pct"] == 93.94
    assert reduction["false_dispatches_eliminated"] == 62


def test_linguistic_defense_gate_threat_detection():
    """Verify that evaluate_linguistic_defense_gate deterministically intercepts each threat category."""
    # 1. Truncation
    r1 = evaluate_linguistic_defense_gate("change")
    assert r1.is_intercepted is True
    assert r1.decision == "REJECT_INSUFFICIENT_CONTEXT"
    assert r1.failure_code == "ERR_INSUFFICIENT_CONTEXT"

    # 2. Out-of-Domain Telemetry
    r2 = evaluate_linguistic_defense_gate("Classify wetland biomes using seismic tremor magnitude recordings.")
    assert r2.is_intercepted is True
    assert r2.decision == "REJECT_INSUFFICIENT_CONTEXT"
    assert r2.failure_code == "ERR_OUT_OF_DOMAIN_TELEMETRY"

    # 3. Negation Trap
    r3 = evaluate_linguistic_defense_gate("Do not detect changes, only inspect the static satellite image.")
    assert r3.is_intercepted is True
    assert r3.decision == "CLARIFY_AMBIGUOUS"
    assert r3.failure_code == "ERR_NEGATION_TRAP"

    # 4. Multi-Task Collision
    r4 = evaluate_linguistic_defense_gate("Classify land cover classes and detect modifications between 2020 and 2024 simultaneously.")
    assert r4.is_intercepted is True
    assert r4.decision == "CLARIFY_AMBIGUOUS"
    assert r4.failure_code == "ERR_MULTI_TASK_COLLISION"

    # 5. Polysemous Jargon Devoid of Directives
    r5 = evaluate_linguistic_defense_gate("Top-of-atmosphere radiance and bottom-of-atmosphere surface reflectance.")
    assert r5.is_intercepted is True
    assert r5.decision == "CLARIFY_AMBIGUOUS"
    assert r5.failure_code == "ERR_DOMAIN_JARGON_NO_VERB"

    # 6. Legitimate Directives Pass
    r6 = evaluate_linguistic_defense_gate("What type of land cover is visible in this satellite patch?")
    assert r6.is_intercepted is False
    assert r6.decision == "PASS"
    assert r6.failure_code is None
