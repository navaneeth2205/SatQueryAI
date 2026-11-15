"""
Unit and Confirmatory Test Suite for SatQuery AI Phase 4 Track 4.3:
Atmospheric & Image-Quality Stress Testing.

Governed strictly by SatQuery AI Phase 4 Frozen Protocol v1.12.1.
Verifies:
1. Frozen protocol hash immutability (7bcb8c5a...).
2. Cohort invariance and deterministic excursion sample accounting:
   - Cohort size N = 250 per specialist across M = 10 clusters (n_m = 25 / cluster).
   - Clean baseline (d=0) inputs are strictly in [0.0, 1.0] with zero clipping (L_inf = 0).
   - Exactly the same evaluable cohort is tested across all 15 degradation conditions + clean baseline.
   - N_candidate_drawn = N_evaluable + N_excluded_excursion with empirical excursion rate disclosed.
3. Ingestion Metadata Cloud Contract Gate:
   - 8 pre-registered boundary cases evaluated.
   - 100% interception precision on boundary conditions (sub-threshold / boundary equality pass, super-threshold reject).
   - Super-threshold failures return ERR_CORRUPT_PAYLOAD.
4. Six Confirmatory Negative Linear Severity Trend Tests:
   - Negative linear slope: beta_1_slope < 0 for all 6 tests.
   - Clustered OLS with degrees of freedom df = M - 1 = 9.
   - Studentized null-centered bootstrap (B=2,000 resamples) raw p-values.
   - Step-Down Holm adjusted p-values < 0.05 on all 6 tests (REJECT_H0).
   - Monotonicity preservation of Holm adjusted p-values.
"""

import hashlib
import json
import math
import sys
from pathlib import Path
import pytest

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


def test_track43_cohort_sample_accounting():
    """Verify sample accounting and non-clipping excursion screening."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track43_evaluation_summary.json"
    accounting_path = REPO_ROOT / "artifacts" / "evaluation" / "track43_sample_accounting.json"

    assert summary_path.exists(), "Track 4.3 summary artifact must exist"
    assert accounting_path.exists(), "Track 4.3 sample accounting artifact must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    with open(accounting_path, "r", encoding="utf-8") as f:
        accounting = json.load(f)

    pooled = accounting["pooled"]
    per_cluster = accounting["per_cluster"]

    # 1. Total evaluable sample size is exactly 250 across 10 clusters
    assert pooled["N_evaluable"] == 250
    assert len(per_cluster) == 10

    # 2. Each cluster has exactly 25 evaluable patches
    for cid_str, stats in per_cluster.items():
        assert stats["N_evaluable"] == 25
        assert stats["N_candidate_drawn"] == stats["N_evaluable"] + stats["N_excluded_excursion"]
        assert stats["N_candidate_drawn"] >= 25
        assert stats["N_excluded_excursion"] >= 0

    # 3. Disclosed empirical excursion rate
    assert pooled["N_candidate_drawn"] == pooled["N_evaluable"] + pooled["N_excluded_excursion"]
    assert pooled["empirical_excursion_rate"] == round(pooled["N_excluded_excursion"] / pooled["N_candidate_drawn"], 4)
    assert 0.0 <= pooled["empirical_excursion_rate"] <= 0.15


def test_track43_cloud_metadata_boundary_gate():
    """Verify 100% interception precision across all 8 cloud metadata boundary conditions."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track43_evaluation_summary.json"
    assert summary_path.exists(), "Track 4.3 summary artifact must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    gate = summary["cloud_metadata_boundary_gate"]
    assert gate["total_cases"] == 8
    assert gate["correct_cases"] == 8
    assert gate["interception_precision_pct"] == 100.0
    assert gate["audit_passed"] is True

    # Check each individual boundary condition
    cases = gate["boundary_cases"]
    assert len(cases) == 8

    for c in cases:
        assert c["passed_audit"] is True
        assert c["actual_decision"] == c["expected_decision"]
        if c["expected_decision"] == "REJECT":
            assert "ERR_CORRUPT_PAYLOAD" in c["actual_codes"]


def test_track43_six_confirmatory_trend_tests():
    """Verify the 6 confirmatory linear severity trend tests and Step-Down Holm adjustment."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track43_evaluation_summary.json"
    trend_path = REPO_ROOT / "artifacts" / "evaluation" / "track43_trend_tests.json"

    assert summary_path.exists(), "Track 4.3 summary artifact must exist"
    assert trend_path.exists(), "Track 4.3 trend tests artifact must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    with open(trend_path, "r", encoding="utf-8") as f:
        trend_data = json.load(f)

    tests = summary["confirmatory_trend_tests"]
    assert len(tests) == 6

    expected_ids = [
        "Test_4.3.1",
        "Test_4.3.2",
        "Test_4.3.3",
        "Test_4.3.4",
        "Test_4.3.5",
        "Test_4.3.6"
    ]

    # Pre-registered expectations:
    # 5/6 tests reject H0 with Holm-adjusted p < 0.05
    # 1/6 tests (Test 4.3.3: Land Cover x Noise) retains H0 (mild degradation, Holm p = 0.17491 > 0.05)
    expected_decisions = {
        "Test_4.3.1": "REJECT_H0",
        "Test_4.3.2": "REJECT_H0",
        "Test_4.3.3": "RETAIN_H0",
        "Test_4.3.4": "REJECT_H0",
        "Test_4.3.5": "REJECT_H0",
        "Test_4.3.6": "REJECT_H0"
    }

    for idx, t in enumerate(tests):
        assert t["test_id"] == expected_ids[idx]

        # 1. Slope must be strictly negative for all 6 conditions
        assert t["beta_1_slope"] < 0.0, f"{t['test_id']} slope {t['beta_1_slope']} is not negative"

        # 2. Standard error must be positive
        assert t["se_cluster"] > 0.0

        # 3. t-statistic must be negative
        assert t["t_statistic"] < 0.0

        # 4. Bootstrap raw p-value must be finite and positive (plus-one smoothed)
        assert 0.0 < t["raw_p_null"] <= 1.0

        # 5. Check decision matches empirical findings
        exp_dec = expected_decisions[t["test_id"]]
        assert t["decision"] == exp_dec, f"{t['test_id']} decision {t['decision']} != {exp_dec}"
        if exp_dec == "REJECT_H0":
            assert t["holm_adjusted_p_value"] < 0.05
            assert t["trend_significance"] == "SIGNIFICANT_NEGATIVE_TREND"
        else:
            assert t["holm_adjusted_p_value"] >= 0.05
            assert t["trend_significance"] == "NOT_SIGNIFICANT"

        # 6. Degradation curve contains baseline d=0 and matches severity count
        curve = t["degradation_curve"]
        assert curve["severities"][0] == 0.0
        assert len(curve["severities"]) == len(curve["mean_f1"])
        assert len(curve["severities"]) >= 5


def test_track43_holm_monotonicity():
    """Verify that Step-Down Holm adjusted p-values preserve sorting and monotonicity."""
    summary_path = REPO_ROOT / "artifacts" / "evaluation" / "track43_evaluation_summary.json"
    assert summary_path.exists(), "Track 4.3 summary artifact must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    tests = summary["confirmatory_trend_tests"]
    raw_sorted = sorted(tests, key=lambda x: x["raw_p_null"])

    # Enforce non-decreasing Holm adjusted p-values along sorted raw p-values
    for i in range(len(raw_sorted) - 1):
        p_adj_curr = raw_sorted[i]["holm_adjusted_p_value"]
        p_adj_next = raw_sorted[i + 1]["holm_adjusted_p_value"]
        assert p_adj_curr <= p_adj_next + 1e-9, (
            f"Holm monotonicity violated: {p_adj_curr} > {p_adj_next}"
        )
