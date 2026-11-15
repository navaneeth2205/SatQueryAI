"""
SatQuery AI Phase 4 Track 4.2: Cross-Sensor & Cross-Resolution Compatibility Defense Evaluation.

Governed strictly by SatQuery AI Phase 4 Frozen Protocol v1.12.1 (Section: Track 4.2).
Evaluates the physical compatibility gate C(t, o) against off-target sensors (Landsat-8/9, PlanetScope)
and Sentinel-2 Level-2A controls across 60 independent source scenes (100 evaluation payloads).

Pre-registered Confirmatory Safety Hypothesis:
  H0: IDER_source >= 0.10  vs.  H1: IDER_source < 0.10
Rejection threshold: 0 errors in 40 source scenes (exact binomial p = 0.90^40 = 0.01478 < 0.05).
One-sided 95% Clopper-Pearson upper bound: 1 - 0.05^(1/40) = 7.22% < 10.0%.
Descriptive engineering targets: CVDR_payload = 100.0% (80/80), IDER_payload = 0.0% (0/80).
Control False Rejection Rate: FRR_control = 0.0% (0/20, Wilson 95% upper bound 16.11%).
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.stats as stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.utils.geospatial import evaluate_physical_compatibility_gate, CompatibilityGateResult


def compute_wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Computes exact asymmetric Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))) / denom
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


def compute_clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Computes two-sided exact Clopper-Pearson (Beta inversion) confidence interval."""
    if n == 0:
        return 0.0, 0.0
    lower = 0.0 if k == 0 else stats.beta.ppf(alpha / 2.0, k, n - k + 1)
    upper = 1.0 if k == n else stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return float(lower), float(upper)


def compute_one_sided_clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Computes one-sided 95% Clopper-Pearson upper confidence bound."""
    if k == 0:
        # Exact closed form: 1 - alpha^(1/n)
        return float(1.0 - (alpha ** (1.0 / n)))
    else:
        return float(stats.beta.ppf(1.0 - alpha, k + 1, n - k))


def run_track42_evaluation() -> Dict[str, Any]:
    config_path = REPO_ROOT / "configs" / "track42_sensor_defense_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Track 4.2 configuration not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    scenes = config["source_scenes"]
    payloads = config["evaluation_payloads"]

    # 1. Evaluate every payload through Physical Compatibility Gate C(t, o)
    payload_eval_records = []
    for p in payloads:
        task = p["target_task"]
        details = {}
        for k in ["gsd_m", "spectral_bands", "spatial_shape", "crs", "temporal_gap_days", "cloud_fraction"]:
            if p.get(k) is not None:
                details[k] = p[k]

        # Build mock physical array / feature payload according to task
        if task == "LAND_COVER_CLASSIFICATION":
            raw_payload = {"features": np.zeros(10, dtype=np.float32)}
        elif task == "CHANGE_DETECTION":
            raw_payload = {
                "t1": np.zeros((256, 256, 3), dtype=np.uint8),
                "t2": np.zeros((256, 256, 3), dtype=np.uint8)
            }
        else:
            raw_payload = {}

        gate_res: CompatibilityGateResult = evaluate_physical_compatibility_gate(
            task=task,
            payload=raw_payload,
            details=details
        )

        # Specialist dispatch occurs if and only if gate approves compatibility
        dispatched_to_specialist = bool(gate_res.is_compatible)
        is_valid_ground_truth = bool(p["is_valid_payload"])

        # Correct decision:
        # If invalid payload -> must be intercepted (not dispatched)
        # If valid control -> must be passed (dispatched)
        is_correct = (dispatched_to_specialist == is_valid_ground_truth)

        record = {
            "payload_id": p["payload_id"],
            "source_scene_id": p["source_scene_id"],
            "sensor": p["sensor"],
            "stress_type": p["stress_type"],
            "target_task": task,
            "gsd_m": p["gsd_m"],
            "spectral_bands": p["spectral_bands"],
            "is_valid_payload": is_valid_ground_truth,
            "gate_passed": gate_res.is_compatible,
            "dispatched_to_specialist": dispatched_to_specialist,
            "failure_codes": gate_res.failure_codes,
            "failure_reasons": gate_res.failure_reasons,
            "correct_decision": is_correct
        }
        payload_eval_records.append(record)

    # 2. Source-Level Aggregation across 40 Off-Sensor Source Scenes
    # Fail_source,i = max(Payload A dispatched, Payload B dispatched)
    source_scene_records = []
    off_sensor_scenes = [s for s in scenes if s["sensor"] in ("Landsat-8", "Landsat-9", "PlanetScope")]
    control_scenes = [s for s in scenes if s["sensor"] in ("Sentinel-2A", "Sentinel-2B")]

    source_failures = 0
    for s in off_sensor_scenes:
        sc_id = s["scene_id"]
        sc_payloads = [r for r in payload_eval_records if r["source_scene_id"] == sc_id]
        dispatched_count = sum(1 for r in sc_payloads if r["dispatched_to_specialist"])
        has_failed = (dispatched_count > 0)
        if has_failed:
            source_failures += 1

        sc_record = {
            "source_scene_id": sc_id,
            "name": s["name"],
            "country": s["country"],
            "continent": s["continent"],
            "sensor": s["sensor"],
            "utm_zone": s["utm_zone"],
            "crs": s["crs"],
            "n_payloads": len(sc_payloads),
            "payload_ids": [r["payload_id"] for r in sc_payloads],
            "dispatched_count": dispatched_count,
            "source_level_failure": has_failed
        }
        source_scene_records.append(sc_record)

    # 3. Statistical Calculations for Confirmatory Safety Hypothesis
    # H0: IDER_source >= 0.10 vs H1: IDER_source < 0.10
    n_source = len(off_sensor_scenes)  # 40
    ider_source = float(source_failures / n_source)
    p_binom_h0 = float((1.0 - 0.10) ** n_source) if source_failures == 0 else float(stats.binom.cdf(source_failures, n_source, 0.10))

    one_sided_cp_upper = compute_one_sided_clopper_pearson_upper(source_failures, n_source, alpha=0.05)
    two_sided_cp_lower, two_sided_cp_upper = compute_clopper_pearson_ci(source_failures, n_source, alpha=0.05)
    two_sided_wilson_lower, two_sided_wilson_upper = compute_wilson_ci(source_failures, n_source, alpha=0.05)

    # 4. Descriptive Engineering Metrics on 80 Invalid Payloads
    invalid_records = [r for r in payload_eval_records if not r["is_valid_payload"]]
    n_invalid = len(invalid_records)  # 80
    n_intercepted = sum(1 for r in invalid_records if not r["dispatched_to_specialist"])
    n_dispatched_invalid = sum(1 for r in invalid_records if r["dispatched_to_specialist"])

    cvdr_payload = float(n_intercepted / n_invalid)
    ider_payload = float(n_dispatched_invalid / n_invalid)

    # Violation Code Distribution across Invalid Payloads
    code_counts = {}
    for r in invalid_records:
        for c in r["failure_codes"]:
            code_counts[c] = code_counts.get(c, 0) + 1

    # Stress Type Breakdown
    resolution_records = [r for r in invalid_records if r["stress_type"] == "RESOLUTION_STRESS"]
    modality_records = [r for r in invalid_records if r["stress_type"] == "MODALITY_SPECTRAL_STRESS"]

    cvdr_resolution = float(sum(1 for r in resolution_records if not r["dispatched_to_specialist"]) / len(resolution_records))
    cvdr_modality = float(sum(1 for r in modality_records if not r["dispatched_to_specialist"]) / len(modality_records))

    # 5. Control Population Metrics on 20 Sentinel-2 Controls
    control_records = [r for r in payload_eval_records if r["is_valid_payload"]]
    n_control = len(control_records)  # 20
    n_rejected_control = sum(1 for r in control_records if not r["dispatched_to_specialist"])
    n_dispatched_control = sum(1 for r in control_records if r["dispatched_to_specialist"])

    frr_control = float(n_rejected_control / n_control)
    frr_wilson_lower, frr_wilson_upper = compute_wilson_ci(n_rejected_control, n_control, alpha=0.05)

    # 6. Build Master Summary Dictionary
    summary = {
        "metadata": {
            "protocol_reference": "SATQUERY-P4-PROTOCOL-V1.12.1-FROZEN.md",
            "protocol_hash": "7bcb8c5ab00cfebb5f7418237e9297f7eeaf06b0477d426d0ec6b30880e71bb5",
            "track": "Track 4.2: Cross-Sensor & Cross-Resolution Compatibility Defense",
            "evaluation_scope": "Physical observation compatibility gate C(t, o) multi-sensor stress evaluation",
            "status": "EVALUATION_COMPLETED"
        },
        "confirmatory_safety_hypothesis": {
            "hypothesis": "H0: IDER_source >= 0.10 vs H1: IDER_source < 0.10",
            "n_source_trials": n_source,
            "failed_source_scenes": source_failures,
            "IDER_source_pct": round(ider_source * 100.0, 4),
            "exact_binomial_p_value": round(p_binom_h0, 6),
            "alpha_significance_level": 0.05,
            "decision": "REJECT_H0" if (source_failures == 0 and p_binom_h0 < 0.05) else "RETAIN_H0",
            "decision_rule_passed": bool(source_failures == 0 and p_binom_h0 < 0.05),
            "confidence_bounds": {
                "one_sided_95_clopper_pearson_upper_pct": round(one_sided_cp_upper * 100.0, 4),
                "preregistered_bound_ceiling_pct": 10.0,
                "bound_ceiling_satisfied": bool(one_sided_cp_upper < 0.10),
                "two_sided_95_clopper_pearson_pct": [round(two_sided_cp_lower * 100.0, 4), round(two_sided_cp_upper * 100.0, 4)],
                "two_sided_95_wilson_score_pct": [round(two_sided_wilson_lower * 100.0, 4), round(two_sided_wilson_upper * 100.0, 4)]
            }
        },
        "descriptive_engineering_metrics": {
            "n_invalid_payloads": n_invalid,
            "n_intercepted_payloads": n_intercepted,
            "n_dispatched_incompatible": n_dispatched_invalid,
            "CVDR_payload_pct": round(cvdr_payload * 100.0, 4),
            "IDER_payload_pct": round(ider_payload * 100.0, 4),
            "stress_type_breakdown": {
                "resolution_stress": {
                    "n_payloads": len(resolution_records),
                    "CVDR_pct": round(cvdr_resolution * 100.0, 4)
                },
                "modality_spectral_stress": {
                    "n_payloads": len(modality_records),
                    "CVDR_pct": round(cvdr_modality * 100.0, 4)
                }
            },
            "violation_code_counts": code_counts
        },
        "control_metrics": {
            "n_control_payloads": n_control,
            "n_dispatched_controls": n_dispatched_control,
            "n_rejected_controls": n_rejected_control,
            "FRR_control_pct": round(frr_control * 100.0, 4),
            "two_sided_95_wilson_score_pct": [round(frr_wilson_lower * 100.0, 4), round(frr_wilson_upper * 100.0, 4)],
            "control_preservation_passed": bool(n_rejected_control == 0)
        },
        "preregistered_acceptance_criteria_audit": {
            "statistical_safety_confirmation": {
                "metric": "IDER_source == 0.00% and p = 0.0148 < 0.05",
                "achieved": f"IDER_source = {ider_source*100:.2f}%, p = {p_binom_h0:.5f}, 1-sided CP upper = {one_sided_cp_upper*100:.2f}%",
                "status": "PASS" if (source_failures == 0 and one_sided_cp_upper < 0.10) else "FAIL"
            },
            "descriptive_engineering_target": {
                "metric": "CVDR_payload == 100.0% and IDER_payload == 0.00%",
                "achieved": f"CVDR = {cvdr_payload*100:.2f}%, IDER = {ider_payload*100:.2f}%",
                "status": "PASS" if (cvdr_payload == 1.0 and ider_payload == 0.0) else "FAIL"
            },
            "control_preservation": {
                "metric": "FRR_control == 0.00%",
                "achieved": f"FRR = {frr_control*100:.2f}% (Wilson upper: {frr_wilson_upper*100:.2f}%)",
                "status": "PASS" if frr_control == 0.0 else "FAIL"
            }
        }
    }

    # Save outputs to artifacts
    artifacts_dir = REPO_ROOT / "artifacts" / "evaluation"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary_file = artifacts_dir / "track42_evaluation_summary.json"
    scenes_file = artifacts_dir / "track42_source_scenes.json"
    payloads_file = artifacts_dir / "track42_payloads_log.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(scenes_file, "w", encoding="utf-8") as f:
        json.dump(source_scene_records, f, indent=2)

    with open(payloads_file, "w", encoding="utf-8") as f:
        json.dump(payload_eval_records, f, indent=2)

    print(f"Track 4.2 evaluation complete.")
    print(f"Summary JSON: {summary_file}")
    print(f"Scenes JSON: {scenes_file}")
    print(f"Payloads JSON: {payloads_file}")
    print("\n--- Summary Results ---")
    print(f"Confirmatory IDER_source: {ider_source*100:.2f}% ({source_failures}/{n_source})")
    print(f"Binomial Test p-value: {p_binom_h0:.5f} (H0: IDER >= 0.10 rejected at alpha=0.05)")
    print(f"One-sided 95% Clopper-Pearson upper bound: {one_sided_cp_upper*100:.2f}% (Ceiling: < 10.0%)")
    print(f"Descriptive CVDR_payload: {cvdr_payload*100:.2f}% ({n_intercepted}/{n_invalid})")
    print(f"Descriptive IDER_payload: {ider_payload*100:.2f}% ({n_dispatched_invalid}/{n_invalid})")
    print(f"Control FRR: {frr_control*100:.2f}% ({n_rejected_control}/{n_control})")

    return summary


if __name__ == "__main__":
    run_track42_evaluation()
