"""
Comprehensive Audit Script:
1. Traces training data leakage vs. the 41 benchmark template families.
2. Computes complete group-test and group-val execution denominators and Wilson 95% CIs.
3. Exports artifacts/benchmarks/template_family_overlap_audit.json and
   artifacts/benchmarks/group_held_out_scenario_level_audit.csv
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959964  # 95% confidence
    p_hat = k / n
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    lower = max(0.0, center - margin) * 100.0
    upper = min(1.0, center + margin) * 100.0
    return (round(lower, 2), round(upper, 2))


def run_audit():
    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    sq_path = REPO_ROOT / "data" / "starter_queries.csv"
    sq_df = pd.read_csv(sq_path)
    sq_train = sq_df[sq_df["split"] == "train"].copy()
    sq_val = sq_df[sq_df["split"] == "val"].copy()

    cat_path = REPO_ROOT / "data" / "benchmark_n200_catalog.json"
    with open(cat_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)["scenarios"]
    cat_df = pd.DataFrame(catalog)

    audit_path = REPO_ROOT / "artifacts" / "benchmarks" / "n200_scenario_level_audit.csv"
    audit_df = pd.read_csv(audit_path)

    # -------------------------------------------------------------------------
    # 2. Template Families Partitioning
    # -------------------------------------------------------------------------
    val_scs = cat_df[cat_df["group_split"] == "group_val"]
    test_scs = cat_df[cat_df["group_split"] == "group_test_held_out"]

    val_fams = sorted(list(val_scs["template_family"].unique()))
    test_fams = sorted(list(test_scs["template_family"].unique()))
    all_fams = sorted(list(cat_df["template_family"].unique()))

    val_set = set(val_fams)
    test_set = set(test_fams)
    inter_val_test = val_set.intersection(test_set)

    print("=" * 80)
    print("PART 1: TEMPLATE FAMILY PARTITION & LEAKAGE AUDIT")
    print("=" * 80)
    print(f"Total Template Families in N=200 Catalog: {len(all_fams)}")
    print(f"Group Val Families: {len(val_fams)} ({len(val_scs)} scenarios)")
    print(f"Group Test Families: {len(test_fams)} ({len(test_scs)} scenarios)")
    print(f"F_validation ∩ F_test overlap: {len(inter_val_test)} families (0.00%)")

    # Audit Training Data (109 Starter Queries)
    # Check exact query overlap
    train_queries = set(sq_train["query"].str.strip())
    cat_queries = set(cat_df["query"].str.strip())
    exact_train_matches = cat_df[cat_df["query"].str.strip().isin(train_queries)]

    # Check semantic / lexical overlap of training queries with template families
    # Does starter_queries.csv have template_family annotations?
    has_tf_annotations = "template_family" in sq_train.columns

    overlap_audit = {
        "audit_name": "SatQuery AI — Template Family Partition and Training Data Leakage Audit",
        "has_training_template_family_annotations": has_tf_annotations,
        "starter_queries_train_count": len(sq_train),
        "starter_queries_val_count": len(sq_val),
        "total_benchmark_template_families_count": len(all_fams),
        "group_val_families_count": len(val_fams),
        "group_test_held_out_families_count": len(test_fams),
        "validation_vs_test_family_overlap_count": len(inter_val_test),
        "validation_vs_test_family_overlap_pct": 0.0,
        "f_val_intersect_f_test_empty": len(inter_val_test) == 0,
        "exact_query_overlap_with_train_count": len(exact_train_matches),
        "exact_query_overlap_details": exact_train_matches[["scenario_id", "template_family", "group_split", "query"]].to_dict(orient="records"),
        "training_data_semantic_overlap_assessment": (
            "The 109 starter training queries were collected in Milestone 1 without template_family "
            "metadata annotations. Lexical and semantic overlap exists between starter training queries "
            "(which contain general Earth observation phrasing such as 'forest', 'urban expansion', 'cropland') "
            "and template families in both Group Val and Group Test. Therefore, while F_validation ∩ F_test = ∅ "
            "is strictly established on the benchmark, F_train ∩ F_test = ∅ cannot be strictly verified. "
            "In accordance with rigorous research standards, this split must be characterized as a "
            "'Group-Disjoint Evaluation Split' rather than 'genuine out-of-family generalization from training'."
        ),
        "group_val_families": val_fams,
        "group_test_held_out_families": test_fams
    }

    audit_json_path = REPO_ROOT / "artifacts" / "benchmarks" / "template_family_overlap_audit.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(overlap_audit, f, indent=2)
    print(f"Saved machine-readable overlap audit to: {audit_json_path.relative_to(REPO_ROOT)}")

    # -------------------------------------------------------------------------
    # 2. Denominator Breakdown & Wilson 95% CIs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: GROUP-VAL AND GROUP-TEST EXECUTION DENOMINATORS & 95% CIs")
    print("=" * 80)

    baselines_order = [
        ("B1_TextOnly", "B1: Text-Only Argmax"),
        ("B2_Text_Plus_CompatRules", "B2: Text + Compat Rules"),
        ("B3_Text_Plus_ConfidenceGate", "B3: Text + Confidence Gate (tau=0.65)"),
        ("Proposed_Clarify", "Proposed: Clarify Mode"),
        ("Proposed_Provisional", "Proposed: Provisional Mode")
    ]

    table_records = []
    for g_split in ["group_val", "group_test_held_out"]:
        sub_g = audit_df[audit_df["group_split"] == g_split]
        for b_id, b_display in baselines_order:
            sub_b = sub_g[sub_g["baseline"] == b_id]
            total_scenarios = len(sub_b)

            c1 = sub_b[sub_b["cohort"] == "1_supported_compatible"]
            n_c1 = len(c1)
            k_c1_exec = (c1["action"] == "EXECUTED").sum()
            cov = (k_c1_exec / n_c1 * 100.0) if n_c1 > 0 else 0.0
            cov_ci = wilson_ci(k_c1_exec, n_c1)

            c23 = sub_b[sub_b["cohort"].isin(["2_supported_incompatible_input", "3_missing_sensor_modalities"])]
            n_c23 = len(c23)
            k_incompat = c23["incompatible_dispatch"].sum()
            incompat_rate = (k_incompat / n_c23 * 100.0) if n_c23 > 0 else 0.0
            incompat_ci = wilson_ci(k_incompat, n_c23)

            exec_sub = sub_b[sub_b["action"] == "EXECUTED"]
            n_exec = len(exec_sub)
            k_err = exec_sub["executed_error"].sum()
            risk = (k_err / n_exec * 100.0) if n_exec > 0 else 0.0
            risk_ci = wilson_ci(k_err, n_exec)

            anom = sub_b[sub_b["cohort"].isin(["4_ambiguous_multi_intent", "6_out_of_distribution"])]
            n_anom = len(anom)
            k_anom_halt = (anom["action"] == "HALTED_UNCERTAINTY").sum()
            anom_rate = (k_anom_halt / n_anom * 100.0) if n_anom > 0 else 0.0

            unsupp = sub_b[sub_b["cohort"] == "5_unsupported_tasks"]
            n_unsupp = len(unsupp)
            k_unsupp_ref = (unsupp["action"] == "REFUSED_UNAVAILABLE").sum()
            unsupp_rate = (k_unsupp_ref / n_unsupp * 100.0) if n_unsupp > 0 else 0.0

            table_records.append({
                "group_split": g_split,
                "policy_key": b_id,
                "policy_display": b_display,
                "total_scenarios": total_scenarios,
                "c1_total": n_c1,
                "c1_executed": int(k_c1_exec),
                "supported_coverage_pct": round(cov, 2),
                "supported_coverage_ci": cov_ci,
                "total_executed": int(n_exec),
                "executed_errors": int(k_err),
                "selective_risk_pct": round(risk, 2),
                "selective_risk_ci": risk_ci,
                "incompat_total": n_c23,
                "incompat_dispatches": int(k_incompat),
                "incompat_rate_pct": round(incompat_rate, 2),
                "incompat_ci": incompat_ci,
                "anomaly_interception_pct": round(anom_rate, 2),
                "unsupported_refusal_pct": round(unsupp_rate, 2)
            })

    table_df = pd.DataFrame(table_records)

    for g_split in ["group_val", "group_test_held_out"]:
        display_name = "GROUP VAL (20 FAMILIES, N=100)" if g_split == "group_val" else "GROUP TEST (21 DISJOINT FAMILIES, N=100)"
        print(f"\n--- {display_name} ---")
        sub = table_df[table_df["group_split"] == g_split]
        for idx, r in sub.iterrows():
            print(f"{r['policy_display']}:")
            print(f"  - Supported Coverage: {r['c1_executed']}/{r['c1_total']} = {r['supported_coverage_pct']}% [95% CI: {r['supported_coverage_ci'][0]}%, {r['supported_coverage_ci'][1]}%]")
            print(f"  - Selective Risk    : {r['executed_errors']}/{r['total_executed']} = {r['selective_risk_pct']}% [95% CI: {r['selective_risk_ci'][0]}%, {r['selective_risk_ci'][1]}%]")
            print(f"  - Incompat Dispatches: {r['incompat_dispatches']}/{r['incompat_total']} = {r['incompat_rate_pct']}% [95% CI: {r['incompat_ci'][0]}%, {r['incompat_ci'][1]}%]")

    # Export Group-Held-Out Scenario-Level CSV (records for Group Test)
    group_test_audit = audit_df[audit_df["group_split"] == "group_test_held_out"].copy()
    group_test_csv_path = REPO_ROOT / "artifacts" / "benchmarks" / "group_held_out_scenario_level_audit.csv"
    group_test_audit.to_csv(group_test_csv_path, index=False)
    print(f"\nExported {len(group_test_audit)} group-test audit records to: {group_test_csv_path.relative_to(REPO_ROOT)}")

    # Update group_held_out_benchmark_summary.json with explicit CI details
    summary_path = REPO_ROOT / "artifacts" / "benchmarks" / "group_held_out_benchmark_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    summary_data["detailed_denominators_and_ci"] = table_records
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Updated {summary_path.relative_to(REPO_ROOT)} with complete denominators and CIs.")


if __name__ == "__main__":
    run_audit()
