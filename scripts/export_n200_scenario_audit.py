"""
Exports the complete scenario-level audit dataset for the N=200 benchmark.
Combines data/benchmark_n200_catalog.json with artifacts/benchmarks/n200_routing_baselines_comparison.csv
to provide 100% transparent scenario-by-scenario verification across all 5 routing policies.

Output:
- artifacts/benchmarks/n200_scenario_level_audit.csv
"""

import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

def export_audit_csv():
    # 1. Load catalog
    cat_path = REPO_ROOT / "data" / "benchmark_n200_catalog.json"
    with open(cat_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)["scenarios"]
    cat_df = pd.DataFrame(catalog)

    # 2. Load comparison records
    comp_path = REPO_ROOT / "artifacts" / "benchmarks" / "n200_routing_baselines_comparison.csv"
    comp_df = pd.read_csv(comp_path)

    # 3. Repeat catalog 5 times (once per baseline in exact order)
    n_baselines = len(comp_df["baseline"].unique())
    aligned_cat = pd.concat([cat_df] * n_baselines, ignore_index=True)

    # 4. Construct enriched dataframe
    audit_df = pd.DataFrame()
    audit_df["scenario_id"] = aligned_cat["scenario_id"]
    audit_df["cohort"] = aligned_cat["cohort"]
    audit_df["split"] = aligned_cat["split"]
    audit_df["template_family"] = aligned_cat["template_family"]
    audit_df["query"] = aligned_cat["query"]
    audit_df["attached_input_type"] = aligned_cat["attached_input_type"]
    audit_df["input_asset_key"] = aligned_cat["input_asset_key"]
    audit_df["expected_task"] = aligned_cat["expected_task"]
    audit_df["expected_action"] = aligned_cat["expected_action"]
    audit_df["is_supported"] = aligned_cat["is_supported"]
    audit_df["is_input_compatible"] = aligned_cat["is_input_compatible"]
    audit_df["is_ood_or_ambiguous"] = aligned_cat["is_ood_or_ambiguous"]

    audit_df["baseline"] = comp_df["baseline"]
    audit_df["predicted_task"] = comp_df["predicted_task"]
    audit_df["confidence"] = comp_df["confidence"].round(4)
    audit_df["action"] = comp_df["action"]
    audit_df["execution_status"] = comp_df["execution_status"]
    audit_df["incompatible_dispatch"] = comp_df["incompatible_dispatch"]
    audit_df["executed_error"] = comp_df["executed_error"]
    audit_df["latency_ms"] = comp_df["latency_ms"].round(2)

    # 5. Add granular error classification
    def classify_outcome(row):
        act = row["action"]
        exp_act = row["expected_action"]
        is_err = row["executed_error"]
        cohort = row["cohort"]

        if not is_err and act == exp_act:
            if act == "EXECUTED":
                return "CORRECT_DISPATCH_AND_EXECUTION"
            elif act == "HALTED_UNCERTAINTY":
                return "CORRECT_UNCERTAINTY_HALT"
            elif act == "REJECTED_INCOMPATIBLE":
                return "CORRECT_INPUT_REJECTION"
            elif act == "REFUSED_UNAVAILABLE":
                return "CORRECT_UNSUPPORTED_REFUSAL"

        if is_err:
            if row["incompatible_dispatch"]:
                return "ERROR_INCOMPATIBLE_DISPATCH"
            if cohort == "6_out_of_distribution":
                return "ERROR_OOD_EXECUTION"
            if cohort == "4_ambiguous_multi_intent":
                return "ERROR_AMBIGUOUS_EXECUTION"
            if cohort == "5_unsupported_tasks":
                return "ERROR_UNSUPPORTED_EXECUTION"
            if cohort == "1_supported_compatible" and row["predicted_task"] != row["expected_task"]:
                return "ERROR_TASK_MISCLASSIFICATION"
            return "ERROR_OTHER_EXECUTION_FAILURE"

        return "POLICY_INTERCEPTION_OR_DISAMBIGUATION"

    audit_df["error_classification"] = audit_df.apply(classify_outcome, axis=1)

    out_csv = REPO_ROOT / "artifacts" / "benchmarks" / "n200_scenario_level_audit.csv"
    audit_df.to_csv(out_csv, index=False)
    print(f"Exported {len(audit_df)} records to: {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    export_audit_csv()
