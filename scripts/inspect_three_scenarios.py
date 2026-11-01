import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
from satquery.orchestrator import SatQueryEngine
from scripts.benchmark_routing_policy_baselines import load_shared_test_assets, load_n200_scenarios

assets = load_shared_test_assets()
scenarios = load_n200_scenarios(assets)
target_ids = ["BENCH-056", "BENCH-059", "BENCH-062"]
target_scs = {s["scenario_id"]: s for s in scenarios if s["scenario_id"] in target_ids}

engine_prov = SatQueryEngine.create_default(routing_policy="provisional")
engine_clar = SatQueryEngine.create_default(routing_policy="clarify")

for sid in target_ids:
    sc = target_scs[sid]
    q = sc["query"]
    resp_prov = engine_prov.process_query(q, input_data=sc["input_data"])
    resp_clar = engine_clar.process_query(q, input_data=sc["input_data"])
    
    print("=" * 70)
    print("SCENARIO:", sid)
    print("Query:", q)
    print("Expected Task:", sc.get("expected_task"))
    print("Expected Action:", sc.get("expected_action"))
    print("Attached Input Type:", sc.get("attached_input_type"))
    print("Ground Truth is_input_compatible:", sc.get("is_input_compatible"))
    print("PROVISIONAL MODE:")
    print("  Routed Task:", resp_prov.routed_task)
    print("  Status:", resp_prov.status)
    print("  Execution Policy:", resp_prov.execution_policy)
    print("  Router Confidence:", resp_prov.router_confidence)
    print("  Reported Input Compatibility:", resp_prov.input_compatibility)
    print("  Routing Reason:", resp_prov.routing_reason)
    print("CLARIFY MODE:")
    print("  Routed Task:", resp_clar.routed_task)
    print("  Status:", resp_clar.status)
    print("  Execution Policy:", resp_clar.execution_policy)
    print("TRACE (Provisional):")
    for tr in resp_prov.execution_trace:
        print("   ", tr)
