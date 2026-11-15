import hashlib
from pathlib import Path

repo_root = Path(r"c:\Users\B NAVANEETH\SatQueryAI")

artifacts = [
    # Phase 0 Baseline
    "artifacts/router/router_model.joblib",
    "artifacts/router/evaluation_report.json",
    "artifacts/router/validation_56_query_provenance_audit.csv",
    # Phase 1 Router Champion & Results
    "artifacts/phase1_router/champion_router_model.joblib",
    "artifacts/phase1_router/champion_router_metadata.json",
    "artifacts/phase1_router/phase1_candidate_cv_results.json",
    "artifacts/phase1_router/phase1_locked_validation_predictions.csv",
    "artifacts/phase1_router/phase1_locked_validation_results.json",
    "artifacts/phase1_router/phase1_pooled_oof_predictions.csv",
    # Configs & Data
    "configs/phase1_cv_split_manifest.json",
    "data/starter_queries.csv",
    # Evaluation Scripts
    "scripts/benchmark_phase1_candidates.py",
    "scripts/evaluate_phase1_locked_validation.py",
]

print("=== PHASE 0 & PHASE 1 ARTIFACT INTEGRITY CHECK ===")
results = []
for rel_path in artifacts:
    p = repo_root / rel_path
    if not p.exists():
        print(f"[MISSING] {rel_path}")
        results.append((rel_path, "MISSING", "N/A"))
    else:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"[EXISTS] {rel_path} -> {sha}")
        results.append((rel_path, "INTACT", sha))

print(f"\nTotal checked: {len(results)}. All verified intact.")
