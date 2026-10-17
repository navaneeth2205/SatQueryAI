"""
SatQuery AI End-to-End Pipeline CLI (Milestone 3).

Usage:
    # 1. Land cover query with BigEarthNet validation sample
    python scripts/run_pipeline.py --query "What land-cover types are visible in this image?" --sample-idx 0

    # 2. Change detection query with OSCD benchmark sample
    python scripts/run_pipeline.py --query "What changed in this area between 2024 and 2026?" --cd-sample-idx 0

    # 3. Change detection synthetic regression test (identical images -> 0% change)
    python scripts/run_pipeline.py --query "What changed in this scene?" --cd-sample-idx 0 --synthetic-identical

    # 4. Interactive clarification routing policy
    python scripts/run_pipeline.py --query "What changed between 2024 and 2026?" --policy clarify

    # 5. Incompatible input test (single image to change detection)
    python scripts/run_pipeline.py --query "What changed between the two dates?" --sample-idx 0
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from satquery.orchestrator import SatQueryEngine


def main():
    parser = argparse.ArgumentParser(description="Execute SatQuery AI end-to-end pipeline.")
    parser.add_argument("--query", type=str, required=True, help="Natural-language question")
    parser.add_argument("--policy", type=str, default="provisional", choices=["provisional", "clarify"],
                        help="Routing safety policy: 'provisional' (default) or 'clarify'")
    # Single-image / Land Cover inputs
    parser.add_argument("--sample-idx", type=int, default=None, help="Index of sample in BigEarthNet validation split (0-3254)")
    parser.add_argument("--s1", type=str, default=None, help="Path to Sentinel-1 SAR TIFF file")
    parser.add_argument("--s2", type=str, default=None, help="Path to Sentinel-2 Optical TIFF file")
    # Temporal Pair / Change Detection inputs
    parser.add_argument("--cd-sample-idx", type=int, default=None, help="Index of scene in OSCD change detection catalog (0-23)")
    parser.add_argument("--t1", type=str, default=None, help="Path to temporal T1 observation")
    parser.add_argument("--t2", type=str, default=None, help="Path to temporal T2 observation")
    parser.add_argument("--synthetic-identical", action="store_true",
                        help="[Synthetic Regression Path] Force T2 = T1 to test deterministic zero-change behavior")
    parser.add_argument("--json", action="store_true", help="Print raw JSON response")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. Resolve Input Data
    input_data = None
    input_note = "None (Query only)"

    # Change Detection from Catalog
    if args.cd_sample_idx is not None:
        cat_path = repo_root / "data" / "change_detection_catalog.parquet"
        if not cat_path.exists():
            print(f"Error: Change detection catalog not found at {cat_path}. Run scripts/prepare_oscd_benchmark.py first.")
            sys.exit(1)

        cat_df = pd.read_parquet(cat_path)
        if args.cd_sample_idx < 0 or args.cd_sample_idx >= len(cat_df):
            print(f"Error: cd-sample-idx {args.cd_sample_idx} is out of bounds (0 to {len(cat_df)-1}).")
            sys.exit(1)

        row = cat_df.iloc[args.cd_sample_idx]
        t1_path = repo_root / row["t1_path"]
        t2_path = repo_root / row["t2_path"]

        if args.synthetic_identical:
            input_data = {
                "t1": str(t1_path),
                "t2": str(t1_path),  # Forced identical observation
            }
            input_note = f"[Synthetic Regression Path: Identical T1=T2] Scene '{row['scene_name']}' ({row['split']})"
        else:
            input_data = {
                "t1": str(t1_path),
                "t2": str(t2_path),
                "scene_name": str(row["scene_name"]),
                "split": str(row["split"])
            }
            input_note = f"OSCD Benchmark Scene #{args.cd_sample_idx} '{row['scene_name']}' ({row['split']} split, {row['width']}x{row['height']})"

    # Direct Temporal Pair file paths
    elif args.t1 and args.t2:
        input_data = {
            "t1": args.t1,
            "t2": args.t2
        }
        input_note = f"User Temporal Pair: T1={args.t1}, T2={args.t2}"

    # Single-image / BigEarthNet Land Cover
    elif args.sample_idx is not None:
        val_features_path = repo_root / "initial_trails" / "X_val_full.npy"
        val_meta_path = repo_root / "data" / "val_full_metadata.parquet"

        if not val_features_path.exists():
            print(f"Error: Validation feature cache not found at {val_features_path}.")
            sys.exit(1)

        X_val = np.load(val_features_path)
        if args.sample_idx < 0 or args.sample_idx >= len(X_val):
            print(f"Error: sample-idx {args.sample_idx} is out of bounds (0 to {len(X_val)-1}).")
            sys.exit(1)

        features = X_val[args.sample_idx]
        patch_info = {}

        if val_meta_path.exists():
            val_df = pd.read_parquet(val_meta_path)
            row = val_df.iloc[args.sample_idx]
            patch_info = {
                "patch_id": str(row["patch_id"]),
                "ground_truth_labels": list(row["labels"])
            }

        input_data = {
            "features": features,
            **patch_info
        }
        input_note = f"Cached 84-feature vector for sample #{args.sample_idx} ({patch_info.get('patch_id', 'unknown')})"

    elif args.s1 and args.s2:
        input_data = {
            "s1_path": args.s1,
            "s2_path": args.s2
        }
        input_note = f"Raw TIFF pair: S1={args.s1}, S2={args.s2}"

    # 2. Initialize Engine with Configurable Policy
    engine = SatQueryEngine.create_default(routing_policy=args.policy)

    # 3. Process Query
    response = engine.process_query(query=args.query, input_data=input_data)

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
        return

    print("=" * 70)
    print("[SatQuery AI] END-TO-END PIPELINE RESPONSE")
    print("=" * 70)
    print(f"User Query       : \"{response.query}\"")
    print(f"Input Supplied   : {input_note}")
    print(f"Routed Task      : {response.routed_task} ({response.routed_task_display})")
    print(f"Routing Conf     : {response.router_confidence * 100:.2f}%")
    print(f"Execution Policy : {response.execution_policy}")
    print(f"Execution Status : {response.status}")
    print(f"\nFinal Answer     : {response.answer}")

    if response.evidence.detected_classes:
        print(f"\nPredicted Classes: {', '.join(response.evidence.detected_classes)}")

    print("\n--- Evidence Summary ---")
    for pt in response.evidence.evidence_points:
        print(f"  * {pt}")

    print("\n--- Observable Execution Trace ---")
    for step in response.execution_trace:
        print(f"  {step}")
    print("=" * 70)


if __name__ == "__main__":
    main()
