"""
Comprehensive End-to-End Evaluation Benchmark Suite for SatQuery AI.

Evaluates the integrated pipeline:
  Natural Language Query + Imagery Input -> Task Router -> Compatibility Arbitration -> Specialist -> Evidence Synthesis

Tests 28 diverse scenarios across:
1. Supported Change Detection (Real OSCD scenes, synthetic zero-change, localized perturbation)
2. Supported Land Cover (Phase 2 LightGBM 488 features, Baseline RF 84 features, Raw TIFFs)
3. Unsupported Specialized Tasks (Change VQA, Single-Image VQA, Captioning, Visual Grounding)
4. Input Incompatibilities & Safeguards (Single img for change, temporal pair for land cover, unscaled floats, double-scaling guard)
5. Safety Review & Policy Arbitration (Low confidence under provisional vs clarify policies)
6. Paraphrase Invariance

Outputs:
- artifacts/benchmarks/end_to_end_evaluation_summary.json
- artifacts/benchmarks/end_to_end_scenarios.csv
"""

import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from satquery.orchestrator import SatQueryEngine
from satquery.taxonomy import TaskType


def build_scenario_catalog() -> List[Dict[str, Any]]:
    """Build the curated 28-scenario test matrix."""
    scenarios = []

    # Load benchmark data sources
    oscd_path = REPO_ROOT / "data" / "change_detection_catalog.parquet"
    oscd_df = pd.read_parquet(oscd_path) if oscd_path.exists() else pd.DataFrame()

    val_meta_path = REPO_ROOT / "data" / "val_full_metadata.parquet"
    val_df = pd.read_parquet(val_meta_path) if val_meta_path.exists() else pd.DataFrame()

    p2_feats_path = REPO_ROOT / "data" / "X_val_phase2.npy"
    p2_feats = np.load(p2_feats_path) if p2_feats_path.exists() else None

    p1_feats_path = REPO_ROOT / "initial_trails" / "X_val_full.npy"
    p1_feats = np.load(p1_feats_path) if p1_feats_path.exists() else None

    # Helper paths
    pisa_row = oscd_df[oscd_df["scene_name"] == "pisa"].iloc[0] if not oscd_df.empty else None
    beirut_row = oscd_df[oscd_df["scene_name"] == "beirut"].iloc[0] if not oscd_df.empty else None
    mumbai_row = oscd_df[oscd_df["scene_name"] == "mumbai"].iloc[0] if not oscd_df.empty else None

    # ----------------------------------------------------
    # Category 1: Supported Change Detection (6 scenarios)
    # ----------------------------------------------------
    if pisa_row is not None:
        scenarios.append({
            "id": "CD-01",
            "category": "Change Detection",
            "description": "Real held-out test scene (Pisa) bi-temporal change detection",
            "query": "What changed in this area between the acquisition dates?",
            "input_data": {
                "t1": str(REPO_ROOT / pisa_row["t1_path"]),
                "t2": str(REPO_ROOT / pisa_row["t2_path"]),
                "product_type": "oscd_rgb"
            },
            "policy": "provisional",
            "expected_task": "CHANGE_DETECTION",
            "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
        })

        scenarios.append({
            "id": "CD-02",
            "category": "Change Detection",
            "description": "Synthetic identical pair (T1=T2) zero false-positive change verification",
            "query": "Detect surface modifications across these two dates.",
            "input_data": {
                "t1": str(REPO_ROOT / pisa_row["t1_path"]),
                "t2": str(REPO_ROOT / pisa_row["t1_path"]),
                "product_type": "oscd_rgb"
            },
            "policy": "provisional",
            "expected_task": "CHANGE_DETECTION",
            "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"],
            "custom_check": lambda res: res.evidence.sensor_feature_summary.get("changed_area_pct_of_valid", -1) == 0.0
        })

    if mumbai_row is not None:
        scenarios.append({
            "id": "CD-03",
            "category": "Change Detection",
            "description": "Held-out test scene (Mumbai) coastal urban alteration detection",
            "query": "Identify newly developed or modified ground structures between 2017 and 2018.",
            "input_data": {
                "t1": str(REPO_ROOT / mumbai_row["t1_path"]),
                "t2": str(REPO_ROOT / mumbai_row["t2_path"]),
                "product_type": "oscd_rgb"
            },
            "policy": "provisional",
            "expected_task": "CHANGE_DETECTION",
            "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
        })

    if beirut_row is not None:
        scenarios.append({
            "id": "CD-04",
            "category": "Change Detection",
            "description": "Training scene (Beirut) infrastructure change verification",
            "query": "Map all significant surface variations observed in this pair.",
            "input_data": {
                "t1": str(REPO_ROOT / beirut_row["t1_path"]),
                "t2": str(REPO_ROOT / beirut_row["t2_path"]),
                "product_type": "oscd_rgb"
            },
            "policy": "provisional",
            "expected_task": "CHANGE_DETECTION",
            "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
        })

    # Synthetic perturbation
    np.random.seed(42)
    syn_t1 = np.full((64, 64, 3), 0.25, dtype=np.float32)
    syn_t2 = syn_t1.copy()
    syn_t2[20:40, 20:40, :] = 0.85  # 400 pixel perturbation
    scenarios.append({
        "id": "CD-05",
        "category": "Change Detection",
        "description": "Controlled synthetic perturbation cluster & bounding-box detection",
        "query": "Detect and highlight localized ground alterations in this temporal pair.",
        "input_data": {"t1": syn_t1, "t2": syn_t2, "product_type": "pre_normalized"},
        "policy": "provisional",
        "expected_task": "CHANGE_DETECTION",
        "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"],
        "custom_check": lambda res: res.evidence.sensor_feature_summary.get("changed_pixels", 0) > 0
    })

    scenarios.append({
        "id": "CD-06",
        "category": "Change Detection",
        "description": "Paraphrase query for change vector analysis",
        "query": "Compute multitemporal delta and show modified zones between T1 and T2.",
        "input_data": {"t1": syn_t1, "t2": syn_t2, "product_type": "pre_normalized"},
        "policy": "provisional",
        "expected_task": "CHANGE_DETECTION",
        "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
    })

    # ----------------------------------------------------
    # Category 2: Supported Land Cover (6 scenarios)
    # ----------------------------------------------------
    if p2_feats is not None:
        row0 = val_df.iloc[0] if not val_df.empty else {}
        scenarios.append({
            "id": "LC-01",
            "category": "Land Cover Classification",
            "description": "Phase 2 LightGBM model with 488 multimodal features on validation patch",
            "query": "What land-cover classes are present in this satellite scene?",
            "input_data": {
                "features": p2_feats[0],
                "patch_id": str(row0.get("patch_id", "patch_0")),
                "ground_truth_labels": list(row0.get("labels", []))
            },
            "policy": "provisional",
            "expected_task": "LAND_COVER_CLASSIFICATION",
            "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"],
            "custom_check": lambda res: res.evidence.sensor_feature_summary.get("feature_count") == 488
        })

        row1 = val_df.iloc[1] if not val_df.empty else {}
        scenarios.append({
            "id": "LC-02",
            "category": "Land Cover Classification",
            "description": "Agricultural land-cover query with Phase 2 488 features",
            "query": "Identify arable land, pastures, and vegetation categories in this area.",
            "input_data": {
                "features": p2_feats[1],
                "patch_id": str(row1.get("patch_id", "patch_1")),
                "ground_truth_labels": list(row1.get("labels", []))
            },
            "policy": "provisional",
            "expected_task": "LAND_COVER_CLASSIFICATION",
            "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
        })

        row2 = val_df.iloc[2] if not val_df.empty else {}
        scenarios.append({
            "id": "LC-03",
            "category": "Land Cover Classification",
            "description": "Urban and artificial surface classification query",
            "query": "Classify artificial surfaces and urban fabric in this satellite tile.",
            "input_data": {
                "features": p2_feats[2],
                "patch_id": str(row2.get("patch_id", "patch_2")),
                "ground_truth_labels": list(row2.get("labels", []))
            },
            "policy": "provisional",
            "expected_task": "LAND_COVER_CLASSIFICATION",
            "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
        })

    if p1_feats is not None:
        scenarios.append({
            "id": "LC-04",
            "category": "Land Cover Classification",
            "description": "Baseline Random Forest fallback with 84 global statistical features",
            "query": "Predict multi-label surface categories visible in this footprint.",
            "input_data": {"features": p1_feats[0]},
            "policy": "provisional",
            "expected_task": "LAND_COVER_CLASSIFICATION",
            "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"],
            "custom_check": lambda res: res.evidence.sensor_feature_summary.get("feature_count") == 84
        })

    # Raw TIFF input
    s1_sample = REPO_ROOT / "tests" / "data" / "raw_tiffs" / "s1_sample.tif"
    s2_sample = REPO_ROOT / "tests" / "data" / "raw_tiffs" / "s2_sample.tif"
    if s1_sample.exists() and s2_sample.exists():
        scenarios.append({
            "id": "LC-05",
            "category": "Land Cover Classification",
            "description": "Direct raw TIFF inference parity extraction (S1 2-band + S2 10-band)",
            "query": "Analyze surface reflectance and backscatter to classify land cover.",
            "input_data": {"s1_path": str(s1_sample), "s2_path": str(s2_sample)},
            "policy": "provisional",
            "expected_task": "LAND_COVER_CLASSIFICATION",
            "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
        })

    scenarios.append({
        "id": "LC-06",
        "category": "Land Cover Classification",
        "description": "Forest and woodland query with Phase 2 features",
        "query": "Detect broad-leaved, coniferous, or mixed forests across this region.",
        "input_data": {"features": p2_feats[3] if p2_feats is not None else np.zeros(488)},
        "policy": "provisional",
        "expected_task": "LAND_COVER_CLASSIFICATION",
        "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
    })

    # ----------------------------------------------------
    # Category 3: Unsupported Specialized Tasks (5 scenarios)
    # ----------------------------------------------------
    scenarios.append({
        "id": "UNSUP-01",
        "category": "Unsupported Task Stubs",
        "description": "Change VQA query (honest UNAVAILABLE_SPECIALIST status)",
        "query": "Did the reservoir water boundary shrink between these two dates?",
        "input_data": None,
        "policy": "provisional",
        "expected_task": "CHANGE_VQA",
        "expected_status": ["UNAVAILABLE_SPECIALIST"]
    })

    scenarios.append({
        "id": "UNSUP-02",
        "category": "Unsupported Task Stubs",
        "description": "Single-Image VQA query (honest UNAVAILABLE_SPECIALIST status)",
        "query": "How many commercial airplanes are parked at this airport runway?",
        "input_data": None,
        "policy": "provisional",
        "expected_task": "SINGLE_IMAGE_VQA",
        "expected_status": ["UNAVAILABLE_SPECIALIST"]
    })

    scenarios.append({
        "id": "UNSUP-03",
        "category": "Unsupported Task Stubs",
        "description": "Scene Captioning query (honest UNAVAILABLE_SPECIALIST status)",
        "query": "Generate a detailed descriptive natural-language caption for this satellite scene.",
        "input_data": None,
        "policy": "provisional",
        "expected_task": "SCENE_CAPTIONING",
        "expected_status": ["UNAVAILABLE_SPECIALIST"]
    })

    scenarios.append({
        "id": "UNSUP-04",
        "category": "Unsupported Task Stubs",
        "description": "Visual Grounding query (honest UNAVAILABLE_SPECIALIST status)",
        "query": "Locate the primary solar farm and generate a polygon segmentation mask around it.",
        "input_data": None,
        "policy": "provisional",
        "expected_task": "VISUAL_GROUNDING",
        "expected_status": ["UNAVAILABLE_SPECIALIST"]
    })

    scenarios.append({
        "id": "UNSUP-05",
        "category": "Unsupported Task Stubs",
        "description": "Optical-SAR fusion synthesis stub (honest UNAVAILABLE_SPECIALIST status)",
        "query": "Fuse SAR backscatter and optical spectral bands into a unified composite image.",
        "input_data": None,
        "policy": "provisional",
        "expected_task": "OPTICAL_SAR_FUSION",
        "expected_status": ["UNAVAILABLE_SPECIALIST"]
    })

    # ----------------------------------------------------
    # Category 4: Input Incompatibilities & Safeguards (5 scenarios)
    # ----------------------------------------------------
    scenarios.append({
        "id": "INCOMPAT-01",
        "category": "Input Compatibility",
        "description": "Single image passed with a change detection query (missing T2)",
        "query": "What changed in this scene over time?",
        "input_data": np.zeros((64, 64, 3), dtype=np.float32),  # Single 3D array
        "policy": "provisional",
        "expected_task": "CHANGE_DETECTION",
        "expected_status": ["INCOMPATIBLE_INPUT"]
    })

    scenarios.append({
        "id": "INCOMPAT-02",
        "category": "Input Compatibility",
        "description": "Mismatched dimensions in temporal observation pair",
        "query": "Detect surface changes between these two images.",
        "input_data": (np.zeros((64, 64, 3), dtype=np.float32), np.zeros((64, 80, 3), dtype=np.float32)),
        "policy": "provisional",
        "expected_task": "CHANGE_DETECTION",
        "expected_status": ["INCOMPATIBLE_INPUT"]
    })

    scenarios.append({
        "id": "INCOMPAT-03",
        "category": "Input Compatibility",
        "description": "Ambiguous raw unscaled float array (> 1.0) without declared product_type",
        "query": "Find modifications between these two floating-point observations.",
        "input_data": (np.full((32, 32, 3), 150.0, dtype=np.float32), np.full((32, 32, 3), 150.0, dtype=np.float32)),
        "policy": "provisional",
        "expected_task": "CHANGE_DETECTION",
        "expected_status": ["INCOMPATIBLE_INPUT"],
        "custom_check": lambda res: "Ambiguous floating-point" in res.evidence.answer or "Ambiguous" in str(res.execution_trace)
    })

    scenarios.append({
        "id": "INCOMPAT-04",
        "category": "Input Compatibility",
        "description": "Sentinel-2 reflectance values exceeding physical BOA ceiling (> 10000)",
        "query": "Detect change using these Sentinel-2 L2A observations.",
        "input_data": {
            "t1": np.full((32, 32, 3), 13000.0, dtype=np.float32),
            "t2": np.full((32, 32, 3), 5000.0, dtype=np.float32),
            "product_type": "s2_l2a"
        },
        "policy": "provisional",
        "expected_task": "CHANGE_DETECTION",
        "expected_status": ["INCOMPATIBLE_INPUT"],
        "custom_check": lambda res: "Sentinel-2 L2A reflectance range" in res.evidence.answer or "Sentinel-2" in str(res.execution_trace)
    })

    scenarios.append({
        "id": "INCOMPAT-05",
        "category": "Input Compatibility",
        "description": "Pre-normalized float observation (in [0, 1]) passed under product_type='s2_l2a'",
        "query": "Detect modifications between these two Sentinel-2 observations.",
        "input_data": {
            "t1": np.full((32, 32, 3), 0.45, dtype=np.float32),
            "t2": np.full((32, 32, 3), 0.45, dtype=np.float32),
            "product_type": "s2_l2a"
        },
        "policy": "provisional",
        "expected_task": "CHANGE_DETECTION",
        "expected_status": ["INCOMPATIBLE_INPUT"],
        "custom_check": lambda res: "pre_normalized" in res.evidence.answer or "pre_normalized" in str(res.execution_trace)
    })

    # ----------------------------------------------------
    # Category 5: Safety Policies & Low-Confidence Review (4 scenarios)
    # ----------------------------------------------------
    scenarios.append({
        "id": "SAFETY-01",
        "category": "Safety Policy",
        "description": "Vague / low-confidence query under 'provisional' policy (executes with disclaimer)",
        "query": "Is there anything interesting happening around this place?",
        "input_data": None,
        "policy": "provisional",
        "expected_task": None,  # Any task
        "expected_status": ["PROVISIONAL_SUCCESS", "UNAVAILABLE_SPECIALIST", "NEEDS_CLARIFICATION"],
        "custom_check": lambda res: res.requires_human_review is True
    })

    scenarios.append({
        "id": "SAFETY-02",
        "category": "Safety Policy",
        "description": "Vague / low-confidence query under 'clarify' policy (halts and requests clarification)",
        "query": "Analyze this place.",
        "input_data": None,
        "policy": "clarify",
        "expected_task": None,
        "expected_status": ["NEEDS_CLARIFICATION"],
        "custom_check": lambda res: res.status == "NEEDS_CLARIFICATION" and res.requires_human_review is True
    })

    scenarios.append({
        "id": "SAFETY-03",
        "category": "Safety Policy",
        "description": "Out-of-distribution non-EO question under 'clarify' policy",
        "query": "What will the temperature and rainfall be tomorrow morning?",
        "input_data": None,
        "policy": "clarify",
        "expected_task": None,
        "expected_status": ["NEEDS_CLARIFICATION"],
        "custom_check": lambda res: res.status == "NEEDS_CLARIFICATION"
    })

    scenarios.append({
        "id": "SAFETY-04",
        "category": "Safety Policy",
        "description": "High-confidence clear query under 'clarify' policy (executes normally)",
        "query": "What land-cover classes are present in this satellite scene?",
        "input_data": {"features": p2_feats[0] if p2_feats is not None else np.zeros(488)},
        "policy": "clarify",
        "expected_task": "LAND_COVER_CLASSIFICATION",
        "expected_status": ["SUCCESS"]
    })

    # ----------------------------------------------------
    # Category 6: Input-Aware Disambiguation (2 scenarios)
    # ----------------------------------------------------
    scenarios.append({
        "id": "DISAMBIG-01",
        "category": "Input-Aware Disambiguation",
        "description": "Ambiguous query with temporal pair attached resolves to Change Detection",
        "query": "Analyze surface differences across these observations.",
        "input_data": {"t1": syn_t1, "t2": syn_t2, "product_type": "pre_normalized"},
        "policy": "provisional",
        "expected_task": "CHANGE_DETECTION",
        "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
    })

    scenarios.append({
        "id": "DISAMBIG-02",
        "category": "Input-Aware Disambiguation",
        "description": "Ambiguous query with single feature vector attached resolves to Land Cover",
        "query": "Identify and list all terrain categories.",
        "input_data": {"features": p2_feats[0] if p2_feats is not None else np.zeros(488)},
        "policy": "provisional",
        "expected_task": "LAND_COVER_CLASSIFICATION",
        "expected_status": ["SUCCESS", "PROVISIONAL_SUCCESS"]
    })

    return scenarios


def run_end_to_end_evaluation():
    print("================================================================================")
    print("      SATQUERY AI — END-TO-END PIPELINE EVALUATION BENCHMARK (N=28)")
    print("================================================================================")
    
    scenarios = build_scenario_catalog()
    print(f"Total evaluation scenarios loaded: {len(scenarios)}")
    
    # Pre-instantiate engines for efficiency
    engine_provisional = SatQueryEngine.create_default(routing_policy="provisional")
    engine_clarify = SatQueryEngine.create_default(routing_policy="clarify")

    results = []
    latencies = []
    passes = 0
    failures = 0

    print("\nExecuting test matrix...\n")
    print(f"{'ID':<12} | {'Category':<24} | {'Policy':<11} | {'Status':<22} | {'Latency':<8} | {'Pass'}")
    print("-" * 88)

    for sc in scenarios:
        sid = sc["id"]
        cat = sc["category"]
        policy = sc["policy"]
        engine = engine_clarify if policy == "clarify" else engine_provisional

        t0 = time.perf_counter()
        response = engine.process_query(sc["query"], input_data=sc["input_data"])
        lat = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(lat)

        # Verification checks
        passed = True
        fail_reasons = []

        if sc["expected_task"] is not None:
            actual_task = response.selected_task or response.routed_task
            if actual_task != sc["expected_task"]:
                passed = False
                fail_reasons.append(f"Task mismatch: expected {sc['expected_task']}, got {actual_task}")

        if sc["expected_status"] is not None:
            if response.status not in sc["expected_status"]:
                passed = False
                fail_reasons.append(f"Status mismatch: expected {sc['expected_status']}, got {response.status}")

        if "custom_check" in sc:
            try:
                if not sc["custom_check"](response):
                    passed = False
                    fail_reasons.append("Custom validation condition failed")
            except Exception as e:
                passed = False
                fail_reasons.append(f"Custom validation exception: {e}")

        if passed:
            passes += 1
            verdict = "[PASS]"
        else:
            failures += 1
            verdict = "[FAIL]"

        print(f"{sid:<12} | {cat:<24} | {policy:<11} | {response.status:<22} | {lat:6.1f}ms | {verdict}")
        if not passed:
            for r in fail_reasons:
                print(f"   -> Reason: {r}")

        results.append({
            "scenario_id": sid,
            "category": cat,
            "description": sc["description"],
            "query": sc["query"],
            "policy": policy,
            "passed": passed,
            "fail_reasons": "; ".join(fail_reasons) if fail_reasons else "None",
            "latency_ms": round(lat, 2),
            "routed_task": response.routed_task,
            "selected_task": response.selected_task,
            "router_confidence": round(response.router_confidence, 4),
            "requires_review": response.requires_human_review,
            "status": response.status,
            "execution_policy": response.execution_policy,
            "answer_snippet": response.answer[:120].replace("\n", " "),
            "evidence_count": len(response.evidence.evidence_points)
        })

    # Summary metrics
    pass_rate = (passes / len(scenarios)) * 100.0
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    mean_lat = float(np.mean(latencies))

    print("\n================================================================================")
    print("                         BENCHMARK EVALUATION SUMMARY")
    print("================================================================================")
    print(f"Total Scenarios Tested : {len(scenarios)}")
    print(f"Scenarios Passed       : {passes} / {len(scenarios)} ({pass_rate:.1f}%)")
    print(f"Scenarios Failed       : {failures}")
    print(f"Mean Latency           : {mean_lat:.1f} ms")
    print(f"Median Latency (p50)   : {p50_lat:.1f} ms")
    print(f"95th Percentile (p95)  : {p95_lat:.1f} ms")
    print("================================================================================")

    # Breakdown by category
    df = pd.DataFrame(results)
    cat_summary = df.groupby("category").agg(
        total=("passed", "count"),
        passed=("passed", "sum"),
        mean_latency_ms=("latency_ms", "mean")
    ).reset_index()
    cat_summary["pass_rate_pct"] = (cat_summary["passed"] / cat_summary["total"]) * 100.0
    print("\nPerformance Breakdown by Category:")
    print(cat_summary.to_string(index=False))

    # Save artifacts
    artifacts_dir = REPO_ROOT / "artifacts" / "benchmarks"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "end_to_end_scenarios.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved scenario execution records to: {csv_path}")

    summary_json = {
        "benchmark_title": "SatQuery AI End-to-End Pipeline Evaluation Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_scenarios": len(scenarios),
        "passed": passes,
        "failed": failures,
        "pass_rate_pct": round(pass_rate, 2),
        "latency_stats_ms": {
            "mean": round(mean_lat, 2),
            "median_p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "min": round(float(np.min(latencies)), 2),
            "max": round(float(np.max(latencies)), 2)
        },
        "category_breakdown": cat_summary.to_dict(orient="records"),
        "scenarios": results
    }

    json_path = artifacts_dir / "end_to_end_evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved evaluation summary JSON to: {json_path}")

    return pass_rate == 100.0


if __name__ == "__main__":
    success = run_end_to_end_evaluation()
    sys.exit(0 if success else 1)
