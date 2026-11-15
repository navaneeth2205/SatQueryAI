"""
Milestone M3.3: System Demonstration Runner & Test Infrastructure.
Script: scripts/run_m33_system_demonstration.py

Executes the standardized 20-query demonstration suite across:
- Group 1: Multitemporal Change Detection (Real OSCD test scenes & synthetic parity)
- Group 2: Land-Cover Classification (Multimodal features, per-class thresholds)
- Group 3: Input-Aware Modality Disambiguation (Contract safety, dimension & file checks)
- Group 4: Ambiguous & Borderline Instructions (Safety policy arbitration: provisional vs clarify)
- Group 5: Unsupported Capabilities / VQA Stubs (Zero-hallucination honest stubs)

Profiles hardware, software versions, per-stage latency (P50/P95), and peak RAM/VRAM.
Provides strict failure isolation so individual query issues are recorded without crashing the run.
Includes explicit CUDA synchronization around all GPU timing boundaries and a warm-up pass.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Ensure stdout uses UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import psutil
import torch

from satquery.orchestrator import SatQueryEngine, SatQueryResponse
from satquery.taxonomy import TaskType
from satquery.router import TaskRouter
from satquery.evidence import format_evidence, EvidenceReport
from satquery.specialists.base import SpecialistResult, SpecialistStatus
from satquery.utils.metadata import extract_observation_metadata


def get_system_hardware_info(device_choice: str = "auto") -> Dict[str, Any]:
    """Capture comprehensive hardware, OS, and framework environment metadata."""
    cuda_available = torch.cuda.is_available()
    
    if device_choice == "cpu":
        active_device = "cpu"
    elif device_choice == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was explicitly requested, but torch.cuda.is_available() is False.")
        active_device = "cuda"
    else:
        active_device = "cuda" if cuda_available else "cpu"

    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "active_device": active_device,
        "total_system_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
    }

    if cuda_available and active_device == "cuda":
        dev_idx = 0
        props = torch.cuda.get_device_properties(dev_idx)
        info["gpu_name"] = torch.cuda.get_device_name(dev_idx)
        info["total_vram_mb"] = round(props.total_memory / (1024 ** 2), 2)
        info["cuda_compute_capability"] = f"{props.major}.{props.minor}"
    else:
        info["gpu_name"] = None
        info["total_vram_mb"] = 0.0
        info["cuda_compute_capability"] = None

    return info


def resolve_query_input(
    input_spec: Dict[str, Any],
    ben_features: Optional[np.ndarray],
    ben_df: Optional[pd.DataFrame]
) -> Tuple[Any, Optional[str]]:
    """
    Resolve raw query input assets based on input_specification in the suite.
    Returns (input_data, data_description).
    """
    spec_type = input_spec.get("type", "none")

    if spec_type == "temporal_pair_dict":
        t1 = REPO_ROOT / input_spec["t1_path"]
        t2 = REPO_ROOT / input_spec["t2_path"]
        prod_type = input_spec.get("product_type", "oscd_rgb")
        input_data = {
            "t1": str(t1),
            "t2": str(t2),
            "product_type": prod_type,
            "scene_name": input_spec.get("scene_name", "unknown")
        }
        desc = f"Temporal pair: scene '{input_spec.get('scene_name')}' (split: {input_spec.get('split')})"
        return input_data, desc

    elif spec_type == "multimodal_features":
        s_idx = input_spec.get("sample_index", 0)
        if ben_features is not None and 0 <= s_idx < len(ben_features):
            feats = ben_features[s_idx]
            patch_id = f"patch_{s_idx}"
            labels = []
            if ben_df is not None and 0 <= s_idx < len(ben_df):
                patch_id = str(ben_df.iloc[s_idx].get("patch_id", patch_id))
                labels = list(ben_df.iloc[s_idx].get("labels", []))
            input_data = {
                "features": feats,
                "patch_id": patch_id,
                "ground_truth_labels": labels
            }
            desc = f"Multimodal feature vector (dim: {len(feats)}, patch: {patch_id})"
            return input_data, desc
        else:
            feats = np.zeros(488, dtype=np.float32)
            input_data = {"features": feats, "patch_id": f"synthetic_{s_idx}"}
            return input_data, "Synthetic 488-dim multimodal vector"

    elif spec_type == "incompatible_single_image":
        shape = tuple(input_spec.get("shape", [64, 64, 3]))
        img = np.zeros(shape, dtype=np.uint8)
        return img, f"Single image array (shape: {shape}, dtype: uint8) passed to temporal query"

    elif spec_type == "incompatible_temporal_pair":
        t1 = REPO_ROOT / input_spec.get("t1_path", "data/oscd/test/austin/t1.png")
        t2 = REPO_ROOT / input_spec.get("t2_path", "data/oscd/test/austin/t2.png")
        input_data = {"t1": str(t1), "t2": str(t2)}
        return input_data, "Temporal image pair dictionary passed to single-observation land-cover query"

    elif spec_type == "missing_file_paths":
        t1 = REPO_ROOT / input_spec.get("t1_path", "data/oscd/test/non_existent_scene_t1.png")
        t2 = REPO_ROOT / input_spec.get("t2_path", "data/oscd/test/non_existent_scene_t2.png")
        input_data = {
            "t1": str(t1),
            "t2": str(t2),
            "product_type": input_spec.get("product_type", "oscd_rgb")
        }
        return input_data, "Temporal dictionary with non-existent file paths"

    elif spec_type == "mismatched_dimensions":
        s1 = tuple(input_spec.get("t1_shape", [64, 64, 3]))
        s2 = tuple(input_spec.get("t2_shape", [128, 128, 3]))
        im1 = np.zeros(s1, dtype=np.uint8)
        im2 = np.zeros(s2, dtype=np.uint8)
        return (im1, im2), f"Tuple of arrays with mismatched dimensions: {s1} vs {s2}"

    elif spec_type == "single_image":
        shape = tuple(input_spec.get("shape", [64, 64, 3]))
        img = np.zeros(shape, dtype=np.uint8)
        return img, f"Single image array (shape: {shape})"

    elif spec_type == "none":
        return None, "No input imagery attached (query-only)"

    return None, f"Unknown specification type: {spec_type}"


def execute_single_query_timed(
    engine: SatQueryEngine,
    query_text: str,
    input_data: Any,
    device: str = "cpu"
) -> Tuple[SatQueryResponse, Dict[str, float]]:
    """
    Executes a query with microsecond-resolution timing and CUDA synchronization across:
    - Stage 1: Router latency (observation extraction + linguistic routing)
    - Stage 2: Specialist execution latency (model forward pass / inference)
    - Stage 3: Evidence synthesis & formatting latency
    - Total end-to-end pipeline latency
    """
    timings: Dict[str, float] = {}
    is_cuda = (device == "cuda" and torch.cuda.is_available())

    # --- Stage 1: Router ---
    if is_cuda:
        torch.cuda.synchronize()
    t_r_start = time.perf_counter()

    obs_meta = extract_observation_metadata(input_data)
    route = engine.router.predict(query_text, image_metadata=obs_meta)

    if is_cuda:
        torch.cuda.synchronize()
    t_r_end = time.perf_counter()
    timings["router_ms"] = (t_r_end - t_r_start) * 1000.0

    trace = [
        f'[Trace 1/5] Ingested query: "{query_text}"',
        f"[Trace 2/5] Task Router classified intent as '{route.task}' with {route.confidence*100:.1f}% confidence."
    ]
    if not route.input_compatibility:
        trace.append(f"[Disambiguation Trace] {route.routing_reason}")

    # Check clarification policy early-exit
    if route.requires_review and engine.routing_policy == "clarify":
        timings["specialist_ms"] = 0.0

        if is_cuda:
            torch.cuda.synchronize()
        t_fmt_start = time.perf_counter()

        execution_policy = "CLARIFICATION_REQUIRED"
        trace.append(
            f"[Safety Trace] Routing confidence {route.confidence*100:.1f}% is below threshold ({engine.router.confidence_threshold*100:.1f}%). "
            f"Policy: CLARIFICATION_REQUIRED — Automated execution halted. User clarification requested."
        )
        clarification_answer = (
            f"[CLARIFICATION REQUIRED: Task routing confidence is {route.confidence*100:.1f}%, which is below the review threshold of {engine.router.confidence_threshold*100:.1f}%. "
            f"The router tentatively identified '{route.display_name}', but please clarify or confirm your analytical goal before automated specialist execution.]"
        )
        empty_report = EvidenceReport(
            answer=clarification_answer,
            status="NEEDS_CLARIFICATION",
            detected_classes=[],
            class_probabilities={},
            thresholds_applied={},
            evidence_points=[
                f"[Status] NEEDS_CLARIFICATION",
                f"[Confidence] {route.confidence*100:.1f}% (threshold: {engine.router.confidence_threshold*100:.1f}%)",
                f"[Tentative Task] {route.display_name} ({route.task})"
            ],
            sensor_feature_summary={"route": route.to_dict()}
        )
        response = SatQueryResponse(
            query=query_text,
            status="NEEDS_CLARIFICATION",
            execution_policy=execution_policy,
            routed_task=route.task,
            routed_task_display=route.display_name,
            router_confidence=route.confidence,
            requires_human_review=True,
            answer=clarification_answer,
            evidence=empty_report,
            execution_trace=trace,
            timestamp=datetime.now(timezone.utc).isoformat(),
            text_task=route.text_task,
            text_confidence=route.text_confidence,
            selected_task=route.selected_task,
            input_compatibility=route.input_compatibility,
            routing_reason=route.routing_reason,
            attached_modality=route.attached_modality
        )

        if is_cuda:
            torch.cuda.synchronize()
        t_fmt_end = time.perf_counter()
        timings["formatting_ms"] = (t_fmt_end - t_fmt_start) * 1000.0
        timings["total_pipeline_ms"] = timings["router_ms"] + timings["formatting_ms"]

        return response, timings

    # Execution policy for non-clarification routes
    execution_policy = "PROVISIONAL_EXECUTION" if route.requires_review else "STANDARD_EXECUTION"
    if route.requires_review:
        trace.append(
            f"[Safety Trace] Routing confidence {route.confidence*100:.1f}% is below threshold ({engine.router.confidence_threshold*100:.1f}%). "
            f"Policy: PROVISIONAL_EXECUTION — specialist executes provisionally, but routing intent is flagged for human review."
        )

    # --- Stage 2: Specialist Execution ---
    try:
        task_enum = TaskType(route.task)
    except ValueError:
        task_enum = None

    if is_cuda:
        torch.cuda.synchronize()
    t_s_start = time.perf_counter()

    if task_enum is None:
        specialist_result = SpecialistResult(
            status=SpecialistStatus.UNAVAILABLE,
            task=route.task,
            unavailable_reason=f"Task '{route.task}' is not recognized in the active taxonomy."
        )
    else:
        trace.append(f"[Trace 3/5] Dispatching to Specialist Registry for task '{route.task}'...")
        specialist_result = engine.registry.dispatch(task_enum, input_data)

    if is_cuda:
        torch.cuda.synchronize()
    t_s_end = time.perf_counter()
    timings["specialist_ms"] = (t_s_end - t_s_start) * 1000.0

    # --- Stage 3: Evidence Formatting & Synthesis ---
    if is_cuda:
        torch.cuda.synchronize()
    t_fmt_start = time.perf_counter()

    if specialist_result.status == SpecialistStatus.UNAVAILABLE:
        status = "UNAVAILABLE_SPECIALIST"
        trace.append(f"[Trace 4/5] Task Unavailable: {specialist_result.unavailable_reason}")
        evidence = format_evidence(specialist_result)
    elif specialist_result.status == SpecialistStatus.INCOMPATIBLE_INPUT:
        status = "INCOMPATIBLE_INPUT"
        trace.append(f"[Trace 4/5] Input Error: {specialist_result.unavailable_reason}.")
        evidence = format_evidence(specialist_result)
    elif specialist_result.status == SpecialistStatus.SUCCESS:
        status = "PROVISIONAL_SUCCESS" if execution_policy == "PROVISIONAL_EXECUTION" else "SUCCESS"
        specialist_name = specialist_result.execution_metadata.get("specialist_name", "RegisteredSpecialist")
        trace.append(f"[Trace 4/5] Specialist '{specialist_name}' executed successfully.")
        evidence = format_evidence(specialist_result)
        if execution_policy == "PROVISIONAL_EXECUTION":
            evidence.answer = (
                f"[PROVISIONAL ROUTE: Routing intent confidence is {route.confidence*100:.1f}%, below the review threshold of {engine.router.confidence_threshold*100:.1f}%. "
                f"Specialist executed provisionally. Scores shown are model voting probabilities, not routing confidence.] "
                + evidence.answer
            )
    else:
        status = "FAILED"
        trace.append(f"[Trace 4/5] Unknown specialist status: {specialist_result.status}.")
        evidence = format_evidence(specialist_result)

    trace.append(f"[Trace 5/5] Formatted explainable answer with {len(evidence.evidence_points)} evidence points.")
    response = SatQueryResponse(
        query=query_text,
        status=status,
        execution_policy=execution_policy,
        routed_task=route.task,
        routed_task_display=route.display_name,
        router_confidence=route.confidence,
        requires_human_review=route.requires_review,
        answer=evidence.answer,
        evidence=evidence,
        execution_trace=trace,
        timestamp=datetime.now(timezone.utc).isoformat(),
        text_task=route.text_task,
        text_confidence=route.text_confidence,
        selected_task=route.selected_task,
        input_compatibility=route.input_compatibility,
        routing_reason=route.routing_reason,
        attached_modality=route.attached_modality
    )

    if is_cuda:
        torch.cuda.synchronize()
    t_fmt_end = time.perf_counter()
    timings["formatting_ms"] = (t_fmt_end - t_fmt_start) * 1000.0
    timings["total_pipeline_ms"] = timings["router_ms"] + timings["specialist_ms"] + timings["formatting_ms"]

    return response, timings


def run_demonstration_suite(
    suite_path: Path,
    device: str = "auto",
    policy: str = "provisional",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes all 20 queries in the suite, capturing performance and safety metrics.
    Includes a 1-query warm-up pass before starting timing measurements.
    """
    sys_info = get_system_hardware_info(device_choice=device)
    target_device = sys_info["active_device"]
    
    if target_device == "cuda":
        torch.cuda.reset_peak_memory_stats(0)

    # Initialize SatQueryEngine under specified policy and target compute device
    print(f"\n[SatQuery AI Demo Runner] Initializing Engine on device '{target_device}' with policy '{policy}'...", flush=True)
    engine = SatQueryEngine.create_default(routing_policy=policy, device=target_device)

    # Warm-up pass (1 dummy inference to compile kernels & load CUDA context)
    if verbose:
        print("[SatQuery AI Demo Runner] Executing warm-up inference pass...", flush=True)
    try:
        dummy_pair = (np.full((64, 64, 3), 128, dtype=np.uint8), np.full((64, 64, 3), 128, dtype=np.uint8))
        _ = engine.registry.dispatch(TaskType.CHANGE_DETECTION, dummy_pair)
        if target_device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        print(f"[Warning] Warm-up pass encountered: {e}", flush=True)

    # Load suite definition
    with open(suite_path, "r", encoding="utf-8") as f:
        suite = json.load(f)

    # Preload feature matrices if available
    ben_p2_path = REPO_ROOT / "data" / "X_val_phase2.npy"
    ben_features = np.load(ben_p2_path) if ben_p2_path.exists() else None
    
    ben_meta_path = REPO_ROOT / "data" / "val_full_metadata.parquet"
    ben_df = pd.read_parquet(ben_meta_path) if ben_meta_path.exists() else None

    # Track memory
    process = psutil.Process(os.getpid())
    ram_initial_mb = process.memory_info().rss / (1024 ** 2)

    results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    print("=" * 100, flush=True)
    print(f"  SATQUERY AI — MILESTONE M3.3 SYSTEM DEMONSTRATION SUITE ({suite.get('suite_id')})", flush=True)
    print(f"  Runtime: Python {sys_info['python_version']} | PyTorch {sys_info['pytorch_version']} | Device: {target_device} ({sys_info.get('gpu_name') or 'CPU'})", flush=True)
    print(f"  Safety Policy: {policy.upper()} | Frozen Intent Threshold: {engine.router.confidence_threshold:.2f}", flush=True)
    print("=" * 100, flush=True)

    query_count = 0
    success_count = 0
    incompatible_count = 0
    clarification_count = 0
    unavailable_count = 0
    error_count = 0

    for group in suite.get("groups", []):
        gid = group.get("group_id")
        gname = group.get("group_name")
        print(f"\n--- Group {gid}: {gname} ---", flush=True)

        for q_item in group.get("queries", []):
            qid = q_item["query_id"]
            q_text = q_item["query_text"]
            input_spec = q_item.get("input_specification", {})
            desc_split = q_item.get("dataset_split_attribution", "unknown")
            query_count += 1

            # Prepare input data safely
            try:
                input_data, input_desc = resolve_query_input(input_spec, ben_features, ben_df)
            except Exception as e:
                input_data = None
                input_desc = f"Failed to resolve input: {e}"

            # Memory before query
            vram_before_mb = (torch.cuda.memory_allocated(0) / (1024 ** 2)) if target_device == "cuda" else 0.0

            # Execute with failure isolation
            record: Dict[str, Any] = {
                "query_id": qid,
                "group_id": gid,
                "group_name": gname,
                "query_text": q_text,
                "dataset_split_attribution": desc_split,
                "input_description": input_desc,
                "policy": policy,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            try:
                response, timings = execute_single_query_timed(engine, q_text, input_data, device=target_device)
                
                vram_after_mb = (torch.cuda.memory_allocated(0) / (1024 ** 2)) if target_device == "cuda" else 0.0
                ram_after_mb = process.memory_info().rss / (1024 ** 2)

                record.update({
                    "execution_status": response.status,
                    "execution_policy": response.execution_policy,
                    "routed_task": response.routed_task,
                    "routed_task_display": response.routed_task_display,
                    "router_confidence": round(response.router_confidence, 4),
                    "requires_human_review": response.requires_human_review,
                    "input_compatibility": response.input_compatibility,
                    "timings_ms": {k: round(v, 2) for k, v in timings.items()},
                    "vram_allocated_mb": round(vram_after_mb, 2),
                    "process_ram_mb": round(ram_after_mb, 2),
                    "evidence_summary": {
                        "detected_classes": response.evidence.detected_classes,
                        "num_evidence_points": len(response.evidence.evidence_points),
                        "answer_snippet": response.answer[:160] + ("..." if len(response.answer) > 160 else "")
                    },
                    "error": None
                })

                latencies.append(timings["total_pipeline_ms"])

                # Tally mutually exclusive terminal outcomes
                if response.status in ("SUCCESS", "PROVISIONAL_SUCCESS"):
                    success_count += 1
                elif response.status == "INCOMPATIBLE_INPUT":
                    incompatible_count += 1
                elif response.status == "NEEDS_CLARIFICATION":
                    clarification_count += 1
                elif response.status == "UNAVAILABLE_SPECIALIST":
                    unavailable_count += 1
                else:
                    error_count += 1

                # Format CLI summary line
                status_color = response.status
                print(
                    f"  [{qid}] {q_text[:48]:<48} | Task: {response.routed_task:<24} | "
                    f"Conf: {response.router_confidence*100:5.1f}% | Status: {status_color:<22} | "
                    f"Time: {timings['total_pipeline_ms']:7.1f} ms",
                    flush=True
                )

                if verbose and response.requires_human_review:
                    print(f"        -> [Safety Policy Review Triggered] Policy Action: {response.execution_policy}", flush=True)

            except Exception as e:
                error_count += 1
                err_msg = str(e)
                tb = traceback.format_exc()
                record.update({
                    "execution_status": "EXECUTION_ERROR",
                    "execution_policy": "ERROR_ABORT",
                    "error": {
                        "message": err_msg,
                        "traceback": tb
                    },
                    "timings_ms": {"total_pipeline_ms": 0.0}
                })
                print(f"  [{qid}] EXCEPTION ENCOUNTERED: {err_msg}", flush=True)

            results.append(record)

    # Compute summary statistics
    ram_peak_mb = process.memory_info().rss / (1024 ** 2)
    vram_peak_mb = (torch.cuda.max_memory_allocated(0) / (1024 ** 2)) if target_device == "cuda" else 0.0

    latency_arr = np.array(latencies) if latencies else np.array([0.0])
    summary_metrics = {
        "total_queries": query_count,
        "successful_executions": success_count,
        "incompatible_inputs_intercepted": incompatible_count,
        "clarifications_required": clarification_count,
        "unavailable_stubs_handled": unavailable_count,
        "unexpected_errors": error_count,
        "outcome_accounting_reconciliation": {
            "success": success_count,
            "incompatible": incompatible_count,
            "clarification": clarification_count,
            "unavailable": unavailable_count,
            "unexpected_errors": error_count,
            "reconciled_total": success_count + incompatible_count + clarification_count + unavailable_count + error_count
        },
        "latency_statistics_ms": {
            "mean": round(float(np.mean(latency_arr)), 2),
            "std": round(float(np.std(latency_arr)), 2),
            "p50": round(float(np.percentile(latency_arr, 50)), 2),
            "p90": round(float(np.percentile(latency_arr, 90)), 2),
            "p95": round(float(np.percentile(latency_arr, 95)), 2),
            "min": round(float(np.min(latency_arr)), 2),
            "max": round(float(np.max(latency_arr)), 2)
        },
        "memory_profile_mb": {
            "initial_process_ram": round(ram_initial_mb, 2),
            "peak_process_ram": round(ram_peak_mb, 2),
            "peak_vram_allocated": round(vram_peak_mb, 2)
        }
    }

    print("\n" + "=" * 100, flush=True)
    print("  DEMONSTRATION SUITE EXECUTION SUMMARY", flush=True)
    print("=" * 100, flush=True)
    print(f"  Total Queries Tested       : {query_count}", flush=True)
    print(f"  Specialist Successes       : {success_count}", flush=True)
    print(f"  Incompatible Interceptions : {incompatible_count}", flush=True)
    print(f"  Clarifications Required    : {clarification_count}", flush=True)
    print(f"  Unavailable Stubs Handled  : {unavailable_count}", flush=True)
    print(f"  Unexpected Failures        : {error_count}", flush=True)
    print(f"  Outcome Reconciliation     : {summary_metrics['outcome_accounting_reconciliation']['reconciled_total']} / {query_count} reconciled", flush=True)
    print(f"  Latency P50                : {summary_metrics['latency_statistics_ms']['p50']} ms", flush=True)
    print(f"  Latency P95                : {summary_metrics['latency_statistics_ms']['p95']} ms", flush=True)
    print(f"  Peak Process RAM           : {summary_metrics['memory_profile_mb']['peak_process_ram']} MB", flush=True)
    if target_device == "cuda":
        print(f"  Peak GPU VRAM Allocated    : {summary_metrics['memory_profile_mb']['peak_vram_allocated']} MB", flush=True)
    print("=" * 100 + "\n", flush=True)

    return {
        "environment": sys_info,
        "policy": policy,
        "frozen_thresholds": suite.get("frozen_parameters", {}),
        "dataset_accounting": suite.get("dataset_accounting", {}),
        "summary": summary_metrics,
        "query_results": results
    }


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Milestone M3.3 System Demonstration Runner.")
    parser.add_argument("--suite", type=str, default="artifacts/benchmarks/m33_query_suite.json", help="Path to query suite JSON.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto", help="Compute device to target.")
    parser.add_argument("--policy", type=str, choices=["provisional", "clarify", "both"], default="both", help="Routing safety policy to evaluate.")
    parser.add_argument("--output-json", type=str, default="artifacts/benchmarks/m33_system_demonstration_results.json", help="Destination path for structured benchmark results.")
    parser.add_argument("--append-result", action="store_true", help="Merge results with existing JSON if file exists.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose execution logs.")
    args = parser.parse_args()

    suite_path = REPO_ROOT / args.suite
    if not suite_path.exists():
        print(f"Error: Suite file '{suite_path}' does not exist.")
        sys.exit(1)

    output_path = REPO_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policies = ["provisional", "clarify"] if args.policy == "both" else [args.policy]
    
    # Load existing benchmark results if appending
    existing_data: Dict[str, Any] = {}
    if args.append_result and output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"[Warning] Could not load existing file for merge: {e}")

    runs_output: Dict[str, Any] = existing_data if existing_data else {
        "suite_id": "M3.3-DEMO-20Q",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": {}
    }

    for pol in policies:
        run_res = run_demonstration_suite(
            suite_path=suite_path,
            device=args.device,
            policy=pol,
            verbose=not args.quiet
        )
        dev_tag = run_res["environment"]["active_device"]
        run_key = f"{dev_tag}_{pol}"
        runs_output["runs"][run_key] = run_res

    # Save output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(runs_output, f, indent=2)

    print(f"[SatQuery AI Demo Runner] Benchmark results exported to: {output_path}")


if __name__ == "__main__":
    main()
