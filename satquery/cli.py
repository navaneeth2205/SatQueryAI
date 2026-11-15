"""
Command-Line Interface (CLI) and Application Entrypoint for SatQuery AI.

Provides production-ready subcommands:
  - query:  End-to-end multi-modal query execution with input validation contracts
  - route:  Fast linguistic task intent routing and policy review without neural loading
  - demo:   Start the interactive web demonstration application
  - verify: Cryptographic artifact integrity verification and healthcheck
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np


class JsonLogFormatter(logging.Formatter):
    """Structured JSON formatter for production container and microservice logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "correlation_id"):
            log_obj["correlation_id"] = record.correlation_id
        return json.dumps(log_obj)


def configure_logging(json_logs: bool = False, verbose: bool = False) -> None:
    """Configure system-wide logging formatting."""
    log_level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)

    if json_logs or os.environ.get("SATQUERY_LOG_FORMAT", "").lower() == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        fmt = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def handle_verify(args: argparse.Namespace) -> int:
    """Verify artifact SHA-256 integrity against the frozen deployment manifest."""
    logger = logging.getLogger("satquery.cli.verify")
    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / "artifacts" / "deployment" / "m33_production_freeze_manifest.json"

    if not manifest_path.exists():
        logger.error(f"Deployment manifest not found at: {manifest_path}")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    import hashlib
    artifacts = manifest.get("artifacts", {})
    all_passed = True
    results = {}

    for name, meta in artifacts.items():
        rel_path = meta.get("path")
        target_path = repo_root / rel_path
        expected_sha = meta.get("sha256")
        expected_size = meta.get("size_bytes")

        if not target_path.exists():
            logger.error(f"[FAIL] Missing artifact: {name} ({rel_path})")
            results[name] = {"status": "MISSING", "path": rel_path}
            all_passed = False
            continue

        actual_size = target_path.stat().st_size
        h = hashlib.sha256()
        with open(target_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual_sha = h.hexdigest()

        if actual_sha == expected_sha and actual_size == expected_size:
            results[name] = {"status": "PASS", "size": actual_size, "sha256": actual_sha[:16] + "..."}
            logger.info(f"[PASS] {name} verified.")
        else:
            logger.error(f"[FAIL] Hash mismatch for {name}: expected {expected_sha}, got {actual_sha}")
            results[name] = {
                "status": "CHECKSUM_MISMATCH",
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha
            }
            all_passed = False

    # Check CUDA device availability if requested
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_info = {
            "torch_version": torch.__version__,
            "cuda_available": cuda_avail,
            "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        }
    except Exception as e:
        device_info = {"torch_version": "unavailable", "error": str(e)}

    output_payload = {
        "manifest_version": manifest.get("manifest_version"),
        "milestone": manifest.get("milestone"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_artifacts_intact": all_passed,
        "device_info": device_info,
        "results": results
    }

    if args.json_output:
        print(json.dumps(output_payload, indent=2))
    else:
        status_str = "SUCCESS (All 13 artifacts verified)" if all_passed else "FAILURE (Artifact verification failed)"
        logger.info(f"Verification status: {status_str}")

    return 0 if all_passed else 1


def handle_route(args: argparse.Namespace) -> int:
    """Fast linguistic routing without loading heavy vision neural backends."""
    logger = logging.getLogger("satquery.cli.route")
    start_time = time.perf_counter()

    repo_root = Path(__file__).resolve().parent.parent
    router_path = repo_root / "artifacts" / "router" / "router_model.joblib"

    if not router_path.exists():
        logger.error(f"Router model not found at {router_path}")
        return 1

    from satquery.router import TaskRouter
    from satquery.utils.metadata import InputObservationMetadata
    from satquery.taxonomy import InputType

    router = TaskRouter.load(router_path)

    image_meta = None
    if args.input_type:
        try:
            inp_type = InputType(args.input_type)
            image_meta = InputObservationMetadata(input_type=inp_type)
        except ValueError:
            logger.error(f"Invalid input type '{args.input_type}'. Valid: {[t.value for t in InputType]}")
            return 1

    route = router.predict(args.query_text, image_metadata=image_meta)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    output_payload = route.to_dict()
    output_payload["latency_ms"] = round(latency_ms, 2)

    if args.json_output:
        print(json.dumps(output_payload, indent=2))
    else:
        logger.info(
            f"Query: \"{args.query_text}\" -> Task: {route.task} (Conf: {route.confidence*100:.1f}%) "
            f"[Review: {route.requires_review}, Compatible: {route.input_compatibility}] in {latency_ms:.2f}ms"
        )

    return 0


def handle_query(args: argparse.Namespace) -> int:
    """Execute end-to-end multimodal query with input validation contracts."""
    logger = logging.getLogger("satquery.cli.query")
    start_time = time.perf_counter()

    # 1. Input Validation Contract
    input_data = None
    if args.t1 or args.t2:
        if not (args.t1 and args.t2):
            logger.error("Both --t1 and --t2 must be provided for bi-temporal change detection.")
            return 1
        t1_path = Path(args.t1)
        t2_path = Path(args.t2)
        if not t1_path.exists():
            logger.error(f"Supplied --t1 path does not exist: {t1_path}")
            return 1
        if not t2_path.exists():
            logger.error(f"Supplied --t2 path does not exist: {t2_path}")
            return 1

        input_data = {
            "t1": str(t1_path),
            "t2": str(t2_path),
            "method": args.cd_method
        }
    elif args.features:
        feat_path = Path(args.features)
        if not feat_path.exists():
            logger.error(f"Supplied --features path does not exist: {feat_path}")
            return 1
        try:
            if feat_path.suffix.lower() == ".npy":
                feats = np.load(feat_path)
            elif feat_path.suffix.lower() == ".json":
                with open(feat_path, "r", encoding="utf-8") as f:
                    feats = np.array(json.load(f), dtype=np.float32)
            else:
                logger.error(f"Unsupported features format '{feat_path.suffix}'. Expected .npy or .json")
                return 1
            input_data = feats
        except Exception as e:
            logger.error(f"Failed to load features from {feat_path}: {e}")
            return 1

    # 2. Instantiate Production Engine
    from satquery.orchestrator import SatQueryEngine
    from satquery.specialists.registry import SpecialistRegistry
    from satquery.specialists.land_cover import LandCoverClassifier
    from satquery.specialists.change_detection import ChangeDetectionSpecialist
    from satquery.router import TaskRouter

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        repo_root = Path(__file__).resolve().parent.parent
        router_path = repo_root / "artifacts" / "router" / "router_model.joblib"
        router = TaskRouter.load(router_path)

        registry = SpecialistRegistry()
        registry.register(LandCoverClassifier())

        # If classical fallback explicitly requested, bypass deep neural checkpoint
        ckpt = Path("force_classical_fallback") if args.cd_method == "gaussian" else None
        registry.register(ChangeDetectionSpecialist(checkpoint_path=ckpt, device=device))

        engine = SatQueryEngine(router=router, registry=registry, routing_policy=args.policy)
    except Exception as e:
        logger.error(f"Failed to initialize SatQueryEngine: {e}")
        return 1

    # 3. Process Query
    response = engine.process_query(args.query_text, input_data=input_data)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    output_payload = response.to_dict()
    output_payload["latency_ms"] = round(latency_ms, 2)

    if args.output_file:
        out_p = Path(args.output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        logger.info(f"Results written to {out_p}")

    if args.json_output:
        print(json.dumps(output_payload, indent=2))
    else:
        logger.info(
            f"Result: [{response.status}] Task: {response.selected_task or response.routed_task} "
            f"| Confidence: {response.router_confidence*100:.1f}% | Latency: {latency_ms:.2f}ms"
        )
        print("\n=== SATQUERY AI RESPONSE ===")
        print(f"Status:   {response.status}")
        print(f"Task:     {response.selected_task or response.routed_task}")
        print(f"Answer:   {response.answer}")
        if response.status == "CLARIFICATION_REQUIRED":
            print(f"Notice:   Linguistic confidence below threshold. Clarification requested.")
        print("============================\n")

    if response.status == "SUCCESS":
        return 0
    elif response.status in ("CLARIFICATION_REQUIRED", "NEEDS_CLARIFICATION"):
        return 2
    else:
        return 1


def handle_demo(args: argparse.Namespace) -> int:
    """Launch the interactive web service."""
    from satquery.demo.app import create_app
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct command line arguments parser."""
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--json-logs", action="store_true", help="Emit logs as structured JSON lines")
    common_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")

    parser = argparse.ArgumentParser(
        prog="satquery",
        description="SatQuery AI: Interactive Vision-Language Assistant for Remote Sensing Imagery",
        parents=[common_parser]
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # verify
    p_verify = subparsers.add_parser(
        "verify",
        parents=[common_parser],
        help="Cryptographic verification of production artifacts"
    )
    p_verify.add_argument("--json-output", action="store_true", help="Output verification report as JSON")
    p_verify.set_defaults(func=handle_verify)

    # route
    p_route = subparsers.add_parser(
        "route",
        parents=[common_parser],
        help="Fast intent routing and policy check"
    )
    p_route.add_argument("query_text", type=str, help="Natural language query string")
    p_route.add_argument(
        "--input-type",
        type=str,
        default=None,
        choices=["temporal_pair", "multimodal_stack", "single_optical", "none"],
        help="Optional input observation modality"
    )
    p_route.add_argument("--json-output", action="store_true", help="Output routing result as JSON")
    p_route.set_defaults(func=handle_route)

    # query
    p_query = subparsers.add_parser(
        "query",
        parents=[common_parser],
        help="Execute multimodal query end-to-end"
    )
    p_query.add_argument("query_text", type=str, help="Natural language query string")
    p_query.add_argument("--t1", type=str, default=None, help="Path to pre-event image (T1)")
    p_query.add_argument("--t2", type=str, default=None, help="Path to post-event image (T2)")
    p_query.add_argument("--features", type=str, default=None, help="Path to .npy or .json feature vector")
    p_query.add_argument(
        "--cd-method",
        type=str,
        default="standard",
        choices=["standard", "gaussian"],
        help="Change detection algorithm (standard deep Siamese or gaussian CVA fallback)"
    )
    p_query.add_argument(
        "--policy",
        type=str,
        default="provisional",
        choices=["provisional", "clarify"],
        help="Routing safety review policy"
    )
    p_query.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Hardware execution target"
    )
    p_query.add_argument("--json-output", action="store_true", help="Output result as JSON to stdout")
    p_query.add_argument("--output-file", "-o", type=str, default=None, help="Save result JSON to file")
    p_query.set_defaults(func=handle_query)

    # demo
    p_demo = subparsers.add_parser(
        "demo",
        parents=[common_parser],
        help="Launch interactive web demonstration service"
    )
    p_demo.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    p_demo.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    p_demo.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    p_demo.set_defaults(func=handle_demo)

    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(json_logs=args.json_logs, verbose=args.verbose)
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
