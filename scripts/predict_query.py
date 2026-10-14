"""
Inference CLI for SatQuery AI Task Router (Milestone 1).

Usage:
    python scripts/predict_query.py "What land-cover types are visible in this image?"
    python scripts/predict_query.py "Compare optical and radar imagery." --json
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from satquery.router import TaskRouter


def main():
    parser = argparse.ArgumentParser(description="Classify a natural language satellite query.")
    parser.add_argument("query", type=str, help="Natural language query string")
    parser.add_argument(
        "--model",
        type=str,
        default=str(repo_root / "artifacts" / "router" / "router_model.joblib"),
        help="Path to trained router_model.joblib"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable text"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of attached images (1 for single, 2 for temporal pair)"
    )
    parser.add_argument(
        "--sensors",
        type=str,
        default="optical",
        help="Comma-separated sensor types, e.g., 'optical,sar'"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model artifact not found at {model_path}.")
        print("Please train the model first by running:")
        print("  python scripts/train_router.py")
        sys.exit(1)

    router = TaskRouter.load(model_path)

    metadata = {
        "image_count": args.num_images,
        "sensor_types": [s.strip().lower() for s in args.sensors.split(",")]
    }

    route = router.predict(args.query, image_metadata=metadata)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.json:
        print(json.dumps(route.to_dict(), indent=2))
        return

    print("=" * 65)
    print("[SatQuery AI] QUERY ROUTING RESULT")
    print("=" * 65)
    print(f"Query            : \"{route.query}\"")
    print(f"Target Task      : {route.task} ({route.display_name})")
    print(f"Required Input   : {route.required_input.upper()}")
    print(f"Specialist Model : {route.specialist_model_family}")
    print(f"Confidence Score : {route.confidence * 100:.2f}%")
    
    review_status = "[FLAGGED FOR REVIEW] (Low Confidence)" if route.requires_review else "[AUTOMATED DISPATCH] (Confident)"
    print(f"Review Status    : {review_status}")

    print("\n--- Observable Execution Trace ---")
    for step in route.execution_trace:
        print(f"  {step}")

    print("\n--- Top Class Probabilities ---")
    sorted_probs = sorted(route.class_probabilities.items(), key=lambda x: x[1], reverse=True)[:4]
    for task_name, prob in sorted_probs:
        print(f"  - {task_name:<30}: {prob * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
