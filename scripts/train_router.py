"""
Training Script for SatQuery AI Task Router (Milestone 1).

Usage:
    python scripts/train_router.py [--config configs/router_config.json]
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from satquery.dataset import load_queries_dataset, get_dataset_statistics
from satquery.router import TaskRouter
from satquery.evaluation import evaluate_router


def main():
    parser = argparse.ArgumentParser(description="Train SatQuery AI Task Router.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "configs" / "router_config.json"),
        help="Path to router config JSON"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data_path = repo_root / config.get("dataset_paths", {}).get("starter_queries", "data/starter_queries.csv")
    artifacts_dir = repo_root / config.get("artifacts_dir", "artifacts/router")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[SatQuery AI] TRAINING TASK ROUTING LAYER (Milestone 1)")
    print("=" * 60)
    print(f"Config Version    : {config.get('version')}")
    print(f"Dataset Path      : {data_path}")
    print(f"Artifacts Output  : {artifacts_dir}")

    # 1. Load Data
    train_df, val_df = load_queries_dataset(data_path)
    train_stats = get_dataset_statistics(train_df)
    val_stats = get_dataset_statistics(val_df)

    print(f"\n[Data] Loaded {train_stats['total_samples']} training samples, {val_stats['total_samples']} validation samples.")
    print(f"[Data] Number of target task classes: {len(train_stats['task_counts'])}")

    # 2. Initialize Router
    vec_cfg = config.get("vectorizer", {})
    clf_cfg = config.get("classifier", {})

    router = TaskRouter(
        confidence_threshold=config.get("confidence_threshold", 0.65),
        version=config.get("version", "0.1.0"),
        ngram_range=tuple(vec_cfg.get("ngram_range", [1, 2])),
        C=clf_cfg.get("C", 1.0),
        random_state=clf_cfg.get("random_state", 42)
    )

    # 3. Train
    print("\n[Training] Fitting TF-IDF Vectorizer and Calibrated Logistic Classifier...")
    router.fit(train_df["query"].values, train_df["task"].values)

    # 4. Save Model Artifact
    model_path = artifacts_dir / "router_model.joblib"
    router.save(model_path)
    print(f"[Artifact] Saved model to: {model_path}")

    # 5. Evaluate on Validation Set
    print("\n[Evaluation] Evaluating on held-out validation queries...")
    metrics = evaluate_router(router, val_df, artifacts_dir=artifacts_dir)

    print("\n" + "=" * 60)
    print("[SatQuery AI] EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy   : {metrics['overall_accuracy']*100:.2f}%")
    print(f"Macro F1 Score     : {metrics['macro_f1']:.4f}")
    print(f"Micro F1 Score     : {metrics['micro_f1']:.4f}")
    print(f"Average Confidence : {metrics['average_confidence']*100:.2f}%")
    print(f"Review Rate (< {router.confidence_threshold:.2f}): {metrics['review_rate']*100:.2f}%")
    print(f"Confusion Matrix   : {metrics['confusion_matrix_path']}")
    print(f"Report JSON        : {artifacts_dir / 'evaluation_report.json'}")
    print("=" * 60)
    print("\n[SUCCESS] Router training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
