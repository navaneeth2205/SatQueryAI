"""
Benchmark evaluation of SatQuery Task Router on fixed labeled query testbed.

Measures:
  - Task-level precision, recall, F1
  - Confusion matrix
  - Safety review rate (fraction of queries with confidence < 0.65)
  - Orthogonal input-aware compatibility disambiguation
"""

import sys
from pathlib import Path

# Force UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from satquery.router import TaskRouter
from satquery.taxonomy import TaskType


def main():
    queries_path = Path("data/starter_queries.csv")
    df = pd.read_parquet(queries_path) if queries_path.suffix == ".parquet" else pd.read_csv(queries_path)

    router = TaskRouter(confidence_threshold=0.65)
    train_df = df[df["split"] == "train"]
    router.fit(train_df["query"].tolist(), train_df["task"].tolist())

    val_df = df[df["split"] == "val"].reset_index(drop=True)
    val_queries = val_df["query"].tolist()
    val_true = val_df["task"].tolist()

    routes = [router.predict(q) for q in val_queries]
    val_pred = [r.selected_task if isinstance(r.selected_task, str) else r.selected_task.value for r in routes]
    val_conf = [r.text_confidence for r in routes]
    val_review = [r.requires_review for r in routes]

    classes = sorted(list(set(val_true + val_pred)))
    prec, rec, f1, supp = precision_recall_fscore_support(val_true, val_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(val_true, val_pred, labels=classes)

    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(val_true, val_pred, average="weighted", zero_division=0)
    macro_f1 = f1.mean()
    review_rate = float(np.mean(val_review))
    mean_confidence = float(np.mean(val_conf))

    print(f"\n================ Task Router Benchmark (Validation Set N={len(val_queries)}) ================")
    print(f"Weighted Precision: {overall_prec:.4f}")
    print(f"Weighted Recall:    {overall_rec:.4f}")
    print(f"Weighted F1:        {overall_f1:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Mean Confidence:    {mean_confidence:.4f}")
    print(f"Safety Review Rate: {review_rate*100:.1f}% (queries flagged for human review)")

    df_metrics = pd.DataFrame({
        "Task": classes,
        "Support": supp,
        "Precision": np.round(prec, 4),
        "Recall": np.round(rec, 4),
        "F1": np.round(f1, 4)
    })
    print("\nPer-Task Breakdown:")
    print(df_metrics.to_string(index=False))

    df_cm = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print("\nConfusion Matrix:")
    print(df_cm.to_string())

    out_dir = Path("artifacts/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_dir / "router_validation_metrics.csv", index=False)
    df_cm.to_csv(out_dir / "router_confusion_matrix.csv")

    payload = {
        "overall_precision": round(overall_prec, 4),
        "overall_recall": round(overall_rec, 4),
        "overall_f1": round(overall_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "mean_confidence": round(mean_confidence, 4),
        "safety_review_rate": round(review_rate, 4),
        "confusion_matrix": cm.tolist(),
        "classes": classes
    }
    with open(out_dir / "router_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved router benchmark results to {out_dir / 'router_benchmark_summary.json'}")


if __name__ == "__main__":
    main()
