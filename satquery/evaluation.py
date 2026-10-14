"""
Evaluation and Metrics Generation for SatQuery AI Task Router.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from satquery.router import TaskRouter


def evaluate_router(
    router: TaskRouter,
    val_df: pd.DataFrame,
    artifacts_dir: str | Path = "artifacts/router"
) -> Dict[str, Any]:
    """
    Evaluate fitted router on validation set, export metrics JSON and confusion matrix plot.
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    y_true = val_df["task"].tolist()
    y_pred = []
    confidences = []
    reviews = []

    for query in val_df["query"]:
        route = router.predict(query)
        y_pred.append(route.task)
        confidences.append(route.confidence)
        reviews.append(route.requires_review)

    labels = list(router.classes_)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Plot Confusion Matrix
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[l.replace("_", "\n") for l in labels],
        yticklabels=[l.replace("_", " ") for l in labels]
    )
    plt.title(f"SatQuery AI Task Router Confusion Matrix (Acc: {acc*100:.1f}%)")
    plt.xlabel("Predicted Task")
    plt.ylabel("Ground Truth Task")
    plt.tight_layout()

    cm_path = artifacts_path / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    metrics = {
        "evaluation_timestamp": pd.Timestamp.utcnow().isoformat(),
        "num_validation_samples": len(val_df),
        "overall_accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "micro_f1": round(float(f1_micro), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "average_confidence": round(float(np.mean(confidences)), 4),
        "review_rate": round(float(np.mean(reviews)), 4),
        "classification_report": report_dict,
        "confusion_matrix_path": str(cm_path.resolve())
    }

    report_path = artifacts_path / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
