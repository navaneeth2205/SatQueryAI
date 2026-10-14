"""
Task Router Engine for SatQuery AI.

Classifies natural-language queries into remote-sensing analysis tasks,
determines required input modalities, and produces calibrated confidence
scores alongside execution traces for SIH observable execution.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from satquery.taxonomy import TaskType, InputType, TASK_REGISTRY, get_task_definition


@dataclass
class ExecutionRoute:
    """Detailed routing decision and execution trace metadata."""
    query: str
    task: str
    display_name: str
    required_input: str
    confidence: float
    requires_review: bool
    specialist_model_family: str
    execution_trace: List[str]
    class_probabilities: Dict[str, float]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TaskRouter:
    """
    ML-Driven Task Routing Layer for SatQuery AI.
    Maps natural-language queries and input metadata to specialist models.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        version: str = "0.1.0",
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        random_state: int = 42
    ):
        self.confidence_threshold = confidence_threshold
        self.version = version
        self.ngram_range = ngram_range
        self.C = C
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.training_metadata: Dict[str, Any] = {}
        self.pipeline: Optional[Pipeline] = None

    def fit(self, queries: List[str] | np.ndarray, labels: List[str] | np.ndarray) -> "TaskRouter":
        """Train the router on queries and corresponding task labels."""
        queries = np.asarray(queries, dtype=str)
        labels = np.asarray(labels, dtype=str)

        base_clf = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state
        )

        # Adapt cross-validation folds based on minimum class frequency
        _, counts = np.unique(labels, return_counts=True)
        min_count = int(np.min(counts)) if len(counts) > 0 else 1

        if min_count >= 3:
            clf_step = CalibratedClassifierCV(
                estimator=base_clf,
                method="sigmoid",
                cv=min(min_count, 3)
            )
        elif min_count >= 2:
            clf_step = CalibratedClassifierCV(
                estimator=base_clf,
                method="sigmoid",
                cv=2
            )
        else:
            clf_step = base_clf

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode",
                lowercase=True
            )),
            ("clf", clf_step)
        ])

        self.pipeline.fit(queries, labels)
        self.classes_ = self.pipeline.classes_
        self.is_fitted = True

        self.training_metadata = {
            "version": self.version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "num_training_samples": len(queries),
            "num_classes": len(self.classes_),
            "classes": list(self.classes_),
            "confidence_threshold": self.confidence_threshold
        }
        return self

    def predict(
        self,
        query: str,
        image_metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionRoute:
        """
        Classify query and determine the specialist workflow.
        
        Args:
            query: The user's natural-language question.
            image_metadata: Optional dict containing 'image_count', 'sensor_types' ('optical', 'sar').
        """
        if not self.is_fitted:
            raise RuntimeError("TaskRouter must be fitted before calling predict().")

        trace: List[str] = [
            f"[Trace 1/5] Ingested user query: '{query}'",
            f"[Trace 2/5] Normalized text representation and extracted TF-IDF features."
        ]

        # Extract calibrated class probabilities
        probs = self.pipeline.predict_proba([query])[0]
        prob_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(self.classes_, probs)
        }

        # Select highest-probability task
        best_idx = int(np.argmax(probs))
        predicted_task_name = str(self.classes_[best_idx])
        raw_confidence = float(probs[best_idx])

        trace.append(f"[Trace 3/5] ML Task Classifier routed query to '{predicted_task_name}' with {raw_confidence*100:.1f}% confidence.")

        # Optional metadata arbitration
        task_def = get_task_definition(predicted_task_name)
        required_input = task_def.required_input.value if task_def else "unknown"
        display_name = task_def.display_name if task_def else predicted_task_name
        model_family = task_def.specialist_model_family if task_def else "generic_handler"

        # Check metadata consistency if provided
        if image_metadata:
            num_images = image_metadata.get("image_count", 1)
            sensors = image_metadata.get("sensor_types", ["optical"])

            if "sar" in sensors and "optical" in sensors:
                trace.append("[Metadata Trace] Both Optical and SAR inputs verified available.")
            elif num_images == 2 and "temporal" in str(image_metadata.get("context", "")).lower():
                trace.append("[Metadata Trace] Multitemporal observation pair registered.")

        # Review flag check
        requires_review = raw_confidence < self.confidence_threshold
        if requires_review:
            trace.append(
                f"[Trace 4/5] WARNING: Confidence {raw_confidence:.2f} is below threshold ({self.confidence_threshold:.2f}). Flagged for human-in-the-loop review."
            )
        else:
            trace.append(
                f"[Trace 4/5] Confidence meets automated dispatch threshold (>= {self.confidence_threshold:.2f})."
            )

        trace.append(f"[Trace 5/5] Dispatched request to specialist pipeline: '{model_family}'.")

        return ExecutionRoute(
            query=query,
            task=predicted_task_name,
            display_name=display_name,
            required_input=required_input,
            confidence=round(raw_confidence, 4),
            requires_review=requires_review,
            specialist_model_family=model_family,
            execution_trace=trace,
            class_probabilities=prob_dict,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def save(self, filepath: str | Path) -> None:
        """Persist fitted router to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str | Path) -> "TaskRouter":
        """Load fitted router from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Router model artifact not found at: {path.resolve()}")
        router = joblib.load(path)
        if not isinstance(router, cls):
            raise TypeError(f"Loaded object is of type {type(router)}, expected {cls}.")
        return router
