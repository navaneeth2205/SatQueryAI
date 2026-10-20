"""
ML-Driven Task Routing Layer for SatQuery AI.

Features:
- Calibrated TF-IDF + Logistic Regression linguistic classification
- Input-aware compatibility task arbitration without linguistic probability distortion
- Explicit tracking of text_task, text_confidence, selected_task, and input_compatibility
- Safety policy review flags (preserving 0.65 threshold)
- Full observable 5-step execution trace
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from satquery.taxonomy import TaskType, InputType, TASK_REGISTRY, get_task_definition
from satquery.utils.metadata import (
    InputObservationMetadata,
    extract_observation_metadata,
    TASK_COMPATIBILITY_MAP
)


@dataclass
class ExecutionRoute:
    """
    Detailed routing decision and execution trace metadata.
    
    Preserves separation between:
    - Linguistic model confidence (P(task | text))
    - Input-conditioned compatibility and task selection
    """
    query: str
    task: str                          # Selected task after compatibility evaluation
    display_name: str
    required_input: str
    confidence: float                  # Preserved raw linguistic confidence P(text_task | text)
    requires_review: bool              # Safety review flag based on linguistic confidence
    specialist_model_family: str
    execution_trace: List[str]
    class_probabilities: Dict[str, float]  # Unconditioned calibrated probabilities
    timestamp: str
    # Scientific refinement fields:
    text_task: str = ""                # Original top-ranked linguistic task
    text_confidence: float = 0.0       # Original linguistic model score
    selected_task: str = ""            # Final task selected after compatibility checks
    input_compatibility: bool = True   # Whether linguistic top task matches attached inputs
    routing_reason: str = ""           # Transparent justification of selection
    attached_modality: Optional[str] = None  # Detected modality of attached inputs

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
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
            solver="lbfgs"
        )

        unique_labels, counts = np.unique(labels, return_counts=True)
        min_samples = np.min(counts)

        if min_samples >= 3:
            cv_folds = min(3, min_samples)
            clf_step = CalibratedClassifierCV(
                estimator=base_clf,
                method="sigmoid",
                cv=cv_folds
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
        image_metadata: Optional[Union[InputObservationMetadata, Dict[str, Any], Any]] = None
    ) -> ExecutionRoute:
        """
        Classify query and determine the specialist workflow with compatibility-aware arbitration.
        
        Args:
            query: The user's natural-language question.
            image_metadata: Optional InputObservationMetadata, dict, or raw input object.
        """
        if not self.is_fitted:
            raise RuntimeError("TaskRouter must be fitted before calling predict().")

        trace: List[str] = [
            f"[Trace 1/5] Ingested user query: '{query}'",
            f"[Trace 2/5] Normalized text representation and extracted TF-IDF features."
        ]

        # 1. Extract calibrated linguistic probabilities P(task | text)
        probs = self.pipeline.predict_proba([query])[0]
        prob_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(self.classes_, probs)
        }

        # Original top-ranked linguistic task
        best_idx = int(np.argmax(probs))
        text_task = str(self.classes_[best_idx])
        text_confidence = float(probs[best_idx])

        # 2. Extract or accept input observation metadata
        if isinstance(image_metadata, InputObservationMetadata):
            obs_meta = image_metadata
        elif isinstance(image_metadata, dict):
            modality = image_metadata.get("modality")
            if not modality and image_metadata.get("image_count") == 2:
                modality = InputType.TEMPORAL_PAIR.value
            obs_meta = InputObservationMetadata(
                modality=modality,
                image_count=image_metadata.get("image_count", 0),
                sensors=image_metadata.get("sensor_types", ["optical"]),
                details=image_metadata
            )
        elif image_metadata is not None:
            obs_meta = extract_observation_metadata(image_metadata)
        else:
            obs_meta = InputObservationMetadata(modality=None)

        # 3. Compatibility-Aware Task Arbitration
        selected_task = text_task
        input_compatibility = True
        routing_reason = "Pure natural-language routing without attached observation constraints."

        if obs_meta.modality is not None and obs_meta.modality != "unknown":
            is_compat = obs_meta.is_compatible_with(text_task)
            if is_compat:
                selected_task = text_task
                input_compatibility = True
                routing_reason = (
                    f"Linguistic prediction '{text_task}' ({text_confidence*100:.1f}%) "
                    f"is verified compatible with attached '{obs_meta.modality}' observation."
                )
            else:
                # Top linguistic prediction is incompatible with attached input.
                # If user attached temporal images, disambiguate among temporal tasks if supported:
                if obs_meta.modality == InputType.TEMPORAL_PAIR.value:
                    temporal_candidates = [
                        cls for cls in self.classes_
                        if obs_meta.is_compatible_with(cls)
                    ]
                    cand_probs = [prob_dict[cls] for cls in temporal_candidates] if temporal_candidates else []
                    best_cand_score = max(cand_probs) if cand_probs else 0.0

                    if best_cand_score >= 0.15:
                        best_cand_idx = int(np.argmax(cand_probs))
                        selected_task = temporal_candidates[best_cand_idx]
                        input_compatibility = True
                        routing_reason = (
                            f"Linguistic model tentatively suggested '{text_task}' ({text_confidence*100:.1f}%), "
                            f"disambiguated by verified temporal pair observation to compatible task '{selected_task}' ({best_cand_score*100:.1f}%)."
                        )
                    else:
                        selected_task = text_task
                        input_compatibility = False
                        routing_reason = (
                            f"Linguistic prediction '{text_task}' requires different input than attached '{obs_meta.modality}'."
                        )
                else:
                    # User attached single image or feature vector, but query requested a change or temporal task.
                    # Preserve intended task so input incompatibility is accurately reported by specialist.
                    selected_task = text_task
                    input_compatibility = False
                    routing_reason = (
                        f"Linguistic prediction '{text_task}' ({text_confidence*100:.1f}%) "
                        f"requires different input than attached '{obs_meta.modality}'. "
                        f"Preserved intended task for downstream input validation."
                    )

        if selected_task == text_task:
            trace.append(f"[Trace 3/5] ML Task Classifier routed query to '{selected_task}' with {text_confidence*100:.1f}% confidence.")
        else:
            trace.append(
                f"[Trace 3/5] Linguistic model suggested '{text_task}' ({text_confidence*100:.1f}%), "
                f"disambiguated by attached '{obs_meta.modality}' input to '{selected_task}'."
            )

        # 4. Review flag check: Strictly tied to unconditioned linguistic confidence and threshold
        requires_review = text_confidence < self.confidence_threshold
        if requires_review:
            trace.append(
                f"[Trace 4/5] WARNING: Linguistic confidence {text_confidence:.2f} is below threshold ({self.confidence_threshold:.2f}). Flagged for human review."
            )
        else:
            trace.append(
                f"[Trace 4/5] Linguistic confidence meets automated dispatch threshold (>= {self.confidence_threshold:.2f})."
            )

        # 5. Resolve selected task definition
        task_def = get_task_definition(selected_task)
        required_input = task_def.required_input.value if task_def else "unknown"
        display_name = task_def.display_name if task_def else selected_task
        model_family = task_def.specialist_model_family if task_def else "generic_handler"

        trace.append(f"[Trace 5/5] Dispatched request to specialist pipeline: '{model_family}'.")

        return ExecutionRoute(
            query=query,
            task=selected_task,
            display_name=display_name,
            required_input=required_input,
            confidence=round(text_confidence, 4),
            requires_review=requires_review,
            specialist_model_family=model_family,
            execution_trace=trace,
            class_probabilities=prob_dict,
            timestamp=datetime.now(timezone.utc).isoformat(),
            text_task=text_task,
            text_confidence=round(text_confidence, 4),
            selected_task=selected_task,
            input_compatibility=input_compatibility,
            routing_reason=routing_reason,
            attached_modality=obs_meta.modality
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
