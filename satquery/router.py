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
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import scipy.special as sp

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

# Task-specific distinctive keywords used to protect user intent from task coercion
TASK_SPECIFIC_KEYWORDS: Dict[str, List[str]] = {
    TaskType.LAND_COVER_CLASSIFICATION.value: [
        "land cover", "land-cover", "landcover", "corine", "biome",
        "forest", "arable", "pasture", "crop", "cropland", "urban fabric",
        "surface category", "biophysical", "artificial surfaces", "infrastructure",
        "marine inlets", "lagoons", "salt marshes", "single scene", "scene",
        "patch", "olive groves", "groves", "wetlands", "scrub", "woodland",
        "water bodies", "surfaces"
    ],
    TaskType.CHANGE_DETECTION.value: [
        "change", "changed", "modifications", "alteration", "differencing",
        "between dates", "over time", "deforestation", "pre-event",
        "post-event", "newly built", "sprawl", "temporal sequence",
        "temporal pair", "across years", "surface modifications"
    ],
    TaskType.CHANGE_VQA.value: [
        "shrunk", "damaged", "destroyed", "expanded between", "disappeared"
    ],
    TaskType.OPTICAL_SAR_FUSION.value: [
        "fusion", "fused", "sar", "radar", "backscatter", "polarization",
        "sentinel-1", "s1-s2", "c-band", "optical and sar", "radar and optical"
    ],
    TaskType.SINGLE_IMAGE_VQA.value: [
        "how many", "count", "is there a", "are there any", "airport runway"
    ],
    TaskType.SCENE_CAPTIONING.value: [
        "caption", "describe the scene", "descriptive caption",
        "concise description", "summarize the landscape"
    ],
    TaskType.VISUAL_GROUNDING.value: [
        "bounding box", "bbox", "pixel coordinates", "localize",
        "segmentation mask", "locate the"
    ],
}


def _contains_task_keywords(task_name: str, text: str) -> bool:
    """Check if text contains whole-word matches for task-specific keywords."""
    kws = TASK_SPECIFIC_KEYWORDS.get(str(task_name), [])
    return any(re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE) for kw in kws)


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
    confidence_margin: float = 0.0     # Delta c = c_intent - second_max (M3.3)
    joint_selection_score: float = 0.0 # s_joint(q) = min(c_intent/tau_intent, Delta_c/tau_margin) (M3.3)
    is_ambiguous: bool = False         # Linguistic ambiguity flag (M3.3)

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
        margin_threshold: float = 0.15,
        temperature: float = 1.0,
        version: str = "0.1.0",
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        random_state: int = 42
    ):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.temperature = temperature
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

        # Linguistic intent correction for bag-of-words keyword mismatches:
        # If the statistical prediction has zero keyword evidence in the query, but the query
        # contains explicit task-distinctive keywords for another task, protect the user's intent.
        top_has_keywords = _contains_task_keywords(text_task, query)
        if not top_has_keywords and text_confidence < self.confidence_threshold:
            for cand_task in self.classes_:
                if cand_task != text_task and _contains_task_keywords(str(cand_task), query):
                    text_task = str(cand_task)
                    text_confidence = float(prob_dict.get(cand_task, text_confidence))
                    break

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
        # INVARIANT: An explicit linguistic task must never be overwritten merely because
        # another task is compatible with the attached observation.
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
                text_has_explicit_keywords = _contains_task_keywords(text_task, query)
                has_explicit_intent = (
                    text_confidence >= self.confidence_threshold
                    or text_has_explicit_keywords
                )

                compatible_candidates = [
                    cls for cls in self.classes_
                    if obs_meta.is_compatible_with(cls)
                ]

                # Disambiguation is permitted ONLY when:
                # 1. The query text contains explicitly broad/uncommitted vocabulary (e.g. "analyze", "assess")
                #    AND does not explicitly name concepts of an incompatible task, OR
                # 2. The candidate task itself has explicit keywords in the query.
                can_disambiguate = False
                best_cand = None
                best_cand_score = 0.0

                if not has_explicit_intent and compatible_candidates:
                    cand_probs = [prob_dict[cls] for cls in compatible_candidates]
                    best_cand_idx = int(np.argmax(cand_probs))
                    best_cand = compatible_candidates[best_cand_idx]
                    best_cand_score = cand_probs[best_cand_idx]

                    cand_has_keywords = _contains_task_keywords(str(best_cand), query)
                    broad_words = ["analyze", "assess", "examine", "compare", "report", "process", "pipeline", "survey", "study"]
                    has_broad_vocab = any(re.search(r'\b' + re.escape(w) + r'\b', query, re.IGNORECASE) for w in broad_words)

                    if cand_has_keywords or (has_broad_vocab and not text_has_explicit_keywords):
                        can_disambiguate = True

                if can_disambiguate and best_cand is not None:
                    selected_task = best_cand
                    input_compatibility = True
                    routing_reason = (
                        f"Linguistic model tentatively suggested '{text_task}' ({text_confidence*100:.1f}%), "
                        f"disambiguated under ambiguity (margin {text_confidence - best_cand_score:.2f}) "
                        f"by verified '{obs_meta.modality}' observation to compatible task '{selected_task}' ({best_cand_score*100:.1f}%)."
                    )
                else:
                    # Intended task preserved: DO NOT coerce incompatible requests to wrong specialist!
                    selected_task = text_task
                    input_compatibility = False
                    routing_reason = (
                        f"Linguistic prediction '{text_task}' ({text_confidence*100:.1f}%) "
                        f"requires different input than attached '{obs_meta.modality}'. "
                        f"Preserved intended task to prevent task coercion."
                    )

        if selected_task == text_task:
            trace.append(f"[Trace 3/5] ML Task Classifier routed query to '{selected_task}' with {text_confidence*100:.1f}% confidence.")
        else:
            trace.append(
                f"[Trace 3/5] Linguistic model suggested '{text_task}' ({text_confidence*100:.1f}%), "
                f"disambiguated by attached '{obs_meta.modality}' input to '{selected_task}'."
            )

        # 4. Review flag check & Margin-Aware Selective Scoring (Milestone M3.3)
        sorted_p = np.sort(probs)
        margin_th = getattr(self, "margin_threshold", 0.15)
        conf_th = getattr(self, "confidence_threshold", 0.65)
        delta_c = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else float(sorted_p[-1])
        s_joint = float(min(
            text_confidence / max(conf_th, 1e-6),
            delta_c / max(margin_th, 1e-6)
        ))
        is_ambiguous = (text_confidence < conf_th) or (delta_c < margin_th)
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
            attached_modality=obs_meta.modality,
            confidence_margin=round(delta_c, 4),
            joint_selection_score=round(s_joint, 4),
            is_ambiguous=is_ambiguous
        )

    def predict_logits(self, queries: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Extract raw uncalibrated decision function scores (logits) from the underlying linear classifier.
        Used for clean calibration ablations (R3 vs R4).
        """
        if not self.is_fitted:
            raise RuntimeError("TaskRouter must be fitted before calling predict_logits().")
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        X = tfidf.transform(queries)
        if hasattr(clf, "calibrated_classifiers_"):
            # Ensemble of calibrated base estimators: average the raw linear decision function
            return np.mean([cc.estimator.decision_function(X) for cc in clf.calibrated_classifiers_], axis=0)
        elif hasattr(clf, "decision_function"):
            return clf.decision_function(X)
        else:
            probs = self.pipeline.predict_proba(queries)
            return np.log(np.clip(probs, 1e-12, 1.0))

    def predict_calibrated_proba(
        self,
        queries: Union[List[str], np.ndarray],
        temperature: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute Softmax probabilities with temperature scaling:
            P(t | q) = Softmax(z / T_cal)
        """
        T = temperature if temperature is not None else getattr(self, "temperature", 1.0)
        logits = self.predict_logits(queries)
        scaled_logits = logits / max(T, 1e-6)
        return sp.softmax(scaled_logits, axis=-1)

    def fit_temperature(
        self,
        val_queries: Union[List[str], np.ndarray],
        val_labels: Union[List[str], np.ndarray]
    ) -> float:
        """
        Fit post-hoc calibration temperature T_cal strictly on validation queries via NLL minimization.
        """
        import scipy.optimize as opt
        logits = self.predict_logits(val_queries)
        classes = list(self.classes_)
        y_idx = np.array([classes.index(str(lbl)) for lbl in val_labels])

        def nll(T_arr):
            T = T_arr[0]
            scaled = logits / max(T, 1e-6)
            log_probs = scaled - sp.logsumexp(scaled, axis=1, keepdims=True)
            return -float(np.mean(log_probs[np.arange(len(y_idx)), y_idx]))

        res = opt.minimize(nll, [1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self.temperature

    def compute_ece(
        self,
        queries: Union[List[str], np.ndarray],
        labels: Union[List[str], np.ndarray],
        n_bins: int = 15,
        use_temperature: bool = True
    ) -> float:
        """
        Compute Top-Label Expected Calibration Error (ECE) across M quantile/uniform bins.
        """
        if use_temperature:
            probs = self.predict_calibrated_proba(queries)
        else:
            probs = self.pipeline.predict_proba(queries)
        classes = list(self.classes_)
        y_idx = np.array([classes.index(str(lbl)) for lbl in labels])
        top_pred = np.argmax(probs, axis=1)
        top_conf = np.max(probs, axis=1)
        correct = (top_pred == y_idx).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_total = len(queries)
        for i in range(n_bins):
            b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (top_conf > b_low) & (top_conf <= b_high) if i > 0 else (top_conf >= b_low) & (top_conf <= b_high)
            if np.sum(mask) > 0:
                bin_acc = float(np.mean(correct[mask]))
                bin_conf = float(np.mean(top_conf[mask]))
                ece += (float(np.sum(mask)) / n_total) * abs(bin_acc - bin_conf)
        return float(ece)

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
