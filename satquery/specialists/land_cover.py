"""
Multimodal Land Cover Classification Specialist for SatQuery AI.

Loads the validated 84-feature Random Forest model trained on paired
Sentinel-1 (SAR) and Sentinel-2 (optical) observations from BigEarthNet-14k.
Applies verified class-specific adaptive thresholds.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
import joblib

from satquery.taxonomy import TaskType, InputType
from satquery.specialists.base import BaseSpecialist, SpecialistResult, SpecialistStatus

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class LandCoverClassifier(BaseSpecialist):
    """
    Multilabel Land Cover Specialist using 84 optical-SAR statistical features.
    """

    def __init__(self, artifacts_dir: Optional[str | Path] = None):
        super().__init__(
            name="LandCoverClassifier_RF",
            supported_tasks=[TaskType.LAND_COVER_CLASSIFICATION],
            required_input_type=InputType.SINGLE_IMAGE
        )

        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "specialists" / "land_cover"
        self.artifacts_dir = Path(artifacts_dir)

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained model, label encoder, and threshold configuration."""
        model_path = self.artifacts_dir / "model.joblib"
        mlb_path = self.artifacts_dir / "label_encoder.joblib"
        thresh_path = self.artifacts_dir / "threshold_config.json"
        classes_path = self.artifacts_dir / "classes.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Land cover model artifact not found at {model_path}.")
        if not mlb_path.exists():
            raise FileNotFoundError(f"Label encoder artifact not found at {mlb_path}.")
        if not thresh_path.exists():
            raise FileNotFoundError(f"Threshold configuration not found at {thresh_path}.")

        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(mlb_path)

        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        with open(thresh_path, "r", encoding="utf-8") as f:
            self.threshold_config = json.load(f)

        self.global_threshold = float(self.threshold_config.get("global_threshold", 0.40))
        self.class_thresholds = self.threshold_config.get("class_specific_thresholds", {})

    def supports_input(self, input_data: Any) -> Tuple[bool, Optional[str]]:
        """
        Check if input data provides valid 84-feature vector or readable S1/S2 pair.
        """
        if input_data is None:
            return False, "Input data is None. Must provide an 84-feature vector or S1/S2 image pair."

        # Case 1: Direct numpy array
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 1 and len(input_data) == 84:
                if not np.isfinite(input_data).all():
                    return False, "Feature vector contains NaN or infinite values."
                return True, None
            return False, f"Expected 84 features, got array of shape {input_data.shape}."

        # Case 2: Dictionary input
        if isinstance(input_data, dict):
            if "features" in input_data:
                feats = np.asarray(input_data["features"], dtype=np.float32)
                if len(feats) == 84 and np.isfinite(feats).all():
                    return True, None
                return False, f"Dictionary 'features' must be an 84-element finite vector, got {len(feats)} elements."

            if "s1_path" in input_data and "s2_path" in input_data:
                s1_p = Path(input_data["s1_path"])
                s2_p = Path(input_data["s2_path"])
                if not s1_p.exists() or not s2_p.exists():
                    return False, f"Supplied TIFF paths do not exist: {s1_p}, {s2_p}"
                if not HAS_TIFFFILE:
                    return False, "tifffile library is required to extract features from raw TIFF paths."
                return True, None

        return False, "Unsupported input type. Provide an 84-feature numpy array or dict with 'features' or 's1_path'/'s2_path'."

    def _extract_from_tiffs(self, s1_path: Path, s2_path: Path) -> np.ndarray:
        """Extract 84 statistical features from S1 (2 bands) and S2 (10 bands) TIFFs."""
        s1_arr = tifffile.imread(str(s1_path)).astype(np.float32)
        s2_arr = tifffile.imread(str(s2_path)).astype(np.float32)

        def _stats(arr):
            feats = []
            for band in arr:
                feats.extend([
                    float(np.nanmean(band)), float(np.nanstd(band)),
                    float(np.nanmin(band)), float(np.nanmax(band)),
                    float(np.nanpercentile(band, 25)), float(np.nanpercentile(band, 50)),
                    float(np.nanpercentile(band, 75))
                ])
            return feats

        s1_feats = _stats(s1_arr)
        s2_feats = _stats(s2_arr)
        return np.array(s1_feats + s2_feats, dtype=np.float32)

    def predict(self, input_data: Any) -> SpecialistResult:
        """Execute land cover prediction with adaptive per-class thresholding."""
        # 1. Resolve feature vector
        if isinstance(input_data, np.ndarray):
            features = input_data
        elif isinstance(input_data, dict) and "features" in input_data:
            features = np.asarray(input_data["features"], dtype=np.float32)
        elif isinstance(input_data, dict) and "s1_path" in input_data:
            features = self._extract_from_tiffs(Path(input_data["s1_path"]), Path(input_data["s2_path"]))
        else:
            raise ValueError(f"Incompatible input passed to predict(): {type(input_data)}")

        X = features.reshape(1, -1)

        # 2. Extract model tree voting probabilities
        # predict_proba returns list of length 16 with arrays of shape (1, 2)
        prob_list = self.model.predict_proba(X)
        prob_dict: Dict[str, float] = {}
        thresholds_dict: Dict[str, float] = {}
        predicted_classes: List[str] = []

        for i, class_name in enumerate(self.classes):
            p = prob_list[i]
            # Column 1 corresponds to probability of positive class (label 1)
            score = float(p[0, 1]) if p.shape[1] > 1 else 0.0
            prob_dict[class_name] = round(score, 4)

            # Determine threshold
            thresh = float(self.class_thresholds.get(class_name, self.global_threshold))
            thresholds_dict[class_name] = thresh

            if score >= thresh:
                predicted_classes.append(class_name)

        # Sort predicted classes by descending score
        predicted_classes.sort(key=lambda c: prob_dict[c], reverse=True)

        # Feature diagnostics (S1 vs S2 summary)
        s1_sub = features[:14]
        s2_sub = features[14:]

        meta = {
            "specialist_name": self.name,
            "feature_count": 84,
            "radar_s1_features_summary": {
                "band_1_mean_backscatter_db": round(float(s1_sub[0]), 2),
                "band_2_mean_backscatter_db": round(float(s1_sub[7]), 2)
            },
            "optical_s2_features_summary": {
                "bands_processed": 10,
                "overall_reflectance_mean": round(float(np.mean(s2_sub[0::7])), 2)
            },
            "num_trees_evaluated": int(self.model.n_estimators)
        }

        return SpecialistResult(
            status=SpecialistStatus.SUCCESS,
            task=TaskType.LAND_COVER_CLASSIFICATION.value,
            predictions=predicted_classes,
            probabilities=prob_dict,
            thresholds_applied=thresholds_dict,
            execution_metadata=meta
        )
