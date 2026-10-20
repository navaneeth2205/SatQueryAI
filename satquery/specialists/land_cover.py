"""
Multimodal Land Cover Classification Specialist for SatQuery AI.

Loads the validated 84-feature Random Forest baseline and the Phase 2
488-feature LightGBM gradient-boosted ensemble trained on paired Sentinel-1 (SAR)
and Sentinel-2 (optical) observations from BigEarthNet-14k.
Applies validation-selected class-specific decision thresholds.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
import joblib

from satquery.taxonomy import TaskType, InputType
from satquery.specialists.base import BaseSpecialist, SpecialistResult, SpecialistStatus
from satquery.features.multimodal import extract_multimodal_features_phase2

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class LandCoverClassifier(BaseSpecialist):
    """
    Multilabel Land Cover Specialist supporting both baseline 84-feature RF
    and Phase 2 488-feature LightGBM multimodal spatial-pyramid & spectral models.
    """

    def __init__(
        self,
        artifacts_dir: Optional[str | Path] = None,
        phase2_artifacts_dir: Optional[str | Path] = None
    ):
        super().__init__(
            name="LandCoverClassifier_RF",
            supported_tasks=[TaskType.LAND_COVER_CLASSIFICATION],
            required_input_type=InputType.SINGLE_IMAGE
        )

        repo_root = Path(__file__).resolve().parent.parent.parent
        if artifacts_dir is None:
            artifacts_dir = repo_root / "artifacts" / "specialists" / "land_cover"
        self.artifacts_dir = Path(artifacts_dir)

        if phase2_artifacts_dir is None:
            phase2_artifacts_dir = repo_root / "artifacts" / "specialists" / "land_cover_phase2"
        self.phase2_artifacts_dir = Path(phase2_artifacts_dir)

        self._load_baseline_artifacts()
        self._load_phase2_artifacts()

    def _load_baseline_artifacts(self) -> None:
        """Load trained baseline RF model, label encoder, and threshold configuration."""
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

    def _load_phase2_artifacts(self) -> None:
        """Load Phase 2 LightGBM model and optimized per-class thresholds if available."""
        self.has_phase2 = False
        p2_model_path = self.phase2_artifacts_dir / "model.joblib"
        p2_thresh_path = self.phase2_artifacts_dir / "threshold_config.json"
        p2_classes_path = self.phase2_artifacts_dir / "classes.json"

        if p2_model_path.exists() and p2_thresh_path.exists():
            try:
                self.phase2_model = joblib.load(p2_model_path)
                with open(p2_thresh_path, "r", encoding="utf-8") as f:
                    p2_cfg = json.load(f)
                self.phase2_thresholds = p2_cfg.get("class_specific_thresholds", {})
                self.phase2_global_threshold = float(p2_cfg.get("global_threshold", 0.40))

                if p2_classes_path.exists():
                    with open(p2_classes_path, "r", encoding="utf-8") as f:
                        self.phase2_classes = json.load(f)
                else:
                    self.phase2_classes = self.classes

                self.has_phase2 = True
            except Exception:
                self.has_phase2 = False

    def supports_input(self, input_data: Any) -> Tuple[bool, Optional[str]]:
        """
        Check if input data provides valid 84-feature vector, 488-feature vector, or readable S1/S2 pair.
        """
        if input_data is None:
            return False, "Input data is None. Must provide an 84-feature vector or S1/S2 image pair."

        # Case 1: Direct numpy array
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 1:
                if len(input_data) in (84, 488):
                    if not np.isfinite(input_data).all():
                        return False, "Feature vector contains NaN or infinite values."
                    return True, None
            return False, f"Expected 84 features (or 488 Phase 2 features), got array of shape {input_data.shape}."

        # Case 2: Dictionary input
        if isinstance(input_data, dict):
            if "features" in input_data:
                feats = np.asarray(input_data["features"], dtype=np.float32)
                if len(feats) in (84, 488) and np.isfinite(feats).all():
                    return True, None
                return False, f"Dictionary 'features' must be an 84-element (or 488-element) finite vector, got {len(feats)} elements."

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
        """Extract features from S1 (2 bands) and S2 (10 bands) TIFFs."""
        s1_arr = tifffile.imread(str(s1_path)).astype(np.float32)
        s2_arr = tifffile.imread(str(s2_path)).astype(np.float32)

        if self.has_phase2:
            return extract_multimodal_features_phase2(s1_arr, s2_arr)

        # Fallback to baseline 84 stats
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

        # 2. Branch: Phase 2 LightGBM (488 features) vs Baseline RF (84 features)
        if len(features) == 488 and self.has_phase2:
            prob_dict: Dict[str, float] = {}
            thresholds_dict: Dict[str, float] = {}
            predicted_classes: List[str] = []

            for i, class_name in enumerate(self.phase2_classes):
                clf = self.phase2_model[i]
                p = clf.predict_proba(X)
                score = float(p[0, 1]) if p.shape[1] > 1 else 0.0
                prob_dict[class_name] = round(score, 4)

                thresh = float(self.phase2_thresholds.get(class_name, self.phase2_global_threshold))
                thresholds_dict[class_name] = round(thresh, 4)

                if score >= thresh:
                    predicted_classes.append(class_name)

            predicted_classes.sort(key=lambda c: prob_dict[c], reverse=True)

            meta = {
                "specialist_name": "LandCoverClassifier_LightGBM_Phase2",
                "model_family": "gradient_boosted_decision_trees",
                "feature_count": 488,
                "feature_composition": {
                    "level0_global_stats": 84,
                    "level1_spatial_pyramid_quadrants": 336,
                    "multispectral_indices": 35,
                    "polarimetric_sar_indices": 21,
                    "texture_gradients": 12
                },
                "validation_benchmark": {
                    "micro_f1": 0.7500,
                    "macro_f1": 0.5492,
                    "hamming_loss": 0.0930
                },
                "num_binary_models_evaluated": len(self.phase2_model)
            }

            return SpecialistResult(
                status=SpecialistStatus.SUCCESS,
                task=TaskType.LAND_COVER_CLASSIFICATION.value,
                predictions=predicted_classes,
                probabilities=prob_dict,
                thresholds_applied=thresholds_dict,
                execution_metadata=meta
            )

        # 3. Baseline RF execution (84 features)
        prob_list = self.model.predict_proba(X)
        prob_dict = {}
        thresholds_dict = {}
        predicted_classes = []

        for i, class_name in enumerate(self.classes):
            p = prob_list[i]
            score = float(p[0, 1]) if p.shape[1] > 1 else 0.0
            prob_dict[class_name] = round(score, 4)

            thresh = float(self.class_thresholds.get(class_name, self.global_threshold))
            thresholds_dict[class_name] = thresh

            if score >= thresh:
                predicted_classes.append(class_name)

        predicted_classes.sort(key=lambda c: prob_dict[c], reverse=True)

        s1_sub = features[:14]
        s2_sub = features[14:]

        meta = {
            "specialist_name": self.name,
            "feature_count": 84,
            "radar_s1_features_summary": {
                "channels_processed": 2,
                "band_1_mean_stored_value": round(float(s1_sub[0]), 2),
                "band_2_mean_stored_value": round(float(s1_sub[7]), 2),
                "unit_note": "Stored float32 raster values (uncalibrated; dB scaling unverified)"
            },
            "optical_s2_features_summary": {
                "channels_processed": 10,
                "overall_mean_stored_dn": round(float(np.mean(s2_sub[0::7])), 2),
                "unit_note": "Stored digital numbers (uncalibrated; reflectance scaling unverified)"
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
