"""
Input observation metadata extraction and compatibility definitions for SatQuery AI.

Provides:
- InputObservationMetadata: Structured representation of attached data observations.
- extract_observation_metadata(): Robust inspection of numpy arrays, file paths, and dictionaries.
- TASK_COMPATIBILITY_MAP: Mapping of each TaskType to valid InputTypes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np

from satquery.taxonomy import TaskType, InputType
from satquery.utils.temporal import validate_temporal_pair


# Explicit mapping from task key to set of compatible input modalities
TASK_COMPATIBILITY_MAP: Dict[str, Set[str]] = {
    TaskType.LAND_COVER_CLASSIFICATION.value: {
        InputType.SINGLE_IMAGE.value,
        "feature_vector",
        InputType.OPTICAL_SAR_PAIR.value
    },
    TaskType.CHANGE_DETECTION.value: {
        InputType.TEMPORAL_PAIR.value
    },
    TaskType.CHANGE_VQA.value: {
        InputType.TEMPORAL_PAIR.value
    },
    TaskType.OPTICAL_SAR_FUSION.value: {
        InputType.OPTICAL_SAR_PAIR.value,
        "feature_vector"
    },
    TaskType.SINGLE_IMAGE_VQA.value: {
        InputType.SINGLE_IMAGE.value
    },
    TaskType.SCENE_CAPTIONING.value: {
        InputType.SINGLE_IMAGE.value
    },
    TaskType.VISUAL_GROUNDING.value: {
        InputType.SINGLE_IMAGE.value
    },
}


@dataclass
class InputObservationMetadata:
    """
    Structured metadata extracted from attached user inputs.
    
    Used by the Task Router and Orchestrator to perform orthogonal,
    compatibility-aware task disambiguation without overwriting linguistic scores.
    """
    modality: Optional[str] = None  # 'single_image', 'temporal_pair', 'optical_sar_pair', 'feature_vector', or None
    image_count: int = 0
    is_feature_vector: bool = False
    feature_dim: Optional[int] = None
    has_temporal_context: bool = False
    is_valid_temporal_pair: bool = False
    sensors: List[str] = field(default_factory=lambda: ["optical"])
    spatial_shape: Optional[tuple] = None
    validation_error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def is_compatible_with(self, task: str) -> bool:
        """Check if this attached input modality is compatible with a given task."""
        if self.modality is None:
            # When no input is attached, compatibility cannot be established or rejected
            return True

        allowed_modalities = TASK_COMPATIBILITY_MAP.get(task, set())
        return self.modality in allowed_modalities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "image_count": self.image_count,
            "is_feature_vector": self.is_feature_vector,
            "feature_dim": self.feature_dim,
            "has_temporal_context": self.has_temporal_context,
            "is_valid_temporal_pair": self.is_valid_temporal_pair,
            "sensors": self.sensors,
            "spatial_shape": list(self.spatial_shape) if self.spatial_shape else None,
            "validation_error": self.validation_error,
            "details": self.details
        }


def extract_observation_metadata(input_data: Any) -> InputObservationMetadata:
    """
    Inspect raw input_data and extract validated observation metadata.
    
    Handles:
    - None (Query only)
    - 1D numpy array (Precomputed feature vector)
    - 2D / 3D numpy array (Single satellite image)
    - Tuple / List of length 2 (Temporal image pair)
    - Dictionary with 'features', 't1'/'t2', or 's1_path'/'s2_path'
    """
    if input_data is None:
        return InputObservationMetadata(
            modality=None,
            image_count=0,
            details={"note": "No attached imagery or feature vector (query only)."}
        )

    # Case 1: Direct numpy array
    if isinstance(input_data, np.ndarray):
        if input_data.ndim == 1:
            return InputObservationMetadata(
                modality="feature_vector",
                image_count=0,
                is_feature_vector=True,
                feature_dim=len(input_data),
                details={"length": len(input_data), "dtype": str(input_data.dtype)}
            )
        elif input_data.ndim in (2, 3):
            return InputObservationMetadata(
                modality=InputType.SINGLE_IMAGE.value,
                image_count=1,
                spatial_shape=input_data.shape,
                sensors=["optical"],
                details={"shape": list(input_data.shape), "dtype": str(input_data.dtype)}
            )
        else:
            return InputObservationMetadata(
                modality="unknown",
                validation_error=f"Unexpected array dimension: {input_data.ndim}D."
            )

    # Case 2: Temporal Pair tuple or list
    if isinstance(input_data, (tuple, list)) and len(input_data) == 2:
        val = validate_temporal_pair(input_data[0], input_data[1])
        if val.is_valid:
            pair_shape = (val.height, val.width, val.channels)
            return InputObservationMetadata(
                modality=InputType.TEMPORAL_PAIR.value,
                image_count=2,
                has_temporal_context=True,
                is_valid_temporal_pair=True,
                spatial_shape=pair_shape,
                sensors=["optical"],
                details={
                    "shape": list(pair_shape),
                    "valid_fraction": val.valid_fraction,
                    "total_pixels": val.total_pixels
                }
            )
        else:
            return InputObservationMetadata(
                modality=InputType.TEMPORAL_PAIR.value,
                image_count=2,
                has_temporal_context=True,
                is_valid_temporal_pair=False,
                validation_error=val.error_message,
                details={"validation_failed": True}
            )

    # Case 3: Dictionary representation
    if isinstance(input_data, dict):
        # 3A. Precomputed Feature Vector
        if "features" in input_data:
            feats = np.asarray(input_data["features"])
            return InputObservationMetadata(
                modality="feature_vector",
                image_count=0,
                is_feature_vector=True,
                feature_dim=len(feats),
                details={
                    "length": len(feats),
                    "patch_id": input_data.get("patch_id", "unspecified"),
                    "ground_truth_labels": input_data.get("ground_truth_labels", [])
                }
            )

        # 3B. Temporal Observation Pair
        if "t1" in input_data and "t2" in input_data:
            val = validate_temporal_pair(input_data["t1"], input_data["t2"], nodata_value=input_data.get("nodata_value"))
            if val.is_valid:
                pair_shape = (val.height, val.width, val.channels)
                return InputObservationMetadata(
                    modality=InputType.TEMPORAL_PAIR.value,
                    image_count=2,
                    has_temporal_context=True,
                    is_valid_temporal_pair=True,
                    spatial_shape=pair_shape,
                    sensors=["optical"],
                    details={
                        "scene_name": input_data.get("scene_name", "unspecified"),
                        "shape": list(pair_shape),
                        "valid_fraction": val.valid_fraction
                    }
                )
            else:
                return InputObservationMetadata(
                    modality=InputType.TEMPORAL_PAIR.value,
                    image_count=2,
                    has_temporal_context=True,
                    is_valid_temporal_pair=False,
                    validation_error=val.error_message,
                    details={"validation_failed": True}
                )

        # 3C. Optical-SAR Pair
        if "s1_path" in input_data and "s2_path" in input_data:
            s1_exists = Path(input_data["s1_path"]).exists()
            s2_exists = Path(input_data["s2_path"]).exists()
            return InputObservationMetadata(
                modality=InputType.OPTICAL_SAR_PAIR.value,
                image_count=2,
                sensors=["sar", "optical"],
                details={
                    "s1_path": str(input_data["s1_path"]),
                    "s2_path": str(input_data["s2_path"]),
                    "files_exist": s1_exists and s2_exists
                }
            )

        # 3D. Generic single image dict
        if "image" in input_data or "image_path" in input_data:
            return InputObservationMetadata(
                modality=InputType.SINGLE_IMAGE.value,
                image_count=1,
                sensors=["optical"],
                details={"image_path": str(input_data.get("image_path", "memory"))}
            )

    return InputObservationMetadata(
        modality="unknown",
        validation_error=f"Unrecognized input type: {type(input_data).__name__}."
    )
