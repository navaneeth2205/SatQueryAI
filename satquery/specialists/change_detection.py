"""
Multitemporal Change Detection Specialist for SatQuery AI.

Implements the deterministic TemporalDifferenceBaseline:
- Robust input validation and NoData/finite-value masking
- Normalized per-channel differencing and Change Vector Analysis (CVA)
- Validation-selected decision thresholding
- Connected-component spatial cluster extraction in documented pixel space
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage

from satquery.taxonomy import TaskType, InputType
from satquery.specialists.base import BaseSpecialist, SpecialistResult, SpecialistStatus
from satquery.utils.temporal import validate_temporal_pair, TemporalPairValidation


class TemporalDifferenceBaseline(BaseSpecialist):
    """
    Deterministic Change Detection Specialist using Normalized Change Vector Analysis (CVA).
    
    Operates on co-registered bi-temporal observation pairs (T1, T2) across arbitrary spectral channels.
    Thresholds are selected on validation data against ground truth change masks.
    """

    def __init__(
        self,
        decision_threshold: float = 0.20,
        min_component_area: int = 9,
        artifacts_dir: Optional[Union[str, Path]] = None
    ):
        super().__init__(
            name="temporal_difference_baseline",
            supported_tasks=[TaskType.CHANGE_DETECTION],
            required_input_type=InputType.TEMPORAL_PAIR
        )
        self.decision_threshold = float(decision_threshold)
        self.min_component_area = int(min_component_area)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None

        # Load validation-tuned threshold from disk if artifact exists
        if self.artifacts_dir and (self.artifacts_dir / "threshold_config.json").exists():
            try:
                with open(self.artifacts_dir / "threshold_config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.decision_threshold = float(cfg.get("decision_threshold", self.decision_threshold))
                    self.min_component_area = int(cfg.get("min_component_area", self.min_component_area))
            except Exception:
                pass

    def _normalize_image(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply reproducible, documented scaling across channels.
        - 8-bit imagery ([0, 255]) is scaled by 255.0 to [0, 1].
        - Multi-band 12-bit / 16-bit reflectance ([0, 10000+]) is scaled by 10000.0.
        - Existing float arrays in [0, 1] are preserved.
        """
        max_val = np.nanmax(arr) if arr.size > 0 else 1.0
        if max_val > 255.0:
            return arr / 10000.0
        elif max_val > 1.0:
            return arr / 255.0
        return arr

    def supports_input(self, input_data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate whether input is a compatible temporal pair.
        Supports:
          - Dict: {"t1": ..., "t2": ..., "nodata_value": ...}
          - Tuple / List: (t1, t2)
        """
        if input_data is None:
            return False, "Input data is None. Expected temporal pair (t1, t2)."

        if isinstance(input_data, dict):
            if "t1" not in input_data or "t2" not in input_data:
                return False, "Input dict must contain both 't1' and 't2' keys."
            t1 = input_data["t1"]
            t2 = input_data["t2"]
            nodata = input_data.get("nodata_value")
        elif isinstance(input_data, (tuple, list)) and len(input_data) == 2:
            t1, t2 = input_data[0], input_data[1]
            nodata = None
        else:
            return False, f"Expected temporal pair tuple (t1, t2) or dict with 't1' and 't2', got {type(input_data).__name__}."

        val_res = validate_temporal_pair(t1, t2, nodata_value=nodata)
        if not val_res.is_valid:
            return False, val_res.error_message

        return True, None

    def compute_change_map(
        self,
        t1_arr: np.ndarray,
        t2_arr: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Compute Normalized Change Vector Analysis (CVA) magnitude and binary mask.
        
        Returns:
            cva_norm: 2D float array in [0, 1] representing normalized change magnitude.
            binary_mask: 2D boolean array of detected changed valid pixels.
            mean_deltas: List of mean absolute differences per channel over valid pixels.
        """
        t1_norm = self._normalize_image(t1_arr)
        t2_norm = self._normalize_image(t2_arr)

        # 1. Per-channel absolute differences
        diff = np.abs(t2_norm - t1_norm)
        channels = diff.shape[-1]

        # 2. Change Vector Analysis magnitude rho = sqrt(sum_c diff_c^2)
        rho = np.sqrt(np.sum(diff ** 2, axis=-1))
        # Normalize by sqrt(channels) to bound in [0, 1]
        cva_norm = rho / np.sqrt(channels)
        cva_norm[~valid_mask] = 0.0

        # 3. Apply validation-selected decision threshold
        binary_mask = (cva_norm >= self.decision_threshold) & valid_mask

        # 4. Compute per-channel mean deltas over valid pixels
        mean_deltas: List[float] = []
        for c in range(channels):
            channel_diff = diff[:, :, c]
            valid_vals = channel_diff[valid_mask]
            m_delta = float(np.mean(valid_vals)) if valid_vals.size > 0 else 0.0
            mean_deltas.append(round(m_delta, 4))

        return cva_norm, binary_mask, mean_deltas

    def _extract_largest_cluster_bbox(
        self,
        binary_mask: np.ndarray
    ) -> Tuple[Optional[List[int]], int]:
        """
        Locate the bounding box of the largest connected changed region.
        
        Enforces documented minimum-area rule:
        Returns [ymin, xmin, ymax, xmax] in pixel coordinates if cluster size >= min_component_area,
        or None if no cluster meets the threshold.
        """
        if not np.any(binary_mask):
            return None, 0

        labeled, num_features = scipy.ndimage.label(binary_mask)
        if num_features == 0:
            return None, 0

        # Calculate component sizes
        component_sizes = scipy.ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
        largest_label = int(np.argmax(component_sizes)) + 1
        largest_size = int(component_sizes[largest_label - 1])

        if largest_size < self.min_component_area:
            return None, largest_size

        coords = np.argwhere(labeled == largest_label)
        if len(coords) == 0:
            return None, largest_size

        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0) + 1

        return [int(ymin), int(xmin), int(ymax), int(xmax)], largest_size

    def predict(self, input_data: Any) -> SpecialistResult:
        """
        Execute deterministic change detection on validated temporal pair.
        """
        is_compat, error_msg = self.supports_input(input_data)
        if not is_compat:
            return SpecialistResult(
                status=SpecialistStatus.INCOMPATIBLE_INPUT,
                task=TaskType.CHANGE_DETECTION.value,
                unavailable_reason=f"Incompatible temporal input: {error_msg}",
                required_inputs_description="Requires co-registered temporal pair observation (t1, t2)"
            )

        # Unpack validated arrays
        if isinstance(input_data, dict):
            t1, t2 = input_data["t1"], input_data["t2"]
            nodata = input_data.get("nodata_value")
        else:
            t1, t2 = input_data[0], input_data[1]
            nodata = None

        val = validate_temporal_pair(t1, t2, nodata_value=nodata)
        cva_norm, binary_mask, mean_deltas = self.compute_change_map(
            val.t1_array, val.t2_array, val.valid_mask
        )

        changed_pixels = int(np.sum(binary_mask))
        valid_pixels = val.valid_pixels
        changed_fraction = float(changed_pixels / valid_pixels) if valid_pixels > 0 else 0.0
        changed_pct = changed_fraction * 100.0

        is_significant_change = changed_fraction > 0.005  # > 0.5% of valid scene area altered
        bbox, cluster_area = self._extract_largest_cluster_bbox(binary_mask)

        predictions = (
            ["Surface alteration detected"] if is_significant_change
            else ["No significant surface change detected"]
        )

        return SpecialistResult(
            status=SpecialistStatus.SUCCESS,
            task=TaskType.CHANGE_DETECTION.value,
            predictions=predictions,
            probabilities={
                "change_detected": 1.0 if is_significant_change else 0.0,
                "changed_area_fraction": round(changed_fraction, 6),
                "peak_cva_magnitude": round(float(np.max(cva_norm)), 4) if cva_norm.size > 0 else 0.0
            },
            thresholds_applied={
                "decision_threshold": self.decision_threshold,
                "min_component_area_pixels": float(self.min_component_area),
                "significance_area_fraction": 0.005
            },
            execution_metadata={
                "specialist_name": self.name,
                "specialist_family": "temporal_difference_baseline",
                "total_pixels": val.total_pixels,
                "valid_pixels": valid_pixels,
                "valid_pixel_pct": round(val.valid_fraction * 100.0, 2),
                "changed_pixels": changed_pixels,
                "changed_area_pct_of_valid": round(changed_pct, 4),
                "mean_channel_deltas": mean_deltas,
                "largest_change_cluster_bbox_pixel": bbox,
                "largest_cluster_pixel_area": cluster_area,
                "spatial_coordinate_system": "unprojected_pixel_coordinates_[ymin_xmin_ymax_xmax]",
                "preprocessing": "Normalized CVA over valid non-masked pixels",
                "is_stub": False
            }
        )
