"""
Multitemporal Change Detection Specialist for SatQuery AI.

Implements the deterministic TemporalDifferenceBaseline:
- Robust input validation and NoData/finite-value masking
- Normalized per-channel differencing and Change Vector Analysis (CVA)
- Validation-selected decision thresholding
- Connected-component spatial cluster extraction in documented pixel space
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage

from satquery.taxonomy import TaskType, InputType
from satquery.specialists.base import BaseSpecialist, SpecialistResult, SpecialistStatus
from satquery.utils.temporal import validate_temporal_pair, TemporalPairValidation

logger = logging.getLogger(__name__)


class TemporalDifferenceBaseline(BaseSpecialist):
    """
    Deterministic Change Detection Specialist using Normalized Change Vector Analysis (CVA).
    
    Operates on co-registered bi-temporal observation pairs (T1, T2) across arbitrary spectral channels.
    Thresholds are selected on validation data against ground truth change masks.
    """

    def __init__(
        self,
        decision_threshold: float = 0.28,
        min_component_area: int = 9,
        artifacts_dir: Optional[Union[str, Path]] = None,
        method: str = "standard",
        smoothing_sigma: float = 1.2,
        hysteresis_low_threshold: float = 0.14
    ):
        super().__init__(
            name="temporal_difference_baseline",
            supported_tasks=[TaskType.CHANGE_DETECTION],
            required_input_type=InputType.TEMPORAL_PAIR
        )
        self.decision_threshold = float(decision_threshold)
        self.min_component_area = int(min_component_area)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.method = str(method).lower()
        self.smoothing_sigma = float(smoothing_sigma)
        self.hysteresis_low_threshold = float(hysteresis_low_threshold)

        # Load validation-tuned threshold from disk if artifact exists
        if self.artifacts_dir and (self.artifacts_dir / "threshold_config.json").exists():
            try:
                with open(self.artifacts_dir / "threshold_config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.decision_threshold = float(cfg.get("decision_threshold", self.decision_threshold))
                    self.min_component_area = int(cfg.get("min_component_area", self.min_component_area))
            except Exception:
                pass

    def _normalize_pair(
        self,
        t1_arr: np.ndarray,
        t2_arr: np.ndarray,
        product_type: Optional[str] = None,
        source_dtype: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply product-grounded, documented scaling jointly across both observations in a temporal pair.

        Strict Rules:
        1. 8-bit RGB Digital Values (e.g., OSCD RGB, source_dtype='uint8' or product_type='rgb_8bit'):
           Scaled strictly by 255.0 to yield normalized uncalibrated digital values in [0, 1].
           Rejects negative values or values exceeding 255.0.
        2. Sentinel-2 Level-2A BOA Surface Reflectance (source_dtype='uint16' or product_type='s2_l2a'):
           Scaled strictly by 10,000.0 to yield dimensionless surface reflectance in [0, 1].
           Rejects negative values or values exceeding 10,000.0 (reflectance ceiling 1.0).
        3. Pre-normalized Float Observations (product_type='pre_normalized' or float arrays with all values in [0.0, 1.0]):
           Preserved without scaling (scale = 1.0).
        4. Ambiguous Raw Floating-Point Inputs:
           Floating-point arrays with values > 1.0 without declared product_type are rejected
           to prevent silent value-guessing.
        """
        max_val = max(
            float(np.nanmax(t1_arr)) if t1_arr.size > 0 else 0.0,
            float(np.nanmax(t2_arr)) if t2_arr.size > 0 else 0.0
        )
        min_val = min(
            float(np.nanmin(t1_arr)) if t1_arr.size > 0 else 0.0,
            float(np.nanmin(t2_arr)) if t2_arr.size > 0 else 0.0
        )

        resolved_product = product_type
        if resolved_product is None:
            if source_dtype == "uint8":
                resolved_product = "rgb_8bit"
            elif source_dtype in ("uint16", "int16"):
                resolved_product = "s2_l2a"
            elif max_val <= 1.0 and min_val >= 0.0:
                resolved_product = "pre_normalized"
                logger.info(
                    "Auto-resolved temporal observation with values in [0.0, 1.0] to product_type='pre_normalized'. "
                    "For explicit provenance, provide product_type='pre_normalized' in the input dictionary."
                )
            else:
                raise ValueError(
                    f"Ambiguous floating-point temporal observation with unscaled values (min={min_val:.2f}, max={max_val:.2f}). "
                    f"For unscaled digital numbers or surface reflectance, explicitly specify 'product_type' ('rgb_8bit' or 's2_l2a') "
                    f"in the input dictionary, or pre-normalize observations to [0.0, 1.0]."
                )

        if resolved_product in ("rgb_8bit", "oscd_rgb"):
            if min_val < 0.0 or max_val > 255.0:
                raise ValueError(f"Values outside valid 8-bit digital range [0, 255]: min={min_val}, max={max_val}")
            scale = 255.0
        elif resolved_product in ("s2_l2a", "surface_reflectance_10k"):
            if min_val < 0.0 or max_val > 10000.0:
                raise ValueError(f"Values outside valid Sentinel-2 L2A reflectance range [0, 10000]: min={min_val}, max={max_val}")
            if max_val <= 1.0 and source_dtype in ("float32", "float64") and not np.all(t1_arr == 0):
                raise ValueError(
                    f"Temporal observation with float values in [0.0, 1.0] (max_val={max_val:.4f}) received under product_type='{resolved_product}'. "
                    f"Sentinel-2 L2A ('s2_l2a') expects raw scaled BOA digital numbers in [0, 10000] (scale factor 10000.0). "
                    f"If the observation has already been converted to surface reflectance in [0.0, 1.0], "
                    f"explicitly specify product_type='pre_normalized' to prevent redundant down-scaling."
                )
            scale = 10000.0
        elif resolved_product in ("pre_normalized", "normalized_float"):
            if min_val < 0.0 or max_val > 1.0:
                raise ValueError(f"Values outside valid pre-normalized range [0.0, 1.0]: min={min_val}, max={max_val}")
            scale = 1.0
        else:
            raise ValueError(f"Unsupported product_type '{resolved_product}'. Expected 'rgb_8bit', 's2_l2a', or 'pre_normalized'.")

        return t1_arr / scale, t2_arr / scale

    def _normalize_image(
        self,
        arr: np.ndarray,
        product_type: Optional[str] = None,
        source_dtype: Optional[str] = None
    ) -> np.ndarray:
        """Single-observation wrapper around product-grounded _normalize_pair."""
        t1_norm, _ = self._normalize_pair(arr, arr, product_type=product_type, source_dtype=source_dtype)
        return t1_norm

    def supports_input(self, input_data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate whether input is a compatible temporal pair.
        Supports:
          - Dict: {"t1": ..., "t2": ..., "nodata_value": ..., "product_type": ...}
          - Tuple / List: (t1, t2)
        """
        if input_data is None:
            return False, "Input data is None. Expected temporal pair (t1, t2)."

        product_type = None
        if isinstance(input_data, dict):
            if "t1" not in input_data or "t2" not in input_data:
                return False, "Input dict must contain both 't1' and 't2' keys."
            t1 = input_data["t1"]
            t2 = input_data["t2"]
            nodata = input_data.get("nodata_value")
            product_type = input_data.get("product_type")
        elif isinstance(input_data, (tuple, list)) and len(input_data) == 2:
            t1, t2 = input_data[0], input_data[1]
            nodata = None
        else:
            return False, f"Expected temporal pair tuple (t1, t2) or dict with 't1' and 't2', got {type(input_data).__name__}."

        val_res = validate_temporal_pair(t1, t2, nodata_value=nodata)
        if not val_res.is_valid:
            return False, val_res.error_message

        # Verify product-grounded normalization
        try:
            self._normalize_pair(
                val_res.t1_array, val_res.t2_array,
                product_type=product_type,
                source_dtype=val_res.source_dtype
            )
        except ValueError as err:
            return False, str(err)

        return True, None

    def compute_change_map(
        self,
        t1_arr: np.ndarray,
        t2_arr: np.ndarray,
        valid_mask: np.ndarray,
        product_type: Optional[str] = None,
        source_dtype: Optional[str] = None,
        method: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Compute Normalized Change Vector Analysis (CVA) magnitude and binary mask.
        Supports standard CVA, Gaussian-smoothed CVA (optimal F1/IoU), and dual-threshold hysteresis CVA (high recall).
        
        Returns:
            cva_norm: 2D float array in [0, 1] representing normalized change magnitude.
            binary_mask: 2D boolean array of detected changed valid pixels.
            mean_deltas: List of mean absolute differences per channel over valid pixels.
        """
        t1_norm, t2_norm = self._normalize_pair(
            t1_arr, t2_arr, product_type=product_type, source_dtype=source_dtype
        )

        # 1. Per-channel absolute differences
        diff = np.abs(t2_norm - t1_norm)
        channels = diff.shape[-1]

        # 2. Change Vector Analysis magnitude rho = sqrt(sum_c diff_c^2)
        rho = np.sqrt(np.sum(diff ** 2, axis=-1))
        # Normalize by sqrt(channels) to bound in [0, 1]
        cva_norm = rho / np.sqrt(channels)
        cva_norm[~valid_mask] = 0.0

        # 3. Apply change detection algorithm
        active_method = (method or getattr(self, "method", "standard")).lower()

        if active_method == "smoothed":
            # Method 3: Gaussian-smoothed CVA (Optimal F1 and IoU)
            smoothed = scipy.ndimage.gaussian_filter(cva_norm, sigma=self.smoothing_sigma)
            binary_mask = (smoothed >= self.decision_threshold) & valid_mask
        elif active_method == "hysteresis":
            # Method 4: Dual-threshold hysteresis CVA (High Recall for disaster/emergency monitoring)
            smoothed = scipy.ndimage.gaussian_filter(cva_norm, sigma=1.0)
            seeds = (smoothed >= self.decision_threshold) & valid_mask
            candidates = (smoothed >= self.hysteresis_low_threshold) & valid_mask

            if np.any(seeds):
                labeled, num_features = scipy.ndimage.label(candidates)
                if num_features > 0:
                    seed_labels = np.unique(labeled[seeds])
                    seed_labels = seed_labels[seed_labels > 0]
                    keep = np.zeros(num_features + 1, dtype=bool)
                    keep[seed_labels] = True
                    binary_mask = keep[labeled] & valid_mask
                else:
                    binary_mask = seeds
            else:
                binary_mask = np.zeros_like(seeds, dtype=bool)
        else:
            # Method 1: Standard CVA baseline
            binary_mask = (cva_norm >= self.decision_threshold) & valid_mask

        # Apply minimum connected component area filter if specified
        if self.min_component_area > 1 and np.any(binary_mask):
            labeled, num_features = scipy.ndimage.label(binary_mask)
            if num_features > 0:
                sizes = scipy.ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
                keep = np.zeros(num_features + 1, dtype=bool)
                keep[1:] = sizes >= self.min_component_area
                binary_mask = keep[labeled]

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

        product_type = input_data.get("product_type") if isinstance(input_data, dict) else None
        method = input_data.get("method") if isinstance(input_data, dict) else None
        val = validate_temporal_pair(t1, t2, nodata_value=nodata)
        cva_norm, binary_mask, mean_deltas = self.compute_change_map(
            val.t1_array, val.t2_array, val.valid_mask,
            product_type=product_type,
            source_dtype=val.source_dtype,
            method=method
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
                "change_detection_method": (method or getattr(self, "method", "standard")).lower(),
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
