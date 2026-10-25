"""
Synthetic Regression Tests for TemporalDifferenceBaseline.

Verifies deterministic edge-case behavior under controlled synthetic inputs:
- Identical images (T1 == T2) produce zero detected change.
- Controlled synthetic perturbations produce localized bounding box detections.
- Incompatible input structures and dimension mismatches are rejected cleanly.
"""

import numpy as np
import pytest

from satquery.taxonomy import TaskType
from satquery.specialists.base import SpecialistStatus
from satquery.specialists.change_detection import TemporalDifferenceBaseline
from satquery.utils.temporal import validate_temporal_pair


def test_identical_images_zero_change():
    """Identical observations must yield 0% detected change and null cluster bounding box."""
    np.random.seed(42)
    t1 = np.random.uniform(0.1, 0.9, size=(64, 64, 3)).astype(np.float32)
    t2 = t1.copy()

    baseline = TemporalDifferenceBaseline(decision_threshold=0.20, min_component_area=9)
    res = baseline.predict((t1, t2))

    assert res.status == SpecialistStatus.SUCCESS
    assert res.task == TaskType.CHANGE_DETECTION.value
    assert res.predictions == ["No significant surface change detected"]
    assert res.probabilities["change_detected"] == 0.0
    assert res.probabilities["changed_area_fraction"] == 0.0
    assert res.execution_metadata["changed_pixels"] == 0
    assert res.execution_metadata["largest_change_cluster_bbox_pixel"] is None


def test_controlled_perturbation_localization():
    """Injecting a localized 20x20 perturbation must be detected and bounded in pixel space."""
    np.random.seed(42)
    t1 = np.full((100, 100, 3), 0.2, dtype=np.float32)
    t2 = t1.copy()

    # Inject significant change in region y: [30, 50], x: [40, 60] (400 pixels)
    t2[30:50, 40:60, :] = 0.9

    baseline = TemporalDifferenceBaseline(decision_threshold=0.25, min_component_area=9)
    res = baseline.predict((t1, t2))

    assert res.status == SpecialistStatus.SUCCESS
    assert res.predictions == ["Surface alteration detected"]
    assert res.probabilities["change_detected"] == 1.0
    assert res.execution_metadata["changed_pixels"] == 400

    bbox = res.execution_metadata["largest_change_cluster_bbox_pixel"]
    assert bbox is not None
    ymin, xmin, ymax, xmax = bbox
    assert ymin == 30 and ymax == 50
    assert xmin == 40 and xmax == 60


def test_shape_mismatch_rejection():
    """Observations with mismatched dimensions must return INCOMPATIBLE_INPUT."""
    t1 = np.zeros((50, 50, 3), dtype=np.float32)
    t2 = np.zeros((50, 60, 3), dtype=np.float32)

    baseline = TemporalDifferenceBaseline()
    res = baseline.predict((t1, t2))

    assert res.status == SpecialistStatus.INCOMPATIBLE_INPUT
    assert "Dimension mismatch" in res.unavailable_reason


def test_single_image_input_rejection():
    """Passing a single 2D/3D image array must return INCOMPATIBLE_INPUT."""
    single_img = np.zeros((50, 50, 3), dtype=np.float32)

    baseline = TemporalDifferenceBaseline()
    res = baseline.predict(single_img)

    assert res.status == SpecialistStatus.INCOMPATIBLE_INPUT
    assert "Expected temporal pair" in res.unavailable_reason


def test_nan_infinite_value_masking():
    """Pixels containing NaNs or Infinities must be excluded from valid pixels."""
    t1 = np.ones((20, 20, 3), dtype=np.float32)
    t2 = np.ones((20, 20, 3), dtype=np.float32)

    # Corrupt 10 pixels with NaNs
    t1[0, :10, :] = np.nan

    val = validate_temporal_pair(t1, t2)
    assert val.is_valid
    assert val.valid_pixels == 390
    assert val.total_pixels == 400


def test_oscd_rgb_uint8_scaling():
    """Verify uint8 observations are strictly scaled by 255.0 on both dates."""
    baseline = TemporalDifferenceBaseline()
    t1 = np.full((20, 20, 3), 128, dtype=np.uint8)
    t2 = np.full((20, 20, 3), 128, dtype=np.uint8)

    t1_norm, t2_norm = baseline._normalize_pair(t1.astype(np.float32), t2.astype(np.float32), source_dtype="uint8")
    assert np.isclose(t1_norm.max(), 128.0 / 255.0)
    assert np.isclose(t2_norm.max(), 128.0 / 255.0)
    assert t1_norm.max() <= 1.0 and t2_norm.max() <= 1.0


def test_ambiguous_raw_float_rejection():
    """Verify raw float observations with values > 1.0 without declared product_type are rejected."""
    baseline = TemporalDifferenceBaseline()
    # Floats with values up to 150.0 without product_type
    t1 = np.full((20, 20, 3), 150.0, dtype=np.float32)
    t2 = np.full((20, 20, 3), 150.0, dtype=np.float32)

    res = baseline.predict((t1, t2))
    assert res.status == SpecialistStatus.INCOMPATIBLE_INPUT
    assert "Ambiguous floating-point temporal observation" in res.unavailable_reason


def test_pre_normalized_float_preservation():
    """Verify float observations in [0, 1] remain unchanged with scale factor 1.0."""
    baseline = TemporalDifferenceBaseline()
    t1 = np.full((20, 20, 3), 0.75, dtype=np.float32)
    t2 = np.full((20, 20, 3), 0.75, dtype=np.float32)

    t1_norm, t2_norm = baseline._normalize_pair(t1, t2, product_type="pre_normalized")
    assert np.allclose(t1_norm, 0.75)
    assert np.allclose(t2_norm, 0.75)


def test_invalid_reflectance_range_rejection():
    """Verify Sentinel-2 L2A observations with values exceeding 10,000 or negative are rejected."""
    baseline = TemporalDifferenceBaseline()
    t1 = np.full((20, 20, 3), 12000.0, dtype=np.float32)
    t2 = np.full((20, 20, 3), 5000.0, dtype=np.float32)

    res = baseline.predict({"t1": t1, "t2": t2, "product_type": "s2_l2a"})
    assert res.status == SpecialistStatus.INCOMPATIBLE_INPUT
    assert "Values outside valid Sentinel-2 L2A reflectance range" in res.unavailable_reason


def test_already_normalized_float_with_s2_l2a_rejection():
    """Verify float observations in [0, 1] passed under product_type='s2_l2a' are rejected to prevent double-scaling."""
    baseline = TemporalDifferenceBaseline()
    t1 = np.full((20, 20, 3), 0.45, dtype=np.float32)
    t2 = np.full((20, 20, 3), 0.45, dtype=np.float32)

    res = baseline.predict({"t1": t1, "t2": t2, "product_type": "s2_l2a"})
    assert res.status == SpecialistStatus.INCOMPATIBLE_INPUT
    assert "received under product_type='s2_l2a'" in res.unavailable_reason
    assert "pre_normalized" in res.unavailable_reason
