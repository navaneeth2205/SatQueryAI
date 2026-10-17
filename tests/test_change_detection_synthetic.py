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
