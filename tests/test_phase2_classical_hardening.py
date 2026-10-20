"""
Unit and Integration Tests for Phase 2 Classical Feature Hardening.

Covers:
  1. Multimodal feature extraction integrity (488 features).
  2. Level-0 baseline backward compatibility.
  3. Spatial pyramid quadrant partitioning sensitivity.
  4. Spectral and SAR polarimetric index boundaries.
  5. Texture gradient calculation.
  6. Phase 2 Specialist model loading and prediction with LightGBM.
"""

from pathlib import Path
import numpy as np
import pytest

from satquery.features.multimodal import (
    extract_level0_features,
    extract_spatial_pyramid_features,
    extract_spectral_indices,
    extract_sar_indices,
    extract_texture_features,
    extract_multimodal_features_phase2,
    get_phase2_feature_names,
)
from satquery.specialists.land_cover import LandCoverClassifier
from satquery.specialists.base import SpecialistStatus


@pytest.fixture
def synthetic_s1_s2():
    """Synthetic S1 (2, 120, 120) and S2 (10, 120, 120) observation arrays."""
    np.random.seed(42)
    s1 = np.random.uniform(-25.0, 5.0, size=(2, 120, 120)).astype(np.float32)
    s2 = np.random.uniform(200.0, 4000.0, size=(10, 120, 120)).astype(np.float32)
    return s1, s2


def test_feature_names_length_and_structure():
    names = get_phase2_feature_names()
    assert len(names) == 488
    assert len(set(names)) == 488, "Feature names contain duplicates!"

    # Check prefix categories
    assert any(n.startswith("global_s1_") for n in names)
    assert any(n.startswith("global_s2_") for n in names)
    assert any(n.startswith("quad_top_left_") for n in names)
    assert any(n.startswith("index_s2_ndvi") for n in names)
    assert any(n.startswith("index_s1_cr_linear") for n in names)
    assert any(n.startswith("texture_grad_") for n in names)


def test_multimodal_feature_extraction_shape_and_finite(synthetic_s1_s2):
    s1, s2 = synthetic_s1_s2
    feats = extract_multimodal_features_phase2(s1, s2)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (488,)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "Extracted features contain NaN or Inf values!"


def test_level0_backward_compatibility(synthetic_s1_s2):
    s1, s2 = synthetic_s1_s2
    l0 = extract_level0_features(s1, s2)
    p2 = extract_multimodal_features_phase2(s1, s2)

    assert len(l0) == 84
    # The first 84 features of Phase 2 must match Level-0 exactly
    assert np.allclose(l0, p2[:84])


def test_spatial_pyramid_quadrant_partitioning():
    # Construct an image with signal concentrated strictly in Top-Left quadrant
    s1 = np.zeros((2, 120, 120), dtype=np.float32)
    s2 = np.zeros((10, 120, 120), dtype=np.float32)
    s2[6, :60, :60] = 5000.0  # High NIR in Top-Left quadrant only

    pyr_feats = extract_spatial_pyramid_features(s1, s2)
    assert len(pyr_feats) == 336

    # 4 quadrants x 84 stats.
    # Q0: [0:84], Q1: [84:168], Q2: [168:252], Q3: [252:336]
    q0_feats = pyr_feats[0:84]
    q1_feats = pyr_feats[84:168]

    # Q0 should have non-zero NIR max, Q1 should be all zero
    assert np.max(q0_feats) > 1000.0
    assert np.max(q1_feats) == 0.0


def test_spectral_and_sar_indices(synthetic_s1_s2):
    s1, s2 = synthetic_s1_s2
    spec_feats = extract_spectral_indices(s2)
    sar_feats = extract_sar_indices(s1)

    assert len(spec_feats) == 35  # 5 indices x 7 stats
    assert len(sar_feats) == 21   # 3 indices x 7 stats

    # NDVI is first index (stats 0..6)
    ndvi_mean = spec_feats[0]
    assert -1.0 <= ndvi_mean <= 1.0


def test_texture_gradient_features(synthetic_s1_s2):
    s1, s2 = synthetic_s1_s2
    tex_feats = extract_texture_features(s1, s2)

    assert len(tex_feats) == 12  # 3 channels x 4 stats
    assert np.all(tex_feats >= 0.0), "Gradient magnitudes must be non-negative"


def test_phase2_specialist_prediction_with_488_features(synthetic_s1_s2):
    s1, s2 = synthetic_s1_s2
    feats_488 = extract_multimodal_features_phase2(s1, s2)

    clf = LandCoverClassifier()
    assert clf.has_phase2 is True

    # 1. Input validation support
    is_valid, err = clf.supports_input(feats_488)
    assert is_valid is True
    assert err is None

    # 2. Prediction execution with 488 features
    result = clf.predict(feats_488)
    assert result.status == SpecialistStatus.SUCCESS
    assert len(result.probabilities) == 16
    assert len(result.thresholds_applied) == 16
    assert result.execution_metadata["feature_count"] == 488
    assert result.execution_metadata["specialist_name"] == "LandCoverClassifier_LightGBM_Phase2"
    assert result.execution_metadata["validation_benchmark"]["micro_f1"] == 0.7500
