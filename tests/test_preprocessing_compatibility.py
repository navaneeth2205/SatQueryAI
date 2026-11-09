"""
Preprocessing Compatibility and Regression Tests for Phase 4 Stage B Protocol.

Validates the Stage B specialist preprocessing and tensor contract against the
frozen Phase 3 implementations in satquery.specialists.dataset_land_cover and
satquery.specialists.dataset_change_detection.

Specifically verifies:
1. Sentinel-2 L2A optical reflectance conversion under archival PB < 04.00 (BOA_ADD_OFFSET = 0)
   versus modern PB >= 04.00 (BOA_ADD_OFFSET = -1000).
2. Bitwise parity between Stage B PB < 04.00 standardization and frozen normalize_optical().
3. Detection and quantification of the artificial +0.10 reflectance shift if PB >= 04.00
   is naively processed with DN / 10000.0.
4. Non-clipping physical range inspection (preserves valid physical excursions [-0.05, 1.20]
   and rejects non-finite NaN/Inf or non-physical data).
5. Sentinel-1 SAR normalization bitwise parity with frozen normalize_sar().
6. Change Detection RGB ImageNet standardization bitwise parity with frozen normalize_rgb().
"""

import numpy as np
import pytest
import torch

from satquery.specialists.dataset_land_cover import (
    TRAIN_OPTICAL_MEANS,
    TRAIN_OPTICAL_STDS,
    SAR_DB_MIN,
    SAR_DB_MAX,
    SAR_DB_SPAN,
    normalize_optical,
    normalize_sar,
)
from satquery.specialists.dataset_change_detection import (
    S2_RGB_MEAN,
    S2_RGB_STD,
    normalize_rgb,
)


# ==============================================================================
# Stage B Preprocessing Reference Specification (Preregistration v1.11)
# ==============================================================================

def stage_b_convert_s2_dn_to_reflectance(
    raw_dn: np.ndarray,
    processing_baseline: float = 3.0,
    boa_add_offset: float = 0.0,
) -> np.ndarray:
    """
    Stage B conversion of raw Sentinel-2 L2A Digital Numbers to surface reflectance.

    Formula:
      rho = (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE
      where QUANTIFICATION_VALUE = 10,000.0.

    Baseline handling:
      - PB < 04.00:  BOA_ADD_OFFSET = 0    => rho = DN / 10000.0
      - PB >= 04.00: BOA_ADD_OFFSET = -1000 => rho = (DN - 1000) / 10000.0
    """
    dn_float = raw_dn.astype(np.float32)
    if processing_baseline >= 4.0:
        offset = -1000.0 if boa_add_offset == 0.0 else boa_add_offset
    else:
        offset = 0.0 if boa_add_offset == 0.0 else boa_add_offset

    refl = (dn_float + offset) / 10000.0
    return refl


def stage_b_standardize_optical(
    refl: np.ndarray,
    means: np.ndarray = TRAIN_OPTICAL_MEANS,
    stds: np.ndarray = TRAIN_OPTICAL_STDS,
    expected_range: tuple = (-0.05, 1.20),
) -> np.ndarray:
    """
    Stage B standardization of 10-band optical surface reflectance array (10, H, W).
    Verifies finite values and nominal physical boundaries without silent force-clipping.
    """
    if not np.all(np.isfinite(refl)):
        raise ValueError("Stage B validation failure: non-finite NaN or Inf detected in optical reflectance.")

    min_val = float(np.min(refl))
    max_val = float(np.max(refl))
    if min_val < expected_range[0] or max_val > expected_range[1]:
        raise ValueError(
            f"Stage B validation failure: optical reflectance [{min_val:.4f}, {max_val:.4f}] "
            f"exceeds physical tolerance window [{expected_range[0]}, {expected_range[1]}]."
        )

    means_t = means[:, None, None]
    stds_t = stds[:, None, None]
    return (refl - means_t) / stds_t


def stage_b_normalize_sar(raw_s1: np.ndarray) -> np.ndarray:
    """
    Stage B normalization of 2-band Sentinel-1 SAR array (2, H, W) in float32 dB.
    Clips backscatter to [-35.0, +5.0] dB and linearly scales to [0.0, 1.0].
    """
    if not np.all(np.isfinite(raw_s1)):
        raise ValueError("Stage B validation failure: non-finite NaN or Inf detected in SAR tensor.")
    s1_clipped = np.clip(raw_s1.astype(np.float32), SAR_DB_MIN, SAR_DB_MAX)
    return (s1_clipped - SAR_DB_MIN) / SAR_DB_SPAN


def stage_b_normalize_cd_rgb(img_arr: np.ndarray) -> np.ndarray:
    """
    Stage B normalization of uint8 RGB array (H, W, 3) for Change Detection.
    Scales [0, 255] by 1/255.0, standardizes with ImageNet statistics, and returns (3, H, W).
    """
    if not np.all(np.isfinite(img_arr)):
        raise ValueError("Stage B validation failure: non-finite NaN or Inf detected in CD RGB array.")
    norm = (img_arr.astype(np.float32) / 255.0 - S2_RGB_MEAN) / S2_RGB_STD
    if norm.ndim == 3 and norm.shape[2] == 3:
        return np.transpose(norm, (2, 0, 1))
    return norm


# ==============================================================================
# Unit and Regression Test Suite
# ==============================================================================

class TestPreprocessingCompatibility:
    """Regression tests verifying Stage B protocol compatibility with frozen Phase 3 code."""

    def test_pb_archival_matches_frozen_normalize_optical(self):
        """
        Verify that for archival data (PB < 04.00, BigEarthNet-14k baseline),
        Stage B conversion + standardization is bitwise identical to frozen normalize_optical().
        """
        rng = np.random.RandomState(42)
        # Simulate raw DN in realistic ranges [0, 10000]
        raw_dn = rng.randint(0, 10001, size=(10, 120, 120)).astype(np.float32)

        # Frozen Phase 3 implementation
        frozen_out = normalize_optical(raw_dn)

        # Stage B protocol implementation under PB < 04.00
        refl = stage_b_convert_s2_dn_to_reflectance(raw_dn, processing_baseline=2.04)
        stage_b_out = stage_b_standardize_optical(refl)

        # Exact bitwise / float32 equivalence check
        np.testing.assert_array_equal(frozen_out, stage_b_out)

    def test_pb_ge_04_offset_shift_quantification(self):
        """
        Verify that applying the naive DN / 10000.0 formula to PB >= 04.00 products
        induces exactly a +0.10 reflectance shift (1000 DN / 10000.0 = 0.10),
        and demonstrate that Stage B corrects this shift.
        """
        rng = np.random.RandomState(123)
        # Modern product: DN includes -1000 offset, so a surface with true reflectance 0.20 has DN = 3000
        # (3000 - 1000) / 10000 = 0.20
        true_surface_refl = np.full((10, 120, 120), 0.20, dtype=np.float32)
        modern_dn = true_surface_refl * 10000.0 + 1000.0  # DN = 3000

        # Naive conversion without offset
        naive_refl = modern_dn / 10000.0  # 0.30
        assert np.isclose(naive_refl[0, 0, 0], 0.30, atol=1e-5)

        # Shift magnitude
        refl_bias = float(np.mean(naive_refl - true_surface_refl))
        assert np.isclose(refl_bias, 0.10, atol=1e-6), f"Expected +0.10 bias, got {refl_bias}"

        # Standardized difference in units of sigma for Band 4 (Red, sigma=0.074)
        # Bias of 0.10 on Band 4 is 0.10 / 0.074 = 1.35 sigma!
        sigma_shift_band4 = 0.10 / float(TRAIN_OPTICAL_STDS[2])
        assert sigma_shift_band4 > 1.0, "Bias on Band 4 exceeds one full standard deviation"

        # Stage B baseline-aware conversion
        stage_b_refl = stage_b_convert_s2_dn_to_reflectance(modern_dn, processing_baseline=4.0)
        np.testing.assert_allclose(stage_b_refl, true_surface_refl, rtol=1e-5, atol=1e-5)

    def test_non_clipping_preserves_valid_physical_excursions(self):
        """
        Verify that Stage B does NOT force-clip surface reflectance to [0, 1].
        Slightly negative values (e.g. -0.005 due to atmospheric dark target oversubtraction)
        and values slightly above 1.0 (e.g. 1.05 due to cloud glint or specular reflection)
        are preserved without truncation.
        """
        refl = np.full((10, 120, 120), 0.20, dtype=np.float32)
        refl[0, 0, 0] = -0.005  # Valid slight negative shadow
        refl[1, 0, 0] = 1.050   # Valid specular glint

        # Standardization must proceed without clipping these values
        std_out = stage_b_standardize_optical(refl)
        assert np.all(np.isfinite(std_out))

        # Check that value at [0, 0, 0] was standardized from -0.005, not clipped to 0.0
        expected_neg_std = (-0.005 - TRAIN_OPTICAL_MEANS[0]) / TRAIN_OPTICAL_STDS[0]
        assert np.isclose(std_out[0, 0, 0], expected_neg_std, atol=1e-5)

        # Check that value at [1, 0, 0] was standardized from 1.050, not clipped to 1.0
        expected_high_std = (1.050 - TRAIN_OPTICAL_MEANS[1]) / TRAIN_OPTICAL_STDS[1]
        assert np.isclose(std_out[1, 0, 0], expected_high_std, atol=1e-5)

    def test_extreme_non_physical_or_nan_values_rejected(self):
        """
        Verify that extreme unphysical values (< -0.05 or > 1.20) or NaN/Inf raise validation errors.
        """
        # Non-finite check
        refl_nan = np.full((10, 120, 120), 0.20, dtype=np.float32)
        refl_nan[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite NaN or Inf"):
            stage_b_standardize_optical(refl_nan)

        # Out-of-bounds negative check
        refl_neg = np.full((10, 120, 120), 0.20, dtype=np.float32)
        refl_neg[0, 0, 0] = -0.20  # Far below physical threshold
        with pytest.raises(ValueError, match="exceeds physical tolerance window"):
            stage_b_standardize_optical(refl_neg)

        # Out-of-bounds positive check
        refl_high = np.full((10, 120, 120), 0.20, dtype=np.float32)
        refl_high[0, 0, 0] = 2.50  # Far above physical threshold
        with pytest.raises(ValueError, match="exceeds physical tolerance window"):
            stage_b_standardize_optical(refl_high)

    def test_sar_normalization_matches_frozen_implementation(self):
        """
        Verify that Stage B SAR normalization is bitwise identical to frozen normalize_sar().
        """
        rng = np.random.RandomState(456)
        raw_s1 = rng.uniform(-45.0, 15.0, size=(2, 120, 120)).astype(np.float32)

        frozen_s1 = normalize_sar(raw_s1)
        stage_b_s1 = stage_b_normalize_sar(raw_s1)

        np.testing.assert_array_equal(frozen_s1, stage_b_s1)
        assert np.min(stage_b_s1) >= 0.0
        assert np.max(stage_b_s1) <= 1.0

    def test_change_detection_rgb_normalization_matches_frozen(self):
        """
        Verify that Stage B Change Detection normalization is bitwise identical to frozen normalize_rgb().
        """
        rng = np.random.RandomState(789)
        img_arr = rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)

        frozen_cd = normalize_rgb(img_arr)
        stage_b_cd = stage_b_normalize_cd_rgb(img_arr)

        np.testing.assert_array_equal(frozen_cd, stage_b_cd)
