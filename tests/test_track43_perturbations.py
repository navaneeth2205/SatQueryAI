"""
Unit and Regression Test Suite for Track 4.3 Atmospheric, Cloud, and Noise Perturbations.

Verifies:
1. Native resolution awareness:
   - Land Cover cloud masks generated at (120, 120).
   - Change Detection cloud masks generated at (256, 256).
2. Exact pixel counts matching K(f_c, H, W) = floor(f_c * H * W) for all 7 fractions.
3. Strict nesting guarantee: M(f_1) <= M(f_2) everywhere for f_1 < f_2 (proper subsets).
4. Land Cover channel isolation: all 10 optical channels occluded; 2 SAR channels untouched.
5. Change Detection temporal isolation: t2 post-event RGB occluded; t1 pre-event untouched.
6. Koschmieder atmospheric haze channel and temporal isolation.
7. CRN Gaussian noise tensor shapes, independent channel/temporal generation, and monotonic scaling.
8. Complete forward-pass tensor compatibility across all 6 degradation paths.
"""

from typing import Dict
import numpy as np
import pytest
import torch

from satquery.utils.perturbations import (
    generate_continuous_noise_field,
    generate_nested_cloud_mask,
    generate_nested_cloud_mask_suite,
    apply_cloud_occlusion_land_cover,
    apply_cloud_occlusion_change_detection,
    apply_haze_land_cover,
    apply_haze_change_detection,
    generate_crn_noise_tensors,
    apply_noise_land_cover,
    apply_noise_change_detection,
    CLOUD_SEVERITY_FRACTIONS,
    HAZE_TRANSMITTANCES,
    NOISE_SEVERITY_SIGMAS,
)


EXPECTED_LC_PIXEL_COUNTS = [720, 1440, 2160, 2880, 4320, 5760, 7200]
EXPECTED_CD_PIXEL_COUNTS = [3276, 6553, 9830, 13107, 19660, 26214, 32768]


class TestTrack43Perturbations:
    """Comprehensive test suite for Track 4.3 perturbation operators."""

    def test_native_resolution_cloud_mask_dimensions_and_counts(self):
        """
        Verify that cloud masks are generated independently at native resolutions:
        - Land Cover: (120, 120) with exact pixel counts [720, 1440, 2160, 2880, 4320, 5760, 7200].
        - Change Detection: (256, 256) with exact pixel counts [3276, 6553, 9830, 13107, 19660, 26214, 32768].
        """
        seed = 42

        # 1. Land Cover masks at (120, 120)
        masks_lc = generate_nested_cloud_mask_suite(120, 120, CLOUD_SEVERITY_FRACTIONS, seed=seed)
        assert len(masks_lc) == len(CLOUD_SEVERITY_FRACTIONS)
        actual_lc_counts = [int(np.sum(masks_lc[f])) for f in CLOUD_SEVERITY_FRACTIONS]
        assert actual_lc_counts == EXPECTED_LC_PIXEL_COUNTS, f"LC counts mismatch: {actual_lc_counts}"
        for f, m in masks_lc.items():
            assert m.shape == (120, 120), f"LC mask shape mismatch for f={f}: {m.shape}"

        # 2. Change Detection masks at (256, 256)
        masks_cd = generate_nested_cloud_mask_suite(256, 256, CLOUD_SEVERITY_FRACTIONS, seed=seed)
        assert len(masks_cd) == len(CLOUD_SEVERITY_FRACTIONS)
        actual_cd_counts = [int(np.sum(masks_cd[f])) for f in CLOUD_SEVERITY_FRACTIONS]
        assert actual_cd_counts == EXPECTED_CD_PIXEL_COUNTS, f"CD counts mismatch: {actual_cd_counts}"
        for f, m in masks_cd.items():
            assert m.shape == (256, 256), f"CD mask shape mismatch for f={f}: {m.shape}"

    def test_strict_mask_nesting_proper_subsets(self):
        """
        Verify that for both resolutions, increasing severity fractions produce
        strictly nested proper subsets: M(f_1) <= M(f_2) everywhere for f_1 < f_2.
        """
        seed = 101

        for H, W in [(120, 120), (256, 256)]:
            masks = generate_nested_cloud_mask_suite(H, W, CLOUD_SEVERITY_FRACTIONS, seed=seed)
            fractions = sorted(CLOUD_SEVERITY_FRACTIONS)

            for i in range(len(fractions) - 1):
                f_smaller = fractions[i]
                f_larger = fractions[i + 1]
                m_small = masks[f_smaller]
                m_large = masks[f_larger]

                # Pointwise subset check: wherever m_small is 1, m_large must also be 1
                assert np.all(m_large >= m_small), (
                    f"Nesting violation at resolution ({H}, {W}) between {f_smaller} and {f_larger}."
                )
                # Strict proper subset: m_large must contain strictly more occluded pixels
                assert np.sum(m_large) > np.sum(m_small)

    def test_land_cover_cloud_channel_isolation(self):
        """
        Verify that Land Cover cloud occlusion:
        - Occludes all 10 optical channels identically.
        - Leaves the 2 SAR channels (VH, VV) completely untouched (difference = 0.0).
        """
        rng = np.random.RandomState(42)
        opt_in = rng.uniform(0.1, 0.6, size=(10, 120, 120)).astype(np.float32)
        sar_in = rng.uniform(0.1, 0.8, size=(2, 120, 120)).astype(np.float32)

        mask_lc = generate_nested_cloud_mask(120, 120, severity_fraction=0.20, seed=42)
        opt_out, sar_out = apply_cloud_occlusion_land_cover(opt_in, sar_in, mask_lc)

        # 1. SAR channels must be bitwise identical
        np.testing.assert_array_equal(sar_out, sar_in)
        assert float(np.max(np.abs(sar_out - sar_in))) == 0.0

        # 2. Optical channels must be occluded wherever mask == 1
        for c in range(10):
            # In unmasked pixels (mask == 0), values must match exactly
            np.testing.assert_array_equal(opt_out[c][mask_lc == 0], opt_in[c][mask_lc == 0])
            # In masked pixels (mask == 1), values must be shifted toward cloud albedo (0.95)
            assert np.all(opt_out[c][mask_lc == 1] >= opt_in[c][mask_lc == 1])

        # 3. All 10 optical channels must receive identical blend weights
        blend_diff = (opt_out[0] - opt_in[0]) / (0.95 - opt_in[0] + 1e-7)
        for c in range(1, 10):
            blend_diff_c = (opt_out[c] - opt_in[c]) / (0.95 - opt_in[c] + 1e-7)
            np.testing.assert_allclose(blend_diff_c[mask_lc == 1], blend_diff[mask_lc == 1], atol=1e-5)

    def test_change_detection_cloud_temporal_isolation(self):
        """
        Verify that Change Detection cloud occlusion:
        - Occludes all 3 RGB channels of post-event image t2.
        - Leaves pre-event image t1 completely untouched (difference = 0.0).
        """
        rng = np.random.RandomState(42)
        t1_in = rng.uniform(0.1, 0.6, size=(3, 256, 256)).astype(np.float32)
        t2_in = rng.uniform(0.1, 0.6, size=(3, 256, 256)).astype(np.float32)

        mask_cd = generate_nested_cloud_mask(256, 256, severity_fraction=0.30, seed=42)
        t1_out, t2_out = apply_cloud_occlusion_change_detection(t1_in, t2_in, mask_cd)

        # 1. Pre-event t1 must be bitwise identical
        np.testing.assert_array_equal(t1_out, t1_in)
        assert float(np.max(np.abs(t1_out - t1_in))) == 0.0

        # 2. Post-event t2 must be occluded across all 3 RGB channels
        for c in range(3):
            np.testing.assert_array_equal(t2_out[c][mask_cd == 0], t2_in[c][mask_cd == 0])
            assert np.all(t2_out[c][mask_cd == 1] >= t2_in[c][mask_cd == 1])

    def test_atmospheric_haze_isolation(self):
        """
        Verify Koschmieder haze operator:
        - LC: 10 optical channels affected; SAR untouched (difference = 0.0).
        - CD: t2 affected; t1 untouched (difference = 0.0).
        """
        rng = np.random.RandomState(42)
        opt_in = rng.uniform(0.1, 0.6, size=(10, 120, 120)).astype(np.float32)
        sar_in = rng.uniform(0.1, 0.8, size=(2, 120, 120)).astype(np.float32)
        t1_in = rng.uniform(0.1, 0.6, size=(3, 256, 256)).astype(np.float32)
        t2_in = rng.uniform(0.1, 0.6, size=(3, 256, 256)).astype(np.float32)

        # Land Cover
        opt_haze, sar_haze = apply_haze_land_cover(opt_in, sar_in, transmittance=0.50)
        np.testing.assert_array_equal(sar_haze, sar_in)
        assert float(np.max(np.abs(sar_haze - sar_in))) == 0.0
        assert not np.array_equal(opt_haze, opt_in)

        # Change Detection
        t1_haze, t2_haze = apply_haze_change_detection(t1_in, t2_in, transmittance=0.50)
        np.testing.assert_array_equal(t1_haze, t1_in)
        assert float(np.max(np.abs(t1_haze - t1_in))) == 0.0
        assert not np.array_equal(t2_haze, t2_in)

    def test_crn_noise_shapes_and_monotonic_scaling(self):
        """
        Verify CRN Gaussian noise:
        - Shapes: lc_optical (10, 120, 120), lc_sar (2, 120, 120),
                  cd_t1 (3, 256, 256), cd_t2 (3, 256, 256).
        - Independent channels (cross-channel correlation near zero).
        - Monotonic scaling across sigmas from identical base realizations.
        """
        patch_id = 7
        noise_dict = generate_crn_noise_tensors(patch_id=patch_id)

        assert noise_dict["lc_optical"].shape == (10, 120, 120)
        assert noise_dict["lc_sar"].shape == (2, 120, 120)
        assert noise_dict["cd_t1"].shape == (3, 256, 256)
        assert noise_dict["cd_t2"].shape == (3, 256, 256)

        # Check independence across optical channels (correlation between channel 0 and 1)
        c0 = noise_dict["lc_optical"][0].ravel()
        c1 = noise_dict["lc_optical"][1].ravel()
        corr_c = np.corrcoef(c0, c1)[0, 1]
        assert abs(corr_c) < 0.05, f"Cross-channel correlation too high: {corr_c}"

        # Check independence between t1 and t2
        t1_c0 = noise_dict["cd_t1"][0].ravel()
        t2_c0 = noise_dict["cd_t2"][0].ravel()
        corr_t = np.corrcoef(t1_c0, t2_c0)[0, 1]
        assert abs(corr_t) < 0.05, f"Bitemporal noise correlation too high: {corr_t}"

        # Check CRN monotonic scaling: perturbation at sigma=0.05 is exactly 2.5x perturbation at sigma=0.02
        opt_clean = np.full((10, 120, 120), 0.50, dtype=np.float32)
        sar_clean = np.full((2, 120, 120), 0.50, dtype=np.float32)

        opt_s1, sar_s1 = apply_noise_land_cover(opt_clean, sar_clean, sigma=0.02, noise_dict=noise_dict)
        opt_s2, sar_s2 = apply_noise_land_cover(opt_clean, sar_clean, sigma=0.05, noise_dict=noise_dict)

        pert1 = opt_s1 - opt_clean
        pert2 = opt_s2 - opt_clean
        # Before clipping (values are centered at 0.5 with small sigma), ratio must be exactly 2.5
        np.testing.assert_allclose(pert2, pert1 * 2.5, atol=1e-5)

    def test_all_six_degradation_paths_tensor_contract_compatibility(self):
        """
        Verify that all six Track 4.3 degradation paths produce tensors strictly
        matching the Stage B specialist tensor contracts:
        1. LC x Cloud: opt (10, 120, 120), sar (2, 120, 120)
        2. LC x Haze: opt (10, 120, 120), sar (2, 120, 120)
        3. LC x Noise: opt (10, 120, 120), sar (2, 120, 120)
        4. CD x Cloud: t1 (3, 256, 256), t2 (3, 256, 256)
        5. CD x Haze: t1 (3, 256, 256), t2 (3, 256, 256)
        6. CD x Noise: t1 (3, 256, 256), t2 (3, 256, 256)
        """
        rng = np.random.RandomState(42)
        opt = rng.uniform(0.1, 0.6, size=(10, 120, 120)).astype(np.float32)
        sar = rng.uniform(0.1, 0.8, size=(2, 120, 120)).astype(np.float32)
        t1 = rng.uniform(0.1, 0.6, size=(3, 256, 256)).astype(np.float32)
        t2 = rng.uniform(0.1, 0.6, size=(3, 256, 256)).astype(np.float32)

        mask_lc = generate_nested_cloud_mask(120, 120, severity_fraction=0.15, seed=42)
        mask_cd = generate_nested_cloud_mask(256, 256, severity_fraction=0.15, seed=42)
        noise = generate_crn_noise_tensors(patch_id=42)

        # Path 1: LC x Cloud
        p1_opt, p1_sar = apply_cloud_occlusion_land_cover(opt, sar, mask_lc)
        assert p1_opt.shape == (10, 120, 120) and p1_sar.shape == (2, 120, 120)
        assert p1_opt.dtype == np.float32 and p1_sar.dtype == np.float32

        # Path 2: LC x Haze
        p2_opt, p2_sar = apply_haze_land_cover(opt, sar, transmittance=0.70)
        assert p2_opt.shape == (10, 120, 120) and p2_sar.shape == (2, 120, 120)
        assert p2_opt.dtype == np.float32 and p2_sar.dtype == np.float32

        # Path 3: LC x Noise
        p3_opt, p3_sar = apply_noise_land_cover(opt, sar, sigma=0.05, noise_dict=noise)
        assert p3_opt.shape == (10, 120, 120) and p3_sar.shape == (2, 120, 120)
        assert p3_opt.dtype == np.float32 and p3_sar.dtype == np.float32

        # Path 4: CD x Cloud
        p4_t1, p4_t2 = apply_cloud_occlusion_change_detection(t1, t2, mask_cd)
        assert p4_t1.shape == (3, 256, 256) and p4_t2.shape == (3, 256, 256)
        assert p4_t1.dtype == np.float32 and p4_t2.dtype == np.float32

        # Path 5: CD x Haze
        p5_t1, p5_t2 = apply_haze_change_detection(t1, t2, transmittance=0.70)
        assert p5_t1.shape == (3, 256, 256) and p5_t2.shape == (3, 256, 256)
        assert p5_t1.dtype == np.float32 and p5_t2.dtype == np.float32

        # Path 6: CD x Noise
        p6_t1, p6_t2 = apply_noise_change_detection(t1, t2, sigma=0.05, noise_dict=noise)
        assert p6_t1.shape == (3, 256, 256) and p6_t2.shape == (3, 256, 256)
        assert p6_t1.dtype == np.float32 and p6_t2.dtype == np.float32
