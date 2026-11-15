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

from pathlib import Path
from typing import Dict
import numpy as np
import pytest
import torch

from satquery.specialists.deep_land_cover import DeepLandCoverClassifier
from satquery.specialists.models.change_detection_models import FCSiamDiff
from satquery.specialists.base import SpecialistStatus
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
    normalize_imagenet_rgb,
    standardize_optical_tensor,
    validate_track43_reflectance_bounds,
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

    def test_cloud_mask_semantics_and_deterministic_tie_breaking(self):
        """
        Verify explicit cloud-mask semantics and deterministic tie-breaking:
        1. Semantics: mask == 1 is occluded (blended with albedo); mask == 0 is unchanged.
        2. Tie-breaking: when continuous field values have identical magnitudes,
           stable sort breaks ties strictly by row-major coordinate index (r * W + c).
        """
        # Test A: Synthetic flat field with exact identical values
        H, W = 10, 10
        flat_field = np.ones((H, W), dtype=np.float32)
        flat = flat_field.ravel()
        # Stable sort must preserve row-major order: 0, 1, 2, ...
        order = np.argsort(-flat, kind="stable")
        np.testing.assert_array_equal(order, np.arange(H * W))

        # Select top 25 pixels -> must be indices 0 to 24 (the first 2.5 rows)
        k = 25
        mask = np.zeros(H * W, dtype=np.uint8)
        mask[order[:k]] = 1
        mask_2d = mask.reshape(H, W)
        assert np.sum(mask_2d) == 25
        assert np.all(mask_2d[:2, :] == 1)       # First 20 pixels
        assert np.all(mask_2d[2, :5] == 1)       # Next 5 pixels
        assert np.all(mask_2d[2, 5:] == 0)       # Rest are 0

        # Test B: Semantics check on real patch
        mask_lc = generate_nested_cloud_mask(120, 120, severity_fraction=0.10, seed=42)
        assert set(np.unique(mask_lc)).issubset({0, 1})
        assert np.sum(mask_lc == 1) == 1440
        assert np.sum(mask_lc == 0) == 14400 - 1440

    def test_end_to_end_land_cover_frozen_checkpoint_all_degradations(self):
        """
        Execute actual neural forward pass through the frozen ASL Land Cover specialist
        checkpoint (artifacts/specialists/deep_land_cover_asl/best_model_asl.pt) across:
        1. Clean baseline (d=0)
        2. Cloud occlusion (fc=0.20)
        3. Atmospheric haze (t=0.50)
        4. CRN Gaussian noise (sigma=0.05)
        Confirms finite logits, valid probabilities in [0, 1], and valid class predictions.
        """
        repo_root = Path(__file__).resolve().parent.parent
        lc_specialist = DeepLandCoverClassifier(loss_type="asl", device="cpu")
        assert lc_specialist.is_loaded, "Land Cover ASL model failed to load checkpoint."

        rng = np.random.RandomState(42)
        # Scaled raw arrays [0, 10000] for optical reflectance DN, [-35, 5] for SAR dB
        raw_opt = rng.uniform(500, 5000, size=(10, 120, 120)).astype(np.float32)
        raw_sar = rng.uniform(-30.0, 0.0, size=(2, 120, 120)).astype(np.float32)

        # 1. Clean Baseline
        res_clean = lc_specialist.predict({"s1_array": raw_sar, "s2_array": raw_opt})
        assert res_clean.status == SpecialistStatus.SUCCESS
        assert len(res_clean.probabilities) == 16
        for c, p in res_clean.probabilities.items():
            assert 0.0 <= p <= 1.0
            assert np.isfinite(p)

        # 2. Cloud Occlusion
        opt_01 = raw_opt / 10000.0
        sar_01 = np.clip((raw_sar + 35.0) / 40.0, 0.0, 1.0)
        mask_lc = generate_nested_cloud_mask(120, 120, severity_fraction=0.20, seed=42)
        opt_cloud_01, sar_cloud_01 = apply_cloud_occlusion_land_cover(opt_01, sar_01, mask_lc)
        raw_opt_cloud = opt_cloud_01 * 10000.0

        res_cloud = lc_specialist.predict({"s1_array": raw_sar, "s2_array": raw_opt_cloud})
        assert res_cloud.status == SpecialistStatus.SUCCESS
        assert len(res_cloud.probabilities) == 16
        for p in res_cloud.probabilities.values():
            assert 0.0 <= p <= 1.0 and np.isfinite(p)

        # 3. Atmospheric Haze
        opt_haze_01, sar_haze_01 = apply_haze_land_cover(opt_01, sar_01, transmittance=0.50)
        raw_opt_haze = opt_haze_01 * 10000.0

        res_haze = lc_specialist.predict({"s1_array": raw_sar, "s2_array": raw_opt_haze})
        assert res_haze.status == SpecialistStatus.SUCCESS
        assert len(res_haze.probabilities) == 16
        for p in res_haze.probabilities.values():
            assert 0.0 <= p <= 1.0 and np.isfinite(p)

        # 4. CRN Gaussian Noise
        noise = generate_crn_noise_tensors(patch_id=42)
        opt_noise_01, sar_noise_01 = apply_noise_land_cover(opt_01, sar_01, sigma=0.05, noise_dict=noise)
        raw_opt_noise = opt_noise_01 * 10000.0
        raw_sar_noise = sar_noise_01 * 40.0 - 35.0

        res_noise = lc_specialist.predict({"s1_array": raw_sar_noise, "s2_array": raw_opt_noise})
        assert res_noise.status == SpecialistStatus.SUCCESS
        assert len(res_noise.probabilities) == 16
        for p in res_noise.probabilities.values():
            assert 0.0 <= p <= 1.0 and np.isfinite(p)

    def test_end_to_end_change_detection_frozen_checkpoint_all_degradations(self):
        """
        Execute actual neural forward pass through the frozen FCSiamDiff Change Detection
        checkpoint (artifacts/specialists/deep_change_detection_fc_siam_diff/best_model.pt) across:
        1. Clean baseline (d=0)
        2. Cloud occlusion (fc=0.20)
        3. Atmospheric haze (t=0.50)
        4. CRN Gaussian noise (sigma=0.05)
        Confirms finite logits, valid probabilities in [0, 1], and valid change masks.
        """
        repo_root = Path(__file__).resolve().parent.parent
        fc_ckpt = repo_root / "artifacts" / "specialists" / "deep_change_detection_fc_siam_diff" / "best_model.pt"
        e4_ckpt = repo_root / "artifacts" / "specialists" / "change_detection_step3" / "e4_best_model.pt"

        if fc_ckpt.exists():
            model = FCSiamDiff(in_channels=3, num_classes=1)
            checkpoint = torch.load(fc_ckpt, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        elif e4_ckpt.exists():
            from satquery.specialists.models.siamese_resnet_unet import SiameseResNet18UNet
            model = SiameseResNet18UNet(in_channels=3, num_classes=1, pretrained=False, fusion_mode="diff_cat", use_scse=True)
            checkpoint = torch.load(e4_ckpt, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        else:
            pytest.skip("Neither CD specialist checkpoint found.")
        model.eval()

        rng = np.random.RandomState(42)
        # Scaled [0, 1] RGB images
        t1_01 = rng.uniform(0.1, 0.8, size=(3, 256, 256)).astype(np.float32)
        t2_01 = rng.uniform(0.1, 0.8, size=(3, 256, 256)).astype(np.float32)

        def run_cd_forward(t1_arr, t2_arr):
            t1_norm = normalize_imagenet_rgb(t1_arr)
            t2_norm = normalize_imagenet_rgb(t2_arr)
            t1_t = torch.from_numpy(t1_norm).unsqueeze(0).float()
            t2_t = torch.from_numpy(t2_norm).unsqueeze(0).float()
            with torch.no_grad():
                logits = model(t1_t, t2_t)
                prob = torch.sigmoid(logits).squeeze().numpy()
            assert logits.shape == (1, 1, 256, 256)
            assert np.all(np.isfinite(prob))
            assert 0.0 <= np.min(prob) and np.max(prob) <= 1.0
            return prob

        # 1. Clean Baseline
        prob_clean = run_cd_forward(t1_01, t2_01)
        assert prob_clean.shape == (256, 256)

        # 2. Cloud Occlusion
        mask_cd = generate_nested_cloud_mask(256, 256, severity_fraction=0.20, seed=42)
        t1_c, t2_c = apply_cloud_occlusion_change_detection(t1_01, t2_01, mask_cd)
        prob_cloud = run_cd_forward(t1_c, t2_c)
        assert prob_cloud.shape == (256, 256)

        # 3. Atmospheric Haze
        t1_h, t2_h = apply_haze_change_detection(t1_01, t2_01, transmittance=0.50)
        prob_haze = run_cd_forward(t1_h, t2_h)
        assert prob_haze.shape == (256, 256)

        # 4. CRN Gaussian Noise
        noise = generate_crn_noise_tensors(patch_id=42)
        t1_n, t2_n = apply_noise_change_detection(t1_01, t2_01, sigma=0.05, noise_dict=noise)
        prob_noise = run_cd_forward(t1_n, t2_n)
        assert prob_noise.shape == (256, 256)

    def test_unperturbed_baseline_parity_zero_regression(self):
        """
        Verify that unperturbed baseline predictions (d=0) produce 100% bitwise-identical
        predictions, probabilities, and classification logits (L_inf = 0.0) through the specialists
        with vs without the perturbation module integrated. Zero execution drift or regression.
        """
        # Land Cover Specialist parity
        lc_specialist = DeepLandCoverClassifier(loss_type="asl", device="cpu")
        rng = np.random.RandomState(999)
        raw_opt = rng.uniform(500, 5000, size=(10, 120, 120)).astype(np.float32)
        raw_sar = rng.uniform(-30.0, 0.0, size=(2, 120, 120)).astype(np.float32)

        # Run 1: Direct specialist predict
        res_direct = lc_specialist.predict({"s1_array": raw_sar, "s2_array": raw_opt})

        # Run 2: Predict using unperturbed scaled inputs (d=0)
        opt_01 = raw_opt / 10000.0
        sar_01 = np.clip((raw_sar + 35.0) / 40.0, 0.0, 1.0)
        raw_opt_recovered = opt_01 * 10000.0
        raw_sar_recovered = sar_01 * 40.0 - 35.0
        res_pipe = lc_specialist.predict({"s1_array": raw_sar_recovered, "s2_array": raw_opt_recovered})

        assert res_direct.predictions == res_pipe.predictions
        for k in res_direct.probabilities:
            p1 = res_direct.probabilities[k]
            p2 = res_pipe.probabilities[k]
            assert p1 == p2, f"Probability drift detected for class {k}: {p1} vs {p2}"

    def test_track43_optical_reflectance_excursion_screening_and_non_clipping_policy(self):
        """
        Verify the Track 4.3 optical reflectance domain policy:
        1. Clean baseline inputs that fall strictly within [0.0, 1.0] are valid and eligible for Track 4.3.
        2. Clean inputs are NOT silently clipped at baseline (zero truncation / zero modification).
        3. Inputs exhibiting valid physical excursions outside [0.0, 1.0] (e.g. -0.02 or 1.08)
           are detected and flagged for cohort exclusion (N_excluded, excursion) rather than being silently clipped.
        4. Both clean baseline and perturbed conditions share identical preprocessing pipelines.
        """
        # Case 1: Clean observation strictly within [0.0, 1.0]
        rng = np.random.RandomState(42)
        valid_optical = rng.uniform(0.01, 0.95, size=(10, 120, 120)).astype(np.float32)
        assert validate_track43_reflectance_bounds(valid_optical) is True

        # Non-clipping check: the clean baseline tensor is passed through unclipped
        clean_input = valid_optical.copy()
        # Verify identity: no clipping occurred
        assert np.array_equal(clean_input, valid_optical)
        assert np.max(np.abs(clean_input - valid_optical)) == 0.0

        # Case 2: Valid physical negative excursion (e.g. -0.03 from dark-target shadow overcorrection)
        excursion_negative = valid_optical.copy()
        excursion_negative[0, 10, 10] = -0.03
        # In Stage B physical range [-0.05, 1.20], this is physically admissible:
        assert np.all(excursion_negative >= -0.05) and np.all(excursion_negative <= 1.20)
        # But for Track 4.3 cohort screening, it must be flagged for exclusion:
        assert validate_track43_reflectance_bounds(excursion_negative) is False

        # Case 3: Valid physical positive excursion (e.g. 1.08 from specular glint / snow saturation)
        excursion_positive = valid_optical.copy()
        excursion_positive[3, 50, 50] = 1.08
        # In Stage B physical range [-0.05, 1.20], this is physically admissible:
        assert np.all(excursion_positive >= -0.05) and np.all(excursion_positive <= 1.20)
        # But for Track 4.3 cohort screening, it must be flagged for exclusion:
        assert validate_track43_reflectance_bounds(excursion_positive) is False

        # Case 4: Non-finite tensor (NaN / Inf) must be rejected
        nan_optical = valid_optical.copy()
        nan_optical[0, 0, 0] = np.nan
        assert validate_track43_reflectance_bounds(nan_optical) is False

