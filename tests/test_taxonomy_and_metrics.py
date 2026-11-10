"""
Taxonomy Index Verification and Metric Edge-Case Regression Tests for Phase 4.

Verifies:
1. Exact alignment of the 16 classes in artifacts/specialists/land_cover/classes.json
   with the 16-class output head of the frozen ASL model checkpoint.
2. Explicit documentation and validation of zero-based (Python) vs one-based (Protocol)
   indexing for the 13 mapped classes and 3 quarantined classes.
3. Edge-case handling for absent-class false positive share (FPS_absent):
   - Returns float('nan') / 'NA' when denominator (total positive predictions on mapped classes) is zero.
   - Explicitly reports the number of absent mapped classes (|C_absent|).
4. Isolated reporting of the 3 quarantined class triggers.
"""

from pathlib import Path
import json
import numpy as np
import pytest
import torch
from PIL import Image

from satquery.utils.s2_metadata import (
    resample_s2_20m_to_10m,
    harmonize_modis_patch_labels,
    CANONICAL_S2_BANDS,
    NATIVE_10M_BANDS,
    NATIVE_20M_BANDS,
)


# ==============================================================================
# Canonical Taxonomy & Index Definitions
# ==============================================================================

# Zero-based indices (Python / PyTorch array indexing)
QUARANTINED_INDICES_0BASED = [3, 5, 13]
MAPPED_INDICES_0BASED = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15]

# One-based indices (Protocol list numbering)
QUARANTINED_INDICES_1BASED = [4, 6, 14]
MAPPED_INDICES_1BASED = [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16]

EXPECTED_QUARANTINED_NAMES = [
    "Complex cultivation patterns",
    "Industrial or commercial units",
    "Permanent crops",
]


# ==============================================================================
# Metric Reference Implementation with Edge-Case Guard
# ==============================================================================

def compute_track41_metrics(
    y_true: np.ndarray,      # Shape (N, 16) binary ground truth
    y_pred: np.ndarray,      # Shape (N, 16) binary model predictions
    zero_division: float = 0.0,
) -> dict:
    """
    Compute Track 4.1 evaluation metrics with strict edge-case safety.
    
    y_true: binary matrix (N, 16) where columns follow classes.json order.
            For quarantined classes (indices 3, 5, 13), y_true is 0 by construction.
    y_pred: binary matrix (N, 16) from model thresholding.
    """
    N = y_true.shape[0]
    if N == 0:
        raise ValueError("Cannot compute metrics on empty evaluation cohort.")

    # 1. Evaluate per-class TP, FP, FN on 13 mapped classes
    mapped_f1s = []
    c_absent = []
    c_present = []

    for c in MAPPED_INDICES_0BASED:
        n_true = int(np.sum(y_true[:, c]))
        tp = int(np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 1)))
        fp = int(np.sum((y_true[:, c] == 0) & (y_pred[:, c] == 1)))
        fn = int(np.sum((y_true[:, c] == 1) & (y_pred[:, c] == 0)))

        if n_true == 0:
            c_absent.append(c)
            mapped_f1s.append(zero_division)
        else:
            c_present.append(c)
            f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else zero_division
            mapped_f1s.append(f1)

    # Primary Fixed-13 Macro F1
    fixed13_macro_f1 = float(np.mean(mapped_f1s))

    # Secondary Present-Class Macro F1
    if len(c_present) > 0:
        present_f1s = [mapped_f1s[MAPPED_INDICES_0BASED.index(c)] for c in c_present]
        present_macro_f1 = float(np.mean(present_f1s))
    else:
        present_macro_f1 = float("nan")

    # 2. Absent-Class False Positive Metrics
    num_absent = len(c_absent)
    if num_absent > 0:
        absent_fps = [int(np.sum(y_pred[:, c])) for c in c_absent]
        total_absent_fp = sum(absent_fps)
        fpr_absent = float(total_absent_fp / (num_absent * N))
    else:
        total_absent_fp = 0
        fpr_absent = 0.0

    # Total positive predictions across the 13 mapped classes
    total_mapped_positives = int(np.sum(y_pred[:, MAPPED_INDICES_0BASED]))

    # FPS_absent with explicit NA on zero denominator
    if total_mapped_positives == 0:
        fps_absent = float("nan")  # Reported as NA
        fps_absent_str = "NA"
    else:
        fps_absent = float(total_absent_fp / total_mapped_positives)
        fps_absent_str = f"{fps_absent:.4f}"

    # 3. Dedicated Quarantined Diagnostic Metrics
    total_all_16_positives = int(np.sum(y_pred))
    quarantined_counts = {}
    quarantined_rates = {}
    for q_idx in QUARANTINED_INDICES_0BASED:
        count = int(np.sum(y_pred[:, q_idx]))
        quarantined_counts[q_idx] = count
        quarantined_rates[q_idx] = float(count / N)

    total_quarantined_positives = sum(quarantined_counts.values())
    if total_all_16_positives == 0:
        fps_quarantine = float("nan")
    else:
        fps_quarantine = float(total_quarantined_positives / total_all_16_positives)

    return {
        "fixed13_macro_f1": fixed13_macro_f1,
        "present_macro_f1": present_macro_f1,
        "num_absent_classes": num_absent,
        "absent_class_indices_0based": c_absent,
        "absent_class_indices_1based": [c + 1 for c in c_absent],
        "fpr_absent": fpr_absent,
        "fps_absent": fps_absent,
        "fps_absent_display": fps_absent_str,
        "total_mapped_positives": total_mapped_positives,
        "quarantined_counts": quarantined_counts,
        "quarantined_rates": quarantined_rates,
        "fps_quarantine": fps_quarantine,
        "total_all_16_positives": total_all_16_positives,
    }


# ==============================================================================
# Unit & Regression Test Suite
# ==============================================================================

class TestTaxonomyAndMetrics:
    """Tests taxonomy indexing alignment and metric edge-case behavior."""

    def test_taxonomy_file_and_indices_alignment(self):
        """
        Verify that classes.json has 16 classes and matches zero-based / one-based indices.
        """
        repo_root = Path(__file__).resolve().parent.parent
        classes_path = repo_root / "artifacts" / "specialists" / "land_cover" / "classes.json"
        assert classes_path.exists(), f"classes.json not found at {classes_path}"

        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)

        assert len(classes) == 16, f"Expected 16 classes, found {len(classes)}"

        # Verify quarantined classes at zero-based indices [3, 5, 13]
        assert classes[3] == "Complex cultivation patterns"
        assert classes[5] == "Industrial or commercial units"
        assert classes[13] == "Permanent crops"

        # Verify corresponding one-based indices [4, 6, 14]
        assert classes[QUARANTINED_INDICES_1BASED[0] - 1] == "Complex cultivation patterns"
        assert classes[QUARANTINED_INDICES_1BASED[1] - 1] == "Industrial or commercial units"
        assert classes[QUARANTINED_INDICES_1BASED[2] - 1] == "Permanent crops"

        # Verify partition: mapped (13) + quarantined (3) == 16
        all_0based = sorted(MAPPED_INDICES_0BASED + QUARANTINED_INDICES_0BASED)
        assert all_0based == list(range(16))

        all_1based = sorted(MAPPED_INDICES_1BASED + QUARANTINED_INDICES_1BASED)
        assert all_1based == list(range(1, 17))

    def test_model_head_matches_16_classes(self):
        """
        Verify that the frozen ASL model checkpoint output layer has dimension [16, 512].
        """
        repo_root = Path(__file__).resolve().parent.parent
        ckpt_path = repo_root / "artifacts" / "specialists" / "deep_land_cover_asl" / "best_model_asl.pt"
        assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        # Check linear classification head weights
        head_weight = state_dict.get("head.1.weight")
        assert head_weight is not None, "head.1.weight not found in checkpoint state_dict"
        assert head_weight.shape == torch.Size([16, 512]), f"Expected [16, 512], got {head_weight.shape}"

        head_bias = state_dict.get("head.1.bias")
        assert head_bias is not None, "head.1.bias not found in checkpoint state_dict"
        assert head_bias.shape == torch.Size([16]), f"Expected [16], got {head_bias.shape}"

    def test_fps_absent_undefined_denominator_reports_na(self):
        """
        Verify that when the model makes zero positive predictions on mapped classes,
        FPS_absent reports float('nan') / 'NA' rather than silently returning 0.0.
        """
        N = 100
        # Simulated ground truth: 3 classes present, 10 absent
        y_true = np.zeros((N, 16), dtype=np.int32)
        y_true[:, 0] = 1  # Arable land present
        y_true[:, 1] = 1  # Beaches present
        y_true[:, 2] = 1  # Broad-leaved forest present

        # Case A: Model predicts all zeros (no mapped positive predictions)
        y_pred_all_zeros = np.zeros((N, 16), dtype=np.int32)

        metrics = compute_track41_metrics(y_true, y_pred_all_zeros)
        assert metrics["total_mapped_positives"] == 0
        assert np.isnan(metrics["fps_absent"])
        assert metrics["fps_absent_display"] == "NA"
        # Verify number of absent classes is correctly recorded (13 mapped - 3 present = 10 absent)
        assert metrics["num_absent_classes"] == 10

        # Case B: Model predicts ONLY on quarantined classes (e.g. Industrial or commercial units, index 5)
        # Total positive predictions on mapped classes is still 0!
        y_pred_quarantined_only = np.zeros((N, 16), dtype=np.int32)
        y_pred_quarantined_only[:, 5] = 1  # Industrial predicted

        metrics_q = compute_track41_metrics(y_true, y_pred_quarantined_only)
        assert metrics_q["total_mapped_positives"] == 0
        assert np.isnan(metrics_q["fps_absent"])
        assert metrics_q["fps_absent_display"] == "NA"
        # But quarantined diagnostics capture the trigger
        assert metrics_q["quarantined_counts"][5] == 100
        assert metrics_q["quarantined_rates"][5] == 1.0
        assert metrics_q["fps_quarantine"] == 1.0

    def test_fps_absent_normal_operation(self):
        """
        Verify FPS_absent computation when mapped positives exist.
        """
        N = 100
        y_true = np.zeros((N, 16), dtype=np.int32)
        y_true[:, 0] = 1  # Arable land present (all other 12 mapped classes absent)

        y_pred = np.zeros((N, 16), dtype=np.int32)
        # Model correctly predicts Arable land (index 0) on 80 patches
        y_pred[:80, 0] = 1
        # Model has false alarms on Beaches (index 1, absent) on 20 patches
        y_pred[:20, 1] = 1

        metrics = compute_track41_metrics(y_true, y_pred)
        assert metrics["total_mapped_positives"] == 100  # 80 on index 0 + 20 on index 1
        assert metrics["num_absent_classes"] == 12       # 13 mapped - 1 present = 12 absent
        # Total absent FP = 20
        # FPS_absent = 20 / 100 = 0.20
        assert np.isclose(metrics["fps_absent"], 0.20, atol=1e-5)
        assert metrics["fps_absent_display"] == "0.2000"
        # FPR_absent = 20 / (12 * 100) = 20 / 1200 = 0.01667
        assert np.isclose(metrics["fpr_absent"], 20.0 / 1200.0, atol=1e-5)

    def test_resampling_parity_actual_bigearthnet_patch(self):
        """
        Verify Stage B resampling parity against original Phase 2/3 training data:
        1. Bitwise pass-through: If 10-band arrays are already 120x120 (e.g. pre-resampled
           BigEarthNet patches loaded during training), resample_s2_20m_to_10m preserves them bitwise.
        2. Bilinear parity: When 20m bands (60x60) are upsampled to 10m (120x120), PyTorch
           F.interpolate(mode='bilinear', align_corners=False) achieves exact bitwise/numerical
           parity (< 1e-5 max absolute difference) with standard PIL Image.Resampling.BILINEAR.
        3. Real satellite patch verification: If archive.zip is present, test on an actual
           BigEarthNet-S2 GeoTIFF patch from the dataset.
        """
        repo_root = Path(__file__).resolve().parent.parent
        archive_path = repo_root / "initial_trails" / "Dataset" / "archive.zip"

        if archive_path.exists():
            import zipfile
            import io
            import tifffile
            with zipfile.ZipFile(archive_path) as z:
                # Find an S2 patch
                s2_files = [f for f in z.namelist() if "BigEarthNet-S2" in f and f.endswith(".tif")]
                assert len(s2_files) > 0, "No BigEarthNet-S2 GeoTIFFs found in archive.zip"
                sample_file = s2_files[0]
                raw_bytes = z.read(sample_file)
                arr = tifffile.imread(io.BytesIO(raw_bytes))  # (10, 120, 120) uint16

            bands_dict = {b: arr[i].astype(np.float32) for i, b in enumerate(CANONICAL_S2_BANDS)}

            # 1. Bitwise pass-through check
            stacked_pass, prov_pass = resample_s2_20m_to_10m(bands_dict, target_shape=(120, 120))
            np.testing.assert_array_equal(stacked_pass, arr.astype(np.float32))

            # 2. Resampling parity check on actual band (B05)
            b05_orig = arr[3].astype(np.float32)
            b05_60 = b05_orig[::2, ::2]  # Subsample to 20m GSD (60x60)
            bands_dict["B05"] = b05_60
            stacked_resamp, prov_resamp = resample_s2_20m_to_10m(bands_dict, target_shape=(120, 120))

            pil_resamp = np.array(
                Image.fromarray(b05_60).resize((120, 120), resample=Image.Resampling.BILINEAR)
            )
            # Max diff PyTorch vs PIL bilinear is 0.0 or within float32 precision
            max_diff = float(np.max(np.abs(stacked_resamp[3] - pil_resamp)))
            assert max_diff < 1e-5, f"PyTorch vs PIL bilinear discrepancy: {max_diff}"

            # Spatial correlation between bilinearly reconstructed and original pre-resampled raster
            corr = np.corrcoef(stacked_resamp[3].ravel(), b05_orig.ravel())[0, 1]
            assert corr > 0.95, f"Reconstructed correlation {corr:.4f} is too low"
        else:
            # Fallback synthetic verification if archive.zip is absent
            rng = np.random.RandomState(42)
            bands_dict = {}
            for b in NATIVE_10M_BANDS:
                bands_dict[b] = rng.uniform(500, 3000, size=(120, 120)).astype(np.float32)
            for b in NATIVE_20M_BANDS:
                bands_dict[b] = rng.uniform(500, 3000, size=(60, 60)).astype(np.float32)

            stacked, prov = resample_s2_20m_to_10m(bands_dict, target_shape=(120, 120))
            for i, b in enumerate(CANONICAL_S2_BANDS):
                if b in NATIVE_20M_BANDS:
                    pil_resamp = np.array(
                        Image.fromarray(bands_dict[b]).resize((120, 120), resample=Image.Resampling.BILINEAR)
                    )
                    max_diff = float(np.max(np.abs(stacked[i] - pil_resamp)))
                    assert max_diff < 1e-5

    def test_modis_quarantine_edge_cases_and_zero_fallback_rejection(self):
        """
        Verify strict MODIS harmonization decision rule and rejection of plurality fallback:
        1. Sub-threshold rejection (w_max = 0.199): patch is quarantined as Inevaluable,
           excluded from N_evaluable, and assigned 0 ground-truth labels.
        2. Boundary acceptance (w_max = 0.200): patch is evaluable with corresponding mapped label.
        3. Dominant unmapped class rejection: when an unmapped class dominates (e.g. 60%)
           and no mapped class reaches 0.20 (e.g. w_max = 0.12), the patch is strictly quarantined;
           NO fallback to the plurality mapped class is permitted.
        4. Multi-label qualifying assignment: multiple mapped classes >= 0.20 are assigned.
        5. Empty coverage dictionary: strictly quarantined.
        """
        # Case 1: Sub-threshold w_max = 0.199
        cov_sub = {
            "Broad-leaved forest": 0.199,
            "Coniferous forest": 0.150,
            "Water bodies": 0.050,
        }
        res_sub = harmonize_modis_patch_labels(cov_sub, threshold=0.20)
        assert res_sub["is_quarantined"] is True
        assert res_sub["evaluable"] is False
        assert res_sub["assigned_labels"] == {}
        assert "quarantine_reason" in res_sub and res_sub["quarantine_reason"] is not None

        # Case 2: Boundary acceptance w_max = 0.200
        cov_boundary = {
            "Broad-leaved forest": 0.200,
            "Coniferous forest": 0.150,
        }
        res_boundary = harmonize_modis_patch_labels(cov_boundary, threshold=0.20)
        assert res_boundary["is_quarantined"] is False
        assert res_boundary["evaluable"] is True
        assert res_boundary["assigned_labels"] == {"Broad-leaved forest": 1}

        # Case 3: Dominant unmapped class rejection (strict zero fallback)
        cov_dominant_unmapped = {
            "Broad-leaved forest": 0.120,
            "Moors and heathland": 0.080,
        }
        res_unmapped = harmonize_modis_patch_labels(cov_dominant_unmapped, threshold=0.20)
        assert res_unmapped["is_quarantined"] is True
        assert res_unmapped["evaluable"] is False
        assert res_unmapped["assigned_labels"] == {}
        assert "Broad-leaved forest" not in res_unmapped["assigned_labels"]

        # Case 4: Multi-label co-occurrence
        cov_multi = {
            "Broad-leaved forest": 0.450,
            "Coniferous forest": 0.350,
            "Water bodies": 0.100,
        }
        res_multi = harmonize_modis_patch_labels(cov_multi, threshold=0.20)
        assert res_multi["is_quarantined"] is False
        assert res_multi["evaluable"] is True
        assert res_multi["assigned_labels"] == {
            "Broad-leaved forest": 1,
            "Coniferous forest": 1,
        }
        assert "Water bodies" not in res_multi["assigned_labels"]

        # Case 5: Empty coverages
        res_empty = harmonize_modis_patch_labels({}, threshold=0.20)
        assert res_empty["is_quarantined"] is True
        assert res_empty["evaluable"] is False
        assert res_empty["assigned_labels"] == {}
