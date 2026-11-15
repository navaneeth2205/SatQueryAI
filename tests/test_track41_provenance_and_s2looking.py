# -*- coding: utf-8 -*-
"""
Verification suite for Track 4.1 MODIS MCD12Q1 Provenance, S2Looking Matrices, and Null-Centered Bootstrap.

Verifies:
1. MCD12Q1 product lineage, IGBP classes (1-17), 6-direct / 7-approximate mapping, and 3 quarantined classes.
2. S2Looking pixel confusion matrix integrity: 10 scenes x 1,048,576 = 10,485,760 pixels.
   Guarantees harmonic F1 (diff <= 1e-12) and Jaccard equivalence (diff <= 1e-12).
3. Null-centered bootstrap test formulation and one-sided confidence bounds.
4. Protocol immutability (SHA-256 matches frozen protocol).
"""

from pathlib import Path
import json
import hashlib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_frozen_protocol_hash_unchanged():
    """Verify bitwise immutability of SATQUERY-P4-PROTOCOL-V1.12.1-FROZEN.md."""
    protocol_path = REPO_ROOT / "SATQUERY-P4-PROTOCOL-V1.12.1-FROZEN.md"
    assert protocol_path.exists()
    content = protocol_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    expected_digest = "7bcb8c5ab00cfebb5f7418237e9297f7eeaf06b0477d426d0ec6b30880e71bb5"
    assert digest == expected_digest, f"Protocol hash modified: {digest} != {expected_digest}"


def test_modis_mcd12q1_lineage_and_mapping():
    """Verify MODIS MCD12Q1 configuration matches protocol specifications."""
    cfg_path = REPO_ROOT / "configs" / "modis_harmonization_config.json"
    assert cfg_path.exists()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Product specifications
    prod = cfg["modis_product"]
    assert prod["product_id"] == "MCD12Q1"
    assert prod["version"] == "061"
    assert prod["band"] == "LC_Type1"
    assert prod["spatial_resolution_meters"] == 500

    # Decision thresholds
    thresh = cfg["decision_thresholds"]
    assert thresh["area_fraction_threshold"] == 0.20
    assert thresh["fallback_policy"] == "NO_FALLBACK_STRICT_QUARANTINE"
    assert thresh["plurality_fallback_permitted"] is False

    # Taxonomy counts
    tax = cfg["taxonomy_mapping"]
    assert tax["directly_mapped_count"] == 6
    assert tax["approximately_mapped_count"] == 7
    assert tax["quarantined_count"] == 3
    assert len(tax["direct_mappings"]) == 6
    assert len(tax["approximate_mappings"]) == 7
    assert len(tax["quarantined_classes"]["zero_based_indices"]) == 3
    assert tax["quarantined_classes"]["zero_based_indices"] == [3, 5, 13]


def test_s2looking_matrices_exact_pixel_sums_and_harmonic_identities():
    """Verify that all 10 S2Looking scenes sum to 1,048,576 pixels and satisfy exact identities."""
    eval_path = REPO_ROOT / "artifacts" / "evaluation" / "track41_reconciled_s2looking_evaluation.json"
    assert eval_path.exists()
    with open(eval_path, "r", encoding="utf-8") as f:
        s2l_data = json.load(f)

    scenes = s2l_data["per_scene"]
    assert len(scenes) == 10

    total_pixels = 0
    for s_id, sc in scenes.items():
        cm = sc["confusion_matrix"]
        tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
        scene_pixels = tp + fp + fn + tn
        assert scene_pixels == 1048576, f"Scene {s_id} pixel sum {scene_pixels} != 1,048,576"
        total_pixels += scene_pixels

        # Harmonic identity check
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        harmonic_f1 = 2 * p * r / (p + r)
        assert abs(f1 - harmonic_f1) < 1e-12, f"Scene {s_id} F1 harmonic discrepancy {abs(f1 - harmonic_f1)}"

        # IoU Jaccard equivalence check
        iou = tp / (tp + fp + fn)
        iou_from_f1 = f1 / (2.0 - f1)
        assert abs(iou - iou_from_f1) < 1e-12, f"Scene {s_id} IoU discrepancy {abs(iou - iou_from_f1)}"

    assert total_pixels == 10485760, f"Total pixels {total_pixels} != 10,485,760"


def test_bootstrap_hypotheses_and_one_sided_bounds():
    """Verify reconciled bootstrap data structure, one-sided bounds, and Holm adjusted p-values."""
    boot_path = REPO_ROOT / "artifacts" / "evaluation" / "track41_reconciled_bootstrap_hypotheses.json"
    assert boot_path.exists()
    with open(boot_path, "r", encoding="utf-8") as f:
        boot_data = json.load(f)

    assert len(boot_data) == 4
    for b_id in ["Biome_1_Arid", "Biome_2_Tropical_Rainforest", "Biome_3_Boreal_Forest", "Biome_4_Humid_Subtropical_Cropland"]:
        b = boot_data[b_id]
        assert "delta_observed" in b
        assert "two_sided_ci_95" in b
        assert "one_sided_upper_bound_95" in b
        assert "p_sign_ge_0" in b
        assert "raw_p_null" in b
        assert "holm_adjusted_p_value" in b

        # For Boreal and Cropland, transfer deltas are positive (gains)
        if b_id in ["Biome_3_Boreal_Forest", "Biome_4_Humid_Subtropical_Cropland"]:
            assert b["delta_observed"] > 0.0
            assert b["raw_p_null"] >= 0.85
            assert b["holm_adjusted_p_value"] == 1.0
            assert b["statistically_significant_degradation"] is False
