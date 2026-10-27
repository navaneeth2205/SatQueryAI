"""
Tri-Party Scientific Consistency Audit Script for SatQuery AI.
Verifies complete synchronization across:
1. Actual Implementation Code
2. Benchmark & Test Artifacts
3. Frozen Reproducibility Manifest
"""

import sys
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd


def audit_reproducibility_manifest():
    print("--- 1. AUDITING REPRODUCIBILITY MANIFEST & CRYPTOGRAPHIC HASHES ---")
    manifest_path = REPO_ROOT / "artifacts" / "specialists" / "land_cover_phase2" / "reproducibility_manifest.json"
    assert manifest_path.exists(), "Manifest file missing!"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Feature schema check (compact JSON representation of 488 feature names)
    feature_names = manifest["feature_schema"]["feature_names_in_order"]
    assert len(feature_names) == 488, f"Expected 488 features, got {len(feature_names)}"
    feature_bytes = json.dumps(feature_names).encode("utf-8")
    computed_schema_hash = hashlib.sha256(feature_bytes).hexdigest()
    assert computed_schema_hash == manifest["feature_schema_sha256"], (
        f"Schema hash mismatch: computed {computed_schema_hash} vs manifest {manifest['feature_schema_sha256']}"
    )
    print("  [PASS] 488 feature names and schema SHA-256 match perfectly.")

    # 2. Source and benchmark script hashes check
    script_hashes = manifest["source_and_benchmark_script_hashes_sha256"]
    for rel_path, expected_hash in script_hashes.items():
        full_path = REPO_ROOT / rel_path
        assert full_path.exists(), f"File {rel_path} does not exist!"
        content = full_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        assert actual_hash == expected_hash, (
            f"Hash mismatch for {rel_path}:\n  Expected: {expected_hash}\n  Actual:   {actual_hash}"
        )
        print(f"  [PASS] {rel_path} SHA-256 matches ({actual_hash[:12]}...)")

    # 3. Target labels check (compact JSON representation of classes.json)
    classes_path = REPO_ROOT / "artifacts" / "specialists" / "land_cover_phase2" / "classes.json"
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    assert len(classes) == 16, f"Expected 16 classes, got {len(classes)}"
    labels_bytes = json.dumps(classes).encode("utf-8")
    computed_label_hash = hashlib.sha256(labels_bytes).hexdigest()
    assert computed_label_hash == manifest["target_labels_sha256"], (
        f"Target labels hash mismatch: computed {computed_label_hash} vs manifest {manifest['target_labels_sha256']}"
    )
    print("  [PASS] Target label dictionary (16 classes) SHA-256 matches.")


def audit_spatial_buffer_and_geometry():
    print("\n--- 2. AUDITING SPATIAL BUFFER & 23.3M COORDINATE PAIRS ---")
    audit_file = REPO_ROOT / "artifacts" / "benchmarks" / "spatial_buffer_audit.json"
    with open(audit_file, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    summary = audit_data["overall_dataset_summary"]
    assert summary["min_distance_patches"] == 10.0
    assert summary["center_to_center_min_km"] == 12.0
    assert summary["edge_to_edge_min_buffer_km"] == 10.8
    assert summary["patches_with_distance_under_10_units"] == 0
    print("  [PASS] artifacts/benchmarks/spatial_buffer_audit.json reports global minimum 10.0 patch units.")

    # Check live coordinates
    df_test = pd.read_parquet(REPO_ROOT / "data" / "test_full_metadata.parquet")
    df_train = pd.read_parquet(REPO_ROOT / "data" / "train_full_metadata.parquet")

    def parse_row_col(pid):
        parts = pid.split("_")
        return parts[-3], int(parts[-2]), int(parts[-1])

    test_c = [parse_row_col(p) for p in df_test["patch_id"]]
    train_c = [parse_row_col(p) for p in df_train["patch_id"]]

    for tile in ["T34TCR", "T34TEQ", "T34TFN"]:
        t_pts = np.array([(r, c) for t, r, c in test_c if t == tile])
        tr_pts = np.array([(r, c) for t, r, c in train_c if t == tile])
        diffs = np.abs(t_pts[:, None, :] - tr_pts[None, :, :])
        cheb = np.max(diffs, axis=2)
        min_cheb = np.min(cheb)
        assert min_cheb >= 10.0, f"Found Chebyshev distance {min_cheb} < 10.0 in tile {tile}"
        print(f"  [PASS] Tile {tile}: Min Chebyshev distance = {min_cheb:.1f} patch units across {len(t_pts)} test and {len(tr_pts)} train patches.")


def audit_normalization_contracts():
    print("\n--- 3. AUDITING NORMALIZATION CONTRACTS & EDGE CASES ---")
    from satquery.specialists.change_detection import TemporalDifferenceBaseline
    baseline = TemporalDifferenceBaseline()

    # Case A: uint8 scaled by 255
    t1_u8 = np.full((10, 10, 3), 200, dtype=np.uint8)
    t2_u8 = np.full((10, 10, 3), 100, dtype=np.uint8)
    t1_n, t2_n = baseline._normalize_pair(t1_u8.astype(np.float32), t2_u8.astype(np.float32), source_dtype="uint8")
    assert np.isclose(t1_n.max(), 200 / 255.0) and np.isclose(t2_n.max(), 100 / 255.0)
    print("  [PASS] Case A: uint8 scaled strictly by 255.0.")

    # Case B: explicit pre_normalized floats preserved
    t1_fl = np.full((10, 10, 3), 0.65, dtype=np.float32)
    t2_fl = np.full((10, 10, 3), 0.35, dtype=np.float32)
    t1_n, t2_n = baseline._normalize_pair(t1_fl, t2_fl, product_type="pre_normalized")
    assert np.isclose(t1_n.max(), 0.65) and np.isclose(t2_n.max(), 0.35)
    print("  [PASS] Case B: Explicit product_type='pre_normalized' preserves values.")

    # Case C: raw float > 1.0 without product_type rejected
    try:
        baseline._normalize_pair(t1_fl * 50.0, t2_fl * 50.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Ambiguous floating-point temporal observation" in str(e)
        print("  [PASS] Case C: Ambiguous raw float > 1.0 strictly rejected.")

    # Case D: s2_l2a raw BOA scaled by 10000
    t1_s2 = np.full((10, 10, 3), 4000.0, dtype=np.float32)
    t2_s2 = np.full((10, 10, 3), 2000.0, dtype=np.float32)
    t1_n, t2_n = baseline._normalize_pair(t1_s2, t2_s2, product_type="s2_l2a")
    assert np.isclose(t1_n.max(), 0.4) and np.isclose(t2_n.max(), 0.2)
    print("  [PASS] Case D: Sentinel-2 L2A BOA DNs in [0, 10000] scaled by 10000.0.")

    # Case E: s2_l2a with float <= 1.0 rejected to prevent double downscaling
    try:
        baseline._normalize_pair(t1_fl, t2_fl, product_type="s2_l2a", source_dtype="float32")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "pre_normalized" in str(e)
        print("  [PASS] Case E: Float in [0, 1] under s2_l2a rejected to protect against double-downscaling.")


if __name__ == "__main__":
    audit_reproducibility_manifest()
    audit_spatial_buffer_and_geometry()
    audit_normalization_contracts()
    print("\n=======================================================")
    print("TRI-PARTY CONSISTENCY AUDIT PASSED WITH 100% AGREEMENT!")
    print("=======================================================")
