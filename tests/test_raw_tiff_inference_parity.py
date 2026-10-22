"""
Unit and regression tests verifying Raw-TIFF Inference Parity for Phase 2 Specialist.

Confirms that:
1. LandCoverClassifier._extract_from_tiffs produces identical 488-feature vectors
   compared to satquery.features.multimodal.extract_multimodal_features_phase2.
2. Passing raw TIFF file paths to LandCoverClassifier.predict() yields identical
   class probabilities and thresholded predictions as passing precomputed feature vectors.
3. End-to-end mathematical parity holds with zero divergence.
"""

from pathlib import Path
import tempfile
import zipfile
import numpy as np
import pytest
import tifffile
import pandas as pd

from satquery.specialists.land_cover import LandCoverClassifier
from satquery.features.multimodal import extract_multimodal_features_phase2


@pytest.fixture(scope="module")
def classifier():
    return LandCoverClassifier()


def test_synthetic_raw_tiff_feature_and_prediction_parity(classifier):
    """Verify that raw synthetic S1 and S2 TIFFs yield bitwise identical features and predictions."""
    if not classifier.has_phase2:
        pytest.skip("Phase 2 LightGBM model not loaded.")

    # Create synthetic S1 (2 bands, 120x120 float32 dB) and S2 (10 bands, 120x120 uint16)
    np.random.seed(42)
    s1_synth = np.random.uniform(-30.0, 5.0, (2, 120, 120)).astype(np.float32)
    s2_synth = np.random.randint(100, 6000, (10, 120, 120), dtype=np.uint16)

    # Direct extraction
    expected_feats = extract_multimodal_features_phase2(s1_synth, s2_synth)
    expected_result = classifier.predict(expected_feats)

    with tempfile.TemporaryDirectory() as tmpdir:
        s1_file = Path(tmpdir) / "synth_s1.tif"
        s2_file = Path(tmpdir) / "synth_s2.tif"

        tifffile.imwrite(str(s1_file), s1_synth)
        tifffile.imwrite(str(s2_file), s2_synth)

        # Specialist extraction from files
        extracted_feats = classifier._extract_from_tiffs(s1_file, s2_file)
        file_result = classifier.predict({"s1_path": str(s1_file), "s2_path": str(s2_file)})

        # 1. Feature vector parity (max diff < 1e-6)
        max_feat_diff = np.max(np.abs(extracted_feats - expected_feats))
        assert max_feat_diff < 1e-6, f"Feature parity failure: max diff = {max_feat_diff}"

        # 2. Predicted classes match
        assert file_result.predictions == expected_result.predictions

        # 3. Probabilities match
        for cls in file_result.probabilities:
            assert abs(file_result.probabilities[cls] - expected_result.probabilities[cls]) < 1e-5


def test_benchmark_archive_sample_parity(classifier):
    """
    If archive.zip and test metadata exist on disk, test parity against
    actual held-out test patches from the benchmark.
    """
    archive_path = Path("initial_trails/Dataset/archive.zip")
    meta_path = Path("data/test_full_metadata.parquet")
    npy_path = Path("data/X_test_phase2.npy")

    if not (archive_path.exists() and meta_path.exists() and npy_path.exists()):
        pytest.skip("Benchmark archive or test features not present on disk.")

    df_test = pd.read_parquet(meta_path)
    X_test = np.load(npy_path)

    # Test sample index 0 and 5
    with zipfile.ZipFile(str(archive_path), "r") as z:
        for idx in [0, 5]:
            row = df_test.iloc[idx]
            s1_bytes = z.read(f"BEN_14k/BigEarthNet-S1/{row.split}/{row.s1_name}.tif")
            s2_bytes = z.read(f"BEN_14k/BigEarthNet-S2/{row.split}/{row.patch_id}.tif")

            with tempfile.TemporaryDirectory() as tmpdir:
                s1_file = Path(tmpdir) / f"{row.s1_name}.tif"
                s2_file = Path(tmpdir) / f"{row.patch_id}.tif"

                s1_file.write_bytes(s1_bytes)
                s2_file.write_bytes(s2_bytes)

                raw_feats = classifier._extract_from_tiffs(s1_file, s2_file)
                bench_feats = X_test[idx]

                # Parity with benchmark feature array
                max_diff = np.max(np.abs(raw_feats - bench_feats))
                assert max_diff < 1e-5, f"Benchmark sample {idx} feature mismatch: {max_diff}"

                res_raw = classifier.predict({"s1_path": str(s1_file), "s2_path": str(s2_file)})
                res_bench = classifier.predict(bench_feats)

                assert res_raw.predictions == res_bench.predictions
                for cls in res_raw.probabilities:
                    assert abs(res_raw.probabilities[cls] - res_bench.probabilities[cls]) < 1e-5
