"""
Empirical Benchmark Tests for Change Detection Specialist.

Verifies the OSCD benchmark catalog, validation threshold artifacts,
and evaluation metrics on real held-out test scenes:
- Ensures threshold config and evaluation summary artifacts exist with documented provenance.
- Verifies that held-out test metrics (Precision, Recall, F1, IoU) are finite and in [0, 1].
- Verifies inference on actual held-out OSCD test scenes without fabricated claims.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from satquery.specialists.change_detection import TemporalDifferenceBaseline
from satquery.specialists.base import SpecialistStatus

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_catalog_integrity_and_splits():
    """Verify master catalog exists and enforces strict scene-level partitioning."""
    catalog_path = REPO_ROOT / "data" / "change_detection_catalog.parquet"
    assert catalog_path.exists(), f"Change detection catalog missing at {catalog_path}"

    df = pd.read_parquet(catalog_path)
    assert len(df) == 24, f"Expected 24 OSCD scenes, found {len(df)}"

    splits = df["split"].value_counts().to_dict()
    assert splits.get("train") == 10, f"Expected 10 train scenes, got {splits.get('train')}"
    assert splits.get("val") == 4, f"Expected 4 validation scenes, got {splits.get('val')}"
    assert splits.get("test") == 10, f"Expected 10 test scenes, got {splits.get('test')}"

    # Verify scene names are disjoint across splits (zero spatial leakage)
    train_scenes = set(df[df["split"] == "train"]["scene_name"])
    val_scenes = set(df[df["split"] == "val"]["scene_name"])
    test_scenes = set(df[df["split"] == "test"]["scene_name"])

    assert train_scenes.isdisjoint(val_scenes), "Train and validation scenes must not overlap"
    assert train_scenes.isdisjoint(test_scenes), "Train and test scenes must not overlap"
    assert val_scenes.isdisjoint(test_scenes), "Validation and test scenes must not overlap"


def test_artifacts_exist_and_documented():
    """Verify serialized threshold configuration and evaluation summaries."""
    artifacts_dir = REPO_ROOT / "artifacts" / "specialists" / "change_detection"
    thresh_file = artifacts_dir / "threshold_config.json"
    eval_file = artifacts_dir / "evaluation_summary.json"

    assert thresh_file.exists(), f"Threshold config missing at {thresh_file}"
    assert eval_file.exists(), f"Evaluation summary missing at {eval_file}"

    with open(thresh_file, "r", encoding="utf-8") as f:
        thresh_cfg = json.load(f)
    assert "decision_threshold" in thresh_cfg
    assert "validation_scenes" in thresh_cfg
    assert len(thresh_cfg["validation_scenes"]) == 4

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_summary = json.load(f)
    assert eval_summary["model_family"] == "temporal_difference_baseline"
    assert eval_summary["test_scenes_count"] == 10


def test_held_out_metrics_are_finite_and_bounded():
    """Verify reported aggregate test metrics are finite numbers within [0.0, 1.0]."""
    eval_file = REPO_ROOT / "artifacts" / "specialists" / "change_detection" / "evaluation_summary.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = summary["overall_test_metrics"]
    for key in ["precision", "recall", "f1", "iou"]:
        val = metrics.get(key)
        assert val is not None, f"Metric '{key}' missing from evaluation summary"
        assert isinstance(val, (int, float)), f"Metric '{key}' must be numeric"
        assert 0.0 <= val <= 1.0, f"Metric '{key}' must be in [0, 1], got {val}"
        assert np.isfinite(val), f"Metric '{key}' must be finite"


def test_real_scene_inference():
    """Execute baseline on a real held-out test scene from disk."""
    pisa_dir = REPO_ROOT / "data" / "oscd" / "test" / "pisa"
    if not pisa_dir.exists():
        pytest.skip("Pisa test scene not present on disk")

    t1_path = pisa_dir / "t1.png"
    t2_path = pisa_dir / "t2.png"

    baseline = TemporalDifferenceBaseline(
        artifacts_dir=REPO_ROOT / "artifacts" / "specialists" / "change_detection"
    )
    result = baseline.predict({"t1": str(t1_path), "t2": str(t2_path)})

    assert result.status == SpecialistStatus.SUCCESS
    assert result.execution_metadata["valid_pixels"] > 0
    assert result.execution_metadata["valid_pixel_pct"] >= 95.0
    assert len(result.execution_metadata["mean_channel_deltas"]) == 3
    assert result.execution_metadata["spatial_coordinate_system"] == "unprojected_pixel_coordinates_[ymin_xmin_ymax_xmax]"
