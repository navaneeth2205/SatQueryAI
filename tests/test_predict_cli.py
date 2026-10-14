"""
Integration tests for predict_query.py CLI script.
"""

import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent


def test_predict_cli_execution():
    # 1. First ensure model is trained
    train_script = repo_root / "scripts" / "train_router.py"
    train_res = subprocess.run(
        [sys.executable, str(train_script)],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert train_res.returncode == 0, f"Training failed: {train_res.stderr}"

    # 2. Test predict CLI text output
    predict_script = repo_root / "scripts" / "predict_query.py"
    query = "What land-cover types are visible in this image?"
    pred_res = subprocess.run(
        [sys.executable, str(predict_script), query],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert pred_res.returncode == 0, f"Prediction failed: {pred_res.stderr}"
    assert "LAND_COVER_CLASSIFICATION" in pred_res.stdout
    assert "Confidence Score" in pred_res.stdout

    # 3. Test predict CLI JSON output
    pred_json = subprocess.run(
        [sys.executable, str(predict_script), "What changed between 2024 and 2026?", "--json"],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    assert pred_json.returncode == 0
    data = json.loads(pred_json.stdout)
    assert data["task"] == "CHANGE_DETECTION"
    assert data["required_input"] == "temporal_pair"
    assert "confidence" in data
