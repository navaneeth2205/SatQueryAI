"""
Formal Verification and Leakage Audit for OSCD Change Detection Splits.
Milestone M3.2 Pre-Flight Verification.

Enforces:
1. Strict city/scene-level separation across train (10), val (4), and test (10).
2. Zero spatial or geographic overlap across splits.
3. Integrity of image files (t1.png, t2.png, cm.png, metadata.json).
4. Label distribution and binary mask validation.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "oscd"

TRAIN_CITIES = {
    "beirut", "berrida", "bordeaux", "chongqing", "dubai",
    "hong_kong", "lasvegas", "milano", "rennes", "saclay_w"
}
VAL_CITIES = {"montpellier", "paris", "rio", "valencia"}
TEST_CITIES = {
    "aguasclaras", "austin", "beihai", "berlin", "brasilia",
    "budapest", "campinas", "mumbai", "nantes", "pisa"
}


def verify_splits():
    print("=" * 70)
    print("OSCD SPLIT INTEGRITY & LEAKAGE AUDIT")
    print("=" * 70)

    # 1. Disjointness check
    assert TRAIN_CITIES.isdisjoint(VAL_CITIES), "Train and Val cities overlap!"
    assert TRAIN_CITIES.isdisjoint(TEST_CITIES), "Train and Test cities overlap!"
    assert VAL_CITIES.isdisjoint(TEST_CITIES), "Val and Test cities overlap!"
    print("[PASS] City sets are 100% disjoint. Zero geographic overlap.")

    # 2. Directory structure verification
    splits = {
        "train": TRAIN_CITIES,
        "val": VAL_CITIES,
        "test": TEST_CITIES,
    }

    manifest = {
        "dataset": "OSCD (Onera Satellite Change Detection)",
        "protocol_version": "v3.0.0-freeze-rev21",
        "split_counts": {
            "train_scenes": len(TRAIN_CITIES),
            "val_scenes": len(VAL_CITIES),
            "test_scenes": len(TEST_CITIES),
            "total_scenes": len(TRAIN_CITIES) + len(VAL_CITIES) + len(TEST_CITIES)
        },
        "scenes": {}
    }

    for split_name, expected_cities in splits.items():
        split_dir = DATA_DIR / split_name
        assert split_dir.exists(), f"Missing split directory: {split_dir}"
        actual_cities = {p.name for p in split_dir.iterdir() if p.is_dir()}
        assert actual_cities == expected_cities, (
            f"Split {split_name} mismatch: expected {expected_cities}, found {actual_cities}"
        )

        for city in sorted(actual_cities):
            city_dir = split_dir / city
            t1_path = city_dir / "t1.png"
            t2_path = city_dir / "t2.png"
            cm_path = city_dir / "cm.png"
            meta_path = city_dir / "metadata.json"

            assert t1_path.exists(), f"Missing t1.png in {city_dir}"
            assert t2_path.exists(), f"Missing t2.png in {city_dir}"
            assert cm_path.exists(), f"Missing cm.png in {city_dir}"
            assert meta_path.exists(), f"Missing metadata.json in {city_dir}"

            t1 = Image.open(t1_path)
            t2 = Image.open(t2_path)
            cm = np.array(Image.open(cm_path))

            assert t1.size == t2.size, f"Resolution mismatch between t1 and t2 in {city}"
            assert t1.size == (cm.shape[1], cm.shape[0]), f"Shape mismatch between image and mask in {city}"

            unique_vals = np.unique(cm)
            assert set(unique_vals).issubset({0, 1}), f"Non-binary change mask in {city}: {unique_vals}"

            change_pixels = int(np.sum(cm == 1))
            total_pixels = int(cm.size)
            change_ratio = float(change_pixels / total_pixels)

            manifest["scenes"][city] = {
                "split": split_name,
                "width": t1.size[0],
                "height": t1.size[1],
                "total_pixels": total_pixels,
                "change_pixels": change_pixels,
                "change_ratio": round(change_ratio, 6),
            }

    out_manifest = REPO_ROOT / "artifacts" / "specialists" / "oscd_frozen_split_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[PASS] All 24 scenes validated. Frozen split manifest saved to: {out_manifest}")
    return manifest


if __name__ == "__main__":
    verify_splits()
