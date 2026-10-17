"""
Prepare and catalog the official OSCD (Onera Satellite Change Detection) Benchmark.
Step 1 of Milestone 3.

Enforces strict site/scene-level partitioning to prevent spatial data leakage:
- Train: 10 official training scenes
- Validation: 4 official training scenes (reserved for threshold tuning)
- Test: 10 official test scenes (held-out evaluation)
"""

import io
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "oscd"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_TRAIN_CITIES = [
    "berrida", "beirut", "bordeaux", "chongqing", "dubai",
    "hong_kong", "lasvegas", "milano", "rennes", "saclay_w",
    # 4 validation cities:
    "montpellier", "paris", "rio", "valencia"
]

OFFICIAL_TEST_CITIES = [
    "brasilia", "aguasclaras", "austin", "beihai", "berlin",
    "budapest", "campinas", "mumbai", "nantes", "pisa"
]


def download_parquet(split: str) -> pd.DataFrame:
    url = f"https://huggingface.co/datasets/blanchon/OSCD_RGB/resolve/main/data/{split}-00000-of-00001.parquet"
    print(f"Downloading OSCD {split} split from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read()
    print(f"Downloaded {len(content) / (1024*1024):.2f} MB.")
    return pd.read_parquet(io.BytesIO(content))


def extract_and_catalog():
    print("=" * 60)
    print("[Step 1] PREPARING OSCD CHANGE DETECTION BENCHMARK")
    print("=" * 60)

    # 1. Fetch official train split
    train_df = download_parquet("train")
    # 2. Fetch official test split
    test_df = download_parquet("test")

    records = []

    # Map train df rows to cities
    # In official OSCD, 14 cities in alphabetical order:
    # beirut, berrida, bordeaux, chongqing, dubai, hong_kong, lasvegas, milano, montpellier, paris, rennes, rio, saclay_w, valencia
    train_city_order = sorted([
        "beirut", "berrida", "bordeaux", "chongqing", "dubai",
        "hong_kong", "lasvegas", "milano", "montpellier", "paris",
        "rennes", "rio", "saclay_w", "valencia"
    ])

    val_city_set = {"montpellier", "paris", "rio", "valencia"}

    print("\nExtracting and saving training and validation scenes...")
    for idx, row in train_df.iterrows():
        city = train_city_order[idx]
        split_name = "val" if city in val_city_set else "train"
        scene_dir = DATA_DIR / split_name / city
        scene_dir.mkdir(parents=True, exist_ok=True)

        im1 = Image.open(io.BytesIO(row["image1"]["bytes"]))
        im2 = Image.open(io.BytesIO(row["image2"]["bytes"]))
        mask = Image.open(io.BytesIO(row["mask"]["bytes"]))

        im1_path = scene_dir / "t1.png"
        im2_path = scene_dir / "t2.png"
        mask_path = scene_dir / "cm.png"

        im1.save(im1_path)
        im2.save(im2_path)
        mask.save(mask_path)

        mask_np = (np.array(mask) > 0).astype(np.uint8)
        width, height = im1.size
        total_pixels = width * height
        change_pixels = int(mask_np.sum())
        change_pct = round((change_pixels / total_pixels) * 100.0, 4)

        meta = {
            "scene_name": city,
            "split": split_name,
            "width": width,
            "height": height,
            "channels": 3,
            "total_pixels": total_pixels,
            "valid_pixels": total_pixels,
            "change_pixels": change_pixels,
            "change_pct": change_pct,
            "t1_path": str(im1_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "t2_path": str(im2_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "mask_path": str(mask_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        }
        with open(scene_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        records.append(meta)
        print(f"  [{split_name.upper():5s}] {city:15s}: {width}x{height}, {change_pct:.2f}% change ({change_pixels}/{total_pixels} px)")

    # Map test df rows to cities
    test_city_order = sorted([
        "aguasclaras", "austin", "beihai", "berlin", "brasilia",
        "budapest", "campinas", "mumbai", "nantes", "pisa"
    ])

    print("\nExtracting and saving held-out test scenes...")
    for idx, row in test_df.iterrows():
        city = test_city_order[idx]
        split_name = "test"
        scene_dir = DATA_DIR / split_name / city
        scene_dir.mkdir(parents=True, exist_ok=True)

        im1 = Image.open(io.BytesIO(row["image1"]["bytes"]))
        im2 = Image.open(io.BytesIO(row["image2"]["bytes"]))
        mask = Image.open(io.BytesIO(row["mask"]["bytes"]))

        im1_path = scene_dir / "t1.png"
        im2_path = scene_dir / "t2.png"
        mask_path = scene_dir / "cm.png"

        im1.save(im1_path)
        im2.save(im2_path)
        mask.save(mask_path)

        mask_np = (np.array(mask) > 0).astype(np.uint8)
        width, height = im1.size
        total_pixels = width * height
        change_pixels = int(mask_np.sum())
        change_pct = round((change_pixels / total_pixels) * 100.0, 4)

        meta = {
            "scene_name": city,
            "split": split_name,
            "width": width,
            "height": height,
            "channels": 3,
            "total_pixels": total_pixels,
            "valid_pixels": total_pixels,
            "change_pixels": change_pixels,
            "change_pct": change_pct,
            "t1_path": str(im1_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "t2_path": str(im2_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "mask_path": str(mask_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        }
        with open(scene_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        records.append(meta)
        print(f"  [TEST ] {city:15s}: {width}x{height}, {change_pct:.2f}% change ({change_pixels}/{total_pixels} px)")

    catalog_df = pd.DataFrame(records)
    catalog_path = REPO_ROOT / "data" / "change_detection_catalog.parquet"
    catalog_df.to_parquet(catalog_path, index=False)

    manifest_path = DATA_DIR / "manifest.json"
    manifest = {
        "dataset_name": "OSCD (Onera Satellite Change Detection) RGB Subset",
        "provenance": "Daudt et al., 2018; hosted on HuggingFace as blanchon/OSCD_RGB",
        "total_scenes": len(records),
        "split_counts": {
            "train": int((catalog_df["split"] == "train").sum()),
            "val": int((catalog_df["split"] == "val").sum()),
            "test": int((catalog_df["split"] == "test").sum()),
        },
        "leakage_prevention": "Scene-level partitioning; zero geographic overlap between train, val, and test.",
        "catalog_file": str(catalog_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] OSCD Benchmark catalog saved to: {catalog_path}")
    print(f"  - Train scenes     : {manifest['split_counts']['train']}")
    print(f"  - Validation scenes: {manifest['split_counts']['val']}")
    print(f"  - Held-out test    : {manifest['split_counts']['test']}")
    print("=" * 60)


if __name__ == "__main__":
    extract_and_catalog()
