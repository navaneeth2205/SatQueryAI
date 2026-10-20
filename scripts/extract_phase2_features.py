"""
Extraction script for Phase 2 hardened multimodal features (488 dimensions).

Processes paired Sentinel-1 SAR and Sentinel-2 optical imagery from
archive.zip for BigEarthNet-14k train (7,180) and val (3,255) splits.
Guarantees strict row-level alignment and preserves baseline 84 features.
"""

import sys
from pathlib import Path

# Force UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import concurrent.futures
import io
import time
import zipfile
import numpy as np
import pandas as pd
import tifffile

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from satquery.features.multimodal import extract_multimodal_features_phase2


def _process_chunk_worker(args):
    zip_path_str, chunk_items = args
    results = []
    with zipfile.ZipFile(zip_path_str, "r") as z:
        for split, s1_name, patch_id in chunk_items:
            s1_rel = f"BEN_14k/BigEarthNet-S1/{split}/{s1_name}.tif"
            s2_rel = f"BEN_14k/BigEarthNet-S2/{split}/{patch_id}.tif"

            s1_bytes = z.read(s1_rel)
            s2_bytes = z.read(s2_rel)

            s1 = tifffile.imread(io.BytesIO(s1_bytes))
            s2 = tifffile.imread(io.BytesIO(s2_bytes))

            feats = extract_multimodal_features_phase2(s1, s2)
            results.append(feats)

    return np.array(results, dtype=np.float32)


def extract_features_dataset(
    df: pd.DataFrame,
    archive_path: Path,
    max_workers: int = 4,
    chunk_size: int = 100,
    desc: str = "dataset"
) -> np.ndarray:
    """Extract features in parallel chunks across worker processes."""
    print(f"\n[Phase 2 Feature Extraction] Processing {len(df)} samples for {desc}...")
    items = [(r.split, r.s1_name, r.patch_id) for r in df.itertuples()]

    chunks = [
        (str(archive_path), items[i : i + chunk_size])
        for i in range(0, len(items), chunk_size)
    ]

    t0 = time.time()
    arrays = []
    processed_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chunk_result in executor.map(_process_chunk_worker, chunks):
            arrays.append(chunk_result)
            processed_count += len(chunk_result)
            if processed_count % 1000 == 0 or processed_count == len(df):
                elapsed = time.time() - t0
                rate = processed_count / (elapsed + 1e-6)
                print(f"  Processed {processed_count}/{len(df)} samples ({rate:.1f} samples/sec)...")

    full_array = np.vstack(arrays)
    total_time = time.time() - t0
    print(f"[Done] {desc}: shape {full_array.shape} extracted in {total_time:.1f}s ({len(df)/total_time:.1f} samples/sec)")
    return full_array


def main():
    parser = argparse.ArgumentParser(description="Extract Phase 2 multimodal features.")
    parser.add_argument("--archive", type=str, default="initial_trails/Dataset/archive.zip")
    parser.add_argument("--train-meta", type=str, default="data/train_full_metadata.parquet")
    parser.add_argument("--val-meta", type=str, default="data/val_full_metadata.parquet")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    archive_path = Path(args.archive)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # 1. Validation split
    val_df = pd.read_parquet(args.val_meta)
    val_out_path = out_dir / "X_val_phase2.npy"
    if val_out_path.exists():
        print(f"\n[Cache] Loading existing validation features from {val_out_path}...")
        X_val = np.load(val_out_path)
    else:
        X_val = extract_features_dataset(
            val_df, archive_path, max_workers=args.workers, chunk_size=args.chunk_size, desc="validation"
        )
        np.save(val_out_path, X_val)
        print(f"Saved {val_out_path} ({val_out_path.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Train split
    train_df = pd.read_parquet(args.train_meta)
    train_out_path = out_dir / "X_train_phase2.npy"
    if train_out_path.exists():
        print(f"\n[Cache] Loading existing training features from {train_out_path}...")
        X_train = np.load(train_out_path)
    else:
        X_train = extract_features_dataset(
            train_df, archive_path, max_workers=args.workers, chunk_size=args.chunk_size, desc="train"
        )
        np.save(train_out_path, X_train)
        print(f"Saved {train_out_path} ({train_out_path.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Validation checks
    print("\n--- Integrity & Alignment Checks ---")
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"X_val shape:   {X_val.shape}, dtype: {X_val.dtype}")

    assert np.isfinite(X_train).all(), "Non-finite values found in X_train!"
    assert np.isfinite(X_val).all(), "Non-finite values found in X_val!"
    print("✓ All feature values are finite (no NaNs, no Inf)")

    # Check baseline 84 alignment
    baseline_train = np.load("initial_trails/X_train_full.npy")
    baseline_val = np.load("initial_trails/X_val_full.npy")

    train_match = np.allclose(X_train[:, :84], baseline_train, atol=1e-4)
    val_match = np.allclose(X_val[:, :84], baseline_val, atol=1e-4)

    print(f"✓ Level-0 features in X_train align with X_train_full: {train_match}")
    print(f"✓ Level-0 features in X_val align with X_val_full:     {val_match}")

    assert train_match, "Level-0 train features do not match baseline!"
    assert val_match, "Level-0 val features do not match baseline!"

    print("\n[Phase 2 Feature Extraction Complete] Features ready for benchmark comparisons.")


if __name__ == "__main__":
    main()
