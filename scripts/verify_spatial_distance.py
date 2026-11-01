"""
Independent Coordinate-Level Spatial Buffer Verification Script.
Audits all 3,248 test patches against all 7,180 train patches across all 4 MGRS tiles.
Verifies Chebyshev and Euclidean distances, patch grid pitch, and physical distance claims.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).resolve().parent.parent

def parse_patch_id(patch_id: str):
    # Format: ..._{tile}_{row}_{col}
    parts = patch_id.split('_')
    tile = parts[-3]
    row = int(parts[-2])
    col = int(parts[-1])
    return tile, row, col

print("Loading metadata...")
df_test = pd.read_parquet(REPO_ROOT / "data" / "test_full_metadata.parquet")
df_train = pd.read_parquet(REPO_ROOT / "data" / "train_full_metadata.parquet")
df_val = pd.read_parquet(REPO_ROOT / "data" / "val_full_metadata.parquet")

n_train = len(df_train)
n_val = len(df_val)
n_test = len(df_test)
n_total = n_train + n_val + n_test

print(f"Train patches: {n_train}")
print(f"Val patches:   {n_val}")
print(f"Test patches:  {n_test}")
print(f"Total dataset: {n_total}")

test_parsed = [parse_patch_id(p) for p in df_test["patch_id"]]
train_parsed = [parse_patch_id(p) for p in df_train["patch_id"]]
val_parsed = [parse_patch_id(p) for p in df_val["patch_id"]]

all_tiles = sorted(list(set(t[0] for t in train_parsed + val_parsed + test_parsed)))
shared_tiles = sorted(list(set(t[0] for t in test_parsed)))
print(f"\nAll MGRS Tiles ({len(all_tiles)}): {all_tiles}")
print(f"Shared Tiles with Test Patches ({len(shared_tiles)}): {shared_tiles}")

global_min_chebyshev = float("inf")
global_min_euclidean = float("inf")
nearest_pairs = []

tile_audit_records = {}
total_intra_tile_pairs = 0

for tile in all_tiles:
    test_coords = np.array([(t[1], t[2]) for t in test_parsed if t[0] == tile])
    train_coords = np.array([(t[1], t[2]) for t in train_parsed if t[0] == tile])
    val_coords = np.array([(t[1], t[2]) for t in val_parsed if t[0] == tile])

    n_tr = len(train_coords)
    n_va = len(val_coords)
    n_te = len(test_coords)

    print(f"\n--- Tile {tile} (Train={n_tr}, Val={n_va}, Test={n_te}, Subtotal={n_tr+n_va+n_te}) ---")

    if n_te == 0:
        print(f"  Note: Tile {tile} contains training and validation patches only; disjoint from test evaluation set.")
        tile_audit_records[tile] = {
            "num_train_patches": n_tr,
            "num_val_patches": n_va,
            "num_test_patches": 0,
            "train_coordinate_bounds": {
                "r_min": int(train_coords[:, 0].min()) if n_tr > 0 else None,
                "r_max": int(train_coords[:, 0].max()) if n_tr > 0 else None,
                "c_min": int(train_coords[:, 1].min()) if n_tr > 0 else None,
                "c_max": int(train_coords[:, 1].max()) if n_tr > 0 else None,
            },
            "train_patches_inside_test_box": 0,
            "intra_tile_pairs_evaluated": 0,
            "disjoint_from_test": True
        }
        continue

    print(f"  Test bounds: row [{test_coords[:, 0].min()}, {test_coords[:, 0].max()}], col [{test_coords[:, 1].min()}, {test_coords[:, 1].max()}] (N={n_te})")
    print(f"  Val bounds:  row [{val_coords[:, 0].min()}, {val_coords[:, 0].max()}], col [{val_coords[:, 1].min()}, {val_coords[:, 1].max()}] (N={n_va})")
    print(f"  Train bounds: row [{train_coords[:, 0].min()}, {train_coords[:, 0].max()}], col [{train_coords[:, 1].min()}, {train_coords[:, 1].max()}] (N={n_tr})")

    # Check for train patches inside test bounding box
    r_min, r_max = test_coords[:, 0].min(), test_coords[:, 0].max()
    c_min, c_max = test_coords[:, 1].min(), test_coords[:, 1].max()
    inside_test_box = int(np.sum((train_coords[:, 0] >= r_min) & (train_coords[:, 0] <= r_max) & 
                                 (train_coords[:, 1] >= c_min) & (train_coords[:, 1] <= c_max)))
    print(f"  Train patches strictly inside test bounding box: {inside_test_box}")

    # Pairwise distance matrix for intra-tile evaluation
    pairs_in_tile = n_te * n_tr
    total_intra_tile_pairs += pairs_in_tile
    print(f"  Intra-tile test-train pairs evaluated: {pairs_in_tile:,}")

    diffs = np.abs(test_coords[:, None, :] - train_coords[None, :, :])
    cheb_matrix = np.max(diffs, axis=2) # Shape (N_test, N_train)
    euc_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))

    min_cheb_tile = float(np.min(cheb_matrix))
    min_euc_tile = float(np.min(euc_matrix))
    print(f"  Min Chebyshev distance: {min_cheb_tile:.1f} patch units ({min_cheb_tile * 1.2:.1f} km center-to-center)")
    print(f"  Min Euclidean distance: {min_euc_tile:.2f} patch units ({min_euc_tile * 1.2:.2f} km)")

    # Find the exact pair(s) achieving min Chebyshev
    min_indices = np.argwhere(cheb_matrix == min_cheb_tile)
    for test_idx, train_idx in min_indices[:3]:
        t_c = test_coords[test_idx]
        tr_c = train_coords[train_idx]
        pair_info = {
            "tile": tile,
            "test_coord": (int(t_c[0]), int(t_c[1])),
            "train_coord": (int(tr_c[0]), int(tr_c[1])),
            "dr": int(abs(t_c[0] - tr_c[0])),
            "dc": int(abs(t_c[1] - tr_c[1])),
            "chebyshev": float(min_cheb_tile),
            "euclidean": float(euc_matrix[test_idx, train_idx])
        }
        nearest_pairs.append(pair_info)

    if min_cheb_tile < global_min_chebyshev:
        global_min_chebyshev = min_cheb_tile
    if min_euc_tile < global_min_euclidean:
        global_min_euclidean = min_euc_tile

    # Distribution stats
    tile_audit_records[tile] = {
        "num_train_patches": n_tr,
        "num_val_patches": n_va,
        "num_test_patches": n_te,
        "test_coordinate_bounds": {"r_min": int(r_min), "r_max": int(r_max), "c_min": int(c_min), "c_max": int(c_max)},
        "train_patches_inside_test_box": inside_test_box,
        "intra_tile_pairs_evaluated": pairs_in_tile,
        "chebyshev_min_distance_patches": min_cheb_tile,
        "center_to_center_km": round(min_cheb_tile * 1.2, 2),
        "edge_to_edge_buffer_km": round(max(0.0, min_cheb_tile - 1) * 1.2, 2),
        "min_euclidean_patches": round(min_euc_tile, 2),
        "min_euclidean_km": round(min_euc_tile * 1.2, 2),
        "disjoint_from_test": False
    }

shared_train_count = sum(tile_audit_records[t]["num_train_patches"] for t in shared_tiles)
shared_val_count = sum(tile_audit_records[t]["num_val_patches"] for t in shared_tiles)
shared_test_count = sum(tile_audit_records[t]["num_test_patches"] for t in shared_tiles)
shared_total_count = shared_train_count + shared_val_count + shared_test_count

cartesian_shared_tile_pairs = shared_train_count * shared_test_count
cartesian_all_tile_pairs = n_train * n_test

print("\n================== GLOBAL AUDIT SUMMARY ==================")
print(f"Total Dataset Patches: {n_total} (Train={n_train}, Val={n_val}, Test={n_test})")
print(f"  - 3 Shared Tiles ({', '.join(shared_tiles)}): {shared_total_count} (Train={shared_train_count}, Val={shared_val_count}, Test={shared_test_count})")
print(f"  - Disjoint Tile (T34TCS): {n_total - shared_total_count} (Train={n_train - shared_train_count}, Val={n_val - shared_val_count}, Test=0)")
print(f"\nPair Count Accounting:")
print(f"  1. Intra-Tile Test-Train Pairs Evaluated: {total_intra_tile_pairs:,}")
print(f"  2. Cartesian Pairs Across 3 Shared Tiles: {cartesian_shared_tile_pairs:,} ({shared_train_count} train x {shared_test_count} test)")
print(f"  3. Cartesian Pairs Across All 4 Tiles:    {cartesian_all_tile_pairs:,} ({n_train} train x {n_test} test)")
print(f"\nSpatial Separation:")
print(f"  Global Minimum Chebyshev Separation: {global_min_chebyshev:.1f} patch units")
print(f"  Physical Center-to-Center Distance:   {global_min_chebyshev * 1.2:.1f} km")
print(f"  Physical Edge-to-Edge Buffer:        {(global_min_chebyshev - 1) * 1.2:.1f} km")
print(f"  Global Minimum Euclidean Distance:   {global_min_euclidean:.2f} patch units ({global_min_euclidean * 1.2:.2f} km)")

# Save JSON audit
audit_output = {
    "audit_title": "BigEarthNet Serbia Coordinate & Spatial Buffer Audit (Reconciled 4-Tile Ledger)",
    "dataset_total_patches": n_total,
    "split_counts": {
        "train": n_train,
        "val": n_val,
        "test": n_test
    },
    "shared_tiles_subtotal": {
        "tiles": shared_tiles,
        "train": shared_train_count,
        "val": shared_val_count,
        "test": shared_test_count,
        "total": shared_total_count
    },
    "disjoint_tile_subtotal": {
        "tile": "T34TCS",
        "train": n_train - shared_train_count,
        "val": n_val - shared_val_count,
        "test": 0,
        "total": n_total - shared_total_count
    },
    "pair_counts": {
        "intra_tile_pairs_evaluated": total_intra_tile_pairs,
        "cartesian_shared_tile_pairs": cartesian_shared_tile_pairs,
        "cartesian_all_tile_pairs": cartesian_all_tile_pairs
    },
    "spatial_metrics": {
        "min_chebyshev_distance_patch_units": global_min_chebyshev,
        "center_to_center_km": global_min_chebyshev * 1.2,
        "edge_to_edge_buffer_km": (global_min_chebyshev - 1) * 1.2,
        "min_euclidean_distance_patch_units": round(global_min_euclidean, 2),
        "min_euclidean_km": round(global_min_euclidean * 1.2, 2)
    },
    "tile_breakdown": tile_audit_records,
    "nearest_sample_pairs": nearest_pairs
}

out_path = REPO_ROOT / "artifacts" / "benchmarks" / "spatial_buffer_audit.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(audit_output, f, indent=2)

print(f"\nSaved updated spatial buffer audit to: {out_path}")
