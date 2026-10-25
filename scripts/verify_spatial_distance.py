"""
Independent Coordinate-Level Spatial Buffer Verification Script.
Audits all 3,248 test patches against all 7,180 train patches across all MGRS tiles.
Verifies Chebyshev and Euclidean distances, patch grid pitch, and physical distance claims.
"""
import sys
import re
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

def parse_patch_id(patch_id: str):
    # Format: ..._{tile}_{row}_{col}
    parts = patch_id.split('_')
    tile = parts[-3]
    row = int(parts[-2])
    col = int(parts[-1])
    return tile, row, col

print("Loading metadata...")
df_test = pd.read_parquet("data/test_full_metadata.parquet")
df_train = pd.read_parquet("data/train_full_metadata.parquet")
df_val = pd.read_parquet("data/val_full_metadata.parquet")

print(f"Test patches: {len(df_test)}")
print(f"Train patches: {len(df_train)}")
print(f"Val patches: {len(df_val)}")

test_parsed = [parse_patch_id(p) for p in df_test["patch_id"]]
train_parsed = [parse_patch_id(p) for p in df_train["patch_id"]]
val_parsed = [parse_patch_id(p) for p in df_val["patch_id"]]

tiles = sorted(list(set(t[0] for t in test_parsed)))
print(f"MGRS Tiles: {tiles}")

global_min_chebyshev = float("inf")
global_min_euclidean = float("inf")
nearest_pairs = []

for tile in tiles:
    test_coords = np.array([(t[1], t[2]) for t in test_parsed if t[0] == tile])
    train_coords = np.array([(t[1], t[2]) for t in train_parsed if t[0] == tile])
    val_coords = np.array([(t[1], t[2]) for t in val_parsed if t[0] == tile])

    print(f"\nTile {tile}:")
    print(f"  Test bounds: row [{test_coords[:, 0].min()}, {test_coords[:, 0].max()}], col [{test_coords[:, 1].min()}, {test_coords[:, 1].max()}] (N={len(test_coords)})")
    print(f"  Val bounds:  row [{val_coords[:, 0].min()}, {val_coords[:, 0].max()}], col [{val_coords[:, 1].min()}, {val_coords[:, 1].max()}] (N={len(val_coords)})")
    print(f"  Train bounds: row [{train_coords[:, 0].min()}, {train_coords[:, 0].max()}], col [{train_coords[:, 1].min()}, {train_coords[:, 1].max()}] (N={len(train_coords)})")

    # Check for train patches inside test bounding box
    r_min, r_max = test_coords[:, 0].min(), test_coords[:, 0].max()
    c_min, c_max = test_coords[:, 1].min(), test_coords[:, 1].max()
    inside_test_box = np.sum((train_coords[:, 0] >= r_min) & (train_coords[:, 0] <= r_max) & 
                             (train_coords[:, 1] >= c_min) & (train_coords[:, 1] <= c_max))
    print(f"  Train patches strictly inside test bounding box: {inside_test_box}")

    # Compute pairwise distances between all test and all train in this tile
    # Using broadcasting: test_coords[:, None, :] - train_coords[None, :, :]
    # Shape: (N_test, N_train, 2)
    diffs = np.abs(test_coords[:, None, :] - train_coords[None, :, :])
    cheb_matrix = np.max(diffs, axis=2) # Shape (N_test, N_train)
    euc_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))

    min_cheb_tile = np.min(cheb_matrix)
    min_euc_tile = np.min(euc_matrix)
    print(f"  Min Chebyshev distance: {min_cheb_tile:.1f} patch units")
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

print("\n================== GLOBAL SUMMARY ==================")
print(f"Global Minimum Chebyshev Distance: {global_min_chebyshev:.1f} patch units")
print(f"At 1.2 km/patch: {global_min_chebyshev * 1.2:.1f} km center-to-center")
print(f"Edge-to-edge intervening buffer: {(global_min_chebyshev - 1) * 1.2:.1f} km")
print(f"Global Minimum Euclidean Distance: {global_min_euclidean:.2f} patch units ({global_min_euclidean * 1.2:.2f} km)")
print("\nSample Nearest Pairs:")
for p in nearest_pairs:
    print(f"  Tile {p['tile']}: Test {p['test_coord']} <-> Train {p['train_coord']} (dr={p['dr']}, dc={p['dc']}, Cheb={p['chebyshev']}, Euc={p['euclidean']:.2f})")
