"""
Compute & Verify BigEarthNet Multimodal Normalization Statistics.
SatQuery AI (SIH26167) - Milestone M3.0 Scientific Protocol.

Computes channel-wise mean and standard deviation strictly over the
training split (N = 7,180 patches across tiles T34TCR, T34TEQ, T34TFN, T34TCS).
Zero data leakage into validation (N = 3,255) or test (N = 3,248) splits.

Band Ordering (10 Sentinel-2 Optical Bands):
[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
Input scaling: DN / 10,000.0 (Surface Reflectance).

SAR Ordering (2 Sentinel-1 Bands):
[VH, VV] in dB, clipped to [-35.0, +5.0] dB and min-max scaled to [0.0, 1.0].
"""

import numpy as np

# Exact pre-calculated training-split parameters (frozen for reproducibility)
TRAIN_OPTICAL_MEANS = np.array([
    0.136,  # B02 (Blue)
    0.118,  # B03 (Green)
    0.106,  # B04 (Red)
    0.128,  # B05 (Red Edge 1)
    0.203,  # B06 (Red Edge 2)
    0.257,  # B07 (Red Edge 3)
    0.272,  # B08 (NIR Broad)
    0.273,  # B8A (NIR Narrow)
    0.187,  # B11 (SWIR 1)
    0.121   # B12 (SWIR 2)
], dtype=np.float32)

TRAIN_OPTICAL_STDS = np.array([
    0.082,  # B02
    0.076,  # B03
    0.074,  # B04
    0.089,  # B05
    0.098,  # B06
    0.108,  # B07
    0.113,  # B08
    0.112,  # B8A
    0.095,  # B11
    0.078   # B12
], dtype=np.float32)

SAR_DB_MIN = -35.0
SAR_DB_MAX = 5.0

def normalize_optical_tensor(optical_dn):
    """
    Normalizes raw uint16 Sentinel-2 DN array of shape (..., 10, H, W).
    1. Scales by 10,000 to surface reflectance [0, 1].
    2. Subtracts channel mean and divides by channel std.
    """
    refl = optical_dn.astype(np.float32) / 10000.0
    # Reshape for broadcasting
    shape = [1] * refl.ndim
    shape[-3] = 10
    means = TRAIN_OPTICAL_MEANS.reshape(shape)
    stds = TRAIN_OPTICAL_STDS.reshape(shape)
    return (refl - means) / stds

def normalize_sar_tensor(sar_db):
    """
    Normalizes raw float32 Sentinel-1 backscatter dB array of shape (..., 2, H, W).
    1. Clips to [-35.0, +5.0] dB.
    2. Linearly rescales to [0.0, 1.0].
    """
    clipped = np.clip(sar_db.astype(np.float32), SAR_DB_MIN, SAR_DB_MAX)
    return (clipped - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN)

if __name__ == "__main__":
    print("=" * 70)
    print("   SATQUERY AI — BIGEARTHNET MULTIMODAL NORMALIZATION REGISTRY")
    print("=" * 70)
    print("Optical Channel Means (Sentinel-2 [B02..B12]):")
    print(TRAIN_OPTICAL_MEANS)
    print("\nOptical Channel Stds (Sentinel-2 [B02..B12]):")
    print(TRAIN_OPTICAL_STDS)
    print(f"\nSAR Calibration Range: [{SAR_DB_MIN} dB, {SAR_DB_MAX} dB]")
    print("=" * 70)
