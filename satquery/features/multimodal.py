"""
Multimodal Feature Engineering Pipeline for SatQuery AI.

Extracts physically-grounded spatial, spectral, and polarimetric features
from paired Sentinel-1 SAR and Sentinel-2 Multispectral observations:
  1. Level-0 Global band statistics (84 features)
  2. Level-1 Spatial-pyramid quadrant statistics (336 features)
  3. Verified Multispectral indices (NDVI, NDWI, MNDWI, NDBI, NDRE) (35 features)
  4. Polarimetric Radar indices (Cross-pol diff, Linear Cross-Ratio, Radar Veg Index) (21 features)
  5. Spatial Texture gradient statistics (NIR, Red, VV) (12 features)
Total: 488 features.
"""

from typing import List, Tuple
import numpy as np


S1_BAND_NAMES = ["VH", "VV"]
S2_BAND_NAMES = ["B02_Blue", "B03_Green", "B04_Red", "B05_RE1", "B06_RE2", "B07_RE3", "B08_NIR", "B8A_NarrowNIR", "B11_SWIR1", "B12_SWIR2"]
STAT_NAMES_7 = ["mean", "std", "min", "max", "p25", "p50", "p75"]


def band_statistics_7(arr: np.ndarray) -> List[float]:
    """
    Compute 7 statistical summary descriptors: mean, std, min, max, p25, p50, p75.
    Optimized for high-throughput batch extraction with vectorized percentiles.
    """
    clean = arr[np.isfinite(arr)]
    if len(clean) == 0:
        return [0.0] * 7

    p25, p50, p75 = np.percentile(clean, [25, 50, 75])
    return [
        float(np.mean(clean)),
        float(np.std(clean)),
        float(np.min(clean)),
        float(np.max(clean)),
        float(p25),
        float(p50),
        float(p75),
    ]


def extract_level0_features(s1_arr: np.ndarray, s2_arr: np.ndarray) -> np.ndarray:
    """
    Extract baseline 84 global statistical features:
    14 Sentinel-1 (2 bands x 7 stats) + 70 Sentinel-2 (10 bands x 7 stats).
    Preserves exact bit-level alignment with baseline X_train_full and X_val_full.
    """
    feats = []
    for band in s1_arr.astype(np.float32):
        feats.extend(band_statistics_7(band))
    for band in s2_arr.astype(np.float32):
        feats.extend(band_statistics_7(band))
    return np.array(feats, dtype=np.float32)


def extract_spatial_pyramid_features(s1_arr: np.ndarray, s2_arr: np.ndarray) -> np.ndarray:
    """
    Extract Level-1 Spatial Pyramid Pooling features across 2x2 quadrants.
    Subdivides the 120x120 patch into 4 quadrants (60x60 each):
      Q0: Top-Left (y: [0, 60), x: [0, 60))
      Q1: Top-Right (y: [0, 60), x: [60, 120))
      Q2: Bottom-Left (y: [60, 120), x: [0, 60))
      Q3: Bottom-Right (y: [60, 120), x: [60, 120))
    Produces 4 quadrants x (14 S1 + 70 S2) = 336 features.
    """
    h, w = s1_arr.shape[1], s1_arr.shape[2]
    mid_h, mid_w = h // 2, w // 2
    quadrants = [
        (slice(0, mid_h), slice(0, mid_w)),
        (slice(0, mid_h), slice(mid_w, w)),
        (slice(mid_h, h), slice(0, mid_w)),
        (slice(mid_h, h), slice(mid_w, w)),
    ]

    s1_f32 = s1_arr.astype(np.float32)
    s2_f32 = s2_arr.astype(np.float32)

    feats = []
    for qy, qx in quadrants:
        for band in s1_f32:
            feats.extend(band_statistics_7(band[qy, qx]))
        for band in s2_f32:
            feats.extend(band_statistics_7(band[qy, qx]))

    return np.array(feats, dtype=np.float32)


def extract_spectral_indices(s2_arr: np.ndarray) -> np.ndarray:
    """
    Compute 5 multispectral Earth observation indices from Sentinel-2:
      1. NDVI: (B08 - B04) / (B08 + B04 + 1e-6)  [Vegetation / Canopy vigor]
      2. NDWI: (B03 - B08) / (B03 + B08 + 1e-6)  [Open water surface delineation]
      3. MNDWI: (B03 - B11) / (B03 + B11 + 1e-6) [Modified NDWI for wetlands / soil contrast]
      4. NDBI: (B11 - B08) / (B11 + B08 + 1e-6)  [Built-up & bare substrate index]
      5. NDRE: (B08 - B05) / (B08 + B05 + 1e-6)  [Red-edge chlorophyll & canopy density]
    Extracts 7 statistics per index = 35 features.
    """
    s2 = s2_arr.astype(np.float32)
    b_blue  = s2[0]
    b_green = s2[1]
    b_red   = s2[2]
    b_re1   = s2[3]
    b_nir   = s2[6]
    b_swir1 = s2[8]

    eps = 1e-6
    ndvi  = (b_nir - b_red) / (b_nir + b_red + eps)
    ndwi  = (b_green - b_nir) / (b_green + b_nir + eps)
    mndwi = (b_green - b_swir1) / (b_green + b_swir1 + eps)
    ndbi  = (b_swir1 - b_nir) / (b_swir1 + b_nir + eps)
    ndre  = (b_nir - b_re1) / (b_nir + b_re1 + eps)

    feats = []
    for idx_map in [ndvi, ndwi, mndwi, ndbi, ndre]:
        feats.extend(band_statistics_7(idx_map))

    return np.array(feats, dtype=np.float32)


def extract_sar_indices(s1_arr: np.ndarray) -> np.ndarray:
    """
    Compute 3 polarimetric radar metrics from Sentinel-1:
      1. Polarization Difference: VH_dB - VV_dB  (Cross-pol attenuation)
      2. Linear Cross-Ratio: P_VH / (P_VV + 1e-6) where P = 10^(dB/10) (Volume vs surface scattering)
      3. Radar Vegetation Index (RVI): 4 * P_VH / (P_VV + P_VH + 1e-6) (Vegetation canopy roughness)
    Extracts 7 statistics per metric = 21 features.
    """
    s1 = s1_arr.astype(np.float32)
    vh_db = s1[0]
    vv_db = s1[1]

    diff_db = vh_db - vv_db

    eps = 1e-6
    p_vh = np.power(10.0, np.clip(vh_db, -60.0, 30.0) / 10.0)
    p_vv = np.power(10.0, np.clip(vv_db, -60.0, 30.0) / 10.0)
    cr_lin = p_vh / (p_vv + eps)
    rvi = (4.0 * p_vh) / (p_vv + p_vh + eps)

    feats = []
    for r_map in [diff_db, cr_lin, rvi]:
        feats.extend(band_statistics_7(r_map))

    return np.array(feats, dtype=np.float32)


def extract_texture_features(s1_arr: np.ndarray, s2_arr: np.ndarray) -> np.ndarray:
    """
    Compute spatial gradient magnitude statistics across key optical & SAR channels:
      - Sentinel-2 B08 (NIR): high contrast for crop boundaries and forest edges
      - Sentinel-2 B04 (Red): fine urban fabric and bare soil texture
      - Sentinel-1 Band 1 (VV): radar surface roughness and structural texture
    Extracts 4 statistics [mean, std, max, p50] per channel = 12 features.
    """
    s2 = s2_arr.astype(np.float32)
    s1 = s1_arr.astype(np.float32)

    channels = [
        s2[6],  # B08 NIR
        s2[2],  # B04 Red
        s1[1],  # S1 VV
    ]

    feats = []
    for ch in channels:
        gy, gx = np.gradient(ch)
        grad_mag = np.hypot(gx, gy)
        clean = grad_mag[np.isfinite(grad_mag)]
        if len(clean) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([
                float(np.mean(clean)),
                float(np.std(clean)),
                float(np.max(clean)),
                float(np.percentile(clean, 50)),
            ])

    return np.array(feats, dtype=np.float32)


def extract_multimodal_features_phase2(s1_arr: np.ndarray, s2_arr: np.ndarray) -> np.ndarray:
    """
    End-to-end extraction of the 488 Phase 2 hardened features:
      - Level-0 Global: 84
      - Level-1 Quadrants: 336
      - Spectral Indices: 35
      - Radar Indices: 21
      - Texture Gradients: 12
    Returns 1D float32 array of shape (488,).
    """
    l0 = extract_level0_features(s1_arr, s2_arr)
    l1 = extract_spatial_pyramid_features(s1_arr, s2_arr)
    spec = extract_spectral_indices(s2_arr)
    sar = extract_sar_indices(s1_arr)
    tex = extract_texture_features(s1_arr, s2_arr)

    combined = np.concatenate([l0, l1, spec, sar, tex]).astype(np.float32)
    assert len(combined) == 488, f"Expected 488 features, got {len(combined)}"
    return combined


def get_phase2_feature_names() -> List[str]:
    """Return human-readable names for all 488 features."""
    names = []

    # 1. Level-0 Global (84)
    for b in S1_BAND_NAMES:
        for s in STAT_NAMES_7:
            names.append(f"global_s1_{b}_{s}")
    for b in S2_BAND_NAMES:
        for s in STAT_NAMES_7:
            names.append(f"global_s2_{b}_{s}")

    # 2. Level-1 Quadrants (336)
    quad_labels = ["top_left", "top_right", "bottom_left", "bottom_right"]
    for q in quad_labels:
        for b in S1_BAND_NAMES:
            for s in STAT_NAMES_7:
                names.append(f"quad_{q}_s1_{b}_{s}")
        for b in S2_BAND_NAMES:
            for s in STAT_NAMES_7:
                names.append(f"quad_{q}_s2_{b}_{s}")

    # 3. Spectral Indices (35)
    index_names = ["ndvi", "ndwi", "mndwi", "ndbi", "ndre"]
    for idx in index_names:
        for s in STAT_NAMES_7:
            names.append(f"index_s2_{idx}_{s}")

    # 4. Radar Indices (21)
    sar_indices = ["vh_vv_diff_db", "cr_linear", "rvi"]
    for idx in sar_indices:
        for s in STAT_NAMES_7:
            names.append(f"index_s1_{idx}_{s}")

    # 5. Texture Gradients (12)
    tex_channels = ["s2_b08_nir", "s2_b04_red", "s1_vv"]
    tex_stats = ["mean", "std", "max", "p50"]
    for ch in tex_channels:
        for s in tex_stats:
            names.append(f"texture_grad_{ch}_{s}")

    assert len(names) == 488, f"Feature names count mismatch: {len(names)} != 488"
    return names
