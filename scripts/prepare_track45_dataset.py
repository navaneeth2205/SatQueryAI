"""
Prepare and Validate Track 4.5 Adversarial Physical Metadata and Coordinate Perturbations Dataset.

Governed strictly by SatQuery AI Phase 4 Frozen Protocol v1.12.1 (Section: Track 4.5).
Preregistered specifications:
- N_source = 30 independent baseline observations (15 Land Cover, 15 Change Detection).
- N_control = 20 independent control scenes (10 Land Cover, 10 Change Detection).
- Total: 50 independent scenes with pairwise distance d_min >= 100 km and zero IoU.
- Exactly 4 distinct perturbations per perturbed baseline observation (30 x 4 = 120 payloads).
- 20 control payloads.
- Total evaluation payloads: 140 payloads across 8 physical contract dimensions:
  * Dim 1 (Modality & Sensor Structure): 20 payloads
  * Dim 2 (Temporal Revisit Window): 15 payloads
  * Dim 3 (Spectral Band Adequacy): 10 payloads
  * Dim 4 (Spatial Dimensions): 10 payloads
  * Dim 5 (Geospatial CRS Consistency): 10 payloads
  * Dim 6 (Affine Alignment & Coregistration): 20 payloads
  * Dim 7 (Ground Sample Distance / GSD): 10 payloads
  * Dim 8 (Radiometric & Cloud Guard): 25 payloads
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

FROZEN_PROTOCOL_HASH = "7bcb8c5ab00cfebb5f7418237e9297f7eeaf06b0477d426d0ec6b30880e71bb5"


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points on the Earth in kilometers."""
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def compute_bbox_iou(b1: List[float], b2: List[float]) -> float:
    """Compute 2D Intersection-over-Union (IoU) of two bounding boxes [min_lon, min_lat, max_lon, max_lat]."""
    x_left = max(b1[0], b2[0])
    y_bottom = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_top = min(b1[3], b2[3])

    if x_right <= x_left or y_top <= y_bottom:
        return 0.0

    intersection_area = (x_right - x_left) * (y_top - y_bottom)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0
    return float(intersection_area / union_area)


def get_utm_epsg(lon: float, lat: float) -> Tuple[int, str]:
    """Determine UTM zone number and EPSG code."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = f"EPSG:{32600 + zone}"
    else:
        epsg = f"EPSG:{32700 + zone}"
    return zone, epsg


# -----------------------------------------------------------------------------
# 50 Independent Source Scene Definitions (30 Perturbed Baselines + 20 Controls)
# -----------------------------------------------------------------------------

RAW_SCENES: List[Dict[str, Any]] = [
    # Sources 1-5 (Land Cover Baselines)
    {"scene_id": "SRC_01", "name": "Bordeaux", "country": "France", "continent": "Europe", "lat": 44.8378, "lon": -0.5792, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_02", "name": "Valencia", "country": "Spain", "continent": "Europe", "lat": 39.4699, "lon": -0.3763, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_03", "name": "Bologna", "country": "Italy", "continent": "Europe", "lat": 44.4949, "lon": 11.3426, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_04", "name": "Krakow", "country": "Poland", "continent": "Europe", "lat": 50.0647, "lon": 19.9450, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_05", "name": "Cluj-Napoca", "country": "Romania", "continent": "Europe", "lat": 46.7712, "lon": 23.6236, "primary_domain": "LAND_COVER_CLASSIFICATION"},

    # Sources 6-10 (Land Cover Baselines)
    {"scene_id": "SRC_06", "name": "Minneapolis", "country": "USA", "continent": "North America", "lat": 44.9778, "lon": -93.2650, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_07", "name": "Kansas City", "country": "USA", "continent": "North America", "lat": 39.0997, "lon": -94.5786, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_08", "name": "Memphis", "country": "USA", "continent": "North America", "lat": 35.1495, "lon": -90.0490, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_09", "name": "Calgary", "country": "Canada", "continent": "North America", "lat": 51.0447, "lon": -114.0719, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_10", "name": "Guadalajara", "country": "Mexico", "continent": "North America", "lat": 20.6597, "lon": -103.3496, "primary_domain": "LAND_COVER_CLASSIFICATION"},

    # Sources 11-15 (Land Cover Baselines)
    {"scene_id": "SRC_11", "name": "Campinas", "country": "Brazil", "continent": "South America", "lat": -22.9056, "lon": -47.0608, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_12", "name": "Cordoba", "country": "Argentina", "continent": "South America", "lat": -31.4201, "lon": -64.1888, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_13", "name": "Kumasi", "country": "Ghana", "continent": "Africa", "lat": 6.6885, "lon": -1.6244, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_14", "name": "Nagpur", "country": "India", "continent": "Asia", "lat": 21.1458, "lon": 79.0882, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "SRC_15", "name": "Adelaide", "country": "Australia", "continent": "Oceania", "lat": -34.9285, "lon": 138.6007, "primary_domain": "LAND_COVER_CLASSIFICATION"},

    # Sources 16-20 (Change Detection Baselines)
    {"scene_id": "SRC_16", "name": "Stuttgart", "country": "Germany", "continent": "Europe", "lat": 48.7758, "lon": 9.1829, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_17", "name": "Innsbruck", "country": "Austria", "continent": "Europe", "lat": 47.2692, "lon": 11.4041, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_18", "name": "Zaragoza", "country": "Spain", "continent": "Europe", "lat": 41.6488, "lon": -0.8891, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_19", "name": "Gothenburg", "country": "Sweden", "continent": "Europe", "lat": 57.7089, "lon": 11.9746, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_20", "name": "Thessaloniki", "country": "Greece", "continent": "Europe", "lat": 40.6401, "lon": 22.9444, "primary_domain": "CHANGE_DETECTION"},

    # Sources 21-25 (Change Detection Baselines)
    {"scene_id": "SRC_21", "name": "Salt Lake City", "country": "USA", "continent": "North America", "lat": 40.7608, "lon": -111.8910, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_22", "name": "Albuquerque", "country": "USA", "continent": "North America", "lat": 35.0844, "lon": -106.6504, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_23", "name": "Monterrey", "country": "Mexico", "continent": "North America", "lat": 25.6866, "lon": -100.3161, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_24", "name": "Medellin", "country": "Colombia", "continent": "South America", "lat": 6.2442, "lon": -75.5812, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_25", "name": "Curitiba", "country": "Brazil", "continent": "South America", "lat": -25.4284, "lon": -49.2733, "primary_domain": "CHANGE_DETECTION"},

    # Sources 26-30 (Change Detection Baselines)
    {"scene_id": "SRC_26", "name": "Rabat", "country": "Morocco", "continent": "Africa", "lat": 34.0209, "lon": -6.8416, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_27", "name": "Kisumu", "country": "Kenya", "continent": "Africa", "lat": -0.0917, "lon": 34.7680, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_28", "name": "Jaipur", "country": "India", "continent": "Asia", "lat": 26.9124, "lon": 75.7873, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_29", "name": "Kyoto", "country": "Japan", "continent": "Asia", "lat": 35.0116, "lon": 135.7681, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "SRC_30", "name": "Perth", "country": "Australia", "continent": "Oceania", "lat": -31.9505, "lon": 115.8605, "primary_domain": "CHANGE_DETECTION"},

    # Controls 1-10 (Land Cover Controls)
    {"scene_id": "CTRL_LC_01", "name": "Seville", "country": "Spain", "continent": "Europe", "lat": 37.3891, "lon": -5.9845, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "CTRL_LC_02", "name": "Turin", "country": "Italy", "continent": "Europe", "lat": 45.0703, "lon": 7.6869, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "CTRL_LC_03", "name": "Wroclaw", "country": "Poland", "continent": "Europe", "lat": 51.1079, "lon": 17.0385, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "CTRL_LC_04", "name": "Austin", "country": "USA", "continent": "North America", "lat": 30.2672, "lon": -97.7431, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "CTRL_LC_05", "name": "Edmonton", "country": "Canada", "continent": "North America", "lat": 53.5461, "lon": -113.4938, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "CTRL_LC_06", "name": "Puebla", "country": "Mexico", "continent": "North America", "lat": 19.0414, "lon": -98.2063, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "CTRL_LC_07", "name": "Porto Alegre", "country": "Brazil", "continent": "South America", "lat": -30.0346, "lon": -51.2177, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "CTRL_LC_08", "name": "Mombasa", "country": "Kenya", "continent": "Africa", "lat": -4.0435, "lon": 39.6682, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "CTRL_LC_09", "name": "Ahmedabad", "country": "India", "continent": "Asia", "lat": 23.0225, "lon": 72.5714, "primary_domain": "LAND_COVER_CLASSIFICATION"},
    {"scene_id": "CTRL_LC_10", "name": "Brisbane", "country": "Australia", "continent": "Oceania", "lat": -27.4698, "lon": 153.0251, "primary_domain": "LAND_COVER_CLASSIFICATION"},

    # Controls 11-20 (Change Detection Controls)
    {"scene_id": "CTRL_CD_01", "name": "Nantes", "country": "France", "continent": "Europe", "lat": 47.2184, "lon": -1.5536, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "CTRL_CD_02", "name": "Frankfurt", "country": "Germany", "continent": "Europe", "lat": 50.1109, "lon": 8.6821, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "CTRL_CD_03", "name": "Bilbao", "country": "Spain", "continent": "Europe", "lat": 43.2630, "lon": -2.9350, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "CTRL_CD_04", "name": "Charlotte", "country": "USA", "continent": "North America", "lat": 35.2271, "lon": -80.8431, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "CTRL_CD_05", "name": "Ottawa", "country": "Canada", "continent": "North America", "lat": 45.4215, "lon": -75.6972, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "CTRL_CD_06", "name": "Leon", "country": "Mexico", "continent": "North America", "lat": 21.1221, "lon": -101.6820, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "CTRL_CD_07", "name": "Rosario", "country": "Argentina", "continent": "South America", "lat": -32.9468, "lon": -60.6393, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "CTRL_CD_08", "name": "Dakar", "country": "Senegal", "continent": "Africa", "lat": 14.7167, "lon": -17.4677, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "CTRL_CD_09", "name": "Fukuoka", "country": "Japan", "continent": "Asia", "lat": 33.5902, "lon": 130.4017, "primary_domain": "CHANGE_DETECTION"},
    {"scene_id": "CTRL_CD_10", "name": "Auckland", "country": "New Zealand", "continent": "Oceania", "lat": -36.8485, "lon": 174.7633, "primary_domain": "CHANGE_DETECTION"}
]


def validate_spatial_dispersion(scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validates that pairwise distance d_min >= 100 km and pairwise bounding box IoU == 0.0."""
    n = len(scenes)
    min_dist = float("inf")
    closest_pair = None
    max_iou = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            s1 = scenes[i]
            s2 = scenes[j]
            d = haversine(s1["lat"], s1["lon"], s2["lat"], s2["lon"])
            if d < min_dist:
                min_dist = d
                closest_pair = (s1["scene_id"], s2["scene_id"], s1["name"], s2["name"], d)

            # Compute bounding box IoU
            iou = compute_bbox_iou(s1["bbox_lonlat"], s2["bbox_lonlat"])
            if iou > max_iou:
                max_iou = iou

    return {
        "n_scenes": n,
        "min_distance_km": float(min_dist),
        "closest_pair": closest_pair,
        "distance_guarantee_passed": bool(min_dist >= 100.0),
        "max_pairwise_iou": float(max_iou),
        "zero_iou_guarantee_passed": bool(max_iou == 0.0)
    }


def generate_track45_config() -> Dict[str, Any]:
    # 1. Enrich scenes with UTM, bounding boxes, and base geotransforms
    enriched_scenes = []
    for s in RAW_SCENES:
        lat, lon = s["lat"], s["lon"]
        utm_zone, epsg = get_utm_epsg(lon, lat)
        delta_deg = 0.05  # ~5.5 km patch bounding box
        bbox = [round(lon - delta_deg, 4), round(lat - delta_deg, 4), round(lon + delta_deg, 4), round(lat + delta_deg, 4)]
        # Canonical GDAL GeoTransform [origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height]
        # In UTM 10m GSD: pixel_width = 10.0, pixel_height = -10.0
        gt_canonical = [500000.0, 10.0, 0.0, 5000000.0, 0.0, -10.0]

        item = dict(s)
        item["utm_zone"] = utm_zone
        item["epsg"] = epsg
        item["bbox_lonlat"] = bbox
        item["gt_canonical"] = gt_canonical
        enriched_scenes.append(item)

    # 2. Verify spatial dispersion
    disp_metrics = validate_spatial_dispersion(enriched_scenes)
    if not disp_metrics["distance_guarantee_passed"]:
        raise ValueError(f"Spatial dispersion violation! Min distance {disp_metrics['min_distance_km']:.2f} km < 100 km")
    if not disp_metrics["zero_iou_guarantee_passed"]:
        raise ValueError(f"Zero IoU violation! Max pairwise IoU {disp_metrics['max_pairwise_iou']:.4f} > 0.0")

    # 3. Build 120 Perturbed Payloads (Sources 1-30, 4 perturbations each)
    evaluation_payloads: List[Dict[str, Any]] = []

    # Map sources by ID
    sources_map = {s["scene_id"]: s for s in enriched_scenes}

    for idx in range(1, 31):
        src_id = f"SRC_{idx:02d}"
        src = sources_map[src_id]
        gt_base = src["gt_canonical"]
        crs_base = src["epsg"]

        # -------------------------------------------------------------
        # Sources 1-5 (Land Cover)
        # -------------------------------------------------------------
        if 1 <= idx <= 5:
            # P1: Dim 1: Missing SAR / Optical-only (ERR_MISSING_MODALITY)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P1",
                "source_scene_id": src_id,
                "dimension": 1,
                "dimension_name": "Sensor Modality & Structure",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "optical_only_missing_sar",
                "expected_failure_code": "ERR_MISSING_MODALITY",
                "payload_structure": "optical_only_dict",
                "target_specialist": "deep_multimodal",
                "require_multimodal_raster": True,
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [120, 120],
                "gsd_m": 10.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P2: Dim 3: Omission of B08 (ERR_INSUFFICIENT_BANDS)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P2",
                "source_scene_id": src_id,
                "dimension": 3,
                "dimension_name": "Spectral Band Adequacy",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "omission_of_b08_nir",
                "expected_failure_code": "ERR_INSUFFICIENT_BANDS",
                "target_specialist": "deep_multimodal",
                "require_multimodal_raster": False,
                "spectral_bands": ["B02", "B03", "B04"],  # Missing B08
                "spatial_shape": [120, 120],
                "gsd_m": 10.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P3: Dim 4: Undersized 119x120 (ERR_DIMENSION_TOO_SMALL)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P3",
                "source_scene_id": src_id,
                "dimension": 4,
                "dimension_name": "Spatial Dimensions",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "undersized_spatial_grid_119x120",
                "expected_failure_code": "ERR_DIMENSION_TOO_SMALL",
                "target_specialist": "deep_multimodal",
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [119, 120],  # Below 120
                "gsd_m": 10.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P4: Dim 8: Cloud cover fc = 0.201 (ERR_CORRUPT_PAYLOAD)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P4",
                "source_scene_id": src_id,
                "dimension": 8,
                "dimension_name": "Radiometric & Cloud Guard",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "cloud_cover_boundary_violation_0.201",
                "expected_failure_code": "ERR_CORRUPT_PAYLOAD",
                "target_specialist": "deep_multimodal",
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [120, 120],
                "gsd_m": 10.0,
                "cloud_fraction": 0.201,  # Exceeds 0.20
                "is_valid_payload": False
            })

        # -------------------------------------------------------------
        # Sources 6-10 (Land Cover)
        # -------------------------------------------------------------
        elif 6 <= idx <= 10:
            # P1: Dim 1: Bitemporal pair sent to LC (ERR_MISSING_MODALITY)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P1",
                "source_scene_id": src_id,
                "dimension": 1,
                "dimension_name": "Sensor Modality & Structure",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "bitemporal_pair_sent_to_lc",
                "expected_failure_code": "ERR_MISSING_MODALITY",
                "payload_structure": "temporal_pair_dict",
                "target_specialist": "deep_multimodal",
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [120, 120],
                "gsd_m": 10.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P2: Dim 3: Omission of B04 (ERR_INSUFFICIENT_BANDS)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P2",
                "source_scene_id": src_id,
                "dimension": 3,
                "dimension_name": "Spectral Band Adequacy",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "omission_of_b04_red",
                "expected_failure_code": "ERR_INSUFFICIENT_BANDS",
                "target_specialist": "deep_multimodal",
                "spectral_bands": ["B02", "B03", "B08"],  # Missing B04
                "spatial_shape": [120, 120],
                "gsd_m": 10.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P3: Dim 7: GSD boundary 7.9m (ERR_RESOLUTION_INCOMPATIBLE)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P3",
                "source_scene_id": src_id,
                "dimension": 7,
                "dimension_name": "Ground Sample Distance (GSD)",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "gsd_lower_boundary_violation_7.9m",
                "expected_failure_code": "ERR_RESOLUTION_INCOMPATIBLE",
                "target_specialist": "deep_multimodal",
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [120, 120],
                "gsd_m": 7.9,  # Below 8.0m
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P4: Dim 8: Double-scaling trap <= 1.05 (ERR_CORRUPT_PAYLOAD)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P4",
                "source_scene_id": src_id,
                "dimension": 8,
                "dimension_name": "Radiometric & Cloud Guard",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "double_scaling_float_trap_s2_l2a",
                "expected_failure_code": "ERR_CORRUPT_PAYLOAD",
                "target_specialist": "deep_multimodal",
                "product_type": "s2_l2a",
                "is_double_scaling_trap": True,
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [120, 120],
                "gsd_m": 10.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

        # -------------------------------------------------------------
        # Sources 11-15 (Land Cover)
        # -------------------------------------------------------------
        elif 11 <= idx <= 15:
            # P1: Dim 1: 1D Feature vector sent to LC (ERR_MISSING_MODALITY)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P1",
                "source_scene_id": src_id,
                "dimension": 1,
                "dimension_name": "Sensor Modality & Structure",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "1d_feature_vector_sent_to_deep_lc",
                "expected_failure_code": "ERR_MISSING_MODALITY",
                "payload_structure": "1d_feature_vector",
                "target_specialist": "deep_multimodal",
                "require_multimodal_raster": True,
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [120, 120],
                "gsd_m": 10.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P2: Dim 7: GSD boundary 12.1m (ERR_RESOLUTION_INCOMPATIBLE)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P2",
                "source_scene_id": src_id,
                "dimension": 7,
                "dimension_name": "Ground Sample Distance (GSD)",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "gsd_upper_boundary_violation_12.1m",
                "expected_failure_code": "ERR_RESOLUTION_INCOMPATIBLE",
                "target_specialist": "deep_multimodal",
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [120, 120],
                "gsd_m": 12.1,  # Above 12.0m
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P3: Dim 8: Floating NaN / Inf values (ERR_CORRUPT_PAYLOAD)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P3",
                "source_scene_id": src_id,
                "dimension": 8,
                "dimension_name": "Radiometric & Cloud Guard",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "non_finite_floating_nan_inf",
                "expected_failure_code": "ERR_CORRUPT_PAYLOAD",
                "target_specialist": "deep_multimodal",
                "has_nan_inf": True,
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [120, 120],
                "gsd_m": 10.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P4: Dim 8: Cloud cover fc = 0.30 (ERR_CORRUPT_PAYLOAD)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P4",
                "source_scene_id": src_id,
                "dimension": 8,
                "dimension_name": "Radiometric & Cloud Guard",
                "target_task": "LAND_COVER_CLASSIFICATION",
                "perturbation_type": "cloud_cover_extreme_violation_0.30",
                "expected_failure_code": "ERR_CORRUPT_PAYLOAD",
                "target_specialist": "deep_multimodal",
                "spectral_bands": ["B02", "B03", "B04", "B08"],
                "spatial_shape": [120, 120],
                "gsd_m": 10.0,
                "cloud_fraction": 0.30,  # Far above 0.20
                "is_valid_payload": False
            })

        # -------------------------------------------------------------
        # Sources 16-20 (Change Detection)
        # -------------------------------------------------------------
        elif 16 <= idx <= 20:
            # P1: Dim 1: Single image sent to CD (ERR_MISSING_MODALITY)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P1",
                "source_scene_id": src_id,
                "dimension": 1,
                "dimension_name": "Sensor Modality & Structure",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "single_image_sent_to_cd",
                "expected_failure_code": "ERR_MISSING_MODALITY",
                "payload_structure": "single_image_array",
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 30.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P2: Dim 2: Temporal boundary Delta t = 4.9d (ERR_TEMPORAL_INVALID)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P2",
                "source_scene_id": src_id,
                "dimension": 2,
                "dimension_name": "Temporal Revisit Window",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "temporal_revisit_lower_boundary_4.9d",
                "expected_failure_code": "ERR_TEMPORAL_INVALID",
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 4.9,  # Below 5.0d
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P3: Dim 5: CRS mismatch EPSG:32632 vs 4326 (ERR_CRS_MISMATCH)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P3",
                "source_scene_id": src_id,
                "dimension": 5,
                "dimension_name": "Geospatial CRS Consistency",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "crs_mismatch_projected_vs_geographic",
                "expected_failure_code": "ERR_CRS_MISMATCH",
                "gt1": gt_base,
                "gt2": gt_base,
                "crs1": crs_base,
                "crs2": "EPSG:4326",  # Geographic WGS84 vs UTM
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 30.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P4: Dim 6: Unregistered grid delta = 0.051 px (ERR_UNREGISTERED_GRID)
            # 0.051 px shift in X: dx = 0.051 * 10m = 0.51m
            gt_shift = [gt_base[0] + 0.51, gt_base[1], gt_base[2], gt_base[3], gt_base[4], gt_base[5]]
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P4",
                "source_scene_id": src_id,
                "dimension": 6,
                "dimension_name": "Affine Alignment & Coregistration",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "unregistered_grid_displacement_0.051px",
                "expected_failure_code": "ERR_UNREGISTERED_GRID",
                "gt1": gt_base,
                "gt2": gt_shift,
                "crs1": crs_base,
                "crs2": crs_base,
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 30.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

        # -------------------------------------------------------------
        # Sources 21-25 (Change Detection)
        # -------------------------------------------------------------
        elif 21 <= idx <= 25:
            # P1: Dim 2: Temporal boundary Delta t = 730.1d (ERR_TEMPORAL_INVALID)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P1",
                "source_scene_id": src_id,
                "dimension": 2,
                "dimension_name": "Temporal Revisit Window",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "temporal_revisit_upper_boundary_730.1d",
                "expected_failure_code": "ERR_TEMPORAL_INVALID",
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 730.1,  # Exceeds 730.0d
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P2: Dim 4: Undersized 99x100 (ERR_DIMENSION_TOO_SMALL)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P2",
                "source_scene_id": src_id,
                "dimension": 4,
                "dimension_name": "Spatial Dimensions",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "undersized_spatial_grid_99x100",
                "expected_failure_code": "ERR_DIMENSION_TOO_SMALL",
                "spatial_shape": [99, 100],  # Below 100
                "gsd_m": 10.0,
                "temporal_gap_days": 30.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P3: Dim 5: CRS mismatch EPSG:32632 vs 32633 (ERR_CRS_MISMATCH)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P3",
                "source_scene_id": src_id,
                "dimension": 5,
                "dimension_name": "Geospatial CRS Consistency",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "crs_mismatch_adjacent_utm_zones",
                "expected_failure_code": "ERR_CRS_MISMATCH",
                "gt1": gt_base,
                "gt2": gt_base,
                "crs1": "EPSG:32632",
                "crs2": "EPSG:32633",  # Adjacent UTM zone mismatch
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 30.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P4: Dim 6: Anisotropic shear kappa = 1.011 (ERR_ANISOTROPIC_SHEAR)
            # Shear matrix on GeoTransform
            gt_shear = [gt_base[0], 10.055, 0.055, gt_base[3], 0.0, -10.0 / 1.0055]
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P4",
                "source_scene_id": src_id,
                "dimension": 6,
                "dimension_name": "Affine Alignment & Coregistration",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "anisotropic_shear_kappa_1.011",
                "expected_failure_code": "ERR_ANISOTROPIC_SHEAR",
                "gt1": gt_base,
                "gt2": gt_shear,
                "crs1": crs_base,
                "crs2": crs_base,
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 30.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

        # -------------------------------------------------------------
        # Sources 26-30 (Change Detection)
        # -------------------------------------------------------------
        elif 26 <= idx <= 30:
            # P1: Dim 2: Extreme revisit Delta t = 1200d (ERR_TEMPORAL_INVALID)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P1",
                "source_scene_id": src_id,
                "dimension": 2,
                "dimension_name": "Temporal Revisit Window",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "extreme_temporal_drift_1200d",
                "expected_failure_code": "ERR_TEMPORAL_INVALID",
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 1200.0,  # Extreme revisit
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P2: Dim 6: Axis inversion det <= 0 (ERR_INVERTED_AXIS)
            # Positive pixel height flips Y axis
            gt_inv = [gt_base[0], 10.0, 0.0, gt_base[3], 0.0, 10.0]
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P2",
                "source_scene_id": src_id,
                "dimension": 6,
                "dimension_name": "Affine Alignment & Coregistration",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "axis_reflection_inversion_det_neg",
                "expected_failure_code": "ERR_INVERTED_AXIS",
                "gt1": gt_base,
                "gt2": gt_inv,
                "crs1": crs_base,
                "crs2": crs_base,
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 30.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P3: Dim 6: Scale distortion det = 0.979 (ERR_SCALE_DISTORTION)
            s_scale = math.sqrt(0.979)
            gt_scale = [gt_base[0], 10.0 * s_scale, 0.0, gt_base[3], 0.0, -10.0 * s_scale]
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P3",
                "source_scene_id": src_id,
                "dimension": 6,
                "dimension_name": "Affine Alignment & Coregistration",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "scale_distortion_det_0.979",
                "expected_failure_code": "ERR_SCALE_DISTORTION",
                "gt1": gt_base,
                "gt2": gt_scale,
                "crs1": crs_base,
                "crs2": crs_base,
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 30.0,
                "cloud_fraction": 0.05,
                "is_valid_payload": False
            })

            # P4: Dim 8: Cloud cover fc = 0.101 (ERR_CORRUPT_PAYLOAD)
            evaluation_payloads.append({
                "payload_id": f"{src_id}_P4",
                "source_scene_id": src_id,
                "dimension": 8,
                "dimension_name": "Radiometric & Cloud Guard",
                "target_task": "CHANGE_DETECTION",
                "perturbation_type": "cloud_cover_boundary_violation_0.101",
                "expected_failure_code": "ERR_CORRUPT_PAYLOAD",
                "spatial_shape": [256, 256],
                "gsd_m": 10.0,
                "temporal_gap_days": 30.0,
                "cloud_fraction": 0.101,  # Exceeds 0.10 for Change Detection
                "is_valid_payload": False
            })

    # 4. Build 20 Valid Control Payloads (Controls 1-10 for LC, Controls 11-20 for CD)
    for idx in range(1, 11):
        ctrl_id = f"CTRL_LC_{idx:02d}"
        ctrl = sources_map[ctrl_id]
        evaluation_payloads.append({
            "payload_id": f"{ctrl_id}_VALID",
            "source_scene_id": ctrl_id,
            "dimension": 0,
            "dimension_name": "Valid In-Domain Control",
            "target_task": "LAND_COVER_CLASSIFICATION",
            "perturbation_type": "valid_multimodal_s1_s2_control",
            "expected_failure_code": None,
            "target_specialist": "deep_multimodal",
            "payload_structure": "multimodal_optical_sar",
            "spectral_bands": ["B02", "B03", "B04", "B08"],
            "spatial_shape": [120, 120],
            "gsd_m": 10.0,
            "cloud_fraction": 0.04,
            "crs": ctrl["epsg"],
            "is_valid_payload": True
        })

    for idx in range(1, 11):
        ctrl_id = f"CTRL_CD_{idx:02d}"
        ctrl = sources_map[ctrl_id]
        gt_base = ctrl["gt_canonical"]
        crs_base = ctrl["epsg"]
        evaluation_payloads.append({
            "payload_id": f"{ctrl_id}_VALID",
            "source_scene_id": ctrl_id,
            "dimension": 0,
            "dimension_name": "Valid In-Domain Control",
            "target_task": "CHANGE_DETECTION",
            "perturbation_type": "valid_bitemporal_s2_pair_control",
            "expected_failure_code": None,
            "payload_structure": "bitemporal_pair_array",
            "spatial_shape": [256, 256],
            "gsd_m": 10.0,
            "temporal_gap_days": 25.0,  # In [5.0, 730.0]
            "cloud_fraction": 0.03,     # In [0.0, 0.10]
            "gt1": gt_base,
            "gt2": gt_base,
            "crs1": crs_base,
            "crs2": crs_base,
            "is_valid_payload": True
        })

    # 5. Dimension counts audit
    dim_counts = {}
    for p in evaluation_payloads:
        d = p["dimension"]
        dim_counts[d] = dim_counts.get(d, 0) + 1

    expected_dim_counts = {
        1: 20,
        2: 15,
        3: 10,
        4: 10,
        5: 10,
        6: 20,
        7: 10,
        8: 25,
        0: 20  # Controls
    }
    for d, count in expected_dim_counts.items():
        if dim_counts.get(d, 0) != count:
            raise ValueError(f"Dimension {d} payload count mismatch! Expected {count}, got {dim_counts.get(d, 0)}")

    config_data = {
        "metadata": {
            "protocol_reference": "SATQUERY-P4-PROTOCOL-V1.12.1-FROZEN.md",
            "protocol_hash": FROZEN_PROTOCOL_HASH,
            "track": "Track 4.5: Adversarial Physical Metadata and Coordinate Perturbations",
            "n_source_observations": 30,
            "n_control_observations": 20,
            "total_independent_scenes": len(enriched_scenes),
            "n_perturbed_payloads": 120,
            "n_control_payloads": 20,
            "total_evaluation_payloads": len(evaluation_payloads),
            "dimension_breakdown": {f"Dim_{k}": v for k, v in expected_dim_counts.items() if k > 0},
            "spatial_dispersion": disp_metrics
        },
        "source_scenes": enriched_scenes,
        "evaluation_payloads": evaluation_payloads
    }

    out_path = REPO_ROOT / "configs" / "track45_metadata_perturbations_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"Generated Track 4.5 configuration with {len(evaluation_payloads)} payloads at: {out_path}")
    print(f"Spatial dispersion check: Min distance = {disp_metrics['min_distance_km']:.2f} km (>= 100 km), Max IoU = {disp_metrics['max_pairwise_iou']}")
    print(f"Dimension breakdown audit: {dim_counts}")
    return config_data


if __name__ == "__main__":
    generate_track45_config()
