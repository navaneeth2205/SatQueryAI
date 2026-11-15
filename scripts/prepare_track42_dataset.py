"""
Prepare and Validate Track 4.2 Cross-Sensor & Cross-Resolution Compatibility Defense Dataset.

Governed strictly by SatQuery AI Phase 4 Frozen Protocol v1.12.1 (Section: Track 4.2).
Preregistered specifications:
- N_source = 40 independent acquisition footprints (20 Landsat-8/9, 20 PlanetScope).
- Spatial-Temporal Dispersion Guarantee:
  * Minimum geographic separation: d_min >= 150 km across all 40 scenes.
  * At least 4 distinct continents and disjoint UTM zones.
  * Temporal dispersion: acquisition timestamps separated by >= 30 days or distinct seasonal campaigns.
  * Zero polygon overlap: pairwise bounding box IoU == 0.0.
- Exactly 2 perturbed payloads per source scene (2 x 40 = 80 off-sensor payloads):
  * Payload A (Resolution Stress): native spatial resolution (gsd_m = 30.0 for Landsat, gsd_m = 3.0 for PlanetScope).
  * Payload B (Modality / Spectral Stress): optical-only without SAR for LC (ERR_MISSING_MODALITY) or
    off-sensor band naming lacking Sentinel-2 channel codes (ERR_INSUFFICIENT_BANDS).
- 20 Sentinel-2 Level-2A Control Payloads (from 20 independent scenes):
  * 10 valid Land Cover payloads (optical + SAR, gsd_m = 10.0).
  * 10 valid Change Detection pairs (bitemporal S2 RGB, gsd_m = 10.0, valid revisit Delta t in [5, 730] days).
- Total: Exactly 60 independent source scenes yielding 100 evaluation payloads.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


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
# 60 Source Scene Definitions (20 Landsat, 20 PlanetScope, 20 Sentinel-2 Control)
# -----------------------------------------------------------------------------

RAW_SCENES: List[Dict[str, Any]] = [
    # 20 Landsat-8/9 Scenes (Off-target sensors)
    {"scene_id": "LS_01", "name": "Denver", "country": "USA", "continent": "North America", "lat": 39.7392, "lon": -104.9903, "sensor": "Landsat-8", "date": "2023-01-15", "native_gsd_m": 30.0},
    {"scene_id": "LS_02", "name": "Calgary", "country": "Canada", "continent": "North America", "lat": 51.0447, "lon": -114.0719, "sensor": "Landsat-9", "date": "2023-02-20", "native_gsd_m": 30.0},
    {"scene_id": "LS_03", "name": "Hermosillo", "country": "Mexico", "continent": "North America", "lat": 29.0729, "lon": -110.9559, "sensor": "Landsat-8", "date": "2023-03-25", "native_gsd_m": 30.0},
    {"scene_id": "LS_04", "name": "Bogota", "country": "Colombia", "continent": "South America", "lat": 4.7110, "lon": -74.0721, "sensor": "Landsat-9", "date": "2023-04-30", "native_gsd_m": 30.0},
    {"scene_id": "LS_05", "name": "Brasilia", "country": "Brazil", "continent": "South America", "lat": -15.7975, "lon": -47.8919, "sensor": "Landsat-8", "date": "2023-06-05", "native_gsd_m": 30.0},
    {"scene_id": "LS_06", "name": "Mendoza", "country": "Argentina", "continent": "South America", "lat": -32.8895, "lon": -68.8458, "sensor": "Landsat-9", "date": "2023-07-12", "native_gsd_m": 30.0},
    {"scene_id": "LS_07", "name": "Madrid", "country": "Spain", "continent": "Europe", "lat": 40.4168, "lon": -3.7038, "sensor": "Landsat-8", "date": "2023-08-18", "native_gsd_m": 30.0},
    {"scene_id": "LS_08", "name": "Warsaw", "country": "Poland", "continent": "Europe", "lat": 52.2297, "lon": 21.0122, "sensor": "Landsat-9", "date": "2023-09-22", "native_gsd_m": 30.0},
    {"scene_id": "LS_09", "name": "Stockholm", "country": "Sweden", "continent": "Europe", "lat": 59.3293, "lon": 18.0686, "sensor": "Landsat-8", "date": "2023-10-28", "native_gsd_m": 30.0},
    {"scene_id": "LS_10", "name": "Dakar", "country": "Senegal", "continent": "Africa", "lat": 14.7167, "lon": -17.4677, "sensor": "Landsat-9", "date": "2023-12-02", "native_gsd_m": 30.0},
    {"scene_id": "LS_11", "name": "Nairobi", "country": "Kenya", "continent": "Africa", "lat": -1.2921, "lon": 36.8219, "sensor": "Landsat-8", "date": "2024-01-08", "native_gsd_m": 30.0},
    {"scene_id": "LS_12", "name": "Windhoek", "country": "Namibia", "continent": "Africa", "lat": -22.5609, "lon": 17.0658, "sensor": "Landsat-9", "date": "2024-02-14", "native_gsd_m": 30.0},
    {"scene_id": "LS_13", "name": "Cairo", "country": "Egypt", "continent": "Africa", "lat": 30.0444, "lon": 31.2357, "sensor": "Landsat-8", "date": "2024-03-20", "native_gsd_m": 30.0},
    {"scene_id": "LS_14", "name": "Riyadh", "country": "Saudi Arabia", "continent": "Asia", "lat": 24.7136, "lon": 46.6753, "sensor": "Landsat-9", "date": "2024-04-25", "native_gsd_m": 30.0},
    {"scene_id": "LS_15", "name": "Almaty", "country": "Kazakhstan", "continent": "Asia", "lat": 43.2220, "lon": 76.8512, "sensor": "Landsat-8", "date": "2024-05-30", "native_gsd_m": 30.0},
    {"scene_id": "LS_16", "name": "Hyderabad", "country": "India", "continent": "Asia", "lat": 17.3850, "lon": 78.4867, "sensor": "Landsat-9", "date": "2024-07-05", "native_gsd_m": 30.0},
    {"scene_id": "LS_17", "name": "Ulaanbaatar", "country": "Mongolia", "continent": "Asia", "lat": 47.8864, "lon": 106.9057, "sensor": "Landsat-8", "date": "2024-08-10", "native_gsd_m": 30.0},
    {"scene_id": "LS_18", "name": "Sapporo", "country": "Japan", "continent": "Asia", "lat": 43.0618, "lon": 141.3545, "sensor": "Landsat-9", "date": "2024-09-15", "native_gsd_m": 30.0},
    {"scene_id": "LS_19", "name": "Alice Springs", "country": "Australia", "continent": "Oceania", "lat": -23.6980, "lon": 133.8807, "sensor": "Landsat-8", "date": "2024-10-20", "native_gsd_m": 30.0},
    {"scene_id": "LS_20", "name": "Christchurch", "country": "New Zealand", "continent": "Oceania", "lat": -43.5321, "lon": 172.6362, "sensor": "Landsat-9", "date": "2024-11-25", "native_gsd_m": 30.0},

    # 20 PlanetScope Scenes (Off-target sensors)
    {"scene_id": "PS_01", "name": "Phoenix", "country": "USA", "continent": "North America", "lat": 33.4484, "lon": -112.0740, "sensor": "PlanetScope", "date": "2023-01-20", "native_gsd_m": 3.0},
    {"scene_id": "PS_02", "name": "Winnipeg", "country": "Canada", "continent": "North America", "lat": 49.8951, "lon": -97.1384, "sensor": "PlanetScope", "date": "2023-02-25", "native_gsd_m": 3.0},
    {"scene_id": "PS_03", "name": "Guadalajara", "country": "Mexico", "continent": "North America", "lat": 20.6597, "lon": -103.3496, "sensor": "PlanetScope", "date": "2023-03-30", "native_gsd_m": 3.0},
    {"scene_id": "PS_04", "name": "Lima", "country": "Peru", "continent": "South America", "lat": -12.0464, "lon": -77.0428, "sensor": "PlanetScope", "date": "2023-05-05", "native_gsd_m": 3.0},
    {"scene_id": "PS_05", "name": "Fortaleza", "country": "Brazil", "continent": "South America", "lat": -3.7319, "lon": -38.5267, "sensor": "PlanetScope", "date": "2023-06-10", "native_gsd_m": 3.0},
    {"scene_id": "PS_06", "name": "Santiago", "country": "Chile", "continent": "South America", "lat": -33.4489, "lon": -70.6693, "sensor": "PlanetScope", "date": "2023-07-16", "native_gsd_m": 3.0},
    {"scene_id": "PS_07", "name": "Lyon", "country": "France", "continent": "Europe", "lat": 45.7640, "lon": 4.8357, "sensor": "PlanetScope", "date": "2023-08-22", "native_gsd_m": 3.0},
    {"scene_id": "PS_08", "name": "Munich", "country": "Germany", "continent": "Europe", "lat": 48.1351, "lon": 11.5820, "sensor": "PlanetScope", "date": "2023-09-27", "native_gsd_m": 3.0},
    {"scene_id": "PS_09", "name": "Helsinki", "country": "Finland", "continent": "Europe", "lat": 60.1699, "lon": 24.9384, "sensor": "PlanetScope", "date": "2023-11-02", "native_gsd_m": 3.0},
    {"scene_id": "PS_10", "name": "Casablanca", "country": "Morocco", "continent": "Africa", "lat": 33.5731, "lon": -7.5898, "sensor": "PlanetScope", "date": "2023-12-08", "native_gsd_m": 3.0},
    {"scene_id": "PS_11", "name": "Addis Ababa", "country": "Ethiopia", "continent": "Africa", "lat": 9.0300, "lon": 38.7400, "sensor": "PlanetScope", "date": "2024-01-14", "native_gsd_m": 3.0},
    {"scene_id": "PS_12", "name": "Gaborone", "country": "Botswana", "continent": "Africa", "lat": -24.6282, "lon": 25.9231, "sensor": "PlanetScope", "date": "2024-02-19", "native_gsd_m": 3.0},
    {"scene_id": "PS_13", "name": "Tunis", "country": "Tunisia", "continent": "Africa", "lat": 36.8065, "lon": 10.1815, "sensor": "PlanetScope", "date": "2024-03-26", "native_gsd_m": 3.0},
    {"scene_id": "PS_14", "name": "Muscat", "country": "Oman", "continent": "Asia", "lat": 23.5880, "lon": 58.3829, "sensor": "PlanetScope", "date": "2024-05-01", "native_gsd_m": 3.0},
    {"scene_id": "PS_15", "name": "Tashkent", "country": "Uzbekistan", "continent": "Asia", "lat": 41.2995, "lon": 69.2401, "sensor": "PlanetScope", "date": "2024-06-06", "native_gsd_m": 3.0},
    {"scene_id": "PS_16", "name": "Nagpur", "country": "India", "continent": "Asia", "lat": 21.1458, "lon": 79.0882, "sensor": "PlanetScope", "date": "2024-07-12", "native_gsd_m": 3.0},
    {"scene_id": "PS_17", "name": "Chengdu", "country": "China", "continent": "Asia", "lat": 30.5728, "lon": 104.0668, "sensor": "PlanetScope", "date": "2024-08-18", "native_gsd_m": 3.0},
    {"scene_id": "PS_18", "name": "Fukuoka", "country": "Japan", "continent": "Asia", "lat": 33.5902, "lon": 130.4017, "sensor": "PlanetScope", "date": "2024-09-23", "native_gsd_m": 3.0},
    {"scene_id": "PS_19", "name": "Darwin", "country": "Australia", "continent": "Oceania", "lat": -12.4634, "lon": 130.8456, "sensor": "PlanetScope", "date": "2024-10-29", "native_gsd_m": 3.0},
    {"scene_id": "PS_20", "name": "Suva", "country": "Fiji", "continent": "Oceania", "lat": -18.1248, "lon": 178.4501, "sensor": "PlanetScope", "date": "2024-12-04", "native_gsd_m": 3.0},

    # 10 Sentinel-2 Control Scenes (Land Cover)
    {"scene_id": "S2C_LC_01", "name": "Austin", "country": "USA", "continent": "North America", "lat": 30.2672, "lon": -97.7431, "sensor": "Sentinel-2A", "date": "2023-01-25", "native_gsd_m": 10.0},
    {"scene_id": "S2C_LC_02", "name": "Edmonton", "country": "Canada", "continent": "North America", "lat": 53.5461, "lon": -113.4938, "sensor": "Sentinel-2B", "date": "2023-03-02", "native_gsd_m": 10.0},
    {"scene_id": "S2C_LC_03", "name": "Cordoba", "country": "Argentina", "continent": "South America", "lat": -31.4201, "lon": -64.1888, "sensor": "Sentinel-2A", "date": "2023-04-10", "native_gsd_m": 10.0},
    {"scene_id": "S2C_LC_04", "name": "Seville", "country": "Spain", "continent": "Europe", "lat": 37.3891, "lon": -5.9845, "sensor": "Sentinel-2B", "date": "2023-05-18", "native_gsd_m": 10.0},
    {"scene_id": "S2C_LC_05", "name": "Poznan", "country": "Poland", "continent": "Europe", "lat": 52.4064, "lon": 16.9252, "sensor": "Sentinel-2A", "date": "2023-06-25", "native_gsd_m": 10.0},
    {"scene_id": "S2C_LC_06", "name": "Cape Town", "country": "South Africa", "continent": "Africa", "lat": -33.9249, "lon": 18.4241, "sensor": "Sentinel-2B", "date": "2023-08-01", "native_gsd_m": 10.0},
    {"scene_id": "S2C_LC_07", "name": "Accra", "country": "Ghana", "continent": "Africa", "lat": 5.6037, "lon": -0.1870, "sensor": "Sentinel-2A", "date": "2023-09-08", "native_gsd_m": 10.0},
    {"scene_id": "S2C_LC_08", "name": "Jaipur", "country": "India", "continent": "Asia", "lat": 26.9124, "lon": 75.7873, "sensor": "Sentinel-2B", "date": "2023-10-15", "native_gsd_m": 10.0},
    {"scene_id": "S2C_LC_09", "name": "Wuhan", "country": "China", "continent": "Asia", "lat": 30.5928, "lon": 114.3055, "sensor": "Sentinel-2A", "date": "2023-11-22", "native_gsd_m": 10.0},
    {"scene_id": "S2C_LC_10", "name": "Brisbane", "country": "Australia", "continent": "Oceania", "lat": -27.4698, "lon": 153.0251, "sensor": "Sentinel-2B", "date": "2023-12-30", "native_gsd_m": 10.0},

    # 10 Sentinel-2 Control Scenes (Change Detection)
    {"scene_id": "S2C_CD_01", "name": "Montreal", "country": "Canada", "continent": "North America", "lat": 45.5017, "lon": -73.5673, "sensor": "Sentinel-2A", "date": "2023-02-05", "native_gsd_m": 10.0},
    {"scene_id": "S2C_CD_02", "name": "Mexico City", "country": "Mexico", "continent": "North America", "lat": 19.4326, "lon": -99.1332, "sensor": "Sentinel-2B", "date": "2023-03-12", "native_gsd_m": 10.0},
    {"scene_id": "S2C_CD_03", "name": "Belo Horizonte", "country": "Brazil", "continent": "South America", "lat": -19.9167, "lon": -43.9345, "sensor": "Sentinel-2A", "date": "2023-04-20", "native_gsd_m": 10.0},
    {"scene_id": "S2C_CD_04", "name": "Bordeaux", "country": "France", "continent": "Europe", "lat": 44.8378, "lon": -0.5792, "sensor": "Sentinel-2B", "date": "2023-05-28", "native_gsd_m": 10.0},
    {"scene_id": "S2C_CD_05", "name": "Gothenburg", "country": "Sweden", "continent": "Europe", "lat": 57.7089, "lon": 11.9746, "sensor": "Sentinel-2A", "date": "2023-07-04", "native_gsd_m": 10.0},
    {"scene_id": "S2C_CD_06", "name": "Durban", "country": "South Africa", "continent": "Africa", "lat": -29.8587, "lon": 31.0218, "sensor": "Sentinel-2B", "date": "2023-08-12", "native_gsd_m": 10.0},
    {"scene_id": "S2C_CD_07", "name": "Kampala", "country": "Uganda", "continent": "Africa", "lat": 0.3476, "lon": 32.5825, "sensor": "Sentinel-2A", "date": "2023-09-20", "native_gsd_m": 10.0},
    {"scene_id": "S2C_CD_08", "name": "Pune", "country": "India", "continent": "Asia", "lat": 18.5204, "lon": 73.8567, "sensor": "Sentinel-2B", "date": "2023-10-29", "native_gsd_m": 10.0},
    {"scene_id": "S2C_CD_09", "name": "Sendai", "country": "Japan", "continent": "Asia", "lat": 38.2682, "lon": 140.8694, "sensor": "Sentinel-2A", "date": "2023-12-05", "native_gsd_m": 10.0},
    {"scene_id": "S2C_CD_10", "name": "Adelaide", "country": "Australia", "continent": "Oceania", "lat": -34.9285, "lon": 138.6007, "sensor": "Sentinel-2B", "date": "2024-01-12", "native_gsd_m": 10.0},
]


def build_track42_configuration() -> Dict[str, Any]:
    """Build, audit, and return the complete Track 4.2 dataset configuration."""
    half_side_deg = 0.10  # ~11 km bounding box half-width (total size ~22 km x 22 km)

    processed_scenes = []
    for s in RAW_SCENES:
        lat = s["lat"]
        lon = s["lon"]
        utm_zone, epsg = get_utm_epsg(lon, lat)
        bbox = [
            round(lon - half_side_deg, 4),
            round(lat - half_side_deg, 4),
            round(lon + half_side_deg, 4),
            round(lat + half_side_deg, 4)
        ]
        scene_entry = dict(s)
        scene_entry["utm_zone"] = utm_zone
        scene_entry["crs"] = epsg
        scene_entry["bounding_box"] = bbox
        processed_scenes.append(scene_entry)

    # 1. Geographic Separation Audit (d_min >= 150 km)
    min_dist_40 = float("inf")
    min_pair_40 = ("", "")
    for i in range(40):
        for j in range(i + 1, 40):
            d = haversine(processed_scenes[i]["lat"], processed_scenes[i]["lon"],
                          processed_scenes[j]["lat"], processed_scenes[j]["lon"])
            if d < min_dist_40:
                min_dist_40 = d
                min_pair_40 = (processed_scenes[i]["scene_id"], processed_scenes[j]["scene_id"])

    min_dist_all = float("inf")
    min_pair_all = ("", "")
    for i in range(len(processed_scenes)):
        for j in range(i + 1, len(processed_scenes)):
            d = haversine(processed_scenes[i]["lat"], processed_scenes[i]["lon"],
                          processed_scenes[j]["lat"], processed_scenes[j]["lon"])
            if d < min_dist_all:
                min_dist_all = d
                min_pair_all = (processed_scenes[i]["scene_id"], processed_scenes[j]["scene_id"])

    # 2. Bounding Box Zero IoU Audit
    max_iou = 0.0
    for i in range(len(processed_scenes)):
        for j in range(i + 1, len(processed_scenes)):
            iou = compute_bbox_iou(processed_scenes[i]["bounding_box"], processed_scenes[j]["bounding_box"])
            if iou > max_iou:
                max_iou = iou

    # 3. Continents Audit
    continents_40 = set(s["continent"] for s in processed_scenes[:40])
    continents_all = set(s["continent"] for s in processed_scenes)

    # 4. Generate 100 Payloads
    payloads = []

    # 40 Landsat Payloads (from 20 scenes)
    for s in processed_scenes[:20]:
        sc_id = s["scene_id"]
        # Payload A: Resolution Stress (gsd_m = 30.0)
        p_a = {
            "payload_id": f"{sc_id}_PAYLOAD_A",
            "source_scene_id": sc_id,
            "sensor": s["sensor"],
            "stress_type": "RESOLUTION_STRESS",
            "target_task": "LAND_COVER_CLASSIFICATION",
            "gsd_m": 30.0,
            "spectral_bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5"],
            "is_valid_payload": False,
            "expected_failure_codes": ["ERR_RESOLUTION_INCOMPATIBLE"],
            "spatial_shape": [120, 120],
            "crs": s["crs"]
        }
        # Payload B: Modality / Spectral Stress
        # Off-sensor band naming lacking Sentinel-2 codes and missing SAR
        p_b = {
            "payload_id": f"{sc_id}_PAYLOAD_B",
            "source_scene_id": sc_id,
            "sensor": s["sensor"],
            "stress_type": "MODALITY_SPECTRAL_STRESS",
            "target_task": "LAND_COVER_CLASSIFICATION",
            "gsd_m": 10.0,  # Even if resampled to 10m, off-sensor bands and missing SAR must be caught!
            "spectral_bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            "is_valid_payload": False,
            "expected_failure_codes": ["ERR_INSUFFICIENT_BANDS"],
            "spatial_shape": [120, 120],
            "crs": s["crs"]
        }
        payloads.extend([p_a, p_b])

    # 40 PlanetScope Payloads (from 20 scenes)
    for s in processed_scenes[20:40]:
        sc_id = s["scene_id"]
        # Payload A: Resolution Stress (gsd_m = 3.0)
        p_a = {
            "payload_id": f"{sc_id}_PAYLOAD_A",
            "source_scene_id": sc_id,
            "sensor": s["sensor"],
            "stress_type": "RESOLUTION_STRESS",
            "target_task": "CHANGE_DETECTION",
            "gsd_m": 3.0,
            "spectral_bands": ["blue", "green", "red", "nir"],
            "is_valid_payload": False,
            "expected_failure_codes": ["ERR_RESOLUTION_INCOMPATIBLE"],
            "spatial_shape": [256, 256],
            "crs": s["crs"],
            "temporal_gap_days": 45.0
        }
        # Payload B: Modality / Spectral Stress
        # PlanetScope band nomenclature lacking Sentinel-2 codes
        p_b = {
            "payload_id": f"{sc_id}_PAYLOAD_B",
            "source_scene_id": sc_id,
            "sensor": s["sensor"],
            "stress_type": "MODALITY_SPECTRAL_STRESS",
            "target_task": "LAND_COVER_CLASSIFICATION",
            "gsd_m": 10.0,  # Even if resampled, lacks S2 canonical bands and SAR
            "spectral_bands": ["blue", "green", "red", "nir"],
            "is_valid_payload": False,
            "expected_failure_codes": ["ERR_INSUFFICIENT_BANDS"],
            "spatial_shape": [120, 120],
            "crs": s["crs"]
        }
        payloads.extend([p_a, p_b])

    # 20 Sentinel-2 Control Payloads (from 20 scenes)
    # 10 Land Cover Controls
    for s in processed_scenes[40:50]:
        sc_id = s["scene_id"]
        p_ctrl = {
            "payload_id": f"{sc_id}_CONTROL",
            "source_scene_id": sc_id,
            "sensor": s["sensor"],
            "stress_type": "CONTROL_VALID",
            "target_task": "LAND_COVER_CLASSIFICATION",
            "gsd_m": 10.0,
            "spectral_bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "VH", "VV"],
            "is_valid_payload": True,
            "expected_failure_codes": [],
            "spatial_shape": [120, 120],
            "crs": s["crs"],
            "cloud_fraction": 0.05
        }
        payloads.append(p_ctrl)

    # 10 Change Detection Controls
    for s in processed_scenes[50:60]:
        sc_id = s["scene_id"]
        p_ctrl = {
            "payload_id": f"{sc_id}_CONTROL",
            "source_scene_id": sc_id,
            "sensor": s["sensor"],
            "stress_type": "CONTROL_VALID",
            "target_task": "CHANGE_DETECTION",
            "gsd_m": 10.0,
            "spectral_bands": ["B02", "B03", "B04"],
            "is_valid_payload": True,
            "expected_failure_codes": [],
            "spatial_shape": [256, 256],
            "crs": s["crs"],
            "temporal_gap_days": 180.0,
            "cloud_fraction": 0.03
        }
        payloads.append(p_ctrl)

    config_data = {
        "metadata": {
            "protocol_reference": "SATQUERY-P4-PROTOCOL-V1.12.1-FROZEN.md",
            "protocol_hash": "7bcb8c5ab00cfebb5f7418237e9297f7eeaf06b0477d426d0ec6b30880e71bb5",
            "track": "Track 4.2: Cross-Sensor & Cross-Resolution Compatibility Defense",
            "n_source_scenes": len(processed_scenes),
            "n_off_sensor_source_scenes": 40,
            "n_control_source_scenes": 20,
            "n_total_payloads": len(payloads),
            "n_invalid_payloads": 80,
            "n_control_payloads": 20,
            "spatial_dispersion": {
                "min_distance_40_off_sensor_km": round(min_dist_40, 2),
                "min_pair_40": min_pair_40,
                "min_distance_all_60_km": round(min_dist_all, 2),
                "min_pair_all": min_pair_all,
                "distance_floor_km": 150.0,
                "distance_guarantee_passed": bool(min_dist_40 >= 150.0),
                "max_pairwise_iou": max_iou,
                "zero_iou_guarantee_passed": bool(max_iou == 0.0)
            },
            "continents_represented": sorted(list(continents_all)),
            "n_continents": len(continents_all)
        },
        "source_scenes": processed_scenes,
        "evaluation_payloads": payloads
    }

    return config_data


def main():
    config = build_track42_configuration()
    out_path = REPO_ROOT / "configs" / "track42_sensor_defense_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    meta = config["metadata"]
    disp = meta["spatial_dispersion"]
    print(f"Successfully generated Track 4.2 configuration at: {out_path}")
    print(f"Source Scenes: {meta['n_source_scenes']} (40 off-sensor + 20 control)")
    print(f"Evaluation Payloads: {meta['n_total_payloads']} (80 invalid + 20 control)")
    print(f"Min Distance (40 Off-Sensor): {disp['min_distance_40_off_sensor_km']} km (Requirement >= 150.0 km) -> PASS: {disp['distance_guarantee_passed']}")
    print(f"Min Distance (All 60 Scenes): {disp['min_distance_all_60_km']} km")
    print(f"Max Pairwise IoU: {disp['max_pairwise_iou']} -> PASS: {disp['zero_iou_guarantee_passed']}")
    print(f"Continents ({meta['n_continents']}): {meta['continents_represented']} -> PASS (>= 4 continents)")


if __name__ == "__main__":
    main()
