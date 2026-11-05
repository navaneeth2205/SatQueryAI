"""
Geospatial and Physical Observation Compatibility Gate for SatQuery AI.
Implements the 8 physical contract dimensions specified in Phase 3 Milestone M3.0/M3.3:
1. Sensor Modality (ERR_MISSING_MODALITY)
2. Temporal Structure (ERR_TEMPORAL_INVALID)
3. Spectral Bands (ERR_INSUFFICIENT_BANDS)
4. Spatial Dimensions (ERR_DIMENSION_TOO_SMALL)
5. Geospatial CRS & Reprojection (ERR_GEOSPATIAL_MISMATCH / ERR_CRS_MISMATCH)
6. Affine Grid Alignment & Full-Footprint Corner Displacement (ERR_UNREGISTERED_GRID, ERR_INVERTED_AXIS, ERR_ANISOTROPIC_SHEAR)
7. GSD (ERR_RESOLUTION_INCOMPATIBLE)
8. Data Integrity & NaN/Inf / Cloud Cover (ERR_CORRUPT_PAYLOAD)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class CompatibilityGateResult:
    """Detailed evaluation result from the Physical Compatibility Gate C(t, o)."""
    is_compatible: bool
    failure_reasons: List[str] = field(default_factory=list)
    failure_codes: List[str] = field(default_factory=list)
    needs_resampling: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_compatible": self.is_compatible,
            "failure_reasons": self.failure_reasons,
            "failure_codes": self.failure_codes,
            "needs_resampling": self.needs_resampling,
            "details": self.details
        }


def get_affine_matrices(gt: Union[List[float], Tuple[float, ...], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 6-element GDAL GeoTransform to 2x2 linear matrix A and 2x1 origin vector t."""
    # GDAL GT: [origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height]
    t = np.array([gt[0], gt[3]], dtype=np.float64)
    A = np.array([
        [gt[1], gt[2]],
        [gt[4], gt[5]]
    ], dtype=np.float64)
    return A, t


def compute_relative_affine(
    gt1: Union[List[float], Tuple[float, ...], np.ndarray],
    gt2: Union[List[float], Tuple[float, ...], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative affine linear matrix M_rel = A1^{-1} A2 and relative offset d_rel = A1^{-1}(t2 - t1)."""
    A1, t1 = get_affine_matrices(gt1)
    A2, t2 = get_affine_matrices(gt2)

    A1_inv = np.linalg.inv(A1)
    M_rel = A1_inv @ A2
    d_rel = A1_inv @ (t2 - t1)
    return M_rel, d_rel


def compute_svd_metrics(M_rel: np.ndarray) -> Dict[str, float]:
    """Compute SVD singular values, condition number, determinant, and rotation angle."""
    U, S, Vt = np.linalg.svd(M_rel)
    sigma1, sigma2 = float(S[0]), float(S[1])
    det = float(np.linalg.det(M_rel))
    kappa = float(sigma1 / (sigma2 + 1e-12))
    theta_rel = float(np.arctan2(M_rel[1, 0] - M_rel[0, 1], M_rel[0, 0] + M_rel[1, 1]))
    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "det": det,
        "kappa": kappa,
        "theta_deg": float(np.degrees(theta_rel))
    }


def compute_full_footprint_displacements(
    gt1: Union[List[float], Tuple[float, ...], np.ndarray],
    gt2: Union[List[float], Tuple[float, ...], np.ndarray],
    width: int = 120,
    height: int = 120
) -> Dict[str, Any]:
    """
    Map the four corners of raster 2 into raster 1 pixel space,
    and compute physical Euclidean displacement relative to the reference grid coordinate:
        delta_k = ||p_1(C_k) - C_k||_2 = ||(M_rel - I) C_k + d_rel||_2
    """
    M_rel, d_rel = compute_relative_affine(gt1, gt2)
    corners = np.array([
        [0.0, 0.0],                   # C_00 (origin)
        [float(width), 0.0],           # C_W0 (top-right)
        [0.0, float(height)],          # C_0H (bottom-left)
        [float(width), float(height)]  # C_WH (bottom-right)
    ], dtype=np.float64)

    mapped_corners = (M_rel @ corners.T).T + d_rel
    residuals = mapped_corners - corners
    euclidean_displacements = np.sqrt(np.sum(residuals**2, axis=1))

    return {
        "corners": corners,
        "mapped_corners": mapped_corners,
        "residuals": residuals,
        "displacements": euclidean_displacements,
        "origin_displacement": float(euclidean_displacements[0]),
        "max_footprint_displacement": float(np.max(euclidean_displacements)),
        "mean_footprint_displacement": float(np.mean(euclidean_displacements))
    }


def evaluate_affine_alignment_gate(
    gt1: Union[List[float], Tuple[float, ...], np.ndarray],
    gt2: Union[List[float], Tuple[float, ...], np.ndarray],
    width: int = 120,
    height: int = 120,
    shape1: Optional[Tuple[int, ...]] = None,
    shape2: Optional[Tuple[int, ...]] = None,
    crs1: Optional[str] = None,
    crs2: Optional[str] = None,
    max_disp_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluate full-footprint affine gate verifying CRS identity, dimension equality,
    structural SVD properties, and corner displacements.
    """
    passed = True
    failure_reasons = []
    failure_codes = []

    if crs1 is not None and crs2 is not None and crs1 != crs2:
        passed = False
        failure_codes.append("ERR_CRS_MISMATCH")
        failure_reasons.append(f"ERR_CRS_MISMATCH: Reference CRS '{crs1}' != target CRS '{crs2}'")

    if shape1 is not None and shape2 is not None and shape1 != shape2:
        passed = False
        failure_codes.append("ERR_DIMENSION_MISMATCH")
        failure_reasons.append(f"ERR_DIMENSION_MISMATCH: Reference shape {shape1} != target shape {shape2}")

    w = shape2[1] if shape2 is not None else width
    h = shape2[0] if shape2 is not None else height

    M_rel, d_rel = compute_relative_affine(gt1, gt2)
    svd = compute_svd_metrics(M_rel)
    disp = compute_full_footprint_displacements(gt1, gt2, width=w, height=h)

    if svd["det"] <= 0:
        passed = False
        failure_codes.append("ERR_INVERTED_AXIS")
        failure_reasons.append("ERR_INVERTED_AXIS: Negative determinant indicates handedness inversion/reflection")
    elif abs(svd["det"] - 1.0) > 0.02:
        passed = False
        failure_codes.append("ERR_SCALE_DISTORTION")
        failure_reasons.append(f"ERR_SCALE_DISTORTION: Determinant {svd['det']:.4f} deviates >2% from unity")

    if abs(svd["sigma1"] - 1.0) > 0.01 or abs(svd["sigma2"] - 1.0) > 0.01:
        passed = False
        if "ERR_SCALE_DISTORTION" not in failure_codes:
            failure_codes.append("ERR_SCALE_DISTORTION")
        failure_reasons.append(f"ERR_SCALE_DISTORTION: Singular values ({svd['sigma1']:.4f}, {svd['sigma2']:.4f}) deviate >1%")

    if svd["kappa"] > 1.01:
        passed = False
        failure_codes.append("ERR_ANISOTROPIC_SHEAR")
        failure_reasons.append(f"ERR_ANISOTROPIC_SHEAR: Condition number {svd['kappa']:.4f} exceeds 1.01")

    if abs(svd["theta_deg"]) > 0.28:
        passed = False
        failure_codes.append("ERR_ROTATION_MISMATCH")
        failure_reasons.append(f"ERR_ROTATION_MISMATCH: Relative rotation {svd['theta_deg']:.3f} deg exceeds 0.28 deg")

    needs_resampling = False
    if disp["max_footprint_displacement"] > max_disp_threshold:
        passed = False
        needs_resampling = True
        failure_codes.append("ERR_UNREGISTERED_GRID")
        failure_reasons.append(
            f"ERR_UNREGISTERED_GRID: Maximum footprint displacement {disp['max_footprint_displacement']:.4f} px "
            f"exceeds engineering limit {max_disp_threshold} px"
        )

    has_fatal_structural_error = any(
        c in ("ERR_INVERTED_AXIS", "ERR_ANISOTROPIC_SHEAR", "ERR_CRS_MISMATCH", "ERR_DIMENSION_MISMATCH")
        for c in failure_codes
    )
    if has_fatal_structural_error:
        needs_resampling = False

    return {
        "passed": passed,
        "needs_resampling": needs_resampling,
        "failure_reasons": failure_reasons,
        "failure_codes": failure_codes,
        "svd": svd,
        "displacements": disp
    }


def evaluate_physical_compatibility_gate(
    task: str,
    payload: Any,
    details: Optional[Dict[str, Any]] = None
) -> CompatibilityGateResult:
    """
    Programmatic Physical Compatibility Gate C(t, o) in {0, 1} inspecting 8 physical dimensions.
    Returns CompatibilityGateResult with is_compatible=True/False and structured diagnostic error codes.
    """
    failure_reasons = []
    failure_codes = []
    meta = details or {}

    # Extract metadata properties if payload is a dict
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k not in meta:
                meta[k] = v

    # 1. Modality & Sensor check
    if task == "LAND_COVER_CLASSIFICATION":
        # Requires either precomputed features or optical+sar pair
        if isinstance(payload, dict):
            if "features" in payload:
                feats = np.asarray(payload["features"])
                if np.isnan(feats).any() or np.isinf(feats).any():
                    failure_codes.append("ERR_CORRUPT_PAYLOAD")
                    failure_reasons.append("ERR_CORRUPT_PAYLOAD: Feature vector contains NaN or Inf values")
            elif "s1_path" in payload or "s2_path" in payload:
                if not ("s1_path" in payload and "s2_path" in payload):
                    failure_codes.append("ERR_MISSING_MODALITY")
                    failure_reasons.append("ERR_MISSING_MODALITY: Multimodal Land Cover requires optical + SAR observations")
            elif "t1" in payload and "t2" in payload:
                # Temporal pair attached to Land Cover -> Incompatible modality!
                failure_codes.append("ERR_MISSING_MODALITY")
                failure_reasons.append("ERR_MISSING_MODALITY: Land Cover requires single scene / feature vector, not temporal pair")
        elif isinstance(payload, np.ndarray):
            if payload.ndim == 1:
                if np.isnan(payload).any() or np.isinf(payload).any():
                    failure_codes.append("ERR_CORRUPT_PAYLOAD")
                    failure_reasons.append("ERR_CORRUPT_PAYLOAD: Feature vector contains NaN or Inf values")
            elif payload.ndim in (2, 3):
                # Raw single image lacks SAR or feature preprocessing
                failure_codes.append("ERR_MISSING_MODALITY")
                failure_reasons.append("ERR_MISSING_MODALITY: Land Cover specialist requires multimodal features or S1+S2 pair")
        elif isinstance(payload, (tuple, list)) and len(payload) == 2:
            failure_codes.append("ERR_MISSING_MODALITY")
            failure_reasons.append("ERR_MISSING_MODALITY: Temporal pair attached to Land Cover classification")

    elif task in ("CHANGE_DETECTION", "CHANGE_VQA"):
        # Requires valid temporal pair
        if isinstance(payload, np.ndarray) and payload.ndim in (2, 3):
            failure_codes.append("ERR_MISSING_MODALITY")
            failure_reasons.append("ERR_MISSING_MODALITY: Change detection requires bitemporal image pair (N=2), got single image")
        elif isinstance(payload, dict) and "features" in payload:
            failure_codes.append("ERR_MISSING_MODALITY")
            failure_reasons.append("ERR_MISSING_MODALITY: Change detection requires bitemporal image pair, got 1D feature vector")
        elif isinstance(payload, dict) and ("t1" in payload and "t2" in payload):
            t1 = payload["t1"]
            t2 = payload["t2"]
            if isinstance(t1, np.ndarray) and isinstance(t2, np.ndarray):
                if t1.shape != t2.shape:
                    failure_codes.append("ERR_DIMENSION_MISMATCH")
                    failure_reasons.append(f"ERR_DIMENSION_MISMATCH: Temporal pair dimension mismatch {t1.shape} vs {t2.shape}")
                if np.isnan(t1).any() or np.isnan(t2).any() or np.isinf(t1).any() or np.isinf(t2).any():
                    failure_codes.append("ERR_CORRUPT_PAYLOAD")
                    failure_reasons.append("ERR_CORRUPT_PAYLOAD: Temporal pair contains NaN or Inf values")
                if payload.get("product_type") == "s2_l2a" and t1.dtype == np.float32 and np.max(t1) <= 1.05:
                    failure_codes.append("ERR_CORRUPT_PAYLOAD")
                    failure_reasons.append("ERR_CORRUPT_PAYLOAD: Sentinel-2 L2A product expects raw DN [0, 10000], got normalized float [0.0, 1.0] (double scaling trap)")
                if "spatial_shape" not in meta:
                    meta["spatial_shape"] = t1.shape[:2]
        elif isinstance(payload, (tuple, list)) and len(payload) == 2:
            im1, im2 = payload[0], payload[1]
            if isinstance(im1, np.ndarray) and isinstance(im2, np.ndarray):
                if im1.shape != im2.shape:
                    failure_codes.append("ERR_DIMENSION_MISMATCH")
                    failure_reasons.append(f"ERR_DIMENSION_MISMATCH: Temporal pair shape mismatch {im1.shape} vs {im2.shape}")
                if np.isnan(im1).any() or np.isnan(im2).any() or np.isinf(im1).any() or np.isinf(im2).any():
                    failure_codes.append("ERR_CORRUPT_PAYLOAD")
                    failure_reasons.append("ERR_CORRUPT_PAYLOAD: Temporal pair contains NaN or Inf values")
                if "spatial_shape" not in meta:
                    meta["spatial_shape"] = im1.shape[:2]

    elif task == "OPTICAL_SAR_FUSION":
        if isinstance(payload, dict):
            if not ("s1_path" in payload and "s2_path" in payload) and "features" not in payload:
                failure_codes.append("ERR_MISSING_MODALITY")
                failure_reasons.append("ERR_MISSING_MODALITY: Optical-SAR Fusion requires simultaneous S1 and S2 inputs")
        else:
            failure_codes.append("ERR_MISSING_MODALITY")
            failure_reasons.append("ERR_MISSING_MODALITY: Optical-SAR Fusion requires SAR and optical observations")

    # 2. Temporal Revisit / Window constraints
    if "temporal_gap_days" in meta:
        gap = float(meta["temporal_gap_days"])
        if task in ("CHANGE_DETECTION", "CHANGE_VQA"):
            if gap < 5.0:
                failure_codes.append("ERR_TEMPORAL_INVALID")
                failure_reasons.append(f"ERR_TEMPORAL_INVALID: Revisit interval {gap}d < 5d Sentinel-2 minimum revisit")
            elif gap > 730.0:
                failure_codes.append("ERR_TEMPORAL_INVALID")
                failure_reasons.append(f"ERR_TEMPORAL_INVALID: Revisit interval {gap}d > 730d (2 years) baseline drift limit")
        elif task == "OPTICAL_SAR_FUSION":
            is_dynamic = meta.get("is_dynamic", True)
            if is_dynamic and gap > (72.0 / 24.0):
                failure_codes.append("ERR_TEMPORAL_INVALID")
                failure_reasons.append(f"ERR_TEMPORAL_INVALID: Optical-SAR temporal offset {gap*24:.1f}h exceeds dynamic 72h limit")
            elif not is_dynamic and gap > 10.0:
                failure_codes.append("ERR_TEMPORAL_INVALID")
                failure_reasons.append(f"ERR_TEMPORAL_INVALID: Optical-SAR temporal offset {gap:.1f}d exceeds static 10d limit")

    # 3. Spectral Bands
    if "spectral_bands" in meta:
        bands = set(meta["spectral_bands"])
        if task == "LAND_COVER_CLASSIFICATION":
            required_bands = {"B02", "B03", "B04", "B08"}
            if not required_bands.issubset(bands):
                failure_codes.append("ERR_INSUFFICIENT_BANDS")
                failure_reasons.append(f"ERR_INSUFFICIENT_BANDS: Missing required multispectral bands: {required_bands - bands}")

    # 4. Spatial Dimensions
    if "spatial_shape" in meta and meta["spatial_shape"] is not None:
        shape = meta["spatial_shape"]
        h = shape[0] if len(shape) > 0 else 0
        w = shape[1] if len(shape) > 1 else 0
        min_dim = 100 if "CHANGE" in task else 120
        if h < min_dim or w < min_dim:
            failure_codes.append("ERR_DIMENSION_TOO_SMALL")
            failure_reasons.append(f"ERR_DIMENSION_TOO_SMALL: Dimension ({h}, {w}) below minimum ({min_dim}, {min_dim})")

    # 5 & 6. Geotransform / Affine Alignment Gate
    if "gt1" in meta and "gt2" in meta:
        gt_res = evaluate_affine_alignment_gate(
            gt1=meta["gt1"],
            gt2=meta["gt2"],
            width=meta.get("width", 120),
            height=meta.get("height", 120),
            shape1=meta.get("shape1"),
            shape2=meta.get("shape2"),
            crs1=meta.get("crs1"),
            crs2=meta.get("crs2"),
            max_disp_threshold=meta.get("max_disp_threshold", 0.05)
        )
        if not gt_res["passed"]:
            for code in gt_res["failure_codes"]:
                if code not in failure_codes:
                    failure_codes.append(code)
            failure_reasons.extend(gt_res["failure_reasons"])

    # 7. GSD Resolution
    if "gsd_m" in meta:
        gsd = float(meta["gsd_m"])
        if gsd < 8.0 or gsd > 12.0:
            failure_codes.append("ERR_RESOLUTION_INCOMPATIBLE")
            failure_reasons.append(f"ERR_RESOLUTION_INCOMPATIBLE: GSD {gsd:.1f}m outside operational window [8.0m, 12.0m]")

    # 8. Cloud Cover limits
    if "cloud_fraction" in meta:
        cf = float(meta["cloud_fraction"])
        if task == "LAND_COVER_CLASSIFICATION" and cf > 0.20:
            failure_codes.append("ERR_CORRUPT_PAYLOAD")
            failure_reasons.append(f"ERR_CORRUPT_PAYLOAD: Cloud cover {cf*100:.1f}% exceeds 20.0% operational limit for land cover")
        elif task in ("CHANGE_DETECTION", "CHANGE_VQA") and cf > 0.10:
            failure_codes.append("ERR_CORRUPT_PAYLOAD")
            failure_reasons.append(f"ERR_CORRUPT_PAYLOAD: Cloud cover {cf*100:.1f}% exceeds 10.0% operational limit for change detection")

    # Data dtype / scaling
    if isinstance(payload, np.ndarray) and payload.dtype == np.float32:
        if "product_type" in meta and meta["product_type"] == "unscaled_raw_float":
            if np.max(payload) > 1.05 and np.max(payload) < 255.0:
                failure_codes.append("ERR_CORRUPT_PAYLOAD")
                failure_reasons.append("ERR_CORRUPT_PAYLOAD: Float values not normalized to [0.0, 1.0]")

    is_compat = len(failure_reasons) == 0
    return CompatibilityGateResult(
        is_compatible=is_compat,
        failure_reasons=failure_reasons,
        failure_codes=failure_codes,
        needs_resampling=("ERR_UNREGISTERED_GRID" in failure_codes),
        details=meta
    )
