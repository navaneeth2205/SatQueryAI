"""
Tests for Full-Footprint Affine Geospatial Alignment Verification Gate.
SatQuery AI (SIH26167) — Milestone M3.0 Experimental Protocol.

Verifies:
1. Corner displacement calculation across the full raster footprint (C_00, C_W0, C_0H, C_WH).
2. Failure of 1% scale drift to maintain sub-pixel alignment across 120px footprint (1.2px displacement).
3. Failure of 0.28 deg rotation to maintain sub-pixel alignment at footprint corner (0.83px displacement).
4. Rejection of axis reflection / handedness flip (det(M_rel) < 0).
5. Detection of shear distortion via SVD condition number kappa(M_rel) > 1.01.
6. Pure sub-pixel translation gating (<= 0.05px passes, > 0.05px triggers coregistration).
7. Flagging of mandatory resampling requirement prior to specialist execution.
"""

import numpy as np
import pytest


def get_affine_matrices(gt):
    """Convert 6-element GDAL GeoTransform to 2x2 linear matrix A and 2x1 origin vector t."""
    # GDAL GT: [origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height]
    t = np.array([gt[0], gt[3]], dtype=np.float64)
    A = np.array([
        [gt[1], gt[2]],
        [gt[4], gt[5]]
    ], dtype=np.float64)
    return A, t


def compute_relative_affine(gt1, gt2):
    """Compute relative affine linear matrix M_rel and relative offset d_rel."""
    A1, t1 = get_affine_matrices(gt1)
    A2, t2 = get_affine_matrices(gt2)
    
    A1_inv = np.linalg.inv(A1)
    M_rel = A1_inv @ A2
    d_rel = A1_inv @ (t2 - t1)
    return M_rel, d_rel


def compute_svd_metrics(M_rel):
    """Compute SVD singular values, condition number, determinant, and rotation angle."""
    U, S, Vt = np.linalg.svd(M_rel)
    sigma1, sigma2 = S[0], S[1]
    det = np.linalg.det(M_rel)
    kappa = sigma1 / (sigma2 + 1e-12)
    theta_rel = np.arctan2(M_rel[1, 0] - M_rel[0, 1], M_rel[0, 0] + M_rel[1, 1])
    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "det": det,
        "kappa": kappa,
        "theta_deg": np.degrees(theta_rel)
    }


def compute_full_footprint_displacements(gt1, gt2, width=120, height=120):
    """
    Map the four corners of raster 2 into raster 1 pixel space,
    and compute residual Euclidean displacement relative to the nearest integer grid point.
    """
    M_rel, d_rel = compute_relative_affine(gt1, gt2)
    corners = np.array([
        [0.0, 0.0],          # C_00 (origin)
        [float(width), 0.0],  # C_W0 (top-right)
        [0.0, float(height)], # C_0H (bottom-left)
        [float(width), float(height)] # C_WH (bottom-right)
    ], dtype=np.float64)
    
    mapped_corners = (M_rel @ corners.T).T + d_rel
    int_corners = np.round(mapped_corners)
    residuals = mapped_corners - int_corners
    euclidean_displacements = np.sqrt(np.sum(residuals**2, axis=1))
    
    return {
        "corners": corners,
        "mapped_corners": mapped_corners,
        "displacements": euclidean_displacements,
        "origin_displacement": euclidean_displacements[0],
        "max_footprint_displacement": np.max(euclidean_displacements),
        "mean_footprint_displacement": np.mean(euclidean_displacements)
    }


def evaluate_affine_alignment_gate(gt1, gt2, width=120, height=120, max_disp_threshold=0.05):
    """
    Full-footprint affine gate verifying both structural SVD properties
    and maximum footprint displacement.
    """
    M_rel, d_rel = compute_relative_affine(gt1, gt2)
    svd = compute_svd_metrics(M_rel)
    disp = compute_full_footprint_displacements(gt1, gt2, width, height)
    
    passed = True
    failure_reasons = []
    
    # 1. Determinant / Handedness check
    if svd["det"] <= 0:
        passed = False
        failure_reasons.append("ERR_INVERTED_AXIS: Negative determinant indicates handedness inversion/reflection")
    elif abs(svd["det"] - 1.0) > 0.02:
        passed = False
        failure_reasons.append(f"ERR_SCALE_DISTORTION: Determinant {svd['det']:.4f} deviates >2% from unity")
        
    # 2. Scale tolerance (SVD singular values)
    if abs(svd["sigma1"] - 1.0) > 0.01 or abs(svd["sigma2"] - 1.0) > 0.01:
        passed = False
        failure_reasons.append(f"ERR_SCALE_DISTORTION: Singular values ({svd['sigma1']:.4f}, {svd['sigma2']:.4f}) deviate >1%")
        
    # 3. Anisotropy / Aspect ratio
    if svd["kappa"] > 1.01:
        passed = False
        failure_reasons.append(f"ERR_ANISOTROPIC_SHEAR: Condition number {svd['kappa']:.4f} exceeds 1.01")
        
    # 4. Relative rotation
    if abs(svd["theta_deg"]) > 0.28:
        passed = False
        failure_reasons.append(f"ERR_ROTATION_MISMATCH: Relative rotation {svd['theta_deg']:.3f} deg exceeds 0.28 deg")
        
    # 5. Full footprint displacement check (critical invariant)
    needs_resampling = False
    if disp["max_footprint_displacement"] > max_disp_threshold:
        passed = False
        needs_resampling = True
        failure_reasons.append(
            f"ERR_UNREGISTERED_GRID: Maximum footprint displacement {disp['max_footprint_displacement']:.4f} px "
            f"exceeds engineering limit {max_disp_threshold} px"
        )
        
    return {
        "passed": passed,
        "needs_resampling": needs_resampling,
        "failure_reasons": failure_reasons,
        "svd": svd,
        "displacements": disp
    }


# =====================================================================
# Unit Tests
# =====================================================================

def test_perfect_alignment_zero_displacement():
    """Identical geotransforms must yield zero displacement across all 4 corners."""
    gt_ref = (500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    result = evaluate_affine_alignment_gate(gt_ref, gt_ref, width=120, height=120)
    
    assert result["passed"] is True
    assert result["needs_resampling"] is False
    assert result["displacements"]["max_footprint_displacement"] == pytest.approx(0.0, abs=1e-6)
    assert result["svd"]["det"] == pytest.approx(1.0, abs=1e-6)
    assert result["svd"]["kappa"] == pytest.approx(1.0, abs=1e-6)


def test_scale_drift_exceeds_full_footprint_displacement():
    """
    1% scale difference at origin has zero displacement, but accumulates
    to 1.2 px at corner (120, 0) and 1.7 px at (120, 120), failing the gate.
    """
    gt1 = (500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    # Raster 2 has 1% larger pixels (10.1m) with identical origin
    gt2 = (500000.0, 10.1, 0.0, 4500000.0, 0.0, -10.1)
    
    disp = compute_full_footprint_displacements(gt1, gt2, width=120, height=120)
    
    # Origin displacement is 0.0
    assert disp["origin_displacement"] == pytest.approx(0.0, abs=1e-6)
    # But corner displacement at width=120 is 1.2 pixels! (120 * 1.01 = 121.2 -> residual 0.2)
    # Across (120, 120): dx = 0.2, dy = -0.2 -> Euclidean = sqrt(0.08) ~ 0.283 px
    assert disp["max_footprint_displacement"] > 0.05
    
    gate = evaluate_affine_alignment_gate(gt1, gt2, width=120, height=120)
    assert gate["passed"] is False
    assert gate["needs_resampling"] is True
    assert any("ERR_UNREGISTERED_GRID" in r for r in gate["failure_reasons"])


def test_rotation_drift_exceeds_full_footprint_displacement():
    """
    0.28 deg rotation with zero origin translation produces ~0.83 px displacement
    at the corner (120, 120), failing the full footprint gate.
    """
    gt1 = (500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    
    theta_rad = np.radians(0.28)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    # Rotated geotransform: A2 = R * A1
    A2 = np.array([[cos_t, -sin_t], [sin_t, cos_t]]) @ np.array([[10.0, 0.0], [0.0, -10.0]])
    gt2 = (500000.0, A2[0, 0], A2[0, 1], 4500000.0, A2[1, 0], A2[1, 1])
    
    disp = compute_full_footprint_displacements(gt1, gt2, width=120, height=120)
    # Origin displacement is 0
    assert disp["origin_displacement"] == pytest.approx(0.0, abs=1e-6)
    # Corner displacement at (120, 120) exceeds 0.5 pixels
    assert disp["max_footprint_displacement"] > 0.50
    
    gate = evaluate_affine_alignment_gate(gt1, gt2, width=120, height=120)
    assert gate["passed"] is False
    assert gate["needs_resampling"] is True


def test_reflection_handedness_flip_detected():
    """Flipped coordinate axis (negative determinant) triggers ERR_INVERTED_AXIS."""
    gt1 = (500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    # gt2 has flipped Y direction (positive instead of negative)
    gt2 = (500000.0, 10.0, 0.0, 4500000.0, 0.0, 10.0)
    
    gate = evaluate_affine_alignment_gate(gt1, gt2, width=120, height=120)
    assert gate["passed"] is False
    assert any("ERR_INVERTED_AXIS" in r for r in gate["failure_reasons"])


def test_shear_distortion_detected():
    """Shear distortion causing condition number > 1.01 triggers ERR_ANISOTROPIC_SHEAR."""
    gt1 = (500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    # Non-zero off-diagonal shear
    gt2 = (500000.0, 10.0, 0.25, 4500000.0, 0.0, -10.0)
    
    gate = evaluate_affine_alignment_gate(gt1, gt2, width=120, height=120)
    assert gate["passed"] is False
    assert any("ERR_ANISOTROPIC_SHEAR" in r or "ERR_SCALE_DISTORTION" in r for r in gate["failure_reasons"])


def test_subpixel_pure_translation_gating():
    """Sub-pixel translation <= 0.05px passes; > 0.05px fails and triggers resampling."""
    gt1 = (500000.0, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    
    # 0.03 px offset (0.3m on 10m grid) -> should PASS
    gt_pass = (500000.3, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    res_pass = evaluate_affine_alignment_gate(gt1, gt_pass, width=120, height=120)
    assert res_pass["passed"] is True
    assert res_pass["needs_resampling"] is False
    assert res_pass["displacements"]["max_footprint_displacement"] == pytest.approx(0.03, abs=1e-4)
    
    # 0.08 px offset (0.8m on 10m grid) -> should FAIL
    gt_fail = (500000.8, 10.0, 0.0, 4500000.0, 0.0, -10.0)
    res_fail = evaluate_affine_alignment_gate(gt1, gt_fail, width=120, height=120)
    assert res_fail["passed"] is False
    assert res_fail["needs_resampling"] is True
    assert any("ERR_UNREGISTERED_GRID" in r for r in res_fail["failure_reasons"])
