"""
Atmospheric, Cloud Occlusion, and Sensor Noise Perturbation Operators for Phase 4 Track 4.3.

Implements resolution-aware, strictly nested, and Common Random Number (CRN) perturbation operators:
1. Strictly nested cloud occlusion:
   - Evaluated at specialist native resolutions: (120, 120) for Land Cover, (256, 256) for Change Detection.
   - Exact pixel count: K(f_c, H, W) = floor(f_c * H * W).
   - Generated from a single continuous 2D noise field per patch and resolution.
   - For Land Cover: applied identically across all 10 optical channels; SAR channels untouched.
   - For Change Detection: applied across all 3 RGB channels of post-event t2; pre-event t1 untouched.
2. Koschmieder atmospheric haze scattering:
   - Applied to optical channels with transmittance t in {0.90, 0.70, 0.50, 0.30}.
   - SAR channels and pre-event t1 remain untouched.
3. Common Random Numbers (CRN) Gaussian noise:
   - Generated in scaled [0, 1] pre-standardization domain.
   - Explicit tensor shapes: (10, 120, 120) for LC optical, (2, 120, 120) for LC SAR,
     (3, 256, 256) for CD t1, (3, 256, 256) for CD t2.
   - Noise drawn independently across channels and timepoints.
   - Scaled strictly by sigma in {0.01, 0.02, 0.05, 0.10} from identical base noise realizations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


# Frozen default parameters from perturbation_config.json
DEFAULT_CLOUD_ALBEDO: float = 0.95
DEFAULT_CLOUD_OPACITY: float = 0.90
DEFAULT_AIRLIGHT: float = 0.85

CLOUD_SEVERITY_FRACTIONS: List[float] = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
HAZE_TRANSMITTANCES: List[float] = [0.90, 0.70, 0.50, 0.30]
HAZE_SEVERITY_OPACITIES: List[float] = [0.10, 0.30, 0.50, 0.70]
NOISE_SEVERITY_SIGMAS: List[float] = [0.01, 0.02, 0.05, 0.10]


def generate_continuous_noise_field(
    height: int,
    width: int,
    seed: int
) -> np.ndarray:
    """
    Generate a continuous, spatially autocorrelated 2D fractal/Perlin-style noise field in [0, 1].
    Uses multi-scale bilinear octaves for smooth, organic cloud-like structures.
    """
    rng = np.random.RandomState(seed)
    field = np.zeros((height, width), dtype=np.float32)
    scales = [4, 8, 16, 32]
    weights = [0.50, 0.25, 0.125, 0.0625]

    for s, w in zip(scales, weights):
        h_coarse = max(2, height // s)
        w_coarse = max(2, width // s)
        coarse = rng.randn(1, 1, h_coarse, w_coarse).astype(np.float32)
        tensor_coarse = torch.from_numpy(coarse)
        tensor_up = F.interpolate(
            tensor_coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )
        field += w * tensor_up.squeeze().numpy()

    # Normalize to [0, 1]
    f_min = float(field.min())
    f_max = float(field.max())
    if f_max > f_min:
        field = (field - f_min) / (f_max - f_min)
    return field


def generate_nested_cloud_mask(
    height: int,
    width: int,
    severity_fraction: float,
    seed: int
) -> np.ndarray:
    """
    Generate a binary cloud mask M in {0, 1}^(H, W) for a single severity fraction.
    Exact pixel count K(f_c, H, W) = floor(f_c * H * W).
    """
    k = int(np.floor(severity_fraction * height * width))
    field = generate_continuous_noise_field(height, width, seed)
    flat = field.ravel()
    order = np.argsort(-flat, kind="stable")
    mask = np.zeros(height * width, dtype=np.uint8)
    mask[order[:k]] = 1
    return mask.reshape(height, width)


def generate_nested_cloud_mask_suite(
    height: int,
    width: int,
    severity_fractions: List[float],
    seed: int
) -> Dict[float, np.ndarray]:
    """
    Generate a dictionary of strictly nested cloud masks {f_c: M(f_c)}.
    Guarantees:
    1. Exact pixel count: sum(M(f_c)) == floor(f_c * H * W).
    2. Strict nesting: M(f_1) <= M(f_2) everywhere for f_1 < f_2.
    """
    field = generate_continuous_noise_field(height, width, seed)
    flat = field.ravel()
    order = np.argsort(-flat, kind="stable")

    masks: Dict[float, np.ndarray] = {}
    prev_mask: Optional[np.ndarray] = None

    for f_c in severity_fractions:
        k = int(np.floor(f_c * height * width))
        m = np.zeros(height * width, dtype=np.uint8)
        m[order[:k]] = 1
        m_2d = m.reshape(height, width)

        if prev_mask is not None:
            # Enforce proper subset nesting
            if not np.all(m_2d >= prev_mask):
                raise ValueError(f"Nesting violation detected at severity fraction {f_c}.")
        prev_mask = m_2d
        masks[f_c] = m_2d

    return masks


def apply_cloud_occlusion_land_cover(
    optical_tensor: np.ndarray,      # Shape (10, 120, 120) in [0, 1]
    sar_tensor: np.ndarray,          # Shape (2, 120, 120) in [0, 1]
    cloud_mask: np.ndarray,          # Shape (120, 120) binary {0, 1}
    cloud_opacity: float = DEFAULT_CLOUD_OPACITY,
    cloud_albedo: float = DEFAULT_CLOUD_ALBEDO
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply cloud occlusion to Land Cover specialist input.
    - Cloud mask applied identically to all 10 optical channels.
    - SAR channels (VH, VV) remain completely untouched (difference = 0.0).
    """
    if optical_tensor.shape != (10, 120, 120):
        raise ValueError(f"Expected optical tensor shape (10, 120, 120), got {optical_tensor.shape}.")
    if sar_tensor.shape != (2, 120, 120):
        raise ValueError(f"Expected SAR tensor shape (2, 120, 120), got {sar_tensor.shape}.")
    if cloud_mask.shape != (120, 120):
        raise ValueError(f"Expected cloud mask shape (120, 120), got {cloud_mask.shape}.")

    m_float = cloud_mask.astype(np.float32)
    # Blend optical channels: X_opt * (1 - alpha * M) + A_cloud * (alpha * M)
    blend_weight = cloud_opacity * m_float
    perturbed_opt = optical_tensor * (1.0 - blend_weight) + cloud_albedo * blend_weight
    perturbed_opt = np.clip(perturbed_opt, 0.0, 1.0)

    # SAR channels are unaffected by optical clouds
    untouched_sar = sar_tensor.copy()

    return perturbed_opt, untouched_sar


def apply_cloud_occlusion_change_detection(
    t1_rgb: np.ndarray,              # Shape (3, 256, 256) in [0, 1]
    t2_rgb: np.ndarray,              # Shape (3, 256, 256) in [0, 1]
    cloud_mask: np.ndarray,          # Shape (256, 256) binary {0, 1}
    cloud_opacity: float = DEFAULT_CLOUD_OPACITY,
    cloud_albedo: float = DEFAULT_CLOUD_ALBEDO
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply cloud occlusion to Change Detection specialist input.
    - Cloud mask applied across all 3 RGB channels of post-event t2.
    - Pre-event t1 remains completely untouched (difference = 0.0).
    """
    if t1_rgb.shape != (3, 256, 256):
        raise ValueError(f"Expected t1 RGB tensor shape (3, 256, 256), got {t1_rgb.shape}.")
    if t2_rgb.shape != (3, 256, 256):
        raise ValueError(f"Expected t2 RGB tensor shape (3, 256, 256), got {t2_rgb.shape}.")
    if cloud_mask.shape != (256, 256):
        raise ValueError(f"Expected cloud mask shape (256, 256), got {cloud_mask.shape}.")

    # Pre-event t1 is unaffected
    untouched_t1 = t1_rgb.copy()

    # Blend post-event t2
    m_float = cloud_mask.astype(np.float32)
    blend_weight = cloud_opacity * m_float
    perturbed_t2 = t2_rgb * (1.0 - blend_weight) + cloud_albedo * blend_weight
    perturbed_t2 = np.clip(perturbed_t2, 0.0, 1.0)

    return untouched_t1, perturbed_t2


def apply_haze_land_cover(
    optical_tensor: np.ndarray,      # Shape (10, 120, 120) in [0, 1]
    sar_tensor: np.ndarray,          # Shape (2, 120, 120) in [0, 1]
    transmittance: float,
    airlight: float = DEFAULT_AIRLIGHT
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Koschmieder atmospheric haze to Land Cover specialist input.
    - Transmittance applied to all 10 optical channels: X_opt * t + A_atm * (1 - t).
    - SAR channels remain completely untouched.
    """
    if optical_tensor.shape != (10, 120, 120):
        raise ValueError(f"Expected optical tensor shape (10, 120, 120), got {optical_tensor.shape}.")
    if sar_tensor.shape != (2, 120, 120):
        raise ValueError(f"Expected SAR tensor shape (2, 120, 120), got {sar_tensor.shape}.")

    perturbed_opt = optical_tensor * transmittance + airlight * (1.0 - transmittance)
    perturbed_opt = np.clip(perturbed_opt, 0.0, 1.0)
    return perturbed_opt, sar_tensor.copy()


def apply_haze_change_detection(
    t1_rgb: np.ndarray,              # Shape (3, 256, 256) in [0, 1]
    t2_rgb: np.ndarray,              # Shape (3, 256, 256) in [0, 1]
    transmittance: float,
    airlight: float = DEFAULT_AIRLIGHT
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Koschmieder atmospheric haze to Change Detection specialist input.
    - Transmittance applied across all 3 RGB channels of post-event t2.
    - Pre-event t1 remains completely untouched.
    """
    if t1_rgb.shape != (3, 256, 256):
        raise ValueError(f"Expected t1 RGB tensor shape (3, 256, 256), got {t1_rgb.shape}.")
    if t2_rgb.shape != (3, 256, 256):
        raise ValueError(f"Expected t2 RGB tensor shape (3, 256, 256), got {t2_rgb.shape}.")

    perturbed_t2 = t2_rgb * transmittance + airlight * (1.0 - transmittance)
    perturbed_t2 = np.clip(perturbed_t2, 0.0, 1.0)
    return t1_rgb.copy(), perturbed_t2


def generate_crn_noise_tensors(
    patch_id: int
) -> Dict[str, np.ndarray]:
    """
    Generate Common Random Number (CRN) base Gaussian noise tensors for all modalities.
    Uses independent seeds per modality/timepoint stream:
    - lc_optical: shape (10, 120, 120), seed = 10007 + 100 * patch_id
    - lc_sar: shape (2, 120, 120), seed = 20011 + 100 * patch_id
    - cd_t1: shape (3, 256, 256), seed = 30013 + 100 * patch_id
    - cd_t2: shape (3, 256, 256), seed = 40009 + 100 * patch_id
    Noise is drawn independently across channels and timepoints.
    """
    rng_lc_opt = np.random.RandomState(10007 + 100 * patch_id)
    rng_lc_sar = np.random.RandomState(20011 + 100 * patch_id)
    rng_cd_t1 = np.random.RandomState(30013 + 100 * patch_id)
    rng_cd_t2 = np.random.RandomState(40009 + 100 * patch_id)

    return {
        "lc_optical": rng_lc_opt.randn(10, 120, 120).astype(np.float32),
        "lc_sar": rng_lc_sar.randn(2, 120, 120).astype(np.float32),
        "cd_t1": rng_cd_t1.randn(3, 256, 256).astype(np.float32),
        "cd_t2": rng_cd_t2.randn(3, 256, 256).astype(np.float32),
    }


def apply_noise_land_cover(
    optical_tensor: np.ndarray,      # Shape (10, 120, 120) in [0, 1]
    sar_tensor: np.ndarray,          # Shape (2, 120, 120) in [0, 1]
    sigma: float,
    noise_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply scaled CRN Gaussian noise to Land Cover specialist optical and SAR inputs.
    Perturbations are strictly monotonic scalar magnifications of the base noise tensors:
    X' = clip(X + sigma * Z, 0.0, 1.0).
    """
    if optical_tensor.shape != (10, 120, 120):
        raise ValueError(f"Expected optical tensor shape (10, 120, 120), got {optical_tensor.shape}.")
    if sar_tensor.shape != (2, 120, 120):
        raise ValueError(f"Expected SAR tensor shape (2, 120, 120), got {sar_tensor.shape}.")

    z_opt = noise_dict["lc_optical"]
    z_sar = noise_dict["lc_sar"]

    perturbed_opt = np.clip(optical_tensor + sigma * z_opt, 0.0, 1.0)
    perturbed_sar = np.clip(sar_tensor + sigma * z_sar, 0.0, 1.0)

    return perturbed_opt, perturbed_sar


def apply_noise_change_detection(
    t1_rgb: np.ndarray,              # Shape (3, 256, 256) in [0, 1]
    t2_rgb: np.ndarray,              # Shape (3, 256, 256) in [0, 1]
    sigma: float,
    noise_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply scaled CRN Gaussian noise to Change Detection specialist bitemporal inputs.
    Perturbations are strictly monotonic scalar magnifications of the base noise tensors:
    X' = clip(X + sigma * Z, 0.0, 1.0).
    """
    if t1_rgb.shape != (3, 256, 256):
        raise ValueError(f"Expected t1 RGB tensor shape (3, 256, 256), got {t1_rgb.shape}.")
    if t2_rgb.shape != (3, 256, 256):
        raise ValueError(f"Expected t2 RGB tensor shape (3, 256, 256), got {t2_rgb.shape}.")

    z_t1 = noise_dict["cd_t1"]
    z_t2 = noise_dict["cd_t2"]

    perturbed_t1 = np.clip(t1_rgb + sigma * z_t1, 0.0, 1.0)
    perturbed_t2 = np.clip(t2_rgb + sigma * z_t2, 0.0, 1.0)

    return perturbed_t1, perturbed_t2
