"""
Temporal Pair Preprocessing and Validation for SatQuery AI.

Enforces strict input integrity safeguards before multitemporal analysis:
- Compatible dimensions and channel counts
- Spatial alignment
- Finite value verification and NoData masking
- Valid pixel coverage calculation
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import numpy as np
from PIL import Image


@dataclass
class TemporalPairValidation:
    """Outcome of temporal pair validation."""
    is_valid: bool
    error_message: Optional[str] = None
    t1_array: Optional[np.ndarray] = None
    t2_array: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    height: int = 0
    width: int = 0
    channels: int = 0
    total_pixels: int = 0
    valid_pixels: int = 0
    valid_fraction: float = 0.0


def _load_image_array(img_input: Any) -> np.ndarray:
    """Load an image from filepath, PIL Image, or numpy array."""
    if isinstance(img_input, (str, Path)):
        p = Path(img_input)
        if not p.exists():
            raise FileNotFoundError(f"Temporal image file not found: {p}")
        with Image.open(p) as img:
            arr = np.array(img).astype(np.float32)
    elif isinstance(img_input, Image.Image):
        arr = np.array(img_input).astype(np.float32)
    elif isinstance(img_input, np.ndarray):
        arr = img_input.astype(np.float32, copy=False)
    else:
        raise TypeError(f"Unsupported temporal input type: {type(img_input)}. Expected Path, str, PIL.Image, or np.ndarray.")

    # Standardize dimensions to (H, W, C)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        # If channels-first (e.g. C, H, W where C <= 30 and H > C and W > C)
        if arr.shape[0] <= 30 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]:
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")

    return arr


def validate_temporal_pair(
    t1_input: Any,
    t2_input: Any,
    nodata_value: Optional[float] = None,
    min_valid_fraction: float = 0.05
) -> TemporalPairValidation:
    """
    Validate and align two temporal observations (T1, T2).
    
    Args:
        t1_input: First observation (filepath, PIL image, or numpy array).
        t2_input: Second observation (filepath, PIL image, or numpy array).
        nodata_value: Optional sensor-specific NoData marker (e.g. 0, -9999).
        min_valid_fraction: Minimum required fraction of valid finite pixels (default 5%).
        
    Returns:
        TemporalPairValidation dataclass with clean arrays and valid pixel mask.
    """
    if t1_input is None or t2_input is None:
        return TemporalPairValidation(
            is_valid=False,
            error_message="Both T1 (baseline) and T2 (subsequent) observations must be provided for change detection."
        )

    try:
        t1 = _load_image_array(t1_input)
        t2 = _load_image_array(t2_input)
    except Exception as e:
        return TemporalPairValidation(
            is_valid=False,
            error_message=f"Failed to load temporal pair observations: {str(e)}"
        )

    # 1. Dimensional Compatibility
    if t1.shape != t2.shape:
        return TemporalPairValidation(
            is_valid=False,
            error_message=(
                f"Dimension mismatch between temporal observations: "
                f"T1 is {t1.shape} while T2 is {t2.shape}. Both observations must be co-registered with identical dimensions."
            )
        )

    height, width, channels = t1.shape
    total_pixels = height * width

    # 2. Finite Values & NoData Masking
    # Pixel is valid if all channels are finite in both T1 and T2
    valid_t1 = np.all(np.isfinite(t1), axis=-1)
    valid_t2 = np.all(np.isfinite(t2), axis=-1)
    valid_mask = valid_t1 & valid_t2

    if nodata_value is not None:
        nodata_t1 = np.all(t1 == nodata_value, axis=-1)
        nodata_t2 = np.all(t2 == nodata_value, axis=-1)
        valid_mask = valid_mask & (~nodata_t1) & (~nodata_t2)

    valid_pixels = int(np.sum(valid_mask))
    valid_fraction = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0

    # 3. Minimum Valid Pixel Safeguard
    if valid_fraction < min_valid_fraction:
        return TemporalPairValidation(
            is_valid=False,
            error_message=(
                f"Insufficient valid pixels: only {valid_fraction*100:.2f}% of pixels are valid/finite "
                f"(minimum threshold: {min_valid_fraction*100:.2f}%)."
            ),
            height=height,
            width=width,
            channels=channels,
            total_pixels=total_pixels,
            valid_pixels=valid_pixels,
            valid_fraction=valid_fraction
        )

    return TemporalPairValidation(
        is_valid=True,
        t1_array=t1,
        t2_array=t2,
        valid_mask=valid_mask,
        height=height,
        width=width,
        channels=channels,
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        valid_fraction=valid_fraction
    )
