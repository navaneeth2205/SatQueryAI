"""
Utility modules for SatQuery AI.
"""

from satquery.utils.temporal import validate_temporal_pair, TemporalPairValidation
from satquery.utils.metadata import (
    InputObservationMetadata,
    extract_observation_metadata,
    TASK_COMPATIBILITY_MAP
)
from satquery.utils.s2_metadata import (
    S2Metadata,
    parse_s2_metadata,
    convert_dn_to_reflectance,
    resample_s2_20m_to_10m,
    harmonize_modis_patch_labels,
    CANONICAL_S2_BANDS
)
from satquery.utils.perturbations import (
    generate_continuous_noise_field,
    generate_nested_cloud_mask,
    generate_nested_cloud_mask_suite,
    apply_cloud_occlusion_land_cover,
    apply_cloud_occlusion_change_detection,
    apply_haze_land_cover,
    apply_haze_change_detection,
    generate_crn_noise_tensors,
    apply_noise_land_cover,
    apply_noise_change_detection,
    standardize_optical_tensor,
    normalize_imagenet_rgb,
    validate_track43_reflectance_bounds,
    CLOUD_SEVERITY_FRACTIONS,
    HAZE_TRANSMITTANCES,
    NOISE_SEVERITY_SIGMAS,
)

__all__ = [
    "validate_temporal_pair",
    "TemporalPairValidation",
    "InputObservationMetadata",
    "extract_observation_metadata",
    "TASK_COMPATIBILITY_MAP",
    "S2Metadata",
    "parse_s2_metadata",
    "convert_dn_to_reflectance",
    "resample_s2_20m_to_10m",
    "harmonize_modis_patch_labels",
    "CANONICAL_S2_BANDS",
    "generate_continuous_noise_field",
    "generate_nested_cloud_mask",
    "generate_nested_cloud_mask_suite",
    "apply_cloud_occlusion_land_cover",
    "apply_cloud_occlusion_change_detection",
    "apply_haze_land_cover",
    "apply_haze_change_detection",
    "generate_crn_noise_tensors",
    "apply_noise_land_cover",
    "apply_noise_change_detection",
    "standardize_optical_tensor",
    "normalize_imagenet_rgb",
    "validate_track43_reflectance_bounds",
    "CLOUD_SEVERITY_FRACTIONS",
    "HAZE_TRANSMITTANCES",
    "NOISE_SEVERITY_SIGMAS",
]
