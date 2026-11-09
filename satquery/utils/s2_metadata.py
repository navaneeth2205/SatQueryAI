"""
Sentinel-2 Level-2A Metadata Parsing, Baseline Reconciliation, and Resampling Utilities.

Implements Stage B preprocessing and provenance validation for Sentinel-2 optical data:
1. Parsing of processing baseline (PB < 04.00 vs PB >= 04.00) from:
   - Copernicus XML granule/product metadata (MTD_MSIL2A.xml, MTD_TL.xml)
   - Dictionary / JSON manifests
   - SAFE product identifier strings (e.g., _N0400_, _N0205_)
2. Extraction and application of per-band BOA_ADD_OFFSET:
   - Archival PB < 04.00: BOA_ADD_OFFSET = 0.0 for all bands
   - Modern PB >= 04.00: BOA_ADD_OFFSET = -1000.0 for all bands (or per-band list)
   - QUANTIFICATION_VALUE = 10,000.0
   - Formula: rho_lambda = (DN_lambda + BOA_ADD_OFFSET_lambda) / QUANTIFICATION_VALUE
3. Rejection of missing, incomplete, or inconsistent metadata.
4. Bilinear resampling of native 20m bands (B05, B06, B07, B8A, B11, B12) to 10m GSD (120x120),
   matching native 10m bands (B02, B03, B04, B08), and stacking in canonical order.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn.functional as F


# Canonical Sentinel-2 10-band optical order required by MultimodalResNet34
CANONICAL_S2_BANDS: List[str] = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
]

NATIVE_10M_BANDS: List[str] = ["B02", "B03", "B04", "B08"]
NATIVE_20M_BANDS: List[str] = ["B05", "B06", "B07", "B8A", "B11", "B12"]

NOMINAL_QUANTIFICATION_VALUE: float = 10000.0
NOMINAL_REFLECTANCE_RANGE: Tuple[float, float] = (-0.05, 1.20)


@dataclass
class S2Metadata:
    """
    Structured representation of Sentinel-2 Level-2A processing metadata.
    """
    processing_baseline: float
    boa_add_offset: Dict[str, float]
    quantification_value: float = NOMINAL_QUANTIFICATION_VALUE
    product_id: Optional[str] = None
    resampling_provenance: Optional[Dict[str, Any]] = None
    source_format: str = "dictionary"

    @property
    def is_archival(self) -> bool:
        """True if processing baseline is prior to PB 04.00 (e.g. Summer 2017 BigEarthNet)."""
        return self.processing_baseline < 4.0

    @property
    def is_modern(self) -> bool:
        """True if processing baseline is PB 04.00 or later (post-Jan 25, 2022)."""
        return self.processing_baseline >= 4.0

    def get_band_offset(self, band_name: str) -> float:
        """Get BOA_ADD_OFFSET for a specific band code."""
        if band_name not in self.boa_add_offset:
            # Default fallback based on baseline
            return -1000.0 if self.is_modern else 0.0
        return float(self.boa_add_offset[band_name])


def _extract_baseline_from_string(text: str) -> Optional[float]:
    """Extract processing baseline float from a string (e.g. '04.00', '4.0', or '_N0400_')."""
    # Pattern 1: SAFE product naming convention: _N0400_, _N0205_, _N0500_
    safe_match = re.search(r'_N(\d{2})(\d{2})_', text)
    if safe_match:
        major = int(safe_match.group(1))
        minor = int(safe_match.group(2))
        return float(f"{major}.{minor:02d}")

    # Pattern 2: Explicit decimal version e.g. '04.00' or '2.04' or '4.0'
    ver_match = re.search(r'(\d{1,2})\.(\d{1,2})', text)
    if ver_match:
        major = int(ver_match.group(1))
        minor = int(ver_match.group(2))
        return float(f"{major}.{minor:02d}")

    return None


def parse_s2_metadata(metadata_input: Union[str, Path, Dict[str, Any]]) -> S2Metadata:
    """
    Parse and validate Sentinel-2 Level-2A metadata from dictionary, XML, or product ID.
    
    Raises ValueError if metadata is missing, incomplete, or contains contradictory specifications.
    """
    if metadata_input is None:
        raise ValueError("Metadata input is None. Cannot determine Sentinel-2 processing baseline.")

    # --------------------------------------------------------------------------
    # Case 1: Dictionary / JSON input
    # --------------------------------------------------------------------------
    if isinstance(metadata_input, dict):
        if not metadata_input:
            raise ValueError("Metadata dictionary is empty. Missing processing baseline.")

        # Baseline resolution
        pb = None
        for key in ["processing_baseline", "baseline", "PROCESSING_BASELINE", "product_baseline"]:
            if key in metadata_input:
                raw_val = metadata_input[key]
                if isinstance(raw_val, (int, float)):
                    pb = float(raw_val)
                    break
                elif isinstance(raw_val, str):
                    pb = _extract_baseline_from_string(raw_val)
                    if pb is not None:
                        break

        # Check product_id as secondary resolution
        product_id = metadata_input.get("product_id") or metadata_input.get("PRODUCT_ID") or metadata_input.get("granule_id")
        if pb is None and product_id and isinstance(product_id, str):
            pb = _extract_baseline_from_string(product_id)

        if pb is None:
            raise ValueError(
                "Missing processing baseline in Sentinel-2 metadata dictionary. "
                "Must provide 'processing_baseline' (e.g. 2.04 or 4.00) or valid product_id."
            )

        # Quantification value
        qv = float(metadata_input.get("quantification_value", metadata_input.get("QUANTIFICATION_VALUE", NOMINAL_QUANTIFICATION_VALUE)))
        if qv <= 0.0:
            raise ValueError(f"Invalid QUANTIFICATION_VALUE: {qv}. Must be positive.")

        # BOA_ADD_OFFSET resolution
        offsets: Dict[str, float] = {}
        raw_offsets = metadata_input.get("boa_add_offset", metadata_input.get("BOA_ADD_OFFSET", None))

        if raw_offsets is not None:
            if isinstance(raw_offsets, (int, float)):
                scalar_val = float(raw_offsets)
                # Validation: if PB >= 4.0 but offset is explicitly 0.0 without explanation, flag inconsistency
                if pb >= 4.0 and scalar_val == 0.0 and metadata_input.get("force_zero_offset") is not True:
                    raise ValueError(
                        f"Inconsistent metadata: processing_baseline is {pb:.2f} (>= 04.00), "
                        f"but BOA_ADD_OFFSET is explicitly declared as 0.0. "
                        f"PB >= 04.00 products require radiometric offset -1000.0."
                    )
                for b in CANONICAL_S2_BANDS:
                    offsets[b] = scalar_val
            elif isinstance(raw_offsets, dict):
                for b in CANONICAL_S2_BANDS:
                    if b in raw_offsets:
                        offsets[b] = float(raw_offsets[b])
                    else:
                        offsets[b] = -1000.0 if pb >= 4.0 else 0.0
            else:
                raise ValueError(f"Unsupported format for boa_add_offset: {type(raw_offsets)}")
        else:
            # Default based on baseline
            default_val = -1000.0 if pb >= 4.0 else 0.0
            for b in CANONICAL_S2_BANDS:
                offsets[b] = default_val

        return S2Metadata(
            processing_baseline=pb,
            boa_add_offset=offsets,
            quantification_value=qv,
            product_id=str(product_id) if product_id else None,
            source_format="dictionary"
        )

    # --------------------------------------------------------------------------
    # Case 2: XML string or XML file path
    # --------------------------------------------------------------------------
    if isinstance(metadata_input, (str, Path)):
        s_val = str(metadata_input).strip()
        is_xml_path = False
        xml_content = None

        if Path(s_val).exists() and Path(s_val).is_file():
            is_xml_path = True
            with open(s_val, "r", encoding="utf-8", errors="ignore") as f:
                xml_content = f.read()
        elif s_val.startswith("<") and s_val.endswith(">"):
            xml_content = s_val

        if xml_content is not None:
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                raise ValueError(f"Malformed Sentinel-2 XML metadata: {e}")

            # Search for PROCESSING_BASELINE in XML
            pb = None
            # Standard tags: <PROCESSING_BASELINE>, <BASELINE>, etc.
            for elem in root.iter():
                tag_name = elem.tag.split("}")[-1]  # Strip namespace
                if tag_name.upper() in ("PROCESSING_BASELINE", "BASELINE"):
                    if elem.text:
                        pb = _extract_baseline_from_string(elem.text.strip())
                        if pb is not None:
                            break

            # Secondary search: in product name or granule tags
            if pb is None:
                for elem in root.iter():
                    if elem.text and "_N0" in elem.text:
                        pb = _extract_baseline_from_string(elem.text)
                        if pb is not None:
                            break

            if pb is None:
                raise ValueError("Sentinel-2 XML metadata parsed, but PROCESSING_BASELINE tag not found.")

            # Search for QUANTIFICATION_VALUE
            qv = NOMINAL_QUANTIFICATION_VALUE
            for elem in root.iter():
                tag_name = elem.tag.split("}")[-1]
                if tag_name.upper() == "QUANTIFICATION_VALUE":
                    if elem.text:
                        try:
                            qv = float(elem.text.strip())
                        except ValueError:
                            pass

            # Search for BOA_ADD_OFFSET tags
            offsets: Dict[str, float] = {}
            for elem in root.iter():
                tag_name = elem.tag.split("}")[-1]
                if tag_name.upper() in ("BOA_ADD_OFFSET", "BOA_ADD_OFFSET_VALUE"):
                    band_id = elem.attrib.get("band_id") or elem.attrib.get("band_name")
                    if elem.text:
                        try:
                            val = float(elem.text.strip())
                            if band_id:
                                # Normalize band_id (e.g. 'B2' -> 'B02', 'B8A' -> 'B8A')
                                b_norm = band_id.upper()
                                if b_norm.startswith("B") and len(b_norm) == 2 and b_norm[1].isdigit():
                                    b_norm = f"B0{b_norm[1]}"
                                offsets[b_norm] = val
                            else:
                                # Scalar apply
                                for b in CANONICAL_S2_BANDS:
                                    if b not in offsets:
                                        offsets[b] = val
                        except ValueError:
                            pass

            # Fill missing offsets from baseline default
            default_val = -1000.0 if pb >= 4.0 else 0.0
            for b in CANONICAL_S2_BANDS:
                if b not in offsets:
                    offsets[b] = default_val

            return S2Metadata(
                processing_baseline=pb,
                boa_add_offset=offsets,
                quantification_value=qv,
                product_id=str(metadata_input) if is_xml_path else None,
                source_format="xml"
            )

        # ----------------------------------------------------------------------
        # Case 3: Plain string product title (e.g. SAFE product ID)
        # ----------------------------------------------------------------------
        pb = _extract_baseline_from_string(s_val)
        if pb is not None:
            default_val = -1000.0 if pb >= 4.0 else 0.0
            offsets = {b: default_val for b in CANONICAL_S2_BANDS}
            return S2Metadata(
                processing_baseline=pb,
                boa_add_offset=offsets,
                quantification_value=NOMINAL_QUANTIFICATION_VALUE,
                product_id=s_val,
                source_format="product_id"
            )

    raise ValueError(f"Cannot parse Sentinel-2 metadata from input of type {type(metadata_input)}.")


def convert_dn_to_reflectance(
    raw_s2_dn: np.ndarray,
    s2_metadata: S2Metadata,
    bands: Optional[List[str]] = None
) -> np.ndarray:
    """
    Convert 10-band Sentinel-2 Digital Numbers (10, H, W) to physical surface reflectance
    using verified per-band BOA_ADD_OFFSET and QUANTIFICATION_VALUE.
    
    Verifies finite values and checks range against nominal window [-0.05, 1.20].
    Does NOT force-clip valid values to [0.0, 1.0].
    """
    if raw_s2_dn.ndim != 3 or raw_s2_dn.shape[0] != 10:
        raise ValueError(f"raw_s2_dn must have shape (10, H, W), got {raw_s2_dn.shape}.")

    if not np.all(np.isfinite(raw_s2_dn)):
        raise ValueError("raw_s2_dn contains non-finite NaN or Inf values.")

    band_list = bands or CANONICAL_S2_BANDS
    if len(band_list) != 10:
        raise ValueError(f"band_list must contain exactly 10 bands, got {len(band_list)}.")

    refl = np.zeros_like(raw_s2_dn, dtype=np.float32)
    qv = s2_metadata.quantification_value

    for c, band_name in enumerate(band_list):
        offset = s2_metadata.get_band_offset(band_name)
        refl[c] = (raw_s2_dn[c].astype(np.float32) + offset) / qv

    # Verify physical tolerance window
    min_val = float(np.min(refl))
    max_val = float(np.max(refl))
    if min_val < NOMINAL_REFLECTANCE_RANGE[0] or max_val > NOMINAL_REFLECTANCE_RANGE[1]:
        raise ValueError(
            f"Physical range violation: surface reflectance [{min_val:.4f}, {max_val:.4f}] "
            f"exceeds tolerance window [{NOMINAL_REFLECTANCE_RANGE[0]}, {NOMINAL_REFLECTANCE_RANGE[1]}]."
        )

    return refl


def resample_s2_20m_to_10m(
    bands_dict: Dict[str, np.ndarray],
    target_shape: Tuple[int, int] = (120, 120)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resample native 20m Sentinel-2 bands to 10m GSD matching target_shape using bilinear interpolation.
    
    Expects bands_dict to contain all 10 canonical bands:
    - Native 10m bands: B02, B03, B04, B08 (must already have target_shape)
    - Native 20m bands: B05, B06, B07, B8A, B11, B12 (typically target_shape // 2)
    
    Returns:
    - Stacked array of shape (10, H, W) in canonical order:
      [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
    - Provenance dictionary verifying bilinear resampling method and parameter settings.
    """
    missing = [b for b in CANONICAL_S2_BANDS if b not in bands_dict]
    if missing:
        raise ValueError(f"Missing required Sentinel-2 bands: {missing}")

    H_target, W_target = target_shape
    stacked_channels: List[np.ndarray] = []

    for b in CANONICAL_S2_BANDS:
        arr = bands_dict[b]
        if arr.ndim != 2:
            raise ValueError(f"Band {b} must be 2D array (H, W), got {arr.shape}.")

        if b in NATIVE_10M_BANDS:
            if arr.shape != (H_target, W_target):
                raise ValueError(
                    f"Native 10m band {b} shape {arr.shape} does not match target shape {target_shape}."
                )
            stacked_channels.append(arr.astype(np.float32))
        else:
            # Native 20m band: resample using bilinear interpolation
            if arr.shape == (H_target, W_target):
                # Already resampled
                stacked_channels.append(arr.astype(np.float32))
            else:
                tensor_in = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                tensor_out = F.interpolate(
                    tensor_in,
                    size=(H_target, W_target),
                    mode="bilinear",
                    align_corners=False
                )
                upsampled = tensor_out.squeeze(0).squeeze(0).cpu().numpy()
                stacked_channels.append(upsampled)

    stacked_array = np.stack(stacked_channels, axis=0)

    provenance = {
        "resampling_method": "bilinear",
        "align_corners": False,
        "target_shape": list(target_shape),
        "native_10m_bands": NATIVE_10M_BANDS,
        "resampled_20m_bands": NATIVE_20M_BANDS,
        "channel_order": CANONICAL_S2_BANDS,
        "output_shape": list(stacked_array.shape)
    }

    return stacked_array, provenance
