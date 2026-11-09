"""
Integration and Validation Tests for Sentinel-2 Metadata Ingestion and Resampling Pipeline.

Tests:
1. Parsing real Sentinel-2 Level-2A XML metadata structures (archival PB < 04.00 vs modern PB >= 04.00).
2. Per-band BOA_ADD_OFFSET extraction and application across all 10 canonical bands.
3. Safe rejection of missing, incomplete, or contradictory metadata.
4. Native 20m to 10m bilinear resampling provenance and canonical band stacking.
5. Exact parity with frozen Phase 3 preprocessing when processing archival BigEarthNet-14k data.
"""

import numpy as np
import pytest
import torch

from satquery.utils.s2_metadata import (
    CANONICAL_S2_BANDS,
    NATIVE_10M_BANDS,
    NATIVE_20M_BANDS,
    S2Metadata,
    parse_s2_metadata,
    convert_dn_to_reflectance,
    resample_s2_20m_to_10m,
)
from satquery.specialists.dataset_land_cover import (
    TRAIN_OPTICAL_MEANS,
    TRAIN_OPTICAL_STDS,
    normalize_optical,
)


# Sample XML representation of archival Sentinel-2 Level-2A product (PB 02.05, Summer 2017)
SAMPLE_ARCHIVAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd">
    <General_Info>
        <Product_Info>
            <PRODUCT_URI>S2A_MSIL2A_20170613T101031_N0205_R022_T32UND_20170613T101538.SAFE</PRODUCT_URI>
            <PROCESSING_BASELINE>02.05</PROCESSING_BASELINE>
            <QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE>
        </Product_Info>
    </General_Info>
</n1:Level-2A_User_Product>
"""

# Sample XML representation of modern operational Sentinel-2 Level-2A product (PB 04.00, post-Jan 2022)
SAMPLE_MODERN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd">
    <General_Info>
        <Product_Info>
            <PRODUCT_URI>S2A_MSIL2A_20220315T103021_N0400_R108_T32TQM_20220315T141523.SAFE</PRODUCT_URI>
            <PROCESSING_BASELINE>04.00</PROCESSING_BASELINE>
            <QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE>
            <BOA_ADD_OFFSET_VALUES_LIST>
                <BOA_ADD_OFFSET band_id="B02">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="B03">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="B04">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="B05">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="B06">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="B07">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="B08">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="B8A">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="B11">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET band_id="B12">-1000</BOA_ADD_OFFSET>
            </BOA_ADD_OFFSET_VALUES_LIST>
        </Product_Info>
    </General_Info>
</n1:Level-2A_User_Product>
"""


class TestS2MetadataIntegration:
    """Integration test suite for Sentinel-2 metadata parsing and preprocessing."""

    def test_parse_archival_xml_metadata(self):
        """Verify parsing of archival PB < 04.00 XML product."""
        meta = parse_s2_metadata(SAMPLE_ARCHIVAL_XML)
        assert meta.processing_baseline == 2.05
        assert meta.is_archival is True
        assert meta.is_modern is False
        assert meta.quantification_value == 10000.0
        # For archival PB < 04.00, offsets must default to 0.0 for all bands
        for b in CANONICAL_S2_BANDS:
            assert meta.get_band_offset(b) == 0.0

    def test_parse_modern_xml_metadata(self):
        """Verify parsing of modern PB >= 04.00 XML product with per-band offsets."""
        meta = parse_s2_metadata(SAMPLE_MODERN_XML)
        assert meta.processing_baseline == 4.00
        assert meta.is_archival is False
        assert meta.is_modern is True
        assert meta.quantification_value == 10000.0
        # For modern PB >= 04.00, each band must have -1000.0 offset
        for b in CANONICAL_S2_BANDS:
            assert meta.get_band_offset(b) == -1000.0

    def test_parse_from_product_id_string(self):
        """Verify parsing directly from SAFE product title string."""
        archival_title = "S2B_MSIL2A_20180720T103019_N0208_R108_T32TQM_20180720T141830.SAFE"
        modern_title = "S2A_MSIL2A_20230514T103021_N0509_R108_T32TQM_20230514T142510.SAFE"

        meta_arch = parse_s2_metadata(archival_title)
        assert meta_arch.processing_baseline == 2.08
        assert meta_arch.is_archival is True
        assert meta_arch.get_band_offset("B02") == 0.0

        meta_mod = parse_s2_metadata(modern_title)
        assert meta_mod.processing_baseline == 5.09
        assert meta_mod.is_modern is True
        assert meta_mod.get_band_offset("B02") == -1000.0

    def test_missing_or_incomplete_metadata_rejected(self):
        """Verify safe rejection when metadata is missing or baseline is absent."""
        with pytest.raises(ValueError, match="Metadata input is None"):
            parse_s2_metadata(None)

        with pytest.raises(ValueError, match="Metadata dictionary is empty"):
            parse_s2_metadata({})

        with pytest.raises(ValueError, match="Missing processing baseline"):
            parse_s2_metadata({"some_unrelated_key": 123})

        malformed_xml = "<invalid_xml>missing_tags</invalid_xml>"
        with pytest.raises(ValueError, match="PROCESSING_BASELINE tag not found"):
            parse_s2_metadata(malformed_xml)

    def test_contradictory_metadata_rejected(self):
        """Verify that inconsistent metadata (e.g. PB >= 4.0 but offset declared as 0) is flagged."""
        inconsistent_dict = {
            "processing_baseline": 4.00,
            "boa_add_offset": 0.0  # Inconsistent with PB >= 4.00
        }
        with pytest.raises(ValueError, match="Inconsistent metadata"):
            parse_s2_metadata(inconsistent_dict)

    def test_reflectance_conversion_archival_vs_modern(self):
        """Verify reflectance conversion across archival and modern Sentinel-2 products."""
        meta_arch = parse_s2_metadata(SAMPLE_ARCHIVAL_XML)
        meta_mod = parse_s2_metadata(SAMPLE_MODERN_XML)

        # In archival product: raw DN = 2000 => rho = 2000 / 10000 = 0.20
        raw_arch = np.full((10, 120, 120), 2000, dtype=np.float32)
        refl_arch = convert_dn_to_reflectance(raw_arch, meta_arch)
        assert np.isclose(refl_arch[0, 0, 0], 0.20, atol=1e-5)

        # In modern product: same surface with true rho = 0.20 has DN = 3000 (offset -1000)
        # rho = (3000 - 1000) / 10000 = 0.20
        raw_mod = np.full((10, 120, 120), 3000, dtype=np.float32)
        refl_mod = convert_dn_to_reflectance(raw_mod, meta_mod)
        assert np.isclose(refl_mod[0, 0, 0], 0.20, atol=1e-5)

        # Exact bitwise parity of computed surface reflectance
        np.testing.assert_allclose(refl_arch, refl_mod, rtol=1e-6, atol=1e-6)

    def test_frozen_normalization_parity_on_archival(self):
        """
        Verify that converting archival data with parse_s2_metadata + convert_dn_to_reflectance
        and applying channel standardization matches frozen normalize_optical() bitwise.
        """
        meta_arch = parse_s2_metadata(SAMPLE_ARCHIVAL_XML)
        rng = np.random.RandomState(42)
        raw_dn = rng.randint(0, 10001, size=(10, 120, 120)).astype(np.float32)

        # Frozen Phase 3 pipeline
        frozen_out = normalize_optical(raw_dn)

        # Stage B pipeline using parsed metadata
        refl = convert_dn_to_reflectance(raw_dn, meta_arch)
        stage_b_out = (refl - TRAIN_OPTICAL_MEANS[:, None, None]) / TRAIN_OPTICAL_STDS[:, None, None]

        np.testing.assert_array_equal(frozen_out, stage_b_out)

    def test_bilinear_resampling_provenance(self):
        """
        Verify that 20m bands are bilinearly upsampled to 10m GSD (120x120),
        stacked in canonical order, and accompanied by provenance metadata.
        """
        rng = np.random.RandomState(999)
        bands_dict = {}

        # 10m native bands at (120, 120)
        for b in NATIVE_10M_BANDS:
            bands_dict[b] = rng.uniform(500, 3000, size=(120, 120)).astype(np.float32)

        # 20m native bands at (60, 60)
        for b in NATIVE_20M_BANDS:
            bands_dict[b] = rng.uniform(500, 3000, size=(60, 60)).astype(np.float32)

        stacked, prov = resample_s2_20m_to_10m(bands_dict, target_shape=(120, 120))

        # Check output tensor shape
        assert stacked.shape == (10, 120, 120)
        assert stacked.dtype == np.float32

        # Check provenance metadata
        assert prov["resampling_method"] == "bilinear"
        assert prov["align_corners"] is False
        assert prov["target_shape"] == [120, 120]
        assert prov["channel_order"] == CANONICAL_S2_BANDS

        # Check that 10m bands were preserved without alteration
        np.testing.assert_array_equal(stacked[0], bands_dict["B02"])
        np.testing.assert_array_equal(stacked[1], bands_dict["B03"])
        np.testing.assert_array_equal(stacked[2], bands_dict["B04"])
        assert prov["channel_order"][3] == "B05"  # Resampled from (60, 60)
        assert stacked[3].shape == (120, 120)
