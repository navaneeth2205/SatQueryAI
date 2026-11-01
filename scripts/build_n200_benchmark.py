"""
Generates the Expanded Labeled Benchmark Catalog of N=200 Scenarios.

Aligns with Phase 1 Research Objectives for SIH26167:
- 6 Balanced Analytical Cohorts (50, 40, 30, 40, 20, 20)
- Frozen Partition: 100 Validation / 100 Test (Balanced per cohort)
- Explicit Template Families to Prevent In-Sample Overfitting
- Standardized Ground-Truth Actions:
    EXECUTED, REJECTED_INCOMPATIBLE, HALTED_UNCERTAINTY, REFUSED_UNAVAILABLE

Output:
- data/benchmark_n200_catalog.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_n200_scenarios() -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # COHORT 1: Supported In-Domain with Compatible Inputs (N=50)
    # Expected Action: EXECUTED
    # 25 Land Cover Classification + 25 Change Detection
    # -------------------------------------------------------------------------
    lc_queries = [
        # Agricultural & Cultivated (Val)
        ("What land-cover classes are present in this agricultural satellite tile?", "lc_agricultural", "val"),
        ("Determine whether this rural area contains pastures or arable agricultural fields.", "lc_agricultural", "val"),
        ("Catalog crop fields, vineyards, and agricultural plantations across this scene.", "lc_agricultural", "val"),
        ("Detect active farming parcels, irrigated cropland, and agro-forestry zones.", "lc_agricultural", "val"),
        ("Identify whether this sector is dominated by permanent crops or pastures.", "lc_agricultural", "val"),
        # Agricultural & Cultivated (Test)
        ("Quantify the presence of arable farmland and temporary crops in this footprint.", "lc_agricultural", "test"),
        ("Classify agricultural parcels and rural farm boundaries across the observation.", "lc_agricultural", "test"),
        ("Identify pasture land, meadows, and cultivated plots in this Sentinel-2 tile.", "lc_agricultural", "test"),
        ("Survey crop rotation zones and olive groves in this Mediterranean scene.", "lc_agricultural", "test"),
        ("Detect commercial orchards, vineyards, and cereal cultivation patches.", "lc_agricultural", "test"),

        # Forest & Natural (Val)
        ("What proportion of this Sentinel-2 footprint is occupied by coniferous forest?", "lc_forest", "val"),
        ("Identify broadleaved tree canopy, mixed woodland, and shrublands.", "lc_forest", "val"),
        ("Catalog natural biophysical surface categories across this protected reserve.", "lc_forest", "val"),
        ("Distinguish evergreen coniferous forest stands from deciduous vegetation.", "lc_forest", "val"),
        ("Segment natural vegetation, sclerophyllous scrub, and moorland.", "lc_forest", "val"),
        # Forest & Natural (Test)
        ("Map density of dense coniferous forest and alpine timberline zones.", "lc_forest", "test"),
        ("Detect transitional woodland-shrub and natural grassland in this mountainous tile.", "lc_forest", "test"),
        ("Classify broadleaved deciduous forest cover across the river valley.", "lc_forest", "test"),
        ("Identify sparse vegetation, heathlands, and natural forest clearings.", "lc_forest", "test"),
        ("Assess temperate forest biomes and peat bogs across this nature park.", "lc_forest", "test"),

        # Urban & Water (Val)
        ("Segment and report urban fabric, industrial complexes, and inland water bodies.", "lc_urban_water", "val"),
        ("Identify artificial surfaces, airport runways, and transportation infrastructure.", "lc_urban_water", "val"),
        ("Classify coastal lagoons, estuaries, and marine water bodies in this scene.", "lc_urban_water", "val"),
        # Urban & Water (Test)
        ("Map continuous and discontinuous urban fabric, port facilities, and road networks.", "lc_urban_water", "test"),
        ("Detect inland marshes, peatbogs, and freshwater reservoirs in this tile.", "lc_urban_water", "test"),
    ]

    for q, template, split in lc_queries:
        scenarios.append({
            "cohort": "1_supported_compatible",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": "feature_vector",
            "input_asset_key": "valid_lc_features",
            "expected_task": "LAND_COVER_CLASSIFICATION",
            "is_supported": True,
            "is_input_compatible": True,
            "is_ood_or_ambiguous": False,
            "expected_action": "EXECUTED"
        })

    cd_queries = [
        # Urban Sprawl & Construction (Val)
        ("What changed in this area between 2024 and 2026?", "cd_urban_sprawl", "val"),
        ("Map ground modifications and construction activity between the two timestamps.", "cd_urban_sprawl", "val"),
        ("Quantify the surface area of newly built structures across this temporal interval.", "cd_urban_sprawl", "val"),
        ("Detect new building construction and asphalt paving between pre and post dates.", "cd_urban_sprawl", "val"),
        ("Identify expansion of residential subdivisions between the acquisition dates.", "cd_urban_sprawl", "val"),
        # Urban Sprawl & Construction (Test)
        ("Highlight all locations where agricultural land was replaced by urban sprawl.", "cd_urban_sprawl", "test"),
        ("Measure newly erected commercial warehouses and highway extensions over time.", "cd_urban_sprawl", "test"),
        ("Detect industrial infrastructure development between the two temporal observations.", "cd_urban_sprawl", "test"),
        ("Map ground clearing for new housing developments between baseline and follow-up.", "cd_urban_sprawl", "test"),
        ("Quantify converted land area resulting from urban expansion between 2018 and 2022.", "cd_urban_sprawl", "test"),

        # Environmental & Forest Loss (Val)
        ("Did any forest clearing or deforestation occur between 2017 and 2018?", "cd_environmental", "val"),
        ("Detect land-cover modifications and canopy loss between these two acquisition dates.", "cd_environmental", "val"),
        ("Compute Change Vector Analysis to identify altered surface patches.", "cd_environmental", "val"),
        ("Detect significant spectral differences between the pre-event and post-event images.", "cd_environmental", "val"),
        ("Map vegetation destruction following wildfire occurrence between dates.", "cd_environmental", "val"),
        # Environmental & Forest Loss (Test)
        ("Quantify canopy loss and logging track emergence between temporal scenes.", "cd_environmental", "test"),
        ("Identify clear-cut logging patches and deforestation fronts between the two years.", "cd_environmental", "test"),
        ("Detect surface water body shrinkage and shoreline retreat between dates.", "cd_environmental", "test"),
        ("Measure burn scar extent and vegetative recovery between pre-fire and post-fire tiles.", "cd_environmental", "test"),
        ("Map wetland degradation and drainage alterations across the multi-year pair.", "cd_environmental", "test"),

        # Disaster & Hydrology (Val)
        ("Detect flood inundation extent changes between baseline and monsoon images.", "cd_disaster", "val"),
        ("Map coastal erosion and barrier island shifts between seasonal observations.", "cd_disaster", "val"),
        # Disaster & Hydrology (Test)
        ("Assess river channel migration and sedimentation changes over the temporal baseline.", "cd_disaster", "test"),
        ("Quantify post-hurricane coastal flooding modifications using bitemporal differencing.", "cd_disaster", "test"),
        ("Detect landslide scars and slope failures between pre-monsoon and post-monsoon tiles.", "cd_disaster", "test"),
    ]

    for q, template, split in cd_queries:
        scenarios.append({
            "cohort": "1_supported_compatible",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": "temporal_pair",
            "input_asset_key": "valid_temporal_pair",
            "expected_task": "CHANGE_DETECTION",
            "is_supported": True,
            "is_input_compatible": True,
            "is_ood_or_ambiguous": False,
            "expected_action": "EXECUTED"
        })

    # -------------------------------------------------------------------------
    # COHORT 2: Supported In-Domain with Incompatible Inputs (N=40)
    # Expected Action: REJECTED_INCOMPATIBLE
    # 15 Land Cover + Temporal Pair (anti-coercion trap)
    # 15 Change Detection + Single Image / Vector
    # 10 Dtype / Spatial / Scaling Corruption Traps
    # -------------------------------------------------------------------------
    lc_temporal_traps = [
        # Val (7)
        ("Classify all land cover classes visible in this satellite scene.", "trap_lc_temporal", "val"),
        ("Catalog biophysical surface categories across this observation footprint.", "trap_lc_temporal", "val"),
        ("Identify agricultural and forest types in this Sentinel-2 tile.", "trap_lc_temporal", "val"),
        ("What proportion of this landscape is occupied by broadleaved forest?", "trap_lc_temporal", "val"),
        ("Segment and report urban fabric, industrial units, and water bodies.", "trap_lc_temporal", "val"),
        ("Detect arable cropland and olive groves in this single scene.", "trap_lc_temporal", "val"),
        ("Perform automated multilabel land-use survey for this observation.", "trap_lc_temporal", "val"),
        # Test (8)
        ("Determine whether this rural area contains pastures or arable agricultural fields.", "trap_lc_temporal", "test"),
        ("Identify artificial surfaces and transportation infrastructure across this scene.", "trap_lc_temporal", "test"),
        ("Classify all CORINE land-cover classes present in this multispectral observation.", "trap_lc_temporal", "test"),
        ("Survey coniferous woodland and shrubland presence in this footprint.", "trap_lc_temporal", "test"),
        ("Detect marine inlets, lagoons, and salt marshes in this satellite patch.", "trap_lc_temporal", "test"),
        ("Produce multi-label classification probabilities for biophysical categories.", "trap_lc_temporal", "test"),
        ("Quantify percentage of grassland, peatbogs, and transitional woodland.", "trap_lc_temporal", "test"),
        ("Predict presence of natural biomes in this Sentinel-2 multispectral tile.", "trap_lc_temporal", "test"),
    ]
    for q, template, split in lc_temporal_traps:
        scenarios.append({
            "cohort": "2_supported_incompatible_input",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": "temporal_pair",
            "input_asset_key": "valid_temporal_pair",
            "expected_task": "LAND_COVER_CLASSIFICATION",
            "is_supported": True,
            "is_input_compatible": False,
            "is_ood_or_ambiguous": False,
            "expected_action": "REJECTED_INCOMPATIBLE"
        })

    cd_single_traps = [
        # Val (7)
        ("What changed between the baseline and subsequent observation?", "trap_cd_single", "val"),
        ("Map ground modifications between these two timestamps.", "trap_cd_single", "val"),
        ("Detect significant spectral differences across the two observation dates.", "trap_cd_single", "val"),
        ("Quantify changed surface area across the temporal sequence.", "trap_cd_single", "val"),
        ("Compute Change Vector Analysis between pre-event and post-event dates.", "trap_cd_single", "val"),
        ("Did any forest clearing occur between 2017 and 2018?", "trap_cd_single", "val"),
        ("Detect new building construction across the multi-temporal interval.", "trap_cd_single", "val"),
        # Test (8)
        ("Highlight all locations where vegetation cover was replaced by urban sprawl.", "trap_cd_single", "test"),
        ("Detect land-cover modifications and canopy loss between these two acquisition dates.", "trap_cd_single", "test"),
        ("Measure newly erected commercial warehouses over time.", "trap_cd_single", "test"),
        ("Identify altered surface patches between the two satellite timestamps.", "trap_cd_single", "test"),
        ("Map vegetation destruction following wildfire occurrence between dates.", "trap_cd_single", "test"),
        ("Detect flood inundation extent changes between baseline and monsoon images.", "trap_cd_single", "test"),
        ("Compare co-registered observations across time to detect geographical changes.", "trap_cd_single", "test"),
        ("Quantify post-event disaster modifications using temporal differencing.", "trap_cd_single", "test"),
    ]
    for q, template, split in cd_single_traps:
        scenarios.append({
            "cohort": "2_supported_incompatible_input",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": "single_image",
            "input_asset_key": "single_image",
            "expected_task": "CHANGE_DETECTION",
            "is_supported": True,
            "is_input_compatible": False,
            "is_ood_or_ambiguous": False,
            "expected_action": "REJECTED_INCOMPATIBLE"
        })

    corruption_traps = [
        # Val (6)
        ("Detect surface modifications across these two dates.", "trap_mismatched_dims", "val", "mismatched_temporal_pair", "CHANGE_DETECTION"),
        ("Map temporal change vector magnitude across these images.", "trap_unscaled_float", "val", "unscaled_raw_float", "CHANGE_DETECTION"),
        ("Measure bi-temporal surface change between Sentinel-2 acquisitions.", "trap_double_scaling", "val", "double_scaling_float", "CHANGE_DETECTION"),
        ("Classify land cover classes using attached precomputed vector.", "trap_nan_vector", "val", "corrupt_feature_vector_nan", "LAND_COVER_CLASSIFICATION"),
        ("Detect deforestation between dates using corrupt pair.", "trap_inf_pair", "val", "corrupt_temporal_pair_inf", "CHANGE_DETECTION"),
        ("Compute change magnitude across these non-overlapping tiles.", "trap_spatial_mismatch", "val", "mismatched_temporal_pair", "CHANGE_DETECTION"),
        # Test (4)
        ("Quantify urban construction differences across dates.", "trap_unscaled_float", "test", "unscaled_raw_float", "CHANGE_DETECTION"),
        ("Perform change detection across these uncalibrated satellite layers.", "trap_double_scaling", "test", "double_scaling_float", "CHANGE_DETECTION"),
        ("Map land cover categories from this invalid feature vector.", "trap_nan_vector", "test", "corrupt_feature_vector_nan", "LAND_COVER_CLASSIFICATION"),
        ("Measure vegetation change across mismatched spatial resolutions.", "trap_mismatched_dims", "test", "mismatched_temporal_pair", "CHANGE_DETECTION"),
    ]
    for q, template, split, asset_key, exp_task in corruption_traps:
        scenarios.append({
            "cohort": "2_supported_incompatible_input",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": "corrupt_input",
            "input_asset_key": asset_key,
            "expected_task": exp_task,
            "is_supported": True,
            "is_input_compatible": False,
            "is_ood_or_ambiguous": False,
            "expected_action": "REJECTED_INCOMPATIBLE"
        })

    # -------------------------------------------------------------------------
    # COHORT 3: Missing Sensor Modalities (N=30)
    # Expected Action: REJECTED_INCOMPATIBLE
    # 15 Optical-SAR Fusion requests missing SAR (optical only)
    # 15 Land Cover requests with incomplete sensor channels
    # -------------------------------------------------------------------------
    missing_sar_fusion = [
        # Val (7)
        ("Produce a fused multimodal radar-optical composite image.", "fusion_missing_sar", "val"),
        ("Integrate Sentinel-1 C-band SAR backscatter with Sentinel-2 optical bands.", "fusion_missing_sar", "val"),
        ("Jointly analyze Sentinel-1 VV/VH polarizations with Sentinel-2 multispectral reflectance.", "fusion_missing_sar", "val"),
        ("Perform optical-SAR cross-modal analysis to penetrate cloud cover.", "fusion_missing_sar", "val"),
        ("Fuse microwave synthetic aperture radar with multispectral optical channels.", "fusion_missing_sar", "val"),
        ("Compute radar-optical polarization indices and NDVI jointly.", "fusion_missing_sar", "val"),
        ("Combine Sentinel-1 radar backscatter intensity with Sentinel-2 optical bands.", "fusion_missing_sar", "val"),
        # Test (8)
        ("Generate multimodal surface roughness and vegetative index map from optical and SAR.", "fusion_missing_sar", "test"),
        ("Perform Sentinel-1 SAR and Sentinel-2 optical joint decomposition.", "fusion_missing_sar", "test"),
        ("Fuse dual-polarization C-band backscatter with optical RGB for structural analysis.", "fusion_missing_sar", "test"),
        ("Extract fused SAR-optical features for all-weather terrain mapping.", "fusion_missing_sar", "test"),
        ("Integrate radar interferometric coherence with multispectral bands.", "fusion_missing_sar", "test"),
        ("Combine optical reflectance with radar cross-section measurements.", "fusion_missing_sar", "test"),
        ("Compute optical-SAR fusion representations for cloudy scene analysis.", "fusion_missing_sar", "test"),
        ("Perform joint radar backscatter and optical spectral unmixing.", "fusion_missing_sar", "test"),
    ]
    for q, template, split in missing_sar_fusion:
        scenarios.append({
            "cohort": "3_missing_sensor_modalities",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": "optical_only",
            "input_asset_key": "single_image",
            "expected_task": "OPTICAL_SAR_FUSION",
            "is_supported": False,
            "is_input_compatible": False,
            "is_ood_or_ambiguous": False,
            "expected_action": "REJECTED_INCOMPATIBLE"
        })

    missing_sar_lc = [
        # Val (8)
        ("Classify land cover using combined Sentinel-1 SAR and Sentinel-2 optical data.", "lc_missing_sar", "val"),
        ("Map biophysical categories requiring both radar backscatter and optical reflectance.", "lc_missing_sar", "val"),
        ("Perform multimodal 488-feature land cover survey across this scene.", "lc_missing_sar", "val"),
        ("Catalog forest and crop biomes with dual-polarization SAR and multispectral bands.", "lc_missing_sar", "val"),
        ("Survey agricultural parcels requiring radar penetration and optical vegetation index.", "lc_missing_sar", "val"),
        ("Classify wetlands and marshes utilizing microwave backscatter and optical channels.", "lc_missing_sar", "val"),
        ("Differentiate urban structures from natural vegetation using optical-SAR data.", "lc_missing_sar", "val"),
        ("Predict BigEarthNet classes using co-registered S1 and S2 observation layers.", "lc_missing_sar", "val"),
        # Test (7)
        ("Map complex terrain biomes using combined C-band radar and Sentinel-2 MSI.", "lc_missing_sar", "test"),
        ("Perform joint radar-optical multi-label land cover categorization.", "lc_missing_sar", "test"),
        ("Classify land use requiring Sentinel-1 VV/VH backscatter and Sentinel-2 bands.", "lc_missing_sar", "test"),
        ("Detect water bodies and flooded vegetation with dual-sensor radar-optical input.", "lc_missing_sar", "test"),
        ("Map transitional woodland and industrial zones with SAR and optical fusion.", "lc_missing_sar", "test"),
        ("Classify agro-forestry parcels using multimodal satellite observations.", "lc_missing_sar", "test"),
        ("Execute multimodal land cover classification across radar-optical footprint.", "lc_missing_sar", "test"),
    ]
    for q, template, split in missing_sar_lc:
        scenarios.append({
            "cohort": "3_missing_sensor_modalities",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": "missing_sar_dict",
            "input_asset_key": "missing_sar_dict",
            "expected_task": "LAND_COVER_CLASSIFICATION",
            "is_supported": True,
            "is_input_compatible": False,
            "is_ood_or_ambiguous": False,
            "expected_action": "REJECTED_INCOMPATIBLE"
        })

    # -------------------------------------------------------------------------
    # COHORT 4: Ambiguous & Multi-Intent Queries (N=40)
    # Expected Action: HALTED_UNCERTAINTY (NEEDS_CLARIFICATION)
    # (Or safely disambiguated if attached observation cleanly resolves ambiguity)
    # 20 Broad Disambiguation Scenarios + 20 Conflicting Multi-Intent Scenarios
    # -------------------------------------------------------------------------
    broad_queries = [
        # Val (10)
        ("Analyze this satellite observation and report findings.", "broad_disambig", "val", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Analyze this satellite observation and report findings.", "broad_disambig", "val", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
        ("Compare these satellite data layers.", "broad_disambig", "val", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Assess landscape characteristics across this scene.", "broad_disambig", "val", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
        ("Process this imagery data through the specialist pipeline.", "broad_disambig", "val", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Process this imagery data through the specialist pipeline.", "broad_disambig", "val", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
        ("Examine satellite acquisition and extract primary patterns.", "broad_disambig", "val", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Examine satellite acquisition and extract primary patterns.", "broad_disambig", "val", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
        ("Evaluate this observation footprint with analytical models.", "broad_disambig", "val", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Evaluate this observation footprint with analytical models.", "broad_disambig", "val", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
        # Test (10)
        ("Run specialist analysis on the supplied satellite data.", "broad_disambig", "test", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Run specialist analysis on the supplied satellite data.", "broad_disambig", "test", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
        ("Interpret observations and generate geospatial findings.", "broad_disambig", "test", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Interpret observations and generate geospatial findings.", "broad_disambig", "test", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
        ("Perform Earth observation analytical extraction on this input.", "broad_disambig", "test", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Perform Earth observation analytical extraction on this input.", "broad_disambig", "test", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
        ("Inspect attached satellite imagery and provide detailed results.", "broad_disambig", "test", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Inspect attached satellite imagery and provide detailed results.", "broad_disambig", "test", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
        ("Characterize features visible in this remote sensing data.", "broad_disambig", "test", "valid_temporal_pair", "CHANGE_DETECTION", "temporal_pair", "EXECUTED"),
        ("Characterize features visible in this remote sensing data.", "broad_disambig", "test", "valid_lc_features", "LAND_COVER_CLASSIFICATION", "feature_vector", "EXECUTED"),
    ]
    for q, template, split, asset_key, exp_task, in_type, exp_act in broad_queries:
        scenarios.append({
            "cohort": "4_ambiguous_multi_intent",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": in_type,
            "input_asset_key": asset_key,
            "expected_task": exp_task,
            "is_supported": True,
            "is_input_compatible": True,
            "is_ood_or_ambiguous": True,
            "expected_action": exp_act
        })

    multi_intent_queries = [
        # Val (10)
        ("Analyze this scene and tell me what changed or what trees exist.", "multi_intent_compound", "val"),
        ("Show me the satellite picture.", "multi_intent_vague", "val"),
        ("Check the status of this region.", "multi_intent_vague", "val"),
        ("Is there any activity or just normal landscape?", "multi_intent_compound", "val"),
        ("Assess this footprint for environmental and structural patterns.", "multi_intent_vague", "val"),
        ("Compare these images and explain what kind of land this is.", "multi_intent_compound", "val"),
        ("Examine radar and optical variations and report anything noticeable.", "multi_intent_compound", "val"),
        ("Look at the temporal pair and describe the scene in words.", "multi_intent_compound", "val"),
        ("Find the buildings and tell me if they changed over time.", "multi_intent_compound", "val"),
        ("What is happening in this satellite image?", "multi_intent_vague", "val"),
        # Test (10)
        ("Tell me both what changed between dates and what crops are planted.", "multi_intent_compound", "test"),
        ("Identify if there are buildings here and whether new ones appeared.", "multi_intent_compound", "test"),
        ("Give me an overview of the landscape and detect any changes.", "multi_intent_compound", "test"),
        ("Do a complete assessment of this location.", "multi_intent_vague", "test"),
        ("What can you see in this satellite tile?", "multi_intent_vague", "test"),
        ("Are there changes here, or is it forest, or are there runways?", "multi_intent_compound", "test"),
        ("Summarize the scene and locate any modified parcels.", "multi_intent_compound", "test"),
        ("Detect changes, classify land cover, and count the objects.", "multi_intent_compound", "test"),
        ("Report on vegetation health and construction activity simultaneously.", "multi_intent_compound", "test"),
        ("Provide all possible analytical information for this footprint.", "multi_intent_vague", "test"),
    ]
    for q, template, split in multi_intent_queries:
        scenarios.append({
            "cohort": "4_ambiguous_multi_intent",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": "temporal_pair",
            "input_asset_key": "valid_temporal_pair",
            "expected_task": "AMBIGUOUS",
            "is_supported": False,
            "is_input_compatible": True,
            "is_ood_or_ambiguous": True,
            "expected_action": "HALTED_UNCERTAINTY"
        })

    # -------------------------------------------------------------------------
    # COHORT 5: Unsupported Taxonomy Tasks (N=20)
    # Expected Action: REFUSED_UNAVAILABLE (UNAVAILABLE_SPECIALIST)
    # 5 Single-Image VQA, 5 Change VQA, 5 Scene Captioning, 5 Visual Grounding
    # -------------------------------------------------------------------------
    unsupported_tasks = [
        # Single Image VQA (Val 2, Test 3)
        ("How many commercial airplanes are parked at this airport runway?", "unsupported_vqa", "val", "single_image", "SINGLE_IMAGE_VQA"),
        ("Is there an airport runway clearly visible within this satellite image?", "unsupported_vqa", "val", "single_image", "SINGLE_IMAGE_VQA"),
        ("Count the number of oil storage tanks located in this industrial terminal.", "unsupported_vqa", "test", "single_image", "SINGLE_IMAGE_VQA"),
        ("How many cargo ships are docked along the harbor quays?", "unsupported_vqa", "test", "single_image", "SINGLE_IMAGE_VQA"),
        ("Is there a circular irrigation pivot present in this agricultural sector?", "unsupported_vqa", "test", "single_image", "SINGLE_IMAGE_VQA"),

        # Change VQA (Val 3, Test 2)
        ("Has the central water reservoir shrunk in size compared to the previous acquisition?", "unsupported_change_vqa", "val", "valid_temporal_pair", "CHANGE_VQA"),
        ("Were the bridge structures damaged following the flood event?", "unsupported_change_vqa", "val", "valid_temporal_pair", "CHANGE_VQA"),
        ("Did the urban expansion encroach into the protected forest boundary over the interval?", "unsupported_change_vqa", "val", "valid_temporal_pair", "CHANGE_VQA"),
        ("Have the industrial storage facilities expanded toward the north between dates?", "unsupported_change_vqa", "test", "valid_temporal_pair", "CHANGE_VQA"),
        ("Was the residential neighborhood destroyed following the wildfire occurrence?", "unsupported_change_vqa", "test", "valid_temporal_pair", "CHANGE_VQA"),

        # Scene Captioning (Val 3, Test 2)
        ("Provide a comprehensive descriptive caption of this coastal harbor scene.", "unsupported_captioning", "val", "single_image", "SCENE_CAPTIONING"),
        ("Summarize the landscape characteristics and human settlements in this tile.", "unsupported_captioning", "val", "single_image", "SCENE_CAPTIONING"),
        ("Generate a concise natural language description of this mountain village.", "unsupported_captioning", "val", "single_image", "SCENE_CAPTIONING"),
        ("Describe the geographical layout and infrastructure distribution in this scene.", "unsupported_captioning", "test", "single_image", "SCENE_CAPTIONING"),
        ("Produce a descriptive paragraph characterizing land use patterns across this footprint.", "unsupported_captioning", "test", "single_image", "SCENE_CAPTIONING"),

        # Visual Grounding (Val 2, Test 3)
        ("Locate the offshore oil drilling platform and return its pixel bounding box.", "unsupported_grounding", "val", "single_image", "VISUAL_GROUNDING"),
        ("Highlight the solar panel farm in the southern sector with a segmentation mask.", "unsupported_grounding", "val", "single_image", "VISUAL_GROUNDING"),
        ("Detect and return the bounding coordinates of the baseball stadium.", "unsupported_grounding", "test", "single_image", "VISUAL_GROUNDING"),
        ("Localize the circular water treatment plant in the northeastern quadrant.", "unsupported_grounding", "test", "single_image", "VISUAL_GROUNDING"),
        ("Segment the container terminal and outline its perimeter boundary.", "unsupported_grounding", "test", "single_image", "VISUAL_GROUNDING"),
    ]
    for q, template, split, in_type, exp_task in unsupported_tasks:
        scenarios.append({
            "cohort": "5_unsupported_tasks",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": in_type,
            "input_asset_key": in_type if in_type in ("single_image", "valid_temporal_pair") else "valid_temporal_pair",
            "expected_task": exp_task,
            "is_supported": False,
            "is_input_compatible": True,
            "is_ood_or_ambiguous": False,
            "expected_action": "REFUSED_UNAVAILABLE"
        })

    # -------------------------------------------------------------------------
    # COHORT 6: Out-of-Distribution Prompts (N=20)
    # Expected Action: HALTED_UNCERTAINTY (NEEDS_CLARIFICATION)
    # 20 Non-Earth-Observation Prompts
    # -------------------------------------------------------------------------
    ood_queries = [
        # Val (10)
        ("What will the temperature, cloudiness, and rainfall forecast be tomorrow in Delhi?", "ood_weather", "val"),
        ("Write a Python script to compute the Fibonacci sequence up to N terms.", "ood_coding", "val"),
        ("Who is the prime minister of India and when is the next national election?", "ood_politics", "val"),
        ("How do I fix a flat tire on my bicycle?", "ood_general", "val"),
        ("Calculate the square root of 987654321.", "ood_math", "val"),
        ("What are the best tourist attractions to visit in Paris during summertime?", "ood_travel", "val"),
        ("Can you draft a formal job application letter for a data scientist position?", "ood_writing", "val"),
        ("Explain Einstein's theory of general relativity in simple terms.", "ood_science", "val"),
        ("Recommend five movies similar to Interstellar and Contact.", "ood_entertainment", "val"),
        ("What is the recipe for preparing authentic Italian lasagna at home?", "ood_culinary", "val"),
        # Test (10)
        ("How do quantum computers differ from classical binary computing architectures?", "ood_science", "test"),
        ("Write a quicksort implementation in C++ with optimal pivot selection.", "ood_coding", "test"),
        ("What were the primary socio-economic causes leading to the French Revolution?", "ood_history", "test"),
        ("Give me advice on improving cardiovascular endurance for a marathon.", "ood_health", "test"),
        ("How do I troubleshoot a blue screen memory management error in Windows 11?", "ood_tech", "test"),
        ("Can you compose a four-stanza poem celebrating autumn foliage?", "ood_creative", "test"),
        ("What is the current exchange rate between US Dollars and Japanese Yen?", "ood_finance", "test"),
        ("Explain how deep neural networks perform backpropagation during gradient descent.", "ood_education", "test"),
        ("How do solar panels convert sunlight into direct current electricity?", "ood_physics", "test"),
        ("What are the best strategies for managing work-life balance while studying?", "ood_lifestyle", "test"),
    ]
    for q, template, split in ood_queries:
        scenarios.append({
            "cohort": "6_out_of_distribution",
            "split": split,
            "template_family": template,
            "query": q,
            "attached_input_type": "none",
            "input_asset_key": "null_input",
            "expected_task": "OUT_OF_DISTRIBUTION",
            "is_supported": False,
            "is_input_compatible": False,
            "is_ood_or_ambiguous": True,
            "expected_action": "HALTED_UNCERTAINTY"
        })

    # Add scenario IDs
    for idx, sc in enumerate(scenarios, start=1):
        sc["scenario_id"] = f"BENCH-{idx:03d}"

    return scenarios


def main():
    scenarios = build_n200_scenarios()
    print(f"Total scenarios constructed: {len(scenarios)}")

    cohort_counts = {}
    split_counts = {}
    cohort_split = {}
    for sc in scenarios:
        c = sc["cohort"]
        s = sc["split"]
        cohort_counts[c] = cohort_counts.get(c, 0) + 1
        split_counts[s] = split_counts.get(s, 0) + 1
        key = (c, s)
        cohort_split[key] = cohort_split.get(key, 0) + 1

    print("\nCohort distribution:")
    for c, cnt in sorted(cohort_counts.items()):
        print(f"  {c:35s}: {cnt}")

    print("\nSplit distribution:")
    for s, cnt in sorted(split_counts.items()):
        print(f"  {s:10s}: {cnt}")

    print("\nBalanced Cohort x Split breakdown:")
    for (c, s), cnt in sorted(cohort_split.items()):
        print(f"  {c:35s} | {s:5s} : {cnt}")

    output_path = REPO_ROOT / "data" / "benchmark_n200_catalog.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark_version": "2.0", "total_scenarios": len(scenarios), "scenarios": scenarios}, f, indent=2)

    print(f"\nSaved catalog to: {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
