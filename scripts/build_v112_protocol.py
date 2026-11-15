"""
Build Script to Upgrade SatQuery AI Phase 4 Protocol from v1.11 to v1.12.

Executes all 4 audit checks:
1. Sentinel-2 metadata ingestion & resampling integration (satquery/utils/s2_metadata.py, tests/test_s2_metadata_integration.py).
2. Explicit dual zero-based (Python/PyTorch) and one-based (Protocol) taxonomy index documentation.
3. Edge-case definition for FPS_absent (undefined on zero denominator -> NA) and reporting |C_absent|.
4. Comprehensive 35-component cryptographic provenance manifest including all manifests, configs, and test suites.
"""

from pathlib import Path

def build_v112():
    v111_path = Path("SATQUERY-P4-PROTOCOL-V1.11-DRAFT.md")
    v112_path = Path("SATQUERY-P4-PROTOCOL-V1.12-DRAFT.md")

    with open(v111_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Header Updates
    old_header = """**Document Identifier**: `SATQUERY-P4-PROTOCOL-V1.11-DRAFT`  
**Phase**: Phase 4 — Robustness, Generalization, and Operational Envelope Discovery  
**Date**: November 9, 2026  
**Status**: **PROPOSED PREREGISTRATION PROTOCOL — DRAFT v1.11 (Methodological & Statistical Lock — Pending Final User Freeze)**  
**Repository Branch**: `feature/phase4-generalization-protocol`  
**Preceding Milestones**: Phase 3 Closed (M3.1 Multimodal ASL/CB-BCE, M3.2 FC-Siam-diff, M3.3 Selective Routing) with Pre-Registered Limitations; Supersedes `SATQUERY-P4-PROTOCOL-V1.10-DRAFT` with Sentinel-2 processing baseline reconciliation, Stage B preprocessing regression testing, isolated quarantined metrics, expanded 24-component cryptographic provenance manifest, and explicit three-cluster conditional inference scope.  
**Confirmatory Experiment Status**: **ZERO Confirmatory Experiments Executed (Draft Protocol Specification Only)**  
**Model Weight & Threshold Provenance**: Completely Frozen from Phase 3"""

    new_header = """**Document Identifier**: `SATQUERY-P4-PROTOCOL-V1.12-DRAFT`  
**Phase**: Phase 4 — Robustness, Generalization, and Operational Envelope Discovery  
**Date**: November 9, 2026  
**Status**: **PROPOSED PREREGISTRATION PROTOCOL — DRAFT v1.12 (Final Verification Lock — Pending Final User Freeze)**  
**Repository Branch**: `feature/phase4-generalization-protocol`  
**Preceding Milestones**: Phase 3 Closed (M3.1 Multimodal ASL/CB-BCE, M3.2 FC-Siam-diff, M3.3 Selective Routing) with Pre-Registered Limitations; Supersedes `SATQUERY-P4-PROTOCOL-V1.11-DRAFT` with Sentinel-2 product XML/JSON metadata ingestion, bilinear resampling provenance verification, explicit 0-based vs 1-based taxonomy index documentation, metric edge-case resolution ($FPS_{\\text{absent}} \\to \\text{NA}$), and comprehensive 35-component cryptographic provenance manifest.  
**Confirmatory Experiment Status**: **ZERO Confirmatory Experiments Executed (Draft Protocol Specification Only)**  
**Model Weight & Threshold Provenance**: Completely Frozen from Phase 3"""

    assert old_header in content, "Header match failed"
    content = content.replace(old_header, new_header)

    # 2. Section 2 Specialist 1 Taxonomy Indexing (Explicit Zero-Based and One-Based)
    old_taxonomy_list = r"""- **Target Label Taxonomy (Empirically Verified: 16 Classes)**:
  Audited from `artifacts/specialists/land_cover/classes.json`:
  1. `Arable land` ($\tau^* = 0.70$)
  2. `Beaches, dunes, sands` ($\tau^* = 0.50$)
  3. `Broad-leaved forest` ($\tau^* = 0.60$)
  4. `Complex cultivation patterns` ($\tau^* = 0.65$)
  5. `Coniferous forest` ($\tau^* = 0.55$)
  6. `Industrial or commercial units` ($\tau^* = 0.65$)
  7. `Inland waters` ($\tau^* = 0.75$)
  8. `Inland wetlands` ($\tau^* = 0.55$)
  9. `Land principally occupied by agriculture, with significant areas of natural vegetation` ($\tau^* = 0.60$)
  10. `Mixed forest` ($\tau^* = 0.55$)
  11. `Moors, heathland and sclerophyllous vegetation` ($\tau^* = 0.35$)
  12. `Natural grassland and sparsely vegetated areas` ($\tau^* = 0.55$)
  13. `Pastures` ($\tau^* = 0.60$)
  14. `Permanent crops` ($\tau^* = 0.45$)
  15. `Transitional woodland, shrub` ($\tau^* = 0.60$)
  16. `Urban fabric` ($\tau^* = 0.65$)"""

    new_taxonomy_list = r"""- **Target Label Taxonomy (Empirically Verified: 16 Classes)**:
  Audited from `artifacts/specialists/land_cover/classes.json` (SHA-256: `cdb18c8a0a9dc74f52920720340251f14de323bc5c4fb4f07436f90f3500f3e9`), matching model head `head.1.weight: torch.Size([16, 512])`.
  > [!IMPORTANT]
  > **Dual Indexing Disambiguation (0-Based Python vs 1-Based Protocol)**:
  > In Python code and PyTorch tensors, arrays use **zero-based indexing** $c \in \{0, \dots, 15\}$. In mathematical protocol text and human-readable lists, classes are numbered **one-based** $c \in \{1, \dots, 16\}$.
  > - **Three Quarantined Classes**:
  >   - `Complex cultivation patterns`: 0-based index **3** (1-based index **4**), $\tau^* = 0.65$
  >   - `Industrial or commercial units`: 0-based index **5** (1-based index **6**), $\tau^* = 0.65$
  >   - `Permanent crops`: 0-based index **13** (1-based index **14**), $\tau^* = 0.45$
  > - **Thirteen Mapped Classes**:
  >   - 0-based: `[0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15]`
  >   - 1-based: `[1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16]`

  | 0-Based (Code) | 1-Based (Doc) | Class Label | Frozen $\tau^*$ | Evaluation Status |
  | :---: | :---: | :--- | :---: | :--- |
  | `0` | `1` | `Arable land` | 0.70 | **Directly Mapped** |
  | `1` | `2` | `Beaches, dunes, sands` | 0.50 | **Approximate Mapping** |
  | `2` | `3` | `Broad-leaved forest` | 0.60 | **Directly Mapped** |
  | `3` | `4` | `Complex cultivation patterns` | 0.65 | **Quarantined / Inevaluable** |
  | `4` | `5` | `Coniferous forest` | 0.55 | **Directly Mapped** |
  | `5` | `6` | `Industrial or commercial units` | 0.65 | **Quarantined / Inevaluable** |
  | `6` | `7` | `Inland waters` | 0.75 | **Approximate Mapping** |
  | `7` | `8` | `Inland wetlands` | 0.55 | **Directly Mapped** |
  | `8` | `9` | `Land principally occupied by agriculture...` | 0.60 | **Approximate Mapping** |
  | `9` | `10` | `Mixed forest` | 0.55 | **Directly Mapped** |
  | `10` | `11` | `Moors, heathland and sclerophyllous vegetation` | 0.35 | **Approximate Mapping** |
  | `11` | `12` | `Natural grassland and sparsely vegetated areas` | 0.55 | **Approximate Mapping** |
  | `12` | `13` | `Pastures` | 0.60 | **Approximate Mapping** |
  | `13` | `14` | `Permanent crops` | 0.45 | **Quarantined / Inevaluable** |
  | `14` | `15` | `Transitional woodland, shrub` | 0.60 | **Approximate Mapping** |
  | `15` | `16` | `Urban fabric` | 0.65 | **Directly Mapped** |"""

    assert old_taxonomy_list in content, "Section 2 taxonomy list match failed"
    content = content.replace(old_taxonomy_list, new_taxonomy_list)

    # 3. Section 3 Stage B Metadata Ingestion and Resampling Details
    old_stage_b_resamp = """   - *Resampling Provenance*: Verifies that 20m bands (`B05, B06, B07, B8A, B11, B12`) were resampled to 10m GSD via the certified bilinear pipeline prior to stacking."""
    new_stage_b_resamp = """   - *Resampling Provenance & Pipeline*: Verifies that 20m bands (`B05, B06, B07, B8A, B11, B12`) were resampled from native 20m GSD (60x60) to 10m GSD (120x120) via bilinear interpolation (`torch.nn.functional.interpolate` with `mode='bilinear', align_corners=False`), matching native 10m bands (`B02, B03, B04, B08`). Verified by `satquery.utils.s2_metadata.resample_s2_20m_to_10m` which returns structured provenance metadata `{"resampling_method": "bilinear", "align_corners": False, "channel_order": CANONICAL_S2_BANDS}`."""

    assert old_stage_b_resamp in content, "Section 3 resampling match failed"
    content = content.replace(old_stage_b_resamp, new_stage_b_resamp)

    old_stage_b_meta = r"""   - *Processing Baseline Reconciliation & Reflectance Conversion*:
     Sentinel-2 Level-2A surface reflectance conversion is governed by the product processing baseline:
     $$\rho_{\lambda} = \frac{\text{DN} + \text{BOA\_ADD\_OFFSET}_{\lambda}}{\text{QUANTIFICATION\_VALUE}} = \begin{cases} \frac{\text{DN}}{10,000.0}, & \text{for PB } < 04.00 \\ \frac{\text{DN} - 1000}{10,000.0}, & \text{for PB } \ge 04.00 \end{cases}$$
     > [!IMPORTANT]
     > **Baseline Offset Defense**: The frozen BigEarthNet-14k checkpoint was trained on archival Sentinel-2 imagery acquired prior to January 25, 2022 ($\text{PB} < 04.00$, $\text{BOA\_ADD\_OFFSET} = 0$). Modern operational products ($\text{PB} \ge 04.00$) include a radiometric offset of $-1,000\text{ DN}$. Applying the archival $\text{DN}/10000$ formula to $\text{PB} \ge 04.00$ products introduces an artificial $+0.10$ reflectance shift ($+1.35\sigma$ on Band 4 Red), corrupting model inference. Stage B automatically inspects metadata `processing_baseline` and `BOA_ADD_OFFSET` to enforce exact reflectance calibration."""

    new_stage_b_meta = r"""   - *Processing Baseline Reconciliation & Metadata Ingestion*:
     Sentinel-2 Level-2A surface reflectance conversion is governed by the product processing baseline:
     $$\rho_{\lambda} = \frac{\text{DN} + \text{BOA\_ADD\_OFFSET}_{\lambda}}{\text{QUANTIFICATION\_VALUE}} = \begin{cases} \frac{\text{DN}}{10,000.0}, & \text{for PB } < 04.00 \\ \frac{\text{DN} - 1000}{10,000.0}, & \text{for PB } \ge 04.00 \end{cases}$$
     where $\text{QUANTIFICATION\_VALUE} = 10,000.0$.
     - *Automated Metadata Parsing*: Implemented in `satquery.utils.s2_metadata.parse_s2_metadata`, which extracts processing baseline and per-band offsets directly from Copernicus XML metadata (`MTD_MSIL2A.xml`, `MTD_TL.xml`), JSON/dictionary manifests, or SAFE product naming conventions (`_N0400_`, `_N0205_`).
     - *Safe Error Handling*: Payloads missing processing baseline metadata or presenting contradictory declarations (e.g., PB $\ge$ 04.00 with zero offset) are rejected with `ERR_METADATA_INCOMPLETE` / `ValueError` before inference.
     - *Baseline Offset Defense*: Applying the archival $\text{DN}/10000$ formula to $\text{PB} \ge 04.00$ products introduces an artificial $+0.10$ reflectance shift ($+1.35\sigma$ on Band 4 Red). Stage B guarantees complete radiometric alignment across archival and modern products."""

    assert old_stage_b_meta in content, "Section 3 metadata match failed"
    content = content.replace(old_stage_b_meta, new_stage_b_meta)

    old_stage_b_reg = """   - *Regression Verification*: Verified bitwise against frozen implementations in `tests/test_preprocessing_compatibility.py` (SHA-256: `1a8908e3a0b9b2d2a6d9f4ed81686762c08f33f2c272b9bb47c01ff10e2442c9`)."""
    new_stage_b_reg = """   - *Regression & Integration Verification*: Preprocessing compatibility is validated bitwise in `tests/test_preprocessing_compatibility.py` (SHA-256: `1a8908e3a0b9b2d2a6d9f4ed81686762c08f33f2c272b9bb47c01ff10e2442c9`); full XML/dict metadata parsing, baseline reconciliation, and resampling provenance are verified in `tests/test_s2_metadata_integration.py` (SHA-256: `141bed10699d3b6ed716435c1f3cb5503151f87a194043ae4cc4342c079b1e4c`)."""

    assert old_stage_b_reg in content, "Section 3 regression match failed"
    content = content.replace(old_stage_b_reg, new_stage_b_reg)

    # 4. Section 4.1 Track 4.1 Absent-Class Metrics (Edge-Case Guard NA and |C_absent|)
    old_absent_formulas = r"""          - *Absent-Class False Positive Share ($FPS_{\text{absent}}$)*:
            $$FPS_{\text{absent}} = \frac{\sum_{c \in C_{\text{absent}, k}} \text{FP}_c}{\sum_{c \in \mathcal{M}} \sum_{i=1}^{N_{\text{evaluable}, k}} \hat{y}_{i, c}} = \frac{\sum_{c \in C_{\text{absent}, k}} \text{FP}_c}{\sum_{c=1}^{13} \sum_{i=1}^{N_{\text{evaluable}, k}} \hat{y}_{i, c}}$$
            The denominator is **strictly restricted to positive predictions among the 13 mapped classes**, measuring the proportion of mapped positive predictions that fall on absent classes.
          - *Per-Class Absent Alarms*: Tabularize $\text{FP-Count}_c$ and $\text{FPR}_c = \text{FP}_c / N_{\text{evaluable}, k}$ for each absent class in $C_{\text{absent}, k}$."""

    new_absent_formulas = r"""          - *Absent-Class False Positive Share ($FPS_{\text{absent}}$)*:
            $$FPS_{\text{absent}} = \begin{cases} \frac{\sum_{c \in C_{\text{absent}, k}} \text{FP}_c}{\sum_{c \in \mathcal{M}} \sum_{i=1}^{N_{\text{evaluable}, k}} \hat{y}_{i, c}}, & \text{if } \sum_{c \in \mathcal{M}} \sum_{i=1}^{N_{\text{evaluable}, k}} \hat{y}_{i, c} \ge 1 \\ \text{NA}, & \text{if } \sum_{c \in \mathcal{M}} \sum_{i=1}^{N_{\text{evaluable}, k}} \hat{y}_{i, c} = 0 \end{cases}$$
            The denominator is **strictly restricted to positive predictions among the 13 mapped classes**. If the model produces zero positive predictions on all mapped classes across the evaluation cohort, the denominator is zero; the metric is **explicitly reported as NA** (rather than silently assigned 0.0) to avoid false claims of zero false alarms.
          - *Mandatory Reporting of $|C_{\text{absent}, k}|$*: Every evaluation table reporting $FPR_{\text{absent}}$ or $FPS_{\text{absent}}$ must disclose the exact number of absent mapped classes $|C_{\text{absent}, k}|$ used in that calculation. Verified in `tests/test_taxonomy_and_metrics.py` (SHA-256: `548fc2f914df7c88494fd6fa252e4cbe673f53083c05745aa4106ec6a75b5a48`).
          - *Per-Class Absent Alarms*: Tabularize $\text{FP-Count}_c$ and $\text{FPR}_c = \text{FP}_c / N_{\text{evaluable}, k}$ for each absent class in $C_{\text{absent}, k}$."""

    assert old_absent_formulas in content, "Section 4.1 absent formulas match failed"
    content = content.replace(old_absent_formulas, new_absent_formulas)

    # 5. Section 5.2 Expanded Provenance Manifest Table (35 components)
    old_manifest_table = """| Component | Path / Location | SHA-256 Digest |
| :--- | :--- | :--- |
| **Land Cover Specialist Checkpoint** | `artifacts/specialists/deep_land_cover_asl/best_model_asl.pt` | `a3e2d24fc6b1b12e48960bc26f9d13eec2407fafb6a6237531da75038e65aabf` |
| **Land Cover Threshold Config** | `artifacts/specialists/deep_land_cover_asl/threshold_config.json` | `76fa4eda7aec18d4e2df7f9434c603d76effebc2ba569764320dbad58d40ad0d` |
| **Land Cover Class Taxonomy** | `artifacts/specialists/land_cover/classes.json` | `cdb18c8a0a9dc74f52920720340251f14de323bc5c4fb4f07436f90f3500f3e9` |
| **Land Cover Model Architecture** | `satquery/specialists/models/multimodal_resnet.py` | `cc5468f30192947a67c51ac4431d169d0c10ee9136f13815cd7a1677cef1b137` |
| **Land Cover Dataset & Preprocessing**| `satquery/specialists/dataset_land_cover.py` | `a73356eacfbb2ec78d115334712948369ee5613d40063089dc1d31bf073181a9` |
| **Land Cover Specialist Wrapper** | `satquery/specialists/deep_land_cover.py` | `8ef9fbd581adb083e770faa005b0b33fdc0608fc7117704805f815b5c0105d7a` |
| **Change Detection Specialist Checkpoint**| `artifacts/specialists/deep_change_detection_fc_siam_diff/best_model.pt` | `8e42ac5abbb907bac3b18abdbf0367748f3849dbd7dea22aaa8c219da4000df3` |
| **Change Detection Threshold Config** | `artifacts/specialists/deep_change_detection_fc_siam_diff/threshold_config.json` | `c190999eb02f7c6ea043348e9c36b086a6e4ba2d6f2794bd0d889a757553f680` |
| **Change Detection Dataset & Preprocessing**| `satquery/specialists/dataset_change_detection.py` | `554c544d457c9a4f1adfe0db850e57aad0a05cc888ee01b4e2579c3cfdd21c61` |
| **Change Detection Specialist Wrapper** | `satquery/specialists/change_detection.py` | `1aaf30839ad269fa4055688c643ecf9430cc0d3fab78feabb66359c586e3459b` |
| **Router Model Checkpoint** | `artifacts/router/router_model.joblib` | `eaf6c5b2cd8e718980a3496ce416b69ff9793cb1584e176951a9f652941cc218` |
| **Router Hyperparameter Config** | `configs/router_config.json` | `1160e4e4f5c6008c995d2d0cdb63fde55b53d85b329a819ea3e02748c7f04ef9` |
| **Physical Gate Implementation** | `satquery/utils/geospatial.py` | `156b76cd1ab4abc54341e6a405f9cd58f1387096f6cc92917c62ec01a11911f5` |
| **Temporal Gate Implementation** | `satquery/utils/temporal.py` | `2126aaa91280e18f19d440fe8ea54c10fb5438cc34afc57bcc76c6ee47c4db79` |
| **Core Taxonomy Module** | `satquery/taxonomy.py` | `f1c424218f5144621ea5b6ae94065ada326ef3ff0d61e1ed0fe236e23c059997` |
| **Task Router Logic Module** | `satquery/router.py` | `f3bf45452df165ef0a6fd0241fb3973050ac0d69609abf118f8c343491f175b1` |
| **Metadata Parsing Module** | `satquery/utils/metadata.py` | `f3cbbc36ab3ad7c4f8bcad1144ba8235f013f4b84a49ad51989fa6f1aff159c8` |
| **Dataset Orchestration Core** | `satquery/dataset.py` | `a59356dc89be0e94865a34c2538a069c5d2e6cc259857a0e0102a8bb8eb6038c` |
| **System Orchestrator** | `satquery/orchestrator.py` | `254e24ccc508145ac0b715ba88c8cdbe729fcde9d1aec45a67374cfbccc4ac25` |
| **Evaluation Metrics & Verification**| `satquery/evaluation.py` | `5fca3ec08664f96381101bdb2e6711acd43934085e46d50962a9d51757c644fc` |
| **Evidence Package Exporter** | `satquery/evidence.py` | `372ddeac05a5d11dde28155437632f66f1cae206000a16d4a0eb1b5d63e7dd6b` |
| **OSCD Change Detection Catalog** | `data/oscd/manifest.json` | `2ce5800100bf9ee2a004ef6d7cde9d5425357aa0bf66c0539c9762f52ccef7ec` |
| **Stage B Preprocessing Test Suite** | `tests/test_preprocessing_compatibility.py` | `1a8908e3a0b9b2d2a6d9f4ed81686762c08f33f2c272b9bb47c01ff10e2442c9` |
| **Environment Lockfile** | `requirements-lock.txt` | `72c893fb4425dd0c83c77a39338b26325da9cb96d473c2b511630369585b3b25` |"""

    new_manifest_table = """| Component | Path / Location | SHA-256 Digest | Category |
| :--- | :--- | :--- | :--- |
| **Land Cover Specialist Checkpoint** | `artifacts/specialists/deep_land_cover_asl/best_model_asl.pt` | `a3e2d24fc6b1b12e48960bc26f9d13eec2407fafb6a6237531da75038e65aabf` | Model Weights |
| **Land Cover Threshold Config** | `artifacts/specialists/deep_land_cover_asl/threshold_config.json` | `76fa4eda7aec18d4e2df7f9434c603d76effebc2ba569764320dbad58d40ad0d` | Model Thresholds |
| **Land Cover Class Taxonomy** | `artifacts/specialists/land_cover/classes.json` | `cdb18c8a0a9dc74f52920720340251f14de323bc5c4fb4f07436f90f3500f3e9` | Taxonomy |
| **Land Cover Model Architecture** | `satquery/specialists/models/multimodal_resnet.py` | `cc5468f30192947a67c51ac4431d169d0c10ee9136f13815cd7a1677cef1b137` | Source Code |
| **Land Cover Dataset & Preprocessing**| `satquery/specialists/dataset_land_cover.py` | `a73356eacfbb2ec78d115334712948369ee5613d40063089dc1d31bf073181a9` | Preprocessing Code |
| **Land Cover Specialist Wrapper** | `satquery/specialists/deep_land_cover.py` | `8ef9fbd581adb083e770faa005b0b33fdc0608fc7117704805f815b5c0105d7a` | Specialist Wrapper |
| **Change Detection Specialist Checkpoint**| `artifacts/specialists/deep_change_detection_fc_siam_diff/best_model.pt` | `8e42ac5abbb907bac3b18abdbf0367748f3849dbd7dea22aaa8c219da4000df3` | Model Weights |
| **Change Detection Threshold Config** | `artifacts/specialists/deep_change_detection_fc_siam_diff/threshold_config.json` | `c190999eb02f7c6ea043348e9c36b086a6e4ba2d6f2794bd0d889a757553f680` | Model Thresholds |
| **Change Detection Dataset & Preprocessing**| `satquery/specialists/dataset_change_detection.py` | `554c544d457c9a4f1adfe0db850e57aad0a05cc888ee01b4e2579c3cfdd21c61` | Preprocessing Code |
| **Change Detection Specialist Wrapper** | `satquery/specialists/change_detection.py` | `1aaf30839ad269fa4055688c643ecf9430cc0d3fab78feabb66359c586e3459b` | Specialist Wrapper |
| **Router Model Checkpoint** | `artifacts/router/router_model.joblib` | `eaf6c5b2cd8e718980a3496ce416b69ff9793cb1584e176951a9f652941cc218` | Model Weights |
| **Router Hyperparameter Config** | `configs/router_config.json` | `1160e4e4f5c6008c995d2d0cdb63fde55b53d85b329a819ea3e02748c7f04ef9` | Configuration |
| **Perturbation Operators Configuration**| `configs/perturbation_config.json` | `d260def052003959312d3cd1050a8a26b3e1c96d6ab07d1fc6afb009dc91245e` | Configuration |
| **MODIS Harmonization Configuration** | `configs/modis_harmonization_config.json` | `ecaa10b4f83bdc015898874d1ef88cb19b43832e961cf007ed028e2c61eaefdf` | Configuration |
| **S2 Preprocessing Baseline Config** | `configs/s2_preprocessing_config.json` | `020f8e117b67d3da53fcb6e5ebd7c576ca45fe69d19e2223a576e282b4004933` | Configuration |
| **OOD Candidate Manifest Spec** | `configs/ood_candidate_manifest_spec.json` | `3ed8fba913ac557a479c8f6d3428d9dc427910f8bd0f9d600334ce15fdda55ac` | Configuration |
| **Sentinel-2 Metadata Module** | `satquery/utils/s2_metadata.py` | `b1186fd0682e4460bc86b21c08400b59e1dd82ec99bef42afae546db0ccd1dd8` | Source Code |
| **Physical Gate Implementation** | `satquery/utils/geospatial.py` | `156b76cd1ab4abc54341e6a405f9cd58f1387096f6cc92917c62ec01a11911f5` | Source Code |
| **Temporal Gate Implementation** | `satquery/utils/temporal.py` | `2126aaa91280e18f19d440fe8ea54c10fb5438cc34afc57bcc76c6ee47c4db79` | Source Code |
| **Core Taxonomy Module** | `satquery/taxonomy.py` | `f1c424218f5144621ea5b6ae94065ada326ef3ff0d61e1ed0fe236e23c059997` | Source Code |
| **Task Router Logic Module** | `satquery/router.py` | `f3bf45452df165ef0a6fd0241fb3973050ac0d69609abf118f8c343491f175b1` | Source Code |
| **Metadata Parsing Module** | `satquery/utils/metadata.py` | `f3cbbc36ab3ad7c4f8bcad1144ba8235f013f4b84a49ad51989fa6f1aff159c8` | Source Code |
| **Dataset Orchestration Core** | `satquery/dataset.py` | `a59356dc89be0e94865a34c2538a069c5d2e6cc259857a0e0102a8bb8eb6038c` | Source Code |
| **System Orchestrator** | `satquery/orchestrator.py` | `254e24ccc508145ac0b715ba88c8cdbe729fcde9d1aec45a67374cfbccc4ac25` | Source Code |
| **Evaluation Metrics & Verification**| `satquery/evaluation.py` | `5fca3ec08664f96381101bdb2e6711acd43934085e46d50962a9d51757c644fc` | Source Code |
| **Evidence Package Exporter** | `satquery/evidence.py` | `372ddeac05a5d11dde28155437632f66f1cae206000a16d4a0eb1b5d63e7dd6b` | Source Code |
| **BigEarthNet Test Split Metadata** | `data/test_full_metadata.parquet` | `069bca852cce37a28081c20f9fc1ed731fcedfcb129d01400f34f0fdcc5aa59c` | Dataset Manifest |
| **BigEarthNet Test Ground Truth** | `data/y_test_ground_truth.npy` | `d776b34d80fc8ff9b19022ec0d3ccdf938b1c9bbf7867a231d4b7d1058dbfb23` | Ground Truth |
| **BigEarthNet Val Split Metadata** | `data/val_full_metadata.parquet` | `807b23eb45e7438d7f7e930bb88eb2b7724cba68a94f5ac13f8ee7907ce29a83` | Dataset Manifest |
| **BigEarthNet Val Ground Truth** | `data/y_val_ground_truth.npy` | `5faa7a958191c21cbc99888d181e530064ac8f6103214cabd7b0c4d8cd0f08f7` | Ground Truth |
| **OSCD Change Detection Catalog** | `data/oscd/manifest.json` | `2ce5800100bf9ee2a004ef6d7cde9d5425357aa0bf66c0539c9762f52ccef7ec` | Dataset Manifest |
| **Change Detection Benchmark Catalog**| `data/change_detection_catalog.parquet` | `b38f9c226c6222ef64c45c0cdf5763ec109f8b511754cc1715bfebb519342985` | Dataset Manifest |
| **Linguistic Benchmark N=200 Catalog**| `data/benchmark_n200_catalog.json` | `8aa9a8bbc8c811adf719685306fa800706d162a61ec8d2a530e90796c4be346f` | Benchmark Catalog |
| **Stage B Preprocessing Test Suite** | `tests/test_preprocessing_compatibility.py` | `1a8908e3a0b9b2d2a6d9f4ed81686762c08f33f2c272b9bb47c01ff10e2442c9` | Test Suite |
| **S2 Metadata Integration Test Suite**| `tests/test_s2_metadata_integration.py` | `141bed10699d3b6ed716435c1f3cb5503151f87a194043ae4cc4342c079b1e4c` | Test Suite |
| **Taxonomy & Metric Test Suite** | `tests/test_taxonomy_and_metrics.py` | `548fc2f914df7c88494fd6fa252e4cbe673f53083c05745aa4106ec6a75b5a48` | Test Suite |
| **Environment Lockfile** | `requirements-lock.txt` | `72c893fb4425dd0c83c77a39338b26325da9cb96d473c2b511630369585b3b25` | Environment |"""

    assert old_manifest_table in content, "Section 5.2 manifest table match failed"
    content = content.replace(old_manifest_table, new_manifest_table)

    # 6. Section 7 Summary Table
    old_summary_row = """| **Document Version** | `SATQUERY-P4-PROTOCOL-V1.11-DRAFT` | **PROPOSED DRAFT v1.11** |"""
    new_summary_row = """| **Document Version** | `SATQUERY-P4-PROTOCOL-V1.12-DRAFT` | **PROPOSED DRAFT v1.12** |"""
    assert old_summary_row in content, "Section 7 summary row match failed"
    content = content.replace(old_summary_row, new_summary_row)

    # Write out v1.12
    with open(v112_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully generated {v112_path} ({len(content)} characters)")

if __name__ == "__main__":
    build_v112()
