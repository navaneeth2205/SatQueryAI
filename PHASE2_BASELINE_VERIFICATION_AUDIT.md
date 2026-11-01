# SatQuery AI — Phase 2 Baseline Verification Audit & Scientific Defense Package (Final Frozen Baseline)

**Problem Title**: Sensor- and Task-Compatibility-Constrained Selective Routing for Multimodal Earth Observation  
**Problem Code**: SIH26167 (ISRO / Smart India Hackathon)  
**Document Classification**: Scientific Defense & Verification Package (Phase 2 Baseline Audit — Final Frozen Baseline)  
**Date**: November 1, 2026  
**Auditor**: Lead Research Verification Agent  
**Repository Branch**: `main`  
**Git Baseline Commit**: `a02c0a8dfe7ad65556a22a9a469626be0558af21`  
**Verification Status**: **VERIFIED, REPRODUCED & FROZEN**  

---

## Executive Summary

This final audit provides the verified scientific baseline, empirical defense, and prediction-level reproducibility ledger for the **Phase 2 Specialist Baselines** in SatQuery AI. In direct response to the research advisor's rigorous review, all numerical discrepancies, taxonomic classifications, spatial boundaries, and benchmark configurations have been resolved directly against raw dataset metadata, patch coordinates, and cryptographically fingerprinted prediction arrays:

1. **BigEarthNet Sample Accounting & Pair Arithmetic Verified**: The dataset contains **$13,683$ patches** across **four MGRS tiles**: three shared tiles (`T34TCR`, `T34TEQ`, `T34TFN`, $N = 13,056$) and one disjoint tile (`T34TCS`, $N = 627$, containing 573 train and 54 validation patches, with 0 test patches). The spatial audit examined **$7,578,668$ intra-tile pairs** where spatial autocorrelation is physically meaningful, while the Cartesian cross-products are **$21,459,536$ pairs** across the 3 shared tiles and **$23,320,640$ pairs** across all 4 tiles.
2. **Spatial Separation Terminology & Euclidean Distance Verification**:
   * The minimum Chebyshev separation is **$10.0$ patch units**, yielding a **$12.0\text{ km}$ minimum Chebyshev center-distance equivalent**.
   * An exhaustive coordinate-level calculation of actual Euclidean distances across all $7,578,668$ intra-tile test-train pairs proves that the actual minimum Euclidean distance is **identically $10.0000$ patch units ($12.000\text{ km}$)** because the closest test-train pairs (e.g. Test `[46, 68]` to Train `[46, 78]` in `T34TCR`; Test `[23, 68]` to Train `[23, 78]` in `T34TEQ`) are axis-aligned along row coordinates ($\Delta r = 0, \Delta c = 10$).
   * The **$10.8\text{ km}$ buffer** represents the physical **grid-aligned edge-to-edge separation**: $(\Delta c - 1) \times 1.2\text{ km} = 9 \times 1.2\text{ km} = 10.8\text{ km}$, guaranteeing 9 completely vacant patch widths between patch boundaries.
   * Qualified strictly as **spatially buffered intra-regional generalization**, acknowledging that while local boundary overlap and pixel leakage are eliminated, regional agro-climatic continuity prevents claiming complete statistical independence.
3. **CLC Taxonomy Evidence, Code Frequencies & Authoritative Mapping**:
   * Evaluated directly against the raw metadata column `labels` in [`BEN_14k/metadata.parquet`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/initial_trails/Dataset/archive.zip).
   * In the official BigEarthNet 19-class published nomenclature ([Sumbul et al., 2021](https://doi.org/10.1109/MGRS.2021.3089174), Table 1), **CLC 332 (Bare rocks)** is designated as **"Unlabeled" (Excluded)** due to spectral ambiguity with single-date optical Sentinel-2 imagery; it is not part of the active taxonomy.
   * Class `"Natural grassland and sparsely vegetated areas"` strictly aggregates **CLC 321 (Natural grasslands)** and **CLC 333 (Sparsely vegetated areas)**.
   * In the regional Serbian subset ([Aironi, 2023](https://huggingface.co/datasets/narendraaironi/bigearthnet-14k)), exactly 16 of the 19 classes occur, forming a closed vocabulary shared identically across training labels, `label_encoder.joblib`, `model.joblib` $(N, 16)$, and `y_test_ground_truth.npy`.
4. **Dual Macro F1 Evaluation Conventions Formalized**:
   * **Macro F1 Across All 16 Predefined Classes**: **$47.39\%$** (tuned), **$46.13\%$** (default $\tau=0.50$). Divides by 16, penalizing the benchmark for regional zero-support classes.
   * **Macro F1 Across 14 Test-Present Classes**: **$54.16\%$** (tuned), **$52.72\%$** (default $\tau=0.50$). Evaluates performance solely over classes with positive regional ground-truth support.
   * For `Moors, heathland...` (14 false alarms, 0 true positives), Precision is mathematically exact **$0.0000$** ($0/14$), and Recall is indeterminate ($0/0$, imputed to $0.0000$ via `zero_division=0`).
5. **Sentinel-1 SAR Preprocessing Bounds Justified**:
   * Replaced "terrestrial dynamic range" with **implementation-selected clipping range of $-35.0\text{ dB}$ to $+5.0\text{ dB}$**.
   * Documented that these bounds were selected a priori based on empirical quantile inspection strictly of the **training set** (covering $>99.8\%$ of all terrestrial training pixels without clipping valid vegetation, agricultural fields, or built-up surfaces) and established SAR processing literature ([Sumbul et al., 2021](https://doi.org/10.1109/MGRS.2021.3089174); [Schmitt et al., 2019](https://doi.org/10.1109/JSTARS.2019.2949747)), without any look-ahead to or parameter tuning on the validation or test splits.
6. **OSCD Regime Naming & Dual Per-Scene Reporting**:
   * Formally renamed Regime B to **Controlled Parameter Ablation** (since methods use distinct ablation parameters rather than a single fixed threshold).
   * Published full per-scene F1 and IoU tables across all 10 test scenes for **both** Regime A (Validation-Tuned) and Regime B (Controlled Parameter Ablation), preserving historical metrics while documenting ablation dynamics.
7. **Complete Prediction-Level Reproducibility & CLI Execution Verification**:
   * Published SHA-256 cryptographic hashes for raw test prediction arrays (`y_test_pred_lgb_tuned.npy`, `y_test_prob_matrix.npy`), ground-truth arrays (`y_test_ground_truth.npy`), and all 10 OSCD ground-truth masks (`cm.png`) alongside the composite prediction manifest `prediction_manifest.json` (fingerprinting all 80 OSCD prediction masks).
   * Provided full command outputs of `git rev-parse HEAD`, `git status --short`, `python -m pytest tests/ -q` (71 passed), and [`scripts/reproduce_from_saved_predictions.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/reproduce_from_saved_predictions.py) demonstrating 100% precision reproduction directly from disk.

---

## Section 1: Reconciled BigEarthNet Spatial Split & Sample Accounting

### 1.1 Complete 4-Tile Sample Breakdown
The BigEarthNet-14k dataset used in SatQuery AI consists of **$13,683$ patches** distributed across **four MGRS tiles** in the Republic of Serbia:
* **Three Shared Tiles (`T34TCR`, `T34TEQ`, `T34TFN`)**: Patches from these tiles were partitioned into spatially buffered training, validation, and test blocks ($N = 13,056$).
* **One Disjoint Tile (`T34TCS`)**: Located in South Banat north of Belgrade ($N = 627$). All patches were allocated exclusively to training ($573$) and validation ($54$), with **zero test patches**.

This accounts for the 627 patch difference between the shared-tile total ($13,056$) and the complete dataset total ($13,683$):

| MGRS Tile | Geography / Eco-Region | Train Patches | Val Patches | Test Patches | Total Patches | Test Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`T34TCR`** | Central Serbia / Morava River Basin | 2,119 | 1,141 | 1,604 | **4,864** | Evaluated in Test |
| **`T34TEQ`** | Western Serbia / Drina-Sava Plains | 2,782 | 1,286 | 1,278 | **5,346** | Evaluated in Test |
| **`T34TFN`** | Southeastern Banat Plains | 1,706 | 774 | 366 | **2,846** | Evaluated in Test |
| **`T34TCS`** | South Banat (Disjoint Partition) | 573 | 54 | **0** | **627** | *No Test Patches* |
| **Total** | **All 4 Tiles Combined** | **7,180** | **3,255** | **3,248** | **13,683** | **100% Accounted** |

### 1.2 Rigorous Pair Count Accounting
Depending on the experimental scope, there are three mathematically valid pair counts:

1. **Intra-Tile Pairs Evaluated ($7,578,668$ pairs)**:
   The physical spatial buffer is calculated exclusively within shared tiles where patches share a UTM projection plane:
   $$\text{Tile } T34TCR: \quad 1,604 \times 2,119 = 3,398,876 \text{ pairs}$$
   $$\text{Tile } T34TEQ: \quad 1,278 \times 2,782 = 3,555,396 \text{ pairs}$$
   $$\text{Tile } T34TFN: \quad 366 \times 1,706 = 624,396 \text{ pairs}$$
   $$\text{Total Intra-Tile Evaluated Pairs} = 3,398,876 + 3,555,396 + 624,396 = \mathbf{7,578,668 \text{ patch pairs}}$$

2. **Cartesian Pairs Across 3 Shared Tiles ($21,459,536$ pairs)**:
   The unconstrained product of all $6,607$ training patches in the three shared tiles and all $3,248$ test patches:
   $$6,607 \times 3,248 = \mathbf{21,459,536 \text{ patch pairs}}$$

3. **Cartesian Pairs Across All 4 Tiles ($23,320,640$ pairs)**:
   The total unconstrained Cartesian product of all $7,180$ training patches (including `T34TCS`) and all $3,248$ test patches:
   $$7,180 \times 3,248 = \mathbf{23,320,640 \text{ patch pairs}}$$

### 1.3 Spatial Buffer Verification & Mathematical Clarification
Audit executed via [`scripts/verify_spatial_distance.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/verify_spatial_distance.py), saved in [`artifacts/benchmarks/spatial_buffer_audit.json`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/spatial_buffer_audit.json):

* **Chebyshev Distance Formulation**:
  For patch coordinates $(r_1, c_1)$ and $(r_2, c_2)$, the grid Chebyshev distance is:
  $$D_{\text{Chebyshev}} = \max(|\Delta r|, |\Delta c|)$$
  Physical Chebyshev center-distance equivalent: $D_{\text{Chebyshev}} \times 1.2\text{ km} = 10.0 \times 1.2\text{ km} = \mathbf{12.0\text{ km}}$.

* **Euclidean Distance Formulation & Coordinate-Level Verification**:
  For patch coordinates $(r_1, c_1)$ and $(r_2, c_2)$, the Euclidean center-to-center distance is:
  $$D_{\text{Euclidean}} = 1.2 \sqrt{(\Delta r)^2 + (\Delta c)^2}\text{ km}$$
  Across all $7,578,668$ intra-tile pairs, the minimum Euclidean distance is **identically $10.0000$ patch units ($12.000\text{ km}$)**. This identity occurs because the nearest test and train patches align precisely along the same row ($\Delta r = 0, \Delta c = 10$):
  * In tile `T34TCR`: Closest test patch is $[46, 68]$ and closest train patch is $[46, 78]$ ($\Delta r = 0, \Delta c = 10 \implies D_{\text{Euclidean}} = 1.2 \sqrt{0^2 + 10^2} = 12.000\text{ km}$).
  * In tile `T34TEQ`: Closest test patch is $[23, 68]$ and closest train patch is $[23, 78]$ ($\Delta r = 0, \Delta c = 10 \implies D_{\text{Euclidean}} = 1.2 \sqrt{0^2 + 10^2} = 12.000\text{ km}$).
  * In tile `T34TFN`: Closest test patch is $[23, 23]$ and closest train patch is $[12, 23]$ ($\Delta r = 11, \Delta c = 0 \implies D_{\text{Euclidean}} = 1.2 \sqrt{11^2 + 0^2} = 13.200\text{ km}$).

* **Grid-Aligned Edge Separation**:
  Because each patch has an edge dimension of $1.2\text{ km}$ ($120\text{ pixels} \times 10\text{ m} = 1,200\text{ m}$), a coordinate difference of $\Delta c = 10$ patch units provides exactly 9 empty grid cells between patch boundaries:
  $$\text{Edge Buffer} = (\Delta c - 1) \times 1.2\text{ km} = (10 - 1) \times 1.2\text{ km} = \mathbf{10.8\text{ km}}$$

| Tile | Test Bounds $[r, c]$ | Train Bounds $[r, c]$ | Train in Test Box | Min Chebyshev (Units) | Center Chebyshev Equiv. (km) | Actual Min Euclidean (km) | Grid Edge Buffer (km) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T34TCR`** | $r \in [27, 68], c \in [23, 68]$ | $r \in [28, 90], c \in [0, 90]$ | **0** | **10.0** | **12.0 km** | **12.00 km** | **10.8 km** |
| **`T34TEQ`** | $r \in [23, 68], c \in [23, 68]$ | $r \in [0, 90], c \in [0, 90]$ | **0** | **10.0** | **12.0 km** | **12.00 km** | **10.8 km** |
| **`T34TFN`** | $r \in [23, 47], c \in [23, 44]$ | $r \in [0, 51], c \in [0, 90]$ | **0** | **11.0** | **13.2 km** | **13.20 km** | **12.0 km** |
| **`T34TCS`** | *(No test patches)* | $r \in [37, 90], c \in [69, 90]$ | **0** | Disjoint Tile | Disjoint Tile | Disjoint Tile | Disjoint Tile |
| **Overall** | — | — | **0** | **10.0** | **12.0 km** | **12.00 km** | **10.8 km** |

> [!IMPORTANT]
> **Scientific Qualification of Spatial Generalization**:
> A minimum grid-aligned spatial buffer of $10.8\text{ km}$ between patch edges prevents immediate boundary overlap and suppresses local spatial autocorrelation. However, it does **not** prove complete statistical independence. Because train and test blocks reside within the same Serbian Pannonian eco-region, they share regional soil types, climatic regimes, and agricultural calendars. Therefore, this split must be characterized strictly as **spatially buffered intra-regional generalization**, not out-of-domain or continental transfer.

---

## Section 2: Land-Cover Taxonomy Reconciliation & Authoritative Mapping

### 2.1 Authoritative Bibliographic Standards & Metadata Provenance
* **Foundational 19-Class Taxonomy**:
  G. Sumbul, A. de Wall, T. Kreuziger, F. Marcelino, H. Costa, P. Benevides, M. Caetano, B. Demir, V. Markl, *"BigEarthNet-MM: A Large-Scale, Multimodal, Multilabel Benchmark for Remote Sensing Image Understanding"*, **IEEE Geoscience and Remote Sensing Magazine (GRSM)**, vol. 9, no. 3, pp. 174-180, Sept. 2021. DOI: [10.1109/MGRS.2021.3089174](https://doi.org/10.1109/MGRS.2021.3089174). arXiv: [2105.07921](https://arxiv.org/abs/2105.07921).
* **Regional Dataset Source**:
  Narendra Aironi, *"bigearthnet-14k"*, Hugging Face Dataset Repository (`narendraaironi/bigearthnet-14k`), 2023. Contains the summer 2017 multimodal Sentinel-1 GRD and Sentinel-2 L2A paired archive for the Republic of Serbia, serialized in `BEN_14k/metadata.parquet` inside `initial_trails/Dataset/archive.zip`.
* **Exact Metadata Column Used**:
  Column: `labels` in `BEN_14k/metadata.parquet` (stores Python arrays of aggregated BigEarthNet class strings for each patch).

### 2.2 CORINE Land Cover (CLC) to BigEarthNet 19-Class Aggregation Scheme
In Sumbul et al. (2021, Table 1), the original 43 CORINE Land Cover (CLC) Level-3 classes were consolidated into 19 classes to eliminate spatial-spectral ambiguities in Sentinel-2 imagery. Crucially:
* **CLC 332 (Bare rocks)** is designated as **"Unlabeled" (Excluded)** from the 19-class taxonomy because bare rock surfaces cannot be reliably distinguished from other spectrally similar bare surfaces without multitemporal thermal or structural sounding. It is not subsumed into grassland or any other active category.
* **Natural grassland and sparsely vegetated areas** strictly combines **CLC 321 (Natural grasslands)** and **CLC 333 (Sparsely vegetated areas)**.
* **Regional Vocabulary**: Of the 19 standard classes, exactly 16 appear in continental Serbia (`country == 'Serbia'`). Three classes have zero occurrences: `Marine waters` (CLC 521, 522, 523; inland Serbia has no sea), `Coastal wetlands` (CLC 421, 422; no marine coast), and `Agro-forestry areas` (CLC 244; not delineated in Serbian CLC 2018).

The reproducible mapping implementation and exact sample counts across all 13,683 patches are presented below:

| CLC Code | Original CLC Level-3 Class Name | BigEarthNet 19-Class Nomenclature | Total Patches (BEN_14k) | Train ($N=7,180$) | Val ($N=3,255$) | Test ($N=3,248$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **111, 112** | Continuous & Discontinuous urban fabric | **Urban fabric** | 2,076 | 979 | 580 | 517 |
| **121** | Industrial or commercial units | **Industrial or commercial units** | 381 | 201 | 95 | 85 |
| **211, 212, 213** | Non-irrigated arable, Permanently irrigated, Rice fields | **Arable land** | 7,612 | 3,741 | 1,862 | 2,009 |
| **221, 222, 223** | Vineyards, Fruit trees & berries, Olive groves | **Permanent crops** | 271 | 176 | 49 | 46 |
| **231** | Pastures | **Pastures** | 1,512 | 704 | 354 | 454 |
| **242** | Complex cultivation patterns | **Complex cultivation patterns** | 4,192 | 2,356 | 844 | 992 |
| **243** | Land principally occupied by agriculture with natural vegetation | **Land principally occupied by agriculture...** | 4,785 | 2,606 | 1,180 | 999 |
| **311** | Broad-leaved forest | **Broad-leaved forest** | 7,068 | 3,745 | 1,658 | 1,665 |
| **312** | Coniferous forest | **Coniferous forest** | 442 | 272 | 129 | 41 |
| **313** | Mixed forest | **Mixed forest** | 780 | 510 | 195 | 75 |
| **321, 333** | Natural grasslands & Sparsely vegetated areas | **Natural grassland and sparsely vegetated areas** | 1,672 | 1,021 | 395 | 256 |
| **322, 323** | Moors and heathland & Sclerophyllous vegetation | **Moors, heathland and sclerophyllous vegetation** | 8 | 4 | 4 | 0 |
| **324** | Transitional woodland-shrub | **Transitional woodland, shrub** | 4,729 | 2,330 | 1,202 | 1,197 |
| **331** | Beaches, dunes, sands | **Beaches, dunes, sands** | 2 | 2 | 0 | 0 |
| **411, 412** | Inland marshes & Peat bogs | **Inland wetlands** | 356 | 140 | 91 | 125 |
| **511, 512** | Water courses & Water bodies | **Inland waters** | 1,096 | 406 | 329 | 361 |
| **332** | Bare rocks | *Unlabeled / Excluded in Sumbul et al. (2021)* | **0** | 0 | 0 | 0 |
| **244** | Agro-forestry areas | *Agro-forestry areas (0 in regional Serbia)* | **0** | 0 | 0 | 0 |
| **421, 422** | Salt marshes & Salines (Coastal wetlands) | *Coastal wetlands (0 in inland Serbia)* | **0** | 0 | 0 | 0 |
| **521, 522, 523** | Coastal lagoons, Estuaries, Sea & ocean | *Marine waters (0 in inland Serbia)* | **0** | 0 | 0 | 0 |

We verify that:
* Training labels in `data/train_full_metadata.parquet`
* Encoder classes in `artifacts/specialists/land_cover_phase2/label_encoder.joblib`
* Model outputs $(N, 16)$ in `artifacts/specialists/land_cover_phase2/model.joblib`
* Ground-truth evaluation arrays in `data/y_test_ground_truth.npy`
**all share identically this 16-class closed vocabulary**.

---

## Section 3: Reconciled Land-Cover Benchmark Performance

### 3.1 Dual Macro F1 Conventions & Zero-Support Mathematics
In the held-out test partition ($N = 3,248$ patches), two regional classes have zero ground-truth positive occurrences:
1. `Beaches, dunes, sands`: $0$ true positives, $0$ false positives $\implies P = 0/0, R = 0/0$ (imputed to $0.0000$ via `zero_division=0`).
2. `Moors, heathland and sclerophyllous vegetation`: $0$ true positives, $14$ false positives $\implies P = 0/14 = \mathbf{0.0000}$, $R = 0/0$ (indeterminate, imputed to $0.0000$ via `zero_division=0`), resulting in $F_1 = \mathbf{0.0000}$.

To prevent ambiguity, SatQuery AI reports both standard evaluation conventions:
* **Macro F1 Across All 16 Classes**: Arithmetic mean over all 16 predefined model output heads (penalizing the model for zero-support classes).
* **Macro F1 Across 14 Test-Present Classes**: Arithmetic mean evaluated solely across the 14 classes with positive test support ($\text{support} > 0$).

### 3.2 Specialist Performance Summary
Saved in [`artifacts/benchmarks/land_cover_reconciliation_audit.json`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/land_cover_reconciliation_audit.json):

| Metric | Phase 1 Baseline (RF, 84 Feats, $\tau=0.50$) | Phase 2 LightGBM (488 Feats, $\tau=0.50$) | Phase 2 LightGBM (488 Feats, Tuned $\tau$) |
| :--- | :---: | :---: | :---: |
| **Micro F1** | 68.61% | 73.28% | **72.99%** |
| **Macro F1 (All 16 Classes)** | 42.12% | 46.13% | **47.39%** |
| **Macro F1 (14 Test-Present)** | 48.14% | 52.72% | **54.16%** |
| **Hamming Loss** | 0.1084 | **0.0899** | 0.0976 |
| **Exact Subset Match** | 20.35% | **27.74%** | 22.66% |

### 3.3 Per-Class Performance on Held-Out Test Set ($N=3,248$, Tuned $\tau$)
Saved in [`artifacts/benchmarks/land_cover_per_class_test_metrics.csv`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/land_cover_per_class_test_metrics.csv):

| Class Name | Test Support | Precision | Recall | $F_1$ Score | Tuned $\tau$ | Mathematical Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Arable land** | 2,009 | 0.9221 | 0.9074 | **0.9147** | 0.40 | Dominant agricultural class |
| **Beaches, dunes, sands** | 0 | 0.0000 | 0.0000 | **0.0000** | 0.40 | 0 support; 0 predictions ($0/0 \to 0$) |
| **Broad-leaved forest** | 1,665 | 0.8701 | 0.9249 | **0.8967** | 0.40 | Dominant deciduous forest |
| **Complex cultivation patterns** | 992 | 0.6365 | 0.6815 | **0.6582** | 0.35 | Small parcel mosaic |
| **Coniferous forest** | 41 | 0.5000 | 0.4390 | **0.4675** | 0.35 | Minority montane class |
| **Industrial or commercial units** | 85 | 0.2958 | 0.2471 | **0.2692** | 0.35 | Impervious industrial fabric |
| **Inland waters** | 361 | 0.8453 | 0.8172 | **0.8310** | 0.30 | Rivers and open reservoirs |
| **Inland wetlands** | 125 | 0.3333 | 0.5440 | **0.4134** | 0.25 | Marshes and floodplains |
| **Land principally agriculture...** | 999 | 0.6324 | 0.8318 | **0.7185** | 0.35 | Heterogeneous agro-ecological land |
| **Mixed forest** | 75 | 0.2941 | 0.3333 | **0.3125** | 0.30 | Mixed canopy stands |
| **Moors, heathland...** | 0 | 0.0000 | 0.0000 | **0.0000** | 0.40 | 0 support; 14 false positives ($P=0/14=0.0$) |
| **Natural grassland...** | 256 | 0.4926 | 0.2617 | **0.3418** | 0.40 | Steppe and sparse vegetation |
| **Pastures** | 454 | 0.3262 | 0.4714 | **0.3856** | 0.25 | Grazing meadows |
| **Permanent crops** | 46 | 0.0769 | 0.0435 | **0.0556** | 0.20 | Vineyards and orchards (extreme minority) |
| **Transitional woodland, shrub** | 1,197 | 0.5249 | 0.7669 | **0.6232** | 0.35 | Bushland, clear-cuts, young forest |
| **Urban fabric** | 517 | 0.6973 | 0.6905 | **0.6939** | 0.35 | Residential settlements |

---

## Section 4: Sensor Physical Specifications & Implementation Bounds

### 4.1 Sentinel-2 BOA Reflectance Scaling & Radiometric Offset
* **Level-2A BOA Surface Reflectance Formula**:
  $$\rho_{\lambda} = \frac{\text{DN} + \text{BOA\_ADD\_OFFSET}_{\lambda}}{\text{QUANTIFICATION\_VALUE}}$$
* **Operational Product Transition**:
  * **Archival Baseline (PB $<$ 04.00, Summer 2017 Dataset)**: $\text{BOA\_ADD\_OFFSET} = 0, \text{QUANTIFICATION\_VALUE} = 10,000.0 \implies \rho_{\lambda} = \text{DN} / 10,000.0$.
  * **Current Operational Products (PB $\ge$ 04.00, Jan 25, 2022 onward)**: To prevent truncation of negative reflectances in deep shadows or dark water bodies, ESA introduced an explicit negative offset of $-1,000\text{ DN}$ ($0.10$ reflectance units):
    $$\rho_{\lambda} = \frac{\text{DN} - 1000}{10000.0}$$
  SatQuery AI inspects product metadata and applies this offset conditionally.

### 4.2 Sentinel-1 SAR Calibration Pipeline & Implementation Clipping Bounds
* **Acquisition Mode**: Sentinel-1 C-band SAR in Interferometric Wide (IW) swath mode, Ground Range Detected (GRD) high-resolution ($10\text{ m}$ GSD).
* **Exact Preprocessing Sequence**:
  1. **Orbit Vector Application**: Precise orbit ephemerides (POEORB) applied.
  2. **GRD Border Noise Removal**: Edge-of-swath invalid samples masked.
  3. **Thermal Noise Removal**: Subtraction of thermal noise power using the product noise lookup table (LUT).
  4. **Radiometric Calibration**: Raw digital numbers converted to radar backscatter coefficient $\sigma^0$:
     $$\sigma^0 = \frac{\text{DN}^2}{A_i^2}$$
     where $A_i$ is the bilinear interpolation of the calibration vector from the calibration metadata LUT.
  5. **Range-Doppler Terrain Correction**: Geometric orthorectification using the SRTM 1 arc-second global DEM to project backscatter to UTM 34N / WGS84 coordinates.
  6. **Decibel Conversion & Implementation Clipping Bounds**:
     $$\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\sigma^0 + 10^{-6})$$
     Backscatter values are clipped to the **implementation-selected range of $-35.0\text{ dB}$ to $+5.0\text{ dB}$** before normalization via $(\sigma^0_{\text{dB}} + 35.0) / 40.0$.
     * *Justification of Bounds*: These bounds were selected a priori based on empirical quantile inspection strictly of the **training set** (covering $>99.8\%$ of all terrestrial training pixels without clipping valid vegetation, agricultural fields, or built-up surfaces) and established SAR processing conventions ([Sumbul et al., 2021](https://doi.org/10.1109/MGRS.2021.3089174); [Schmitt et al., 2019](https://doi.org/10.1109/JSTARS.2019.2949747)), **without any look-ahead to or optimization on the validation or test splits**. Values below $-35\text{ dB}$ represent radar shadow or calm open water falling into the sensor thermal noise floor (NESZ $\approx -22\text{ dB}$ to $-28\text{ dB}$), while values above $+5\text{ dB}$ represent outsized dihedral metallic corner reflectors.
* **Dual-Polarization vs Polarimetric SAR**:
  Sentinel-1 operates in **dual-polarization mode** (co-polarized VV and cross-polarized VH). This does **not** constitute full polarimetric (quad-pol) SAR, which requires the complex scattering matrix $[S]$ with inter-channel phase. All SAR features in SatQuery AI are rigorously defined as **dual-polarization intensity and texture proxies** (e.g., cross-pol ratio $\text{VH}/\text{VV}$, normalized polarization difference index $\text{NDPI}$).

---

## Section 5: OSCD Change Detection Multi-Regime Reconciliation

### 5.1 Operating Parameters: Regime A vs Regime B
The two benchmark regimes evaluate distinct operational configurations across all 10 held-out test scenes:
1. **Regime A: Validation-Tuned Parameters (Historical Benchmark)**:
   Thresholds tuned on the **4 validation scenes** (`montpellier`, `paris`, `rio`, `valencia`) via [`scripts/experiment_change_detection.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/experiment_change_detection.py):
   * Standard CVA: $\tau = 0.28, \text{min\_area} = 9$
   * Radiometric CVA: $\mathbf{\tau = 0.22}, \text{min\_area} = 9$
   * Gaussian-Smoothed CVA: $\mathbf{\tau = 0.20}, \sigma = 1.2, \text{min\_area} = 9$
   * Dual-Threshold Hysteresis CVA: $\mathbf{\tau_{\text{high}} = 0.24}, \mathbf{\tau_{\text{low}} = 0.14}, \mathbf{\sigma = 1.0}, \text{min\_area} = 9$
2. **Regime B: Controlled Parameter Ablation**:
   Evaluates the direct algorithmic effect when parameters are held near the baseline operating point:
   * Standard CVA: $\tau = 0.28, \text{min\_area} = 9$
   * Radiometric CVA: $\mathbf{\tau = 0.28}, \text{min\_area} = 9$
   * Gaussian-Smoothed CVA: $\mathbf{\tau = 0.26}, \sigma = 1.2, \text{min\_area} = 9$
   * Dual-Threshold Hysteresis CVA: $\mathbf{\tau_{\text{high}} = 0.28}, \mathbf{\tau_{\text{low}} = 0.14}, \mathbf{\sigma = 0.0} \text{ (raw } \rho\text{)}, \text{min\_area} = 9$

### 5.2 Pooled Performance Comparison Across Regimes
Saved in [`artifacts/benchmarks/oscd_methods_pooled_comparison.csv`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/oscd_methods_pooled_comparison.csv):

| Benchmark Regime | Method | Pooled Precision | Pooled Recall | Pooled $F_1$ | Pooled IoU | True Positives (TP) | False Negatives (FN) | False Positives (FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regime A (Validation-Tuned)** | Standard CVA ($\tau=0.28$) | 0.5820 | 0.1824 | **0.2778** | 0.1613 | 29,016 | 130,061 | 20,838 |
| **Regime A (Validation-Tuned)** | Radiometrically Aligned CVA ($\tau=0.22$) | 0.4878 | 0.2158 | **0.2992** | 0.1759 | 34,330 | 124,747 | 36,054 |
| **Regime A (Validation-Tuned)** | Gaussian-Smoothed CVA ($\tau=0.20$) | 0.3970 | 0.2963 | **0.3393** | 0.2043 | 47,133 | 111,944 | 71,594 |
| **Regime A (Validation-Tuned)** | Dual-Threshold Hysteresis CVA | 0.2612 | 0.4430 | **0.3287** | 0.1966 | 70,470 | 88,607 | 199,283 |
| **Regime B (Parameter Ablation)** | Standard CVA ($\tau=0.28$) | 0.5820 | 0.1824 | **0.2778** | 0.1613 | 29,016 | 130,061 | 20,838 |
| **Regime B (Parameter Ablation)** | Radiometrically Aligned CVA ($\tau=0.28$) | 0.5996 | 0.1195 | **0.1992** | 0.1106 | 19,002 | 140,075 | 12,687 |
| **Regime B (Parameter Ablation)** | Gaussian-Smoothed CVA ($\tau=0.26$) | 0.6268 | 0.1818 | **0.2818** | 0.1640 | 28,915 | 130,162 | 17,215 |
| **Regime B (Parameter Ablation)** | Dual-Threshold Hysteresis CVA ($\tau_{\text{high}}=0.28$) | 0.2535 | 0.4360 | **0.3206** | 0.1909 | 69,364 | 89,713 | 204,219 |

### 5.3 Per-Scene Performance: Regime A (Validation-Tuned)
Saved in [`artifacts/benchmarks/oscd_historical_validation_tuned_per_scene.csv`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/oscd_historical_validation_tuned_per_scene.csv):

| Scene | GT Change Pixels | GT Change % | Standard CVA $F_1$ (IoU) | Radiometric CVA $F_1$ (IoU) | Gaussian CVA $F_1$ (IoU) | Hysteresis CVA $F_1$ (IoU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`aguasclaras`** | 8,552 | $5.69\%$ | 0.2341 (0.1325) | 0.3193 (0.1900) | 0.3336 (0.2002) | **0.4136** (0.2607) |
| **`austin`** | 5,018 | $1.14\%$ | 0.0561 (0.0288) | **0.0593** (0.0306) | 0.0298 (0.0151) | 0.0512 (0.0263) |
| **`beihai`** | 969 | $0.44\%$ | 0.0384 (0.0196) | **0.0583** (0.0300) | 0.0266 (0.0135) | 0.0134 (0.0067) |
| **`berlin`** | 28,637 | $7.21\%$ | 0.0623 (0.0321) | 0.0928 (0.0486) | 0.1259 (0.0672) | **0.2645** (0.1524) |
| **`brasilia`** | 13,054 | $6.79\%$ | 0.4490 (0.2895) | 0.5391 (0.3691) | 0.6515 (0.4832) | **0.6958** (0.5336) |
| **`budapest`** | 5,242 | $2.58\%$ | 0.1969 (0.1092) | 0.1290 (0.0689) | **0.4195** (0.2655) | 0.3915 (0.2434) |
| **`campinas`** | 1,227 | $1.32\%$ | 0.0696 (0.0360) | **0.1041** (0.0549) | 0.0607 (0.0313) | 0.0703 (0.0364) |
| **`mumbai`** | 48,689 | $9.92\%$ | 0.1066 (0.0563) | 0.2474 (0.1412) | 0.1892 (0.1045) | **0.3445** (0.2081) |
| **`nantes`** | 2,419 | $0.80\%$ | 0.1560 (0.0846) | 0.2035 (0.1133) | 0.2185 (0.1227) | **0.3590** (0.2188) |
| **`pisa`** | 45,270 | $7.67\%$ | 0.5583 (0.3872) | 0.5136 (0.3455) | **0.6758** (0.5103) | 0.6662 (0.4994) |
| **Macro Avg** | — | — | **0.1927** (0.1166) | **0.2266** (0.1362) | **0.2731** (0.1813) | **0.3270** (0.2186) |

### 5.4 Per-Scene Performance: Regime B (Controlled Parameter Ablation)
Saved in [`artifacts/benchmarks/oscd_fixed_threshold_ablation_per_scene.csv`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/oscd_fixed_threshold_ablation_per_scene.csv):

| Scene | GT Change Pixels | GT Change % | Standard CVA $F_1$ (IoU) | Radiometric CVA $F_1$ (IoU) | Gaussian CVA $F_1$ (IoU) | Hysteresis CVA $F_1$ (IoU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`aguasclaras`** | 8,552 | $5.69\%$ | 0.2341 (0.1325) | 0.2507 (0.1433) | 0.2223 (0.1251) | **0.3752** (0.2309) |
| **`austin`** | 5,018 | $1.14\%$ | 0.0561 (0.0288) | **0.0717** (0.0372) | 0.0457 (0.0234) | 0.0590 (0.0304) |
| **`beihai`** | 969 | $0.44\%$ | 0.0384 (0.0196) | 0.0274 (0.0139) | **0.0470** (0.0241) | 0.0123 (0.0062) |
| **`berlin`** | 28,637 | $7.21\%$ | 0.0623 (0.0321) | 0.0506 (0.0260) | 0.0492 (0.0252) | **0.2759** (0.1601) |
| **`brasilia`** | 13,054 | $6.79\%$ | 0.4490 (0.2895) | 0.3943 (0.2456) | 0.4732 (0.3100) | **0.6088** (0.4376) |
| **`budapest`** | 5,242 | $2.58\%$ | 0.1969 (0.1092) | 0.0773 (0.0402) | 0.2074 (0.1157) | **0.3091** (0.1828) |
| **`campinas`** | 1,227 | $1.32\%$ | 0.0696 (0.0360) | **0.1240** (0.0661) | 0.0633 (0.0327) | 0.0675 (0.0349) |
| **`mumbai`** | 48,689 | $9.92\%$ | 0.1066 (0.0563) | 0.1317 (0.0705) | 0.0774 (0.0402) | **0.3674** (0.2250) |
| **`nantes`** | 2,419 | $0.80\%$ | 0.1560 (0.0846) | 0.1312 (0.0702) | 0.0996 (0.0524) | **0.3269** (0.1954) |
| **`pisa`** | 45,270 | $7.67\%$ | 0.5583 (0.3872) | 0.3281 (0.1963) | 0.5850 (0.4135) | **0.6083** (0.4371) |
| **Macro Avg** | — | — | **0.1927** (0.1166) | **0.1587** (0.0909) | **0.1829** (0.1142) | **0.3010** (0.1940) |

### 5.5 Empirical Qualification of Radiometric Normalization Dynamics
Global band-level radiometric normalization (channel-wise mean and standard-deviation matching) is characterized as an **empirical observation on the tested scenes and implementation**, rather than a universal theorem:
1. **Scenes with Prominent Physical Change**:
   In scenes where real land-use transitions occupy a substantial fraction of the image (e.g. `pisa` at $7.67\%$ change, `budapest` at $2.58\%$), global scene statistics $(\mu, \sigma)$ are materially shifted by the changed pixels themselves. Forcing $\mu_{T2} \to \mu_{T1}$ artificially pulls changed pixels toward the background mean, shrinking the spectral difference magnitude $\|\Delta \vec{\rho}\|_2$ below detection thresholds and suppressing true-positive detections.
2. **Scenes with Negligible Change and Atmospheric Drift**:
   In scenes with very low ground-truth change prevalence (e.g. `campinas` at $1.32\%$, `beihai` at $0.44\%$, `austin` at $1.14\%$), whole-image differences are dominated by atmospheric haze and illumination angles. In these scenes, radiometric alignment successfully suppressed false alarms and improved $F_1$ ($0.0696 \to 0.1041$ in `campinas`; $0.0384 \to 0.0583$ in `beihai`).

---

## Section 6: Complete Prediction-Level Reproducibility Ledger & Execution Evidence

Every dataset manifest, feature matrix, model artifact, raw prediction file, and ground-truth mask has been cryptographically fingerprinted via SHA-256:

### 6.1 Prediction Arrays & Ground-Truth Masks Ledger
| Component Category | Target File Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Land Cover GT (Test)** | `data/y_test_ground_truth.npy` | `d776b34d80fc8ff9b19022ec0d3ccdf938b1c9bbf7867a231d4b7d1058dbfb23` |
| **Land Cover GT (Val)** | `data/y_val_ground_truth.npy` | `5faa7a958191c21cbc99888d181e530064ac8f6103214cabd7b0c4d8cd0f08f7` |
| **Land Cover Probabilities** | `artifacts/specialists/land_cover_phase2/predictions/y_test_prob_matrix.npy` | `b11396cced559df09a21a9ed863221fe7537217858f3d4e0f7b9464cabbd06cc` |
| **Land Cover Pred (Tuned)** | `artifacts/specialists/land_cover_phase2/predictions/y_test_pred_lgb_tuned.npy` | `9de9304ac35ce88e71fbb55ef3ae816334f2737fa3c28defc08a01678efe7e68` |
| **Land Cover Pred (Default)**| `artifacts/specialists/land_cover_phase2/predictions/y_test_pred_lgb_default.npy` | `83e928609c5d007f9af96ae570c67936478cf49dc294830f75158d26832d14d5` |
| **OSCD Prediction Manifest**| `artifacts/specialists/change_detection/predictions/prediction_manifest.json` | `854c277ca26737331477fc262f7b3b4d04ba0bdbd652cb7935504d88cd9c9207` |
| **OSCD GT: aguasclaras** | `data/oscd/test/aguasclaras/cm.png` | `ce69194158f54fa986aa3a41664572c6888cdbe5ac12d9e09813af466a2a4286` |
| **OSCD GT: austin** | `data/oscd/test/austin/cm.png` | `fef6d022c612911d7960303eebbfb411358195dd21bfca72dcfed679bafb2cd2` |
| **OSCD GT: beihai** | `data/oscd/test/beihai/cm.png` | `cc55d25958b5de29384a0fe7c36a8d17e274534c19d3ec4109e263de9ee323c7` |
| **OSCD GT: berlin** | `data/oscd/test/berlin/cm.png` | `03009cab3ef9113e6131503e086eb733f30317223358fb9db10302acdc17971d` |
| **OSCD GT: brasilia** | `data/oscd/test/brasilia/cm.png` | `4c2415173c85647f133ef7bfac5bc0a7d6a67b9a8399472fe6ef464b56a2ab33` |
| **OSCD GT: budapest** | `data/oscd/test/budapest/cm.png` | `d1199ede89faf0dbafbc8ac52ee08fe5cbfb6cfa0006d2a7a96348310f6ac282` |
| **OSCD GT: campinas** | `data/oscd/test/campinas/cm.png` | `605ae8af713f00842e3add08af0b99b8e4063ade1020fe6df1782cd4749366a1` |
| **OSCD GT: mumbai** | `data/oscd/test/mumbai/cm.png` | `1d4d4b6821ad72843c59a621f48318b8174ba2cdaf262876135ce593279b0e0c` |
| **OSCD GT: nantes** | `data/oscd/test/nantes/cm.png` | `790dd22bdfd243fb47b7665b1901afe65ae9d8e079909ba2158d17425196f863` |
| **OSCD GT: pisa** | `data/oscd/test/pisa/cm.png` | `afabd431a75361c49273ddada203958d7e5d9af7656e1cb494c9fb18bca94fe8` |

> [!NOTE]
> All 80 individual OSCD prediction masks across Regime A (40 masks) and Regime B (40 masks) are tracked and individually hashed inside [`prediction_manifest.json`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/specialists/change_detection/predictions/prediction_manifest.json).

### 6.2 Metadata, Features, Models & Scripts Ledger
| File Category | Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Dataset Metadata** | `data/train_full_metadata.parquet` | `268734bf885b085cda740bfff72cb1688599eea5b0710cc131b8f8673e9083c4` |
| **Dataset Metadata** | `data/val_full_metadata.parquet` | `807b23eb45e7438d7f7e930bb88eb2b7724cba68a94f5ac13f8ee7907ce29a83` |
| **Dataset Metadata** | `data/test_full_metadata.parquet` | `069bca852cce37a28081c20f9fc1ed731fcedfcb129d01400f34f0fdcc5aa59c` |
| **Extracted Features** | `data/X_train_phase2.npy` | `b3bc80385d1c625e5e707e2bd503fd81d9ce1b80cf80b136611fd965134ee78a` |
| **Extracted Features** | `data/X_val_phase2.npy` | `79c453fa50ab1ecadd5042e5bf1d25d16d9a2bfad46bfe0f7debc07520409289` |
| **Extracted Features** | `data/X_test_phase2.npy` | `4b4e6aa09f08d43ece6b8fea5512e8ad06111c977c1467a1aec26b09763e0b72` |
| **Specialist Model** | `artifacts/specialists/land_cover_phase2/model.joblib` | `f785ca1f5e9691c17e4a17ebf3f6766f7053059ba871188d24f6dce0ed43d071` |
| **Label Encoder** | `artifacts/specialists/land_cover_phase2/label_encoder.joblib` | `d7e78865b4f21643cd669246dfafb14bb14ad2142b39dcb931ea28becb9af042` |
| **Class Taxonomy** | `artifacts/specialists/land_cover_phase2/classes.json` | `cdb18c8a0a9dc74f52920720340251f14de323bc5c4fb4f07436f90f3500f3e9` |
| **Threshold Config** | `artifacts/specialists/land_cover_phase2/threshold_config.json` | `74f1e5f1bccd8e9751a19ad3936aef528e234e69b33c9a7d34a4b20c6f2a3a10` |
| **Spatial Audit Script** | `scripts/verify_spatial_distance.py` | `a013aac68821d74a2204b3021332107e8742466669bb4d6aa6451cbdf792f298` |
| **Land Cover Evaluator** | `scripts/reproduce_land_cover_evaluation.py` | `4c7d885210d4f1a32094f83dacb656e3a8a9d9727e2a5982223304f455a63d92` |
| **OSCD Evaluator** | `scripts/reproduce_oscd_per_scene_evaluation.py` | `104794db8f9c89270271a280f354684adfa61ea152fec08b5fb49785c0ba2c79` |
| **Prediction Reproduction Script** | `scripts/reproduce_from_saved_predictions.py` | `4f685389652d1dbb7e8f3615703b7d82ee72b373cbba1636d9ac7cd1d49c5f93` |
| **Spatial Audit JSON** | `artifacts/benchmarks/spatial_buffer_audit.json` | `1a873e3d804a0ba5e5d62e87e855ca8d7b90a8d831233f4b1490153d05016acb` |
| **Land Cover Audit JSON** | `artifacts/benchmarks/land_cover_reconciliation_audit.json` | `d529af4dccc53e98e2d3c96b30d8507cb9a52d7ddb02cdfed9d93ca55f714a56` |
| **Per-Class Test CSV** | `artifacts/benchmarks/land_cover_per_class_test_metrics.csv` | `76a84253dc6fcf90f4c2c314b9aade37932fc81b56791eab2c728d1f767af9d8` |
| **OSCD Pooled CSV** | `artifacts/benchmarks/oscd_methods_pooled_comparison.csv` | `8233fef2808fedbad3418850c2545f88d27dc562c171f9e03243ef9be6f3165c` |
| **OSCD Per-Scene CSV** | `artifacts/benchmarks/oscd_per_scene_evaluation_breakdown.csv` | `8dbdaaa0a65789f33352a2186a1d8a58ebd640e21470ec8489b394ed5166ec7b` |

### 6.3 Actual CLI Verification & Reproduction Outputs

#### Command 1: Git Working Tree & Commit Verification
```powershell
PS C:\Users\B NAVANEETH\SatQueryAI> git rev-parse HEAD
a02c0a8dfe7ad65556a22a9a469626be0558af21

PS C:\Users\B NAVANEETH\SatQueryAI> git status --short
(empty - working tree clean)
```

#### Command 2: Regression Test Suite Execution
```powershell
PS C:\Users\B NAVANEETH\SatQueryAI> python -m pytest tests/ -q
.......................................................................  [100%]
71 passed in 47.72s
```

#### Command 3: Independent Metric Reproduction Directly from Saved Predictions
```powershell
PS C:\Users\B NAVANEETH\SatQueryAI> python scripts/reproduce_from_saved_predictions.py
SATQUERY AI: INDEPENDENT PREDICTION-LEVEL REPRODUCTION REPORT

================================================================================
1. CRYPTOGRAPHIC PREDICTION & GROUND TRUTH INTEGRITY AUDIT
================================================================================
  [VERIFIED] data/y_test_ground_truth.npy -> d776b34d80fc8ff9...
  [VERIFIED] data/y_val_ground_truth.npy -> 5faa7a958191c21c...
  [VERIFIED] artifacts/specialists/land_cover_phase2/predictions/y_test_prob_matrix.npy -> b11396cced559df0...
  [VERIFIED] artifacts/specialists/land_cover_phase2/predictions/y_test_pred_lgb_tuned.npy -> 9de9304ac35ce88e...
  [VERIFIED] artifacts/specialists/land_cover_phase2/predictions/y_test_pred_lgb_default.npy -> 83e928609c5d007f...
  [VERIFIED] artifacts/specialists/change_detection/predictions/prediction_manifest.json -> 854c277ca2673733...
  [VERIFIED] data/oscd/test/aguasclaras/cm.png -> ce69194158f54fa9...
  [VERIFIED] data/oscd/test/austin/cm.png -> fef6d022c612911d...
  [VERIFIED] data/oscd/test/beihai/cm.png -> cc55d25958b5de29...
  [VERIFIED] data/oscd/test/berlin/cm.png -> 03009cab3ef9113e...
  [VERIFIED] data/oscd/test/brasilia/cm.png -> 4c2415173c85647f...
  [VERIFIED] data/oscd/test/budapest/cm.png -> d1199ede89faf0db...
  [VERIFIED] data/oscd/test/campinas/cm.png -> 605ae8af713f0084...
  [VERIFIED] data/oscd/test/mumbai/cm.png -> 1d4d4b6821ad7284...
  [VERIFIED] data/oscd/test/nantes/cm.png -> 790dd22bdfd243fb...
  [VERIFIED] data/oscd/test/pisa/cm.png -> afabd431a75361c4...
  [VERIFIED] All 80 OSCD prediction masks match prediction_manifest.json
Integrity Status: ALL MATCH (PASS)

================================================================================
2. REPRODUCING LAND COVER SPECIALIST TEST METRICS FROM SAVED PREDICTION ARRAYS
================================================================================
Test Set Evaluation (N = 3248 patches, 16 classes):
  Validation-Tuned Thresholds:
    - Micro F1:                  72.99%
    - Macro F1 (All 16 Classes):  47.39%
    - Macro F1 (14 Test-Present):54.16%
    - Hamming Loss:              0.0976
    - Exact Subset Match:        22.66%

  Default Fixed Thresholds (tau = 0.50):
    - Micro F1:                  73.28%
    - Macro F1 (All 16 Classes):  46.13%
    - Macro F1 (14 Test-Present):52.72%
    - Hamming Loss:              0.0899
    - Exact Subset Match:        27.74%

================================================================================
3. REPRODUCING OSCD CHANGE DETECTION METRICS DIRECTLY FROM PREDICTION MASKS
================================================================================

--- Regime A (Validation-Tuned Thresholds) ---
         Method Precision (%) Recall (%) Pooled F1 Pooled IoU TP Pixels FP Pixels FN Pixels
   standard_cva        58.20%     18.24%    0.2778     0.1613    29,016    20,838   130,061
radiometric_cva        48.78%     21.58%    0.2992     0.1759    34,330    36,054   124,747
   gaussian_cva        39.70%     29.63%    0.3393     0.2043    47,133    71,594   111,944
 hysteresis_cva        26.12%     44.30%    0.3287     0.1966    70,470   199,283    88,607

--- Regime B (Controlled Parameter Ablation) ---
         Method Precision (%) Recall (%) Pooled F1 Pooled IoU TP Pixels FP Pixels FN Pixels
   standard_cva        58.20%     18.24%    0.2778     0.1613    29,016    20,838   130,061
radiometric_cva        59.96%     11.95%    0.1992     0.1106    19,002    12,687   140,075
   gaussian_cva        62.68%     18.18%    0.2818     0.1640    28,915    17,215   130,162
 hysteresis_cva        25.35%     43.60%    0.3206     0.1909    69,364   204,219    89,713

================================================================================
ALL REPRODUCTION CHECKS COMPLETED SUCCESSFULLY.
================================================================================
```

---

## Section 7: Remaining Scientific Limitations & Scope Boundaries

While the classical baselines are now fully verified and reproducible, the following scientific boundaries remain formally recognized for Phase 2:

1. **Intra-Regional Generalization Boundary**:
   The $10.8\text{ km}$ buffer effectively eliminates spatial overlap and pixel-level boundary leakage, but it does not represent out-of-domain cross-continental generalization. Future phases will evaluate cross-country generalization across European and Asian geographic domains.
2. **Minority Class Deficit in Classical Specialist**:
   Extreme minority classes (`Permanent crops` at $F_1 = 0.0556$, `Moors, heathland...` at $F_1 = 0.0000$) demonstrate the capacity limitations of gradient-boosted decision trees under heavy class imbalance. This directly motivates the transition to deep-learning architectures trained with class-frequency weighting or focal loss.
3. **Change Detection False Positives in Dense Urban Fabics**:
   Classical vector analysis (CVA) exhibits high false-alarm rates on urban shadow dynamics and registration artifacts (e.g. `berlin` and `mumbai`). This establishes the empirical baseline against which deep Siamese feature difference networks (e.g., BIT, ChangeFormer) will be benchmarked.
4. **Separation of Router Novelty from Specialist Capacity**:
   Specialist accuracy improvements ($F_1$, IoU) represent perception-level engineering progress. The central research contribution of SatQuery AI remains **sensor- and task-compatibility-constrained selective routing with calibrated uncertainty**, which will be evaluated independently of specialist model capacity.

---

**Verification Signed**: Lead Verification Agent  
**Baseline Status**: **PERMANENTLY FROZEN** (All 71 tests passing, full prediction-level evidence verified).
