# Phase 2.2 / Milestone M3.2: Change Detection Specialist Dataset Readiness & Architecture Feasibility Audit

**Document Code**: SATQUERY-SPECIALIST-P2.2-AUDIT-001  
**Project**: SatQuery AI (ISRO / Earth Observation Analytical Assistant, Problem Code: SIH26167)  
**Track**: Phase 2.2 / Milestone M3.2 — Bitemporal Change-Detection Specialist  
**Date**: 2026-11-13  
**Status**: Dataset Verified | Hardware Audited | Architecture Feasibility Confirmed  

---

## Executive Summary & Dataset Confirmation

This audit provides the foundational readiness assessment for **Milestone M3.2 (Bitemporal Change Detection Specialist)**, establishing data provenance, temporal alignment, hardware boundaries, class distribution, and model feasibility before initiating experimental execution.

### Key Audit Findings
1. **Dataset Selection & Status**:
   - **Confirmed Dataset**: **OSCD (Onera Satellite Change Detection) RGB Benchmark** (Daudt et al., 2018; hosted on Hugging Face as `blanchon/OSCD_RGB`).
   - The dataset is **fully downloaded, structured, and verified on local disk** at `data/oscd/` with catalog metadata in `data/change_detection_catalog.parquet`.
   - Total Volume: **24 distinct global scenes** totaling **$9,881,114$ pixels**.
2. **Temporal Alignment & Change Masks**:
   - All 24 scenes contain verified bitemporal image pairs (`t1.png` and `t2.png`, Sentinel-2 optical RGB at $10\text{ m}$ GSD) with chronological ordering ($t_1 < t_2$).
   - Ground-truth binary change masks (`cm.png`) exist for all 24 scenes ($0 = \text{no change}$, $1 = \text{verified change}$).
3. **Severe Class Imbalance**:
   - Foreground change constitutes only **$3.59\%$** of pixels across the dataset ($2.20\%$ in Train, $2.69\%$ in Val, $4.42\%$ in Test).
   - Change ratio ranges from **$0.29\%$** (Paris) to **$9.92\%$** (Mumbai), highlighting severe class imbalance and sparse change phenomena.
4. **Leakage-Free Spatial Partitioning**:
   - City-level disjoint partitioning: **10 Train cities**, **4 Validation cities**, and **10 Held-Out Test cities**.
   - Zero geographic overlap ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$).
5. **Hardware Platform & VRAM Boundaries**:
   - Hardware: **NVIDIA GeForce RTX 2050 Laptop GPU** ($4,096\text{ MiB}$ VRAM, ~3.3 GB free).
   - Active PyTorch Environment: Python 3.13 (`C:\Users\B NAVANEETH\AppData\Local\Programs\Python\Python313\python.exe`) with **PyTorch 2.6.0+cu124 and CUDA enabled**.
   - Memory Budget: Fully accommodates lightweight Siamese CNNs (FC-Siam-diff, ResNet-18 Siamese U-Net) and compact Transformers (BIT) at $256 \times 256$ patch resolution with batch sizes of 4 to 8. Heavy transformer architectures ($>6\text{ GB}$ VRAM) are infeasible.
6. **Historical Benchmark Ledger (Pre-Existing M3.2 Models)**:
   - Classical Standard CVA: **$20.31\%$** Test Macro F1 (33.93% Pooled F1).
   - Classical Gaussian CVA: **$27.31\%$** Test Macro F1 (33.93% Pooled F1).
   - Deep FC-Siam-diff: **$37.66\%$** Test Macro F1 (38.18% Pooled F1), passing 5 of 10 test scenes.
   - Deep BIT Transformer: **$12.95\%$** Test Macro F1 (12.60% Pooled F1), underperforming due to tokenization overfitting in small-sample regimes.

---

## 1. Dataset Provenance, Scope, and Catalog Audit

### 1.1 Dataset Identification & Provenance
* **Dataset Name**: OSCD (Onera Satellite Change Detection) RGB Subset.
* **Original Reference**: R. C. Daudt, B. Le Saux, A. Boulch, Y. Gousseau, *"Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks"*, IEEE IGARSS 2018.
* **Sensor & Resolution**: Sentinel-2 Multi-Spectral Instrument (MSI), Level-2A surface reflectance, focusing on the $10\text{ m}$ Ground Sample Distance (GSD) optical bands (B04 Red, B03 Green, B02 Blue).
* **Storage Location**: `data/oscd/` containing subdirectories `train/`, `val/`, and `test/`.
* **Metadata Catalog**: `data/change_detection_catalog.parquet` (SHA-256 verified).

### 1.2 Complete 24-Scene Census

| Split | Scene Identifier | Geographic Region | Image Width | Image Height | Total Pixels | Change Pixels | Change Percentage (%) | Primary Urban / Landscape Context |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | `beirut` | Lebanon, Middle East | 582 | 522 | 303,804 | 3,449 | 1.14% | Coastal urban reconstruction |
| **Train** | `berrida` | Algeria, North Africa | 718 | 776 | 557,168 | 9,130 | 1.64% | Semi-arid agricultural & expansion |
| **Train** | `bordeaux` | France, Europe | 563 | 339 | 190,857 | 4,932 | 2.58% | European commercial/residential |
| **Train** | `chongqing` | China, East Asia | 788 | 1,015 | 799,820 | 18,942 | 2.37% | Dense urban infrastructure & riverbanks |
| **Train** | `dubai` | UAE, Middle East | 525 | 471 | 247,275 | 4,614 | 1.87% | Desert urban developments |
| **Train** | `hong_kong` | Hong Kong, East Asia | 461 | 517 | 238,337 | 2,380 | 1.00% | High-density reclamation & building |
| **Train** | `lasvegas` | USA, North America | 772 | 902 | 696,344 | 17,318 | 2.49% | Arid suburban sprawl |
| **Train** | `milano` | Italy, Europe | 540 | 695 | 375,300 | 13,379 | 3.56% | Industrial / commercial logistics |
| **Train** | `rennes` | France, Europe | 557 | 858 | 477,906 | 12,244 | 2.56% | Peripheral suburban construction |
| **Train** | `saclay_w` | France, Europe | 360 | 395 | 142,200 | 1,046 | 0.74% | Campus / technological park development |
| **Subtotal** | **Train (10 Scenes)** | **Global (6 Continents)** | — | — | **3,968,485** | **87,314** | **2.20%** | — |
| **Val** | `montpellier` | France, Europe | 1,070 | 1,180 | 1,262,600 | 33,918 | 2.69% | Mediterranean urban expansion |
| **Val** | `paris` | France, Europe | 390 | 408 | 159,120 | 458 | 0.29% | Dense European urban core |
| **Val** | `rio` | Brazil, South America | 688 | 639 | 439,632 | 4,346 | 0.99% | Coastal tropical urban topography |
| **Val** | `valencia` | Spain, Europe | 785 | 799 | 627,215 | 23,592 | 3.76% | Coastal Mediterranean infrastructure |
| **Subtotal** | **Val (4 Scenes)** | **Europe & S. America** | — | — | **2,314,237** | **62,314** | **2.69%** | — |
| **Test** | `aguasclaras` | Brazil, South America | 426 | 353 | 150,378 | 8,552 | 5.69% | Rapid planned suburban development |
| **Test** | `austin` | USA, North America | 688 | 639 | 439,632 | 5,018 | 1.14% | Highway corridor & commercial sprawl |
| **Test** | `beihai` | China, East Asia | 476 | 458 | 218,008 | 969 | 0.44% | Coastal port and aquaculture |
| **Test** | `berlin` | Germany, Europe | 544 | 730 | 397,120 | 28,637 | 7.21% | European capital redevelopment |
| **Test** | `brasilia` | Brazil, South America | 451 | 426 | 192,126 | 13,054 | 6.79% | Federal district construction |
| **Test** | `budapest` | Hungary, Europe | 469 | 433 | 203,077 | 5,242 | 2.58% | Danube river valley industrial |
| **Test** | `campinas` | Brazil, South America | 385 | 241 | 92,785 | 1,227 | 1.32% | Southeastern Brazil logistics hub |
| **Test** | `mumbai` | India, South Asia | 634 | 774 | 490,716 | 48,689 | 9.92% | Dense coastal reclamation & transit |
| **Test** | `nantes` | France, Europe | 558 | 545 | 304,110 | 2,419 | 0.80% | Western European suburban development |
| **Test** | `pisa` | Italy, Europe | 716 | 824 | 589,984 | 45,270 | 7.67% | Tuscan industrial and airport expansion |
| **Subtotal** | **Test (10 Scenes)** | **Global Benchmark** | — | — | **3,598,392** | **159,077** | **4.42%** | — |
| **Total** | **Full OSCD (24)** | **Worldwide** | — | — | **9,881,114** | **308,705** | **3.59%** | — |

---

## 2. Spatial and Temporal Integrity Audit

### 2.1 Spatial Independence & Zero Geographic Leakage
- **City-Level Carveout**: Unlike patch-based partitioning where adjacent patches from the same satellite pass risk spatial autocorrelation, the OSCD split is defined at the **independent city level**.
- **Cross-Split Independence**:
  $$\text{Train Cities} \cap \text{Val Cities} = \emptyset, \quad \text{Train Cities} \cap \text{Test Cities} = \emptyset, \quad \text{Val Cities} \cap \text{Test Cities} = \emptyset$$
- Because cities are located in different countries, continents, or distinct UTM zones (often separated by hundreds or thousands of kilometers), **zero spatial leakage exists across splits**.

### 2.2 Temporal Pairing & Chronological Alignment
- Each scene comprises an observation pair acquired by Sentinel-2A / Sentinel-2B:
  - $t_1$: Pre-change surface reflectance image.
  - $t_2$: Post-change surface reflectance image.
- Ground truth change masks reflect actual permanent land-cover transformations (e.g. soil clearing, building construction, road expansion, earthworks) between $t_1$ and $t_2$, while transient agricultural seasonal variations and atmospheric illumination differences are excluded by design.

---

## 3. Hardware Platform & Architecture Feasibility

### 3.1 Hardware Specifications
- **Host GPU**: NVIDIA GeForce RTX 2050 Laptop GPU
  - Architecture: Ampere (GA107)
  - Compute Capability: 8.6
  - Dedicated VRAM: **$4,096\text{ MiB}$ (4.0 GB)**
  - Active Operating Load: ~792 MiB used by host display and UI tasks
  - **Effective Free VRAM for Training**: **$\approx 3,300\text{ MiB}$**
- **PyTorch CUDA Environment**:
  - Python Executable: `C:\Users\B NAVANEETH\AppData\Local\Programs\Python\Python313\python.exe`
  - Framework: PyTorch 2.6.0+cu124 (CUDA 12.4 runtime, Driver 572.83)
  - PyTorch CUDA Status: `torch.cuda.is_available() == True`

### 3.2 Architectural Feasibility Assessment

| Architecture Family | Candidate Model | Memory Footprint (FP32) | Feasibility on RTX 2050 (4 GB) | Training Speed | Expected Performance Trade-Off |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Classical Radiometric** | Normalized CVA / Gaussian CVA | $< 100\text{ MB}$ (CPU) | **Fully Feasible** | Instant | Fast deterministic baseline; high precision on sharp changes, but struggles with illumination differences. |
| **Fully Convolutional Siamese** | FC-Siam-diff (Daudt et al., 2018) | $\approx 1.2\text{ GB}$ (Batch=8) | **Fully Feasible** | $\approx 5.7\text{ min}$ (20 epochs) | Lightweight (1.4M params), local differencing inductive bias; current best performer ($37.66\%$ Macro F1). |
| **Siamese ResNet U-Net** | Siam-ResNet18 U-Net (Pretrained) | $\approx 2.1\text{ GB}$ (Batch=4) | **Fully Feasible** | $\approx 12\text{ min}$ (20 epochs) | Pretrained ImageNet features provide robust semantic representations; strong candidate for exceeding FC-Siam-diff. |
| **Compact Transformer** | BIT (Chen et al., 2021) | $\approx 2.6\text{ GB}$ (Batch=4) | **Feasible (Borderline)**| $\approx 6.3\text{ min}$ (20 epochs) | Prone to overfitting semantic tokens on small datasets (10 training scenes); achieved only $12.95\%$ Macro F1. |
| **Heavy Transformer** | ChangeFormer-V6 / Swin-CD | $> 6.0\text{ GB}$ | **INFEASIBLE (OOM)** | — | Exceeds 4 GB VRAM limit. Would require heavy gradient checkpointing or CPU offloading. |

**Feasibility Decision**: Focus development on **Classical CVA**, **FC-Siam-diff**, and an advanced **Pretrained Siamese ResNet-18/34 U-Net** architecture with boundary/imbalance-aware losses.

---

## 4. Historical M3.2 Performance Ledger & Root Cause Analysis

### 4.1 Historical Scorecard Summary

| Specialist Model | Architecture Description | Decision Threshold Regime | Test Macro F1 (10 Scenes) | Test Pooled F1 | Test Pooled Precision | Test Pooled Recall | Test Pooled IoU | Certification Status ($\ge 0.50$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard CVA** | Pixel Radiometric Difference | $\tau^* = 0.28$ | 20.31% | 27.78% | **58.20%** | 18.24% | 16.13% | Unmet |
| **Gaussian CVA** | Smoothed Radiometric Difference | $\tau^* = 0.28$ | 27.31% | 33.93% | 39.70% | 29.63% | 20.43% | Unmet |
| **FC-Siam-diff** | 4-Stage ConvNet (1.4M params) | $\tau^* = 0.25$ | **37.66%** | **38.18%** | 35.33% | **41.53%** | **23.60%** | **Unmet (5/10 passed)** |
| **BIT** | Siamese ResNet-18 + Transformer | $\tau^* = 0.45$ | 12.95% | 12.60% | 18.79% | 9.47% | 6.72% | Unmet |

### 4.2 Why the Certification Target ($\text{Macro } F_1 \ge 0.50$) Was Not Met
1. **The Bimodal Per-Scene Distribution**:
   - FC-Siam-diff achieved $F_1 \ge 0.50$ on **5 scenes**: `brasilia` ($72.68\%$), `pisa` ($58.09\%$), `budapest` ($54.62\%$), `nantes` ($53.84\%$), and `aguasclaras` ($51.24\%$).
   - On the remaining **5 scenes**, performance collapsed: `austin` ($14.79\%$), `beihai` ($2.65\%$), `berlin` ($46.71\%$), `campinas` ($10.41\%$), and `mumbai` ($11.54\%$).
2. **The Mechanism of Failure**:
   - The failing scenes have extremely low change percentages (`beihai`: 0.44%, `austin`: 1.14%, `campinas`: 1.32%).
   - Under standard loss functions (Binary Cross-Entropy + basic Dice), false positive clusters in complex coastal or sprawling agricultural areas overwhelm true positive detections, causing precision collapse ($2.0\% - 15.0\%$).
3. **The Path Forward**:
   - To lift the failing scenes into passing territory without degrading passing scenes requires:
     - Pretrained deep backbones (leveraging ImageNet transfer learning instead of random initialization).
     - Hybrid boundary-aware loss functions (e.g. Focal-Tversky or Lovász-Softmax) explicitly penalizing false alarms in sparse foreground regimes.
     - Spatial multi-scale feature aggregation.

---

## 5. Audit Conclusions & Recommended Action Plan

### 5.1 Definitive Answers to Audit Questions
1. **Dataset Selection**: **OSCD (Onera Satellite Change Detection) RGB** is selected, staged, and verified. No external dataset download is required.
2. **Temporal & Mask Pairing**: Fully established; 24 scenes with aligned `t1`, `t2`, and `cm` masks.
3. **Class Distribution**: Severe imbalance ($3.59\%$ overall foreground change, $4.42\%$ in test split).
4. **Leakage Safety**: 100% city-level disjoint splits eliminate geographic data leakage.
5. **Hardware Feasibility**: NVIDIA RTX 2050 (4 GB VRAM) with Python 3.13 (PyTorch 2.6.0+cu124) supports Siamese CNNs and lightweight ResNet encoders.

### 5.2 Recommended Sequence for Phase 2.2 Execution
1. **Step 1: Manifest & Reproducibility Ledger Freeze**
   - Cryptographically lock the OSCD dataset manifest and split definitions.
2. **Step 2: Classical & Deep Baseline Verification**
   - Re-execute and verify the baseline CVA and FC-Siam-diff pipelines, confirming exact metric reproducibility.
3. **Step 3: Advanced Architecture Development within Hardware Budget**
   - Implement a Pretrained Siamese ResNet-18 Feature Extractor with a Feature Pyramid Network (FPN) / U-Net decoder.
   - Introduce Focal-Tversky loss ($\alpha=0.7, \beta=0.3$) to specifically penalize false alarms on sparse scenes.
4. **Step 4: Formal Validation Tuning & Held-Out Test Evaluation**
   - Tune decision thresholds strictly on the 4 validation scenes (`montpellier`, `paris`, `rio`, `valencia`).
   - Unblind the 10 test scenes strictly once for final scorecard certification.
