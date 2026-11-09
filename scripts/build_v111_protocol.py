"""
Build Script to Upgrade SatQuery AI Phase 4 Protocol from v1.10 to v1.11.

Executes all 5 mandatory audit corrections:
1. Optical preprocessing baseline reconciliation (PB < 04.00 vs PB >= 04.00), non-clipping policy, and regression test lock.
2. Absent-class metrics restricted to mapped classes (C_absent in M) and isolated quarantined output diagnostics.
3. Three-cluster spatial degrees of freedom, conditional inference scope, and leave-one-tile-out sensitivity.
4. Strictly nested cloud masks (exact increasing integer pixel counts) and CRN noise as normalized-tensor sensitivity.
5. Expanded 24-component cryptographic provenance manifest with exact SHA-256 hashes.
"""

from pathlib import Path

def build_v111():
    v110_path = Path("SATQUERY-P4-PROTOCOL-V1.10-DRAFT.md")
    v111_path = Path("SATQUERY-P4-PROTOCOL-V1.11-DRAFT.md")

    with open(v110_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Header Updates
    old_header = """**Document Identifier**: `SATQUERY-P4-PROTOCOL-V1.10-DRAFT`  
**Phase**: Phase 4 — Robustness, Generalization, and Operational Envelope Discovery  
**Date**: November 9, 2026  
**Status**: **PROPOSED PREREGISTRATION PROTOCOL — DRAFT v1.10 (Methodological & Statistical Lock — Pending Final User Freeze)**  
**Repository Branch**: `feature/phase4-generalization-protocol`  
**Preceding Milestones**: Phase 3 Closed (M3.1 Multimodal ASL/CB-BCE, M3.2 FC-Siam-diff, M3.3 Selective Routing) with Pre-Registered Limitations  
**Model Weight & Threshold Provenance**: Completely Frozen from Phase 3"""

    new_header = """**Document Identifier**: `SATQUERY-P4-PROTOCOL-V1.11-DRAFT`  
**Phase**: Phase 4 — Robustness, Generalization, and Operational Envelope Discovery  
**Date**: November 9, 2026  
**Status**: **PROPOSED PREREGISTRATION PROTOCOL — DRAFT v1.11 (Methodological & Statistical Lock — Pending Final User Freeze)**  
**Repository Branch**: `feature/phase4-generalization-protocol`  
**Preceding Milestones**: Phase 3 Closed (M3.1 Multimodal ASL/CB-BCE, M3.2 FC-Siam-diff, M3.3 Selective Routing) with Pre-Registered Limitations; Supersedes `SATQUERY-P4-PROTOCOL-V1.10-DRAFT` with Sentinel-2 processing baseline reconciliation, Stage B preprocessing regression testing, isolated quarantined metrics, expanded 24-component cryptographic provenance manifest, and explicit three-cluster conditional inference scope.  
**Confirmatory Experiment Status**: **ZERO Confirmatory Experiments Executed (Draft Protocol Specification Only)**  
**Model Weight & Threshold Provenance**: Completely Frozen from Phase 3"""

    assert old_header in content, "Header match failed"
    content = content.replace(old_header, new_header)

    # 2. Section 2 Specialist 1 Optical Input Contract
    old_s1_opt = """  - Optical tensor: Shape `(B, 10, 120, 120)` consisting of 10 Sentinel-2 L2A surface reflectance channels in **exact canonical order**: `[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]`.
    - Native 20m bands (`B05, B06, B07, B8A, B11, B12`) must be resampled to 10m GSD via bilinear interpolation to match the native 10m bands (`B02, B03, B04, B08`).
    - Scaling & Standardization: $\\mathbf{X}_{\\text{opt,raw}} = \\text{DN} / 10000.0 \\in [0.0, 1.0]$, standardized using frozen statistics:
      $$\\mu_{\\text{opt}} = [0.136, 0.118, 0.106, 0.128, 0.203, 0.257, 0.272, 0.273, 0.187, 0.121]$$
      $$\\sigma_{\\text{opt}} = [0.082, 0.076, 0.074, 0.089, 0.098, 0.108, 0.113, 0.112, 0.095, 0.078]$$"""

    new_s1_opt = """  - Optical tensor: Shape `(B, 10, 120, 120)` consisting of 10 Sentinel-2 L2A surface reflectance channels in **exact canonical order**: `[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]`.
    - Native 20m bands (`B05, B06, B07, B8A, B11, B12`) must be resampled to 10m GSD via bilinear interpolation to match the native 10m bands (`B02, B03, B04, B08`).
    - Processing Baseline Reconciliation: Surface reflectance $\\rho_{\\lambda}$ is computed strictly accounting for Sentinel-2 Level-2A processing baselines:
      $$\\rho_{\\lambda} = \\frac{\\text{DN} + \\text{BOA\\_ADD\\_OFFSET}_{\\lambda}}{\\text{QUANTIFICATION\\_VALUE}} = \\begin{cases} \\frac{\\text{DN}}{10,000.0}, & \\text{for Processing Baseline } \\text{PB} < 04.00 \\\\ \\frac{\\text{DN} - 1000}{10,000.0}, & \\text{for Processing Baseline } \\text{PB} \\ge 04.00 \\end{cases}$$
      where $\\text{QUANTIFICATION\\_VALUE} = 10,000.0$.
    - Non-Clipping Physical Policy: Surface reflectance $\\rho_{\\lambda}$ is **not silently clipped to $[0.0, 1.0]$**. Values are verified to be finite (`not np.isfinite` halts execution) and within expected physical tolerances $[-0.05, 1.20]$ to accommodate water dark-target subtraction shadows and cloud/snow specular glint without artificial truncation.
    - Channel Standardization: Scaled via frozen training split statistics:
      $$X_{\\text{opt}, c} = \\frac{\\rho_c - \\mu_{\\text{opt}, c}}{\\sigma_{\\text{opt}, c}}$$
      $$\\mu_{\\text{opt}} = [0.136, 0.118, 0.106, 0.128, 0.203, 0.257, 0.272, 0.273, 0.187, 0.121]$$
      $$\\sigma_{\\text{opt}} = [0.082, 0.076, 0.074, 0.089, 0.098, 0.108, 0.113, 0.112, 0.095, 0.078]$$
    - Regression Test Lock: Bitwise equivalence and shift quantification are validated in `tests/test_preprocessing_compatibility.py` (SHA-256: `1a8908e3a0b9b2d2a6d9f4ed81686762c08f33f2c272b9bb47c01ff10e2442c9`)."""

    assert old_s1_opt in content, "Section 2 Specialist 1 match failed"
    content = content.replace(old_s1_opt, new_s1_opt)

    # 3. Section 3 Stage B Specification
    old_stage_b = """### Stage B: Specialist Tensor & Preprocessing Contract Gate $\\mathcal{C}_{\\text{tensor}}(M_k, \\mathbf{X})$
Executes immediately upon Stage A clearance and prior to passing tensors into the neural forward pass of specialist $M_k$:
1. **Land Cover Specialist Preprocessing & Tensor Contract (`MultimodalResNet34`)**:
   - *Optical Tensor Specification*: Shape must be exactly `(B, 10, 120, 120)` with `dtype == torch.float32`.
   - *Canonical Channel Sequence*: Must strictly correspond to `[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]`. Any omission (e.g. 4-band RGB+NIR) or permutation halts execution with `ERR_SPECIALIST_TENSOR_CONTRACT`.
   - *Resampling Provenance*: Verifies that 20m bands (`B05, B06, B07, B8A, B11, B12`) were resampled to 10m GSD via the certified bilinear pipeline prior to stacking.
   - *Scaling & Standardization Invariants*: Verifies pre-standardization optical range $\\mathbf{X}_{\\text{opt,raw}} \\in [0.0, 1.0]$ and application of frozen channel statistics $(\\mu_{\\text{opt}}, \\sigma_{\\text{opt}})$. Finite floating-point check confirms zero `NaN` or `Inf` values.
   - *SAR Tensor Specification*: Shape must be exactly `(B, 2, 120, 120)` with `dtype == torch.float32`, corresponding to `[VH, VV]`.
   - *SAR Normalization Invariants*: Verifies clipping to $[-35.0, +5.0]\\text{ dB}$ and linear scaling to $[0.0, 1.0]$.
   - *Representation Constraint*: If input is formatted as a 1D feature vector or summary embedding, it is rejected with `ERR_UNSUPPORTED_REPRESENTATION`.
2. **Change Detection Specialist Preprocessing & Tensor Contract (`FCSiamDiff`)**:
   - *Bitemporal Tensor Specification*: Verifies two tensors $(t_1, t_2)$ each of shape `(B, 3, H, W)` with $H, W \\ge 100$, $H_1 = H_2, W_1 = W_2$, and `dtype == torch.float32` (RGB channels).
   - *ImageNet Normalization Invariant*: Verifies standardization with ImageNet channel statistics ($\\mu = [0.485, 0.456, 0.406], \\sigma = [0.229, 0.224, 0.225]$) and finite floating-point values."""

    new_stage_b = """### Stage B: Specialist Tensor & Preprocessing Contract Gate $\\mathcal{C}_{\\text{tensor}}(M_k, \\mathbf{X})$
Executes immediately upon Stage A clearance and prior to passing tensors into the neural forward pass of specialist $M_k$:
1. **Land Cover Specialist Preprocessing & Tensor Contract (`MultimodalResNet34`)**:
   - *Optical Tensor Specification*: Shape must be exactly `(B, 10, 120, 120)` with `dtype == torch.float32`.
   - *Canonical Channel Sequence*: Must strictly correspond to `[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]`. Any omission (e.g. 4-band RGB+NIR) or permutation halts execution with `ERR_SPECIALIST_TENSOR_CONTRACT`.
   - *Resampling Provenance*: Verifies that 20m bands (`B05, B06, B07, B8A, B11, B12`) were resampled to 10m GSD via the certified bilinear pipeline prior to stacking.
   - *Processing Baseline Reconciliation & Reflectance Conversion*:
     Sentinel-2 Level-2A surface reflectance conversion is governed by the product processing baseline:
     $$\\rho_{\\lambda} = \\frac{\\text{DN} + \\text{BOA\\_ADD\\_OFFSET}_{\\lambda}}{\\text{QUANTIFICATION\\_VALUE}} = \\begin{cases} \\frac{\\text{DN}}{10,000.0}, & \\text{for PB } < 04.00 \\\\ \\frac{\\text{DN} - 1000}{10,000.0}, & \\text{for PB } \\ge 04.00 \\end{cases}$$
     > [!IMPORTANT]
     > **Baseline Offset Defense**: The frozen BigEarthNet-14k checkpoint was trained on archival Sentinel-2 imagery acquired prior to January 25, 2022 ($\\text{PB} < 04.00$, $\\text{BOA\\_ADD\\_OFFSET} = 0$). Modern operational products ($\\text{PB} \\ge 04.00$) include a radiometric offset of $-1,000\\text{ DN}$. Applying the archival $\\text{DN}/10000$ formula to $\\text{PB} \\ge 04.00$ products introduces an artificial $+0.10$ reflectance shift ($+1.35\\sigma$ on Band 4 Red), corrupting model inference. Stage B automatically inspects metadata `processing_baseline` and `BOA_ADD_OFFSET` to enforce exact reflectance calibration.
   - *Non-Clipping Physical Inspection*: Surface reflectance $\\rho$ is verified for finite floating-point values (`not np.isfinite` raises `ERR_NON_FINITE_TENSOR`). It is **not silently clipped to $[0.0, 1.0]$**; values within nominal physical tolerance $[-0.05, 1.20]$ are preserved to avoid distorting shadow or glint distributions. Tensors with values outside $[-0.05, 1.20]$ halt execution with `ERR_UNPHYSICAL_REFLECTANCE`.
   - *Standardization Invariants*: Applies channel-wise standardization using frozen training split parameters:
     $$X_{\\text{opt}, c} = \\frac{\\rho_c - \\mu_{\\text{opt}, c}}{\\sigma_{\\text{opt}, c}}$$
     with $\\mu_{\\text{opt}} = [0.136, 0.118, 0.106, 0.128, 0.203, 0.257, 0.272, 0.273, 0.187, 0.121]$ and $\\sigma_{\\text{opt}} = [0.082, 0.076, 0.074, 0.089, 0.098, 0.108, 0.113, 0.112, 0.095, 0.078]$.
   - *SAR Tensor Specification*: Shape must be exactly `(B, 2, 120, 120)` with `dtype == torch.float32`, corresponding to `[VH, VV]`.
   - *SAR Normalization Invariants*: Verifies clipping to $[-35.0, +5.0]\\text{ dB}$ and linear scaling to $[0.0, 1.0]$ via $(\\text{dB} + 35.0) / 40.0$.
   - *Representation Constraint*: If input is formatted as a 1D feature vector or summary embedding, it is rejected with `ERR_UNSUPPORTED_REPRESENTATION`.
   - *Regression Verification*: Verified bitwise against frozen implementations in `tests/test_preprocessing_compatibility.py` (SHA-256: `1a8908e3a0b9b2d2a6d9f4ed81686762c08f33f2c272b9bb47c01ff10e2442c9`).
2. **Change Detection Specialist Preprocessing & Tensor Contract (`FCSiamDiff`)**:
   - *Bitemporal Tensor Specification*: Verifies two tensors $(t_1, t_2)$ each of shape `(B, 3, H, W)` with $H, W \\ge 100$, $H_1 = H_2, W_1 = W_2$, and `dtype == torch.float32` (RGB channels).
   - *ImageNet Normalization Invariant*: Verifies standardization with ImageNet channel statistics ($\\mu = [0.485, 0.456, 0.406], \\sigma = [0.229, 0.224, 0.225]$), canonical `(3, H, W)` channel ordering, and finite floating-point values."""

    assert old_stage_b in content, "Section 3 Stage B match failed"
    content = content.replace(old_stage_b, new_stage_b)

    # 4. Section 4.1 Track 4.1 Absent-Class Metrics and Quarantined Isolation
    old_absent_metrics = """       3. **Mandatory Absent-Class False Positive Metrics**:
          For all absent classes $c \\in C_{\\text{absent}, k} = \\{c \\in \\{1,\\dots,13\\} \\mid N_{\\text{true}, c} = 0\\}$:
          - *Absent-Class False Positive Rate ($FPR_{\\text{absent}}$)*:
            $$FPR_{\\text{absent}} = \\frac{\\sum_{c \\in C_{\\text{absent}, k}} \\text{FP}_c}{|C_{\\text{absent}, k}| \\times N_{\\text{evaluable}, k}}$$
          - *Absent-Class False Positive Share ($FPS_{\\text{absent}}$)*:
            $$FPS_{\\text{absent}} = \\frac{\\sum_{c \\in C_{\\text{absent}, k}} \\text{FP}_c}{\\sum_{c=1}^{16} \\sum_{i=1}^{N_{\\text{evaluable}, k}} \\hat{y}_{i, c}}$$
          - *Per-Class Absent Alarms*: Tabularize $\\text{FP-Count}_c$ and $\\text{FPR}_c = \\text{FP}_c / N_{\\text{evaluable}, k}$ for each absent class.
       4. **Sub-Taxonomy Sensitivity Metrics**: Direct-Six ($F_{1,\\text{direct6}}^{\\text{fixed6}}$) and Approximate-Seven ($F_{1,\\text{approx7}}^{\\text{fixed7}}$).
       5. **Quarantined Diagnostics Table**: Model output frequencies and trigger rates for the 3 unmapped classes (`Industrial`, `Complex cultivation`, `Permanent crops`).
     - **Conditional Spatial Cluster Scope & Leave-One-Tile-Out Sensitivity**:
       > [!WARNING]
       > **Three-Cluster In-Domain Scope**: The in-domain BigEarthNet-14k test population comprises patches derived from exactly **3 geographic tile clusters**: `T34TCR`, `T34TEQ`, and `T34TFN` in Serbia.
       > 1. Resampling over $K_{\\text{in}} = 3$ clusters provides limited representation of in-domain spatial variability ($df \\le 2$). The bootstrap iteration count ($B=2,000$) ensures numerical stability of the resampling distribution but **cannot inflate the underlying spatial sample size**.
       > 2. All geographic transfer inferences and bootstrap confidence intervals are **formally conditional on this three-tile reference population**.
       > 3. To assess sensitivity to individual tile clusters, evaluation mandates reporting **leave-one-tile-out sensitivity analyses**: computing $\\Delta F_{1, k}$ when excluding each in-domain tile in turn ($\\setminus\\text{T34TCR}$, $\\setminus\\text{T34TEQ}$, $\\setminus\\text{T34TFN}$) alongside per-tile metrics."""

    new_absent_metrics = """       3. **Mandatory Absent-Class False Positive Metrics (Restricted to Mapped Taxonomy $\\mathcal{M}$)**:
          The 13 mapped classes define the evaluation ground-truth space $\\mathcal{M} = \\{1, \\dots, 13\\}$. The 3 quarantined classes $\\mathcal{Q} = \\{\\text{Industrial or commercial units (6)}, \\text{Complex cultivation patterns (4)}, \\text{Permanent crops (14)}\\}$ have zero ground truth support by construction and are **strictly excluded from the mapped absent-class set $C_{\\text{absent}, k}$**.
          For all absent mapped classes $c \\in C_{\\text{absent}, k} = \\{c \\in \\mathcal{M} \\mid N_{\\text{true}, c} = 0\\}$:
          - *Absent-Class False Positive Rate ($FPR_{\\text{absent}}$)*:
            $$FPR_{\\text{absent}} = \\frac{\\sum_{c \\in C_{\\text{absent}, k}} \\text{FP}_c}{|C_{\\text{absent}, k}| \\times N_{\\text{evaluable}, k}}$$
            where $\\text{FP}_c = \\sum_{i=1}^{N_{\\text{evaluable}, k}} \\hat{y}_{i, c}$ (since $y_{i, c} = 0$ for all $i$ when $c \\in C_{\\text{absent}, k}$).
          - *Absent-Class False Positive Share ($FPS_{\\text{absent}}$)*:
            $$FPS_{\\text{absent}} = \\frac{\\sum_{c \\in C_{\\text{absent}, k}} \\text{FP}_c}{\\sum_{c \\in \\mathcal{M}} \\sum_{i=1}^{N_{\\text{evaluable}, k}} \\hat{y}_{i, c}} = \\frac{\\sum_{c \\in C_{\\text{absent}, k}} \\text{FP}_c}{\\sum_{c=1}^{13} \\sum_{i=1}^{N_{\\text{evaluable}, k}} \\hat{y}_{i, c}}$$
            The denominator is **strictly restricted to positive predictions among the 13 mapped classes**, measuring the proportion of mapped positive predictions that fall on absent classes.
          - *Per-Class Absent Alarms*: Tabularize $\\text{FP-Count}_c$ and $\\text{FPR}_c = \\text{FP}_c / N_{\\text{evaluable}, k}$ for each absent class in $C_{\\text{absent}, k}$.
        4. **Sub-Taxonomy Sensitivity Metrics**: Direct-Six ($F_{1,\\text{direct6}}^{\\text{fixed6}}$) and Approximate-Seven ($F_{1,\\text{approx7}}^{\\text{fixed7}}$).
        5. **Dedicated Quarantined Outputs Diagnostic Table**:
           The 3 unmapped model outputs are reported in an independent diagnostic table to prevent distorting mapped transfer metrics while monitoring model trigger behavior:
           - *Quarantined Output Trigger Counts*: Tabularize $\\sum_{i=1}^{N_{\\text{evaluable}, k}} \\hat{y}_{i, c}$ for each $c \\in \\mathcal{Q}$.
           - *Quarantined Trigger Rates*: $\\frac{1}{N_{\\text{evaluable}, k}} \\sum_{i=1}^{N_{\\text{evaluable}, k}} \\hat{y}_{i, c}$ for each $c \\in \\mathcal{Q}$.
           - *Quarantined Output Share ($FPS_{\\text{quarantine}}$)*:
             $$FPS_{\\text{quarantine}} = \\frac{\\sum_{c \\in \\mathcal{Q}} \\sum_{i=1}^{N_{\\text{evaluable}, k}} \\hat{y}_{i, c}}{\\sum_{c=1}^{16} \\sum_{i=1}^{N_{\\text{evaluable}, k}} \\hat{y}_{i, c}}$$
             quantifying what fraction of total 16-channel model outputs fall onto unmapped categories.
     - **Conditional Spatial Cluster Scope & Leave-One-Tile-Out Sensitivity**:
       > [!WARNING]
       > **Three-Cluster In-Domain Scope & Statistical Degrees of Freedom**: The in-domain BigEarthNet-14k test population comprises patches derived from exactly **3 geographic tile clusters**: `T34TCR`, `T34TEQ`, and `T34TFN` in Serbia.
       > 1. Resampling over $K_{\\text{in}} = 3$ clusters provides strictly limited representation of in-domain spatial variability ($df \\le 2$). The bootstrap iteration count ($B=2,000$) provides numerical resampling stability ($p_{\\min} = 1/(B+1) = 1/2001 \\approx 0.000500$) but **cannot expand the underlying spatial degrees of freedom**.
       > 2. All geographic transfer inferences, p-values, and bootstrap confidence intervals are **formally conditional on this three-tile reference population**.
       > 3. A non-significant hypothesis test ($\\text{Holm } p \\ge 0.05$) indicates **insufficient evidence to detect degradation relative to this conditional 3-tile reference**; it must **never** be reported as proof of global geographic generalization, domain invariance, or equivalence.
       > 4. To evaluate sensitivity to individual tile clusters, evaluation mandates reporting **leave-one-tile-out sensitivity analyses**: computing $\\Delta F_{1, k}^{(\\setminus\\text{tile})}$ when excluding each in-domain tile in turn ($\\setminus\\text{T34TCR}$, $\\setminus\\text{T34TEQ}$, $\\setminus\\text{T34TFN}$) alongside per-tile metrics. These are strictly categorized as descriptive sensitivity analyses."""

    assert old_absent_metrics in content, "Section 4.1 absent metrics match failed"
    content = content.replace(old_absent_metrics, new_absent_metrics)

    # 5. Section 4.3 Perturbations
    old_track_43_ops = """  All perturbations are implemented in the **scaled $[0, 1]$ pre-standardization input domain**:
  - *Optical domain*: Surface reflectance $\\mathbf{X}_{\\text{opt}} \\in [0.0, 1.0]^{10 \\times 120 \\times 120}$ ($\\text{DN} / 10000.0$).
  - *SAR domain*: Scaled backscatter $\\mathbf{X}_{\\text{SAR}} \\in [0.0, 1.0]^{2 \\times 120 \\times 120}$ ($(\\text{dB} + 35.0) / 40.0$).
  - *Change detection domain*: Bitemporal RGB images $\\mathbf{X}_{t_1}, \\mathbf{X}_{t_2} \\in [0.0, 1.0]^{3 \\times 256 \\times 256}$."""

    new_track_43_ops = """  All perturbations are implemented in the **scaled $[0, 1]$ pre-standardization input domain**:
  - *Optical domain*: Surface reflectance $\\mathbf{X}_{\\text{opt}} \\in [0.0, 1.0]^{10 \\times 120 \\times 120}$ obtained via baseline-aware conversion $\\rho = (\\text{DN} + \\text{offset}) / 10000.0$.
  - *SAR domain*: Scaled backscatter $\\mathbf{X}_{\\text{SAR}} \\in [0.0, 1.0]^{2 \\times 120 \\times 120}$ ($(\\text{dB} + 35.0) / 40.0$).
  - *Change detection domain*: Bitemporal RGB images $\\mathbf{X}_{t_1}, \\mathbf{X}_{t_2} \\in [0.0, 1.0]^{3 \\times 256 \\times 256}$."""

    assert old_track_43_ops in content, "Section 4.3 domains match failed"
    content = content.replace(old_track_43_ops, new_track_43_ops)

    # 5b. Cloud subset guarantee & Noise interpretation in Track 4.3
    old_subset = """       3. For each severity level $f_c \\in \\{0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50\\}$, select the top $K(f_c) = \\lfloor f_c \\times H \\times W \\rfloor$ pixels to form binary mask $\\mathbf{M}_{\\text{cloud}}(f_c) \\in \\{0, 1\\}^{H \\times W}$.
       4. *Strict Subset Guarantee*:
          $$\\mathbf{M}_{\\text{cloud}}(0.05) \\subset \\mathbf{M}_{\\text{cloud}}(0.10) \\subset \\mathbf{M}_{\\text{cloud}}(0.15) \\subset \\dots \\subset \\mathbf{M}_{\\text{cloud}}(0.50)$$
          guaranteeing that increasing severity strictly adds newly occluded pixels while perfectly retaining all previously occluded pixels."""

    new_subset = """       3. For each severity level $f_c \\in \\{0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50\\}$, select the top $K(f_c) = \\lfloor f_c \\times 120 \\times 120 \\rfloor$ pixels to form binary mask $\\mathbf{M}_{\\text{cloud}}(f_c) \\in \\{0, 1\\}^{120 \\times 120}$.
          The pixel counts are strictly increasing integers:
          $$K(0.05) = 720 < K(0.10) = 1440 < K(0.15) = 2160 < K(0.20) = 2880 < K(0.30) = 4320 < K(0.40) = 5760 < K(0.50) = 7200$$
       4. *Strict Proper Subset Guarantee*:
          $$\\mathbf{M}_{\\text{cloud}}(0.05) \\subset \\mathbf{M}_{\\text{cloud}}(0.10) \\subset \\mathbf{M}_{\\text{cloud}}(0.15) \\subset \\dots \\subset \\mathbf{M}_{\\text{cloud}}(0.50)$$
          Because pixel counts are strictly increasing and selection follows a single fixed rank ordering, increasing severity strictly adds newly occluded pixels while perfectly retaining all previously occluded pixels."""

    assert old_subset in content, "Section 4.3 subset match failed"
    content = content.replace(old_subset, new_subset)

    old_noise_scope = """> [!IMPORTANT]
       > **Scope and Interpretation**: This operator evaluates specialist sensitivity to additive numerical noise in the scaled $[0, 1]$ input domain. It does **not** assume or claim equivalent physical sensor noise across optical reflectance ($\\text{DN}/10000$) and radar backscatter ($(\\text{dB}+35)/40$)."""

    new_noise_scope = """> [!IMPORTANT]
       > **Scope and Interpretation**: This operator evaluates specialist sensitivity to additive numerical perturbations in the scaled $[0, 1]$ input domain (normalized-tensor sensitivity). It does **not** simulate physical sensor noise, photon shot noise, or radar speckle, and makes no claim of physical noise equivalence between optical reflectance and radar backscatter."""

    assert old_noise_scope in content, "Section 4.3 noise scope match failed"
    content = content.replace(old_noise_scope, new_noise_scope)

    # 6. Section 5.2 Cryptographic Provenance Manifest Table
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
| **Physical Gate Implementation** | `satquery/utils/geospatial.py` | `156b76cd1ab4abc54341e6a405f9cd58f1387096f6cc92917c62ec01a11911f5` |
| **Core Taxonomy Module** | `satquery/taxonomy.py` | `f1c424218f5144621ea5b6ae94065ada326ef3ff0d61e1ed0fe236e23c059997` |
| **Task Router Logic Module** | `satquery/router.py` | `f3bf45452df165ef0a6fd0241fb3973050ac0d69609abf118f8c343491f175b1` |
| **Metadata Parsing Module** | `satquery/utils/metadata.py` | `f3cbbc36ab3ad7c4f8bcad1144ba8235f013f4b84a49ad51989fa6f1aff159c8` |
| **Dataset Orchestration Core** | `satquery/dataset.py` | `a59356dc89be0e94865a34c2538a069c5d2e6cc259857a0e0102a8bb8eb6038c` |
| **System Orchestrator** | `satquery/orchestrator.py` | `254e24ccc508145ac0b715ba88c8cdbe729fcde9d1aec45a67374cfbccc4ac25` |
| **Evaluation Metrics & Verification**| `satquery/evaluation.py` | `5fca3ec08664f96381101bdb2e6711acd43934085e46d50962a9d51757c644fc` |
| **Evidence Package Exporter** | `satquery/evidence.py` | `372ddeac05a5d11dde28155437632f66f1cae206000a16d4a0eb1b5d63e7dd6b` |
| **Environment Lockfile** | `requirements-lock.txt` | `72c893fb4425dd0c83c77a39338b26325da9cb96d473c2b511630369585b3b25` |"""

    new_manifest_table = """| Component | Path / Location | SHA-256 Digest |
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

    assert old_manifest_table in content, "Section 5.2 manifest table match failed"
    content = content.replace(old_manifest_table, new_manifest_table)

    # 7. Section 7 Summary Table
    old_summary_row = """| **Document Version** | `SATQUERY-P4-PROTOCOL-V1.10-DRAFT` | **PROPOSED DRAFT v1.10** |"""
    new_summary_row = """| **Document Version** | `SATQUERY-P4-PROTOCOL-V1.11-DRAFT` | **PROPOSED DRAFT v1.11** |"""
    assert old_summary_row in content, "Section 7 summary row match failed"
    content = content.replace(old_summary_row, new_summary_row)

    # Write out v1.11
    with open(v111_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully generated {v111_path} ({len(content)} characters)")

if __name__ == "__main__":
    build_v111()
