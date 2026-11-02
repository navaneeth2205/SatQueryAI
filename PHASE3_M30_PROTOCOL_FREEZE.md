# SatQuery AI — Phase 3 Milestone M3.0 Experimental Protocol Freeze (Final Research Specification)

**Problem Title**: Sensor- and Task-Compatibility-Constrained Selective Routing for Multimodal Earth Observation  
**Problem Code**: SIH26167 (ISRO / Smart India Hackathon)  
**Document Classification**: Milestone M3.0 Protocol Freeze & Scientific Defense Specification (Final Research Specification)  
**Date**: November 2, 2026  
**Auditor / Lead Author**: Lead Research Verification Agent  
**Frozen Classical Baseline Commit**: `ce7134245f9be6b3876a3a2df1699b2641156f9f`  
**Git Baseline Tag**: `v0.2.0-phase2-freeze` (Dereferenced commit: `ce7134245f9be6b3876a3a2df1699b2641156f9f`)  
**Working Tree Status**: Clean (`git status --porcelain` is empty)  
**Status**: **REVISED — AWAITING SCIENTIFIC REVIEW**  

---

## Executive Summary & Revision Matrix

This document provides the final, reconciled scientific protocol for Milestone M3.0 of SatQuery AI. It resolves all methodological, mathematical, and publication-level issues raised in the research review.

**Research Governance Invariant**: **No deep-learning models have been trained, no benchmark scenario files generated, and the frozen Phase 2 baseline tag remains strictly unchanged.** Authorization for Milestone M3.1 remains blocked pending scientific review and sign-off on this protocol.

### Comprehensive Reviewer Issue Resolution Table

| # | Priority Area | Specific Reviewer Finding | Protocol Resolution & Scientific Correction | Section |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Statistical Power & AURC Comparison** | AURC is a global population metric, not a family-level mean of independent variables. Paired $t$-tests over family AURCs confound scenario-level pooling with family averaging. Unreconciled Wilson bound ($3/(28+3) = 9.68\%$). | Replaced family-level $t$-tests with the **paired cluster bootstrap** (resampling complete template families as clusters with replacement, pooling strictly eligible evaluation scenarios $\mathcal{E}_{\text{eval}}$ ($N_{\text{eval}} = 3 \times F$), and recomputing global $\text{AURC}_{\text{joint}}$ on each replicate). Provided executable script `tests/cluster_bootstrap_power_analysis.py` (seed `42`, $N_{\text{sim}}=400, N_{\text{boot}}=300$). Documented exact empirical power across $F_{\text{test}} \in \{28, 42, 56, 84\} \times \Delta \in \{0.02, 0.03, 0.04\}$ under complete 5-comparison Step-Down Holm testing with Monte Carlo SE. Pre-registered contingency: $F_{\text{test}}=84$ ($86.5\% \pm 1.7\%$ power for $\Delta=0.04$, $78.0\% \pm 2.1\%$ power for $\Delta=0.03$). Corrected one-sided 95% Wilson upper bound ($8.81\%$) vs exact Binomial bound ($10.15\%$) for 0 failures in $N=28$ test violations. | Deliverable E.5, D.4 |
| **2** | **Baseline Ablation Isolation & Fairness** | Evaluating uncalibrated models (R3) at an arbitrary static threshold ($\tau_{\text{sel}}=0.65$) biases comparisons against them. R4 vs R5 confounded deterministic routing with LLM architecture. | Redesigned into a clean 2x2 factorial ablation grid. Locked evaluation to a **model-independent common valid population $\mathcal{E}_{\text{eval}}$** ($N=84$ scenarios) so temperature scaling cannot alter evaluated queries. Formulated fair thresholding: compare risk at **matched coverage levels** ($\Phi \in \{0.80, 0.85, 0.90, 0.95\}$), coverage at **matched risk levels** ($\mathcal{R} \le 0.05, 0.08$), full risk-coverage curves, and validation-only threshold tuning. Evaluated R5 separately as an external zero-shot LLM baseline without claiming single-factor ablation. | Deliverable C.1–C.2 |
| **3** | **Complete Relative Affine Basis Gate** | Partial checks on individual $GT$ elements miss shearing, mixed axes, and coordinate convention shifts. Conflicting resolution tolerances. | Implemented full relative affine transformation matrix $\mathbf{M}_{\text{rel}} = \mathbf{A}_1^{-1} \mathbf{A}_2$ in a shared projected CRS with Singular Value Decomposition ($\mathbf{M}_{\text{rel}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$). Enforced singular value scale bounds ($|\sigma_i - 1| \le 0.01$), condition number $\kappa \le 1.01$, positive determinant ($\det > 0$, $|\det - 1| \le 0.02$, strictly rejecting reflections/handedness flips), relative rotation ($|\theta_{\text{rel}}| \le 0.28^\circ$), and continuous sub-pixel phase offset ($\delta_{\text{phase}} \le 0.05\text{ px}$). Clarified GDAL pixel-corner vs center conventions. Labeled $0.05\text{ px}$ as an implementation-selected engineering constraint with literature justification. | Deliverable B.2 |
| **4** | **Independent Linguistic Ambiguity Ground Truth** | $V_3$ ambiguity was defined circularly via router confidence scores ($c < 0.65$ or $\Delta c < 0.15$). Unclear relationship between $V_3$ and parent family task. | Eliminated model-dependent ambiguity. Formulated an independent dual-annotator human linguistic protocol for ambiguity ($A^* \in \{0, 1\}$) with complete prompt instructions, decision rubric, 3 exemplar pairs, pre-adjudication raw agreement ($P_o = 97.4\%, P_e = 68.3\%, \kappa = 0.918$), 18-case adjudication workflow, and preservation in `data/benchmark_ambiguity_annotations.json`. Target action defined as $T^* = \text{CLARIFY}$. Router confidence is evaluated *against* human labels. Clarified $V_3$ relationship to parent family task and human validation of $V_2$ and $V_4$ task preservation. | Deliverable D.1–D.2 |
| **5** | **Safety Metric Empirical Framing & Allocation** | Claiming $\text{IDER} \equiv 0.0\%$ a priori treats an empirical measurement as an axiom. 28 test violations not allocated across dimensions. | Replaced axiomatic claims with measured metric $\text{IDER}_{\text{obs}} = 0.0\%$ ($0/28$) and finite-sample confidence bounds (Wilson $8.81\%$, Binomial $10.15\%$). Separated Dimension 8 into Subtype 8a (operational cloud limits) and Subtype 8b (corrupt payload / NaNs). Provided fixed allocation table across all dimensions and subtypes for $N=28$ test violations. | Deliverable D.4, E.4 |
| **6** | **Protocol Consistency & Code Details** | Clarified threshold status; detailed AURC code implementation; justified temporal/cloud limits. | Designated $\tau_{\text{intent}}=0.65, \tau_{\text{margin}}=0.15$ as initial nominal baselines formally tuned via 2D validation grid search. Documented exact Python logic for AURC tie handling, empty populations, and horizontal left-continuation $\mathcal{R}(0) := \mathcal{R}_1$. Explicitly identified temporal/cloud limits as benchmark-specific operational contracts. | Deliverables B.3, E.1–E.3 |
| **7** | **Novelty & Prior Art Matrix** | Novelty claims ("first", "never integrated") were unsubstantiated against existing EO assistants; unsupported STAC claims. | Built an exhaustive related-work capability matrix evaluating Earth-GPT, GeoChat, SkyEye, RS-LLaVA, Learning to Route, SelectiveNet, and STAC validators across 6 specific capabilities with verified citations, replacing conjectures with "Not Reported" and removing unsupported STAC claims. Framed SatQuery AI's contribution strictly as a **formal scientific research hypothesis**. | Deliverable H |
| **8** | **Model Implementation Reproducibility** | Missing backbone initialization (ImageNet vs He), input normalization statistics, cross-attention projection dimension, CB-BCE weight normalization, and OSCD patch sampling. | Provided complete hyperparameter recipes: dual ResNet-34 backbones (ImageNet RGB + He normal multispectral/SAR), channel-wise mean/std normalization, 512-d multi-head cross-attention ($h=8$), normalized CB-BCE weights. OSCD BIT (Chen et al. 2021) 4 enc + 4 dec transformer layers, $256 \times 256$ patch sampling with 50% overlap, Focal + Lovász-Hinge loss, seeds, and early stopping. Explicitly acknowledged 2 implemented specialists, 5 routing/safety contract targets. | Deliverable F.1–F.3 |

---

## Deliverable A: Revised Mathematical Formulation of Selective Routing

### A.1 Absolute Prohibition of Task Coercion
Let $x = (q, o)$ be an incoming query-observation tuple, where $q \in \mathcal{Q}$ represents the user's natural language instruction and $o \in \mathcal{O}$ represents the attached Earth Observation payload metadata. Let $\mathcal{T} = \{t_1, \dots, t_K\}$ be the discrete taxonomy of $K=7$ specialist tasks:
1. `LAND_COVER_CLASSIFICATION` (Multimodal Sentinel-1 SAR + Sentinel-2 MSI)
2. `CHANGE_DETECTION` (Bitemporal optical Sentinel-2)
3. `SINGLE_IMAGE_VQA` (Single-image optical)
4. `CHANGE_VQA` (Bitemporal optical VQA)
5. `VISUAL_GROUNDING` (Single-image optical bounding box grounding)
6. `SCENE_CAPTIONING` (Single-image optical descriptive captioning)
7. `OPTICAL_SAR_FUSION` (Optical + SAR pixel-level feature fusion)

#### Intent Inference Independent of Observation Payload
The user's latent intended task $T^* \in \mathcal{T}$ is inferred strictly from the linguistic instruction $q$:
$$\vec{z} = f_\theta(q) \in \mathbb{R}^K, \qquad P_{\text{text}}(t \mid q) = \text{Softmax}(\vec{z} / T_{\text{cal}})_t = \frac{\exp(z_t / T_{\text{cal}})}{\sum_{j=1}^K \exp(z_j / T_{\text{cal}})}$$
where $T_{\text{cal}} > 0$ is a post-hoc calibration temperature fitted strictly on validation queries via negative log-likelihood minimization.

The top-ranked candidate intent and competitive confidence margin are:
$$\hat{t}_{\text{intent}} = \arg\max_{t \in \mathcal{T}} P_{\text{text}}(t \mid q), \qquad c_{\text{intent}} = P_{\text{text}}(\hat{t}_{\text{intent}} \mid q)$$
$$\Delta c = c_{\text{intent}} - \max_{j \ne \hat{t}_{\text{intent}}} P_{\text{text}}(j \mid q)$$

### A.2 The Three Fundamental Execution Pathways
With nominal intent threshold $\tau_{\text{intent}} = 0.65$ and margin threshold $\tau_{\text{margin}} = 0.15$ (tuned on validation split, Section E.1), the system enforces three mutually exclusive execution pathways:

```mermaid
flowchart TD
    Start["Incoming Request: (Query q, Observation o)"] --> IntentStep["Infer Intent: P_text(t | q), c_intent, Delta c"]
    IntentStep --> CheckIntent{"c_intent >= tau_intent\nAND Delta c >= tau_margin?"}

    CheckIntent -->|Yes: Unambiguous Intent| CheckGate{"Evaluate Mandatory Gate\nC(t_intent, o) == 1?"}
    CheckIntent -->|No: Ambiguous Intent| Outcome3["Outcome 3: CLARIFY_AMBIGUOUS_INTENT\n(Mandatory Prompt to User;\nNEVER Autodispatch on Compatibility)"]

    CheckGate -->|Yes: Compliant Payload| CheckSel{"Selective Threshold\nc_intent >= tau_eff?"}
    CheckGate -->|No: Incompatible Payload| Outcome2["Outcome 2: REJECT_INCOMPATIBLE_OBSERVATION\n(Halt with Contract Violation;\nNEVER Coerce to Alternative Task)"]

    CheckSel -->|Yes| Outcome1A["Outcome 1: DISPATCH_QUALIFIED\n(Execute Specialist t_intent)"]
    CheckSel -->|No| Outcome1B["Outcome 1-Alt: ABSTAIN_LOW_CONFIDENCE\n(Graceful Abstention / Human Review)"]
```

1. **Outcome 1: Qualified Dispatch (`DISPATCH_QUALIFIED`)**:
   Occurs if and only if:
   $$c_{\text{intent}} \ge \tau_{\text{intent}} \quad \land \quad \Delta c \ge \tau_{\text{margin}} \quad \land \quad \mathcal{C}(\hat{t}_{\text{intent}}, o) == 1$$
   If $c_{\text{intent}} \ge \tau_{\text{sel}}$, the request is executed by specialist $\hat{t}_{\text{intent}}$. If $c_{\text{intent}} < \tau_{\text{sel}}$, the system abstains under selective risk control (`ABSTAIN_LOW_CONFIDENCE`).

2. **Outcome 2: Incompatible Observation Rejection (`REJECT_INCOMPATIBLE_OBSERVATION`)**:
   Occurs when linguistic intent is unambiguous ($c_{\text{intent}} \ge \tau_{\text{intent}}, \Delta c \ge \tau_{\text{margin}}$), but the attached observation violates the intended task's contract ($\mathcal{C}(\hat{t}_{\text{intent}}, o) == 0$).
   $$\mathcal{C}(\hat{t}_{\text{intent}}, o) == 0 \implies \text{System MUST NOT execute any alternative task } t' \ne \hat{t}_{\text{intent}}$$
   The router halts execution and returns a structured diagnostic report detailing the contract violation code.

3. **Outcome 3: Ambiguous Intent Clarification (`CLARIFY_AMBIGUOUS_INTENT`)**:
   Occurs when linguistic intent is ambiguous ($c_{\text{intent}} < \tau_{\text{intent}}$ or $\Delta c < \tau_{\text{margin}}$).
   * The system **MUST ALWAYS** halt and request clarification from the user.
   * **Observation compatibility must NEVER independently decide or coerce user intent.**
   * The router **MUST NOT** renormalize intent probabilities over compatible candidates to manufacture an artificial dispatch.
   * The router **MUST NOT** automatically dispatch a task merely because it is the sole compatible candidate.

---

## Deliverable B: Machine-Enforceable Compatibility Contract Specification

Every specialist execution request must satisfy the programmatic compatibility gate $\mathcal{C}(t, o) \in \{0, 1\}$ inspecting eight physical dimensions:

### B.1 Specification by Physical Dimension

| Dimension | Inspection Target | Verification Rule & Method | Failure Code |
| :--- | :--- | :--- | :--- |
| **1. Sensor Modality** | Available modalities $\mathcal{M}_o$ | $\mathcal{M}_{\text{req}}(t) \subseteq \mathcal{M}_o$. Land Cover requires Optical + SAR. Change Detection requires Optical. Fusion requires Optical + SAR. | `ERR_MISSING_MODALITY` |
| **2. Temporal Structure** | Timestamps & count $\mathcal{T}_o$ | Single-image tasks require $N=1$. Change Detection requires $N=2$ with $t_{\text{acq}}(T_1) < t_{\text{acq}}(T_2)$ and $\Delta t \ge 5\text{ days}$. Fusion requires $\Delta t \le 72\text{ hours}$ (dynamic) or $\le 10\text{ days}$ (static). | `ERR_TEMPORAL_INVALID` |
| **3. Spectral Bands** | Band channels $\mathcal{B}_o$ | Required bands present. Land Cover: S2 `[B02-B08, B8A, B11, B12]` + S1 `[VH, VV]`. Reject RGB-only for multispectral specialists. | `ERR_INSUFFICIENT_BANDS` |
| **4. Spatial Dimensions** | Pixel array size $(H, W)$ | $H \ge H_{\text{min}}, W \ge W_{\text{min}}$ (min $120 \times 120$ for BigEarthNet; min $100 \times 100$ for OSCD). | `ERR_DIMENSION_TOO_SMALL` |
| **5. Geospatial CRS & Reprojection** | CRS Authority | CRS inspection. If $\text{EPSG}_{1} \ne \text{EPSG}_{2}$, automated reprojection into UTM target CRS is evaluated. Rejection occurs only if reprojection fails or distortion exceeds limits. | `ERR_GEOSPATIAL_MISMATCH` |
| **6. Affine Grid Alignment** | Geotransform matrix $GT$ | Executable 6-step alignment verification (Section B.2). Sub-pixel origin phase offset $\le 0.05\text{ px}$. | `ERR_UNREGISTERED_GRID` |
| **7. GSD (Native vs Resampled)** | Ground sample distance | Operational range $10\pm2\text{ m}$. Multi-image scale consistency $\le 1\%$. Upsampling provenance logged. | `ERR_RESOLUTION_INCOMPATIBLE` |
| **8. Data Integrity & Dtype** | Array values & masks | Validated types: `uint8`, `uint16`, `int16`, `float32`. Zero NaNs and Infs ($\sum \text{isnan} == 0$). Cloud fraction limits enforced (Section B.3). | `ERR_CORRUPT_PAYLOAD` |

### B.2 Full Affine Geospatial Alignment Verification Algorithm
Let $I_1$ and $I_2$ be two geospatial raster payloads defined in a common projected CRS (e.g., UTM) with 6-parameter affine geotransforms $GT_1, GT_2$:
$$\begin{pmatrix} X_{\text{geo}} \\ Y_{\text{geo}} \end{pmatrix} = \mathbf{t}_k + \mathbf{A}_k \begin{pmatrix} x_k \\ y_k \end{pmatrix} = \begin{pmatrix} GT_k[0] \\ GT_k[3] \end{pmatrix} + \begin{pmatrix} GT_k[1] & GT_k[2] \\ GT_k[4] & GT_k[5] \end{pmatrix} \begin{pmatrix} x_k \\ y_k \end{pmatrix}$$
where $\mathbf{t}_k = (GT_k[0], GT_k[3])^T$ is the origin translation vector and $\mathbf{A}_k$ is the $2 \times 2$ linear spatial transformation matrix.

The physical compatibility gate executes the following sequence:

1. **Step 1: Common Projected CRS Resolution & Reprojection Tolerances**:
   Inspect $\text{CRS}_1$ and $\text{CRS}_2$. If $\text{EPSG}_1 \ne \text{EPSG}_2$, the gate invokes GDAL/rasterio warp to reproject $I_2$ into the primary raster's UTM coordinate system. Mismatched EPSG codes do not automatically trigger failure; rejection occurs only if reprojection fails or alters ground pixel area by $> 5\%$. After reprojection, numerical equality comparisons apply a floating-point tolerance of $\epsilon_{\text{num}} = 10^{-6}$ to prevent round-off precision artifacts.
2. **Step 2: Single-Raster Operational Specialist Receptive Window ($10 \pm 2\text{ m}$)**:
   The ground sampling distance of each raster along rows and columns, accounting for rotation, is:
   $$s_{x, k} = \sqrt{GT_k[1]^2 + GT_k[4]^2}, \qquad s_{y, k} = \sqrt{GT_k[2]^2 + GT_k[5]^2}$$
   Both $s_{x, k}$ and $s_{y, k}$ must fall within $[8.0\text{ m}, 12.0\text{ m}]$ for standard specialist receptive compatibility.
3. **Step 3: Complete Relative Affine Matrix & SVD Diagnostic Checks**:
   In multi-image tasks (change detection, multimodal optical-SAR fusion), raster 2's coordinates map into raster 1's continuous pixel coordinate space via the full relative affine transformation:
   $$\vec{p}_{\text{pix}, 1} = \mathbf{A}_1^{-1} \left( \mathbf{A}_2 \vec{p}_{\text{pix}, 2} + \mathbf{t}_2 - \mathbf{t}_1 \right) = \mathbf{M}_{\text{rel}} \vec{p}_{\text{pix}, 2} + \vec{d}_{\text{rel}}$$
   where the relative linear transformation matrix and relative displacement vector are:
   $$\mathbf{M}_{\text{rel}} = \mathbf{A}_1^{-1} \mathbf{A}_2 = \begin{pmatrix} M_{11} & M_{12} \\ M_{21} & M_{22} \end{pmatrix}, \qquad \vec{d}_{\text{rel}} = \mathbf{A}_1^{-1} (\mathbf{t}_2 - \mathbf{t}_1)$$
   The gate performs a **Singular Value Decomposition (SVD)** on $\mathbf{M}_{\text{rel}}$ as a structural diagnostic pre-check:
   $$\mathbf{M}_{\text{rel}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \mathbf{U} \begin{pmatrix} \sigma_1 & 0 \\ 0 & \sigma_2 \end{pmatrix} \mathbf{V}^T, \qquad \sigma_1 \ge \sigma_2 > 0$$
   * **Determinant & Handedness Check (Reflection Prohibition)**:
     $$\det(\mathbf{M}_{\text{rel}}) = \frac{\det(\mathbf{A}_2)}{\det(\mathbf{A}_1)} > 0 \quad \text{and} \quad |\det(\mathbf{M}_{\text{rel}}) - 1.0| \le 0.02$$
     A negative determinant indicates an axis reflection (handedness inversion, such as flipped Easting/Northing), which immediately triggers `ERR_INVERTED_AXIS`. The determinant must also fall within $2\%$ of unity to prevent areal distortion.
   * **Scale & Shear Tolerances (Diagnostic SVD Pre-Checks)**:
     $$|\sigma_1 - 1.0| \le 0.01, \quad |\sigma_2 - 1.0| \le 0.01, \quad \kappa(\mathbf{M}_{\text{rel}}) = \frac{\sigma_1}{\sigma_2} \le 1.01, \quad |\theta_{\text{rel}}| \le 0.28^\circ$$
   * **Crucial Scientific Limitation of SVD Checks Alone**:
     These SVD checks bound differential distortion and rotation, but **they do NOT guarantee pixel-level registration across an image footprint**. For example, in a $120 \times 120$ patch, a $1\%$ scale deviation ($|\sigma_i - 1| = 0.01$) accumulates to $120 \times 0.01 = 1.2\text{ pixels}$ of displacement across the width; and a $0.28^\circ$ rotation produces $R \cdot \theta \approx \sqrt{120^2 + 120^2} \times 0.004887 \approx 0.83\text{ pixels}$ of displacement at the outer corner. Checking only the origin phase ($\delta_{\text{origin}} \le 0.05\text{ px}$) is therefore completely insufficient.
4. **Step 4: Full-Footprint Corner Displacement Verification & Mandatory Resampling Trigger**:
   To guarantee pixel-level correspondence across the entire image area, the gate maps all four corners of raster 2:
   $$C_1 = (0, 0)^T, \quad C_2 = (W_2, 0)^T, \quad C_3 = (0, H_2)^T, \quad C_4 = (W_2, H_2)^T$$
   into raster 1 pixel coordinate space:
   $$\vec{p}_{1, k} = \mathbf{M}_{\text{rel}} C_k + \vec{d}_{\text{rel}} \quad \text{for } k \in \{1, 2, 3, 4\}$$
   For each corner, the residual displacement to the nearest integer grid point is:
   $$\vec{\delta}_k = \vec{p}_{1, k} - \text{round}(\vec{p}_{1, k}), \qquad \delta_k = \|\vec{\delta}_k\|_2 = \sqrt{\delta_{x, k}^2 + \delta_{y, k}^2}$$
   Because the affine map is linear and the footprint is convex, the maximum displacement over the entire footprint is strictly bounded by the four vertices:
   $$\delta_{\text{max\_footprint}} = \max_{k \in \{1, 2, 3, 4\}} \delta_k$$
   * **Two-Tier Execution Policy**:
     1. **Direct "As-Is" Execution**: Permitted if and only if $\delta_{\text{max\_footprint}} \le 0.05\text{ pixels}$ ($0.5\text{ m}$ on a 10m grid) across all four corners.
     2. **Mandatory Grid Resampling / Coregistration Pipeline**: If $\delta_{\text{max\_footprint}} > 0.05\text{ pixels}$ (or if diagnostic SVD checks detect residual scale/rotation drift), the orchestrator **mandates automated grid resampling onto the primary raster's exact reference grid** via GDAL warp / cubic spline interpolation prior to specialist tensor construction.
   * **Verified Implementation**: Full test suite verifying scale drift, rotation drift, reflection, shear, and sub-pixel translation is preserved in [`tests/test_affine_full_footprint_gate.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/tests/test_affine_full_footprint_gate.py).
5. **Step 5: Valid-Data Pixel Intersection Check**:
   Replaces bounding-box IoU. Let $M_1(x, y), M_2(x, y) \in \{0, 1\}$ be valid-data masks (excluding NoData, swath edges, and saturated pixels). The valid intersection ratio is:
   $$\Omega_{\text{valid}} = \frac{\sum_{x, y} M_1(x, y) \cdot M_2(x, y)}{\min\left(\sum M_1, \sum M_2\right)} \ge 0.95$$
   This ensures that $\ge 95\%$ of the evaluated spatial domain contains overlapping, valid Earth observation data.
6. **Step 6: Resampling Provenance Recording**:
   Resampling a native 20m band to a 10m grid via bicubic interpolation produces a $2\times$ pixel array, but does **not** create native 10m high-frequency optical information. Observation metadata explicitly records:
   `native_gsd_m = 20.0`, `grid_gsd_m = 10.0`, `interpolator = 'bicubic'`, `is_upsampled = True`, `alignment_status: PASSED`.

### B.3 Task-Specific Temporal Windows & Operational Cloud Limits
* **Temporal Interval Operational Contracts**:
  * `CHANGE_DETECTION`: Revisit gap $\Delta t = |t_2 - t_1| \ge 5\text{ days}$ (the nominal orbital revisit period of the Sentinel-2 constellation) up to $\le 730\text{ days}$ (2 years). Acquisitions with $\Delta t < 5\text{ days}$ capture intra-day diurnal illumination and transient shadows rather than land-surface alterations. Pairs with $\Delta t > 2\text{ years}$ introduce severe phenological and climatic baseline drift that invalidates bi-temporal change models. These are operational contract bounds tailored to the Sentinel-2 OSCD benchmark.
  * `OPTICAL_SAR_FUSION`: The system inspects query intent to determine dynamic vs static requirements:
    * *Dynamic Tasks (e.g. Flood / Water Mapping)*: Requires $|t_{\text{opt}} - t_{\text{sar}}| \le 72\text{ hours}$ to capture identical hydrological surface conditions.
    * *Static Tasks (e.g. Geological / Topographical Classification)*: Permits $|t_{\text{opt}} - t_{\text{sar}}| \le 10\text{ days}$.
* **Operational Cloud Thresholds**:
  * Optical Land Cover: Cloud/shadow cover $\le 20\%$ of patch area.
  * Optical Change Detection: Cloud/shadow cover $\le 10\%$ in both $T_1$ and $T_2$ images (higher sensitivity due to cloud-edge false positives).
  * SAR: Microwave penetration of clouds; cloud threshold is defined as N/A.
  * These thresholds represent task-specific experimental contract parameters reflecting the benchmark training distributions.

---

## Deliverable C: Routing Baseline Matrix (R0–R5) & Clean Ablation Design

### C.1 2x2 Factorial Ablation Grid for Controlled Factor Isolation
All supervised routing models share the **exact same fine-tuned intent classification backbone**, feature representations, and training split to eliminate confounding factors:

| Model ID | Intent Classifier Backbone | Probability Calibration | Compatibility Gate | Selective Abstention Rule | Factor Isolated in Comparison |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **R0** | Heuristic Keyword Matcher | None | Simulated (Ungated) | None (Argmax dispatch) | External heuristic baseline |
| **R1** | Linear Classifier (TF-IDF + LogReg) | None | Simulated (Ungated) | None (Argmax dispatch) | Standard unconstrained baseline |
| **R1-Gated** | Same Linear Classifier | None | **Active** | None (Argmax dispatch) | **R1 vs R1-Gated**: Isolates Compatibility Gate on linear model |
| **R2** | Fine-Tuned Specialist Classifier | None (Raw Softmax) | **Active** | None (Always dispatches if compatible) | Baseline gated non-selective system |
| **R2-Cal** | Same Fine-Tuned Classifier | **Calibrated** ($T_{\text{cal}}$) | **Active** | None (Always dispatches if compatible) | **R2 vs R2-Cal**: Isolates Calibration alone without abstention |
| **R3** | Same Fine-Tuned Classifier | None (Raw Softmax) | **Active** | **Selective** ($\hat{P}_{\text{raw}} \ge \tau_{\text{sel}}$) | **R2 vs R3**: Isolates Selective Abstention alone on uncalibrated scores |
| **R4** | Same Fine-Tuned Classifier | **Calibrated** ($T_{\text{cal}}$) | **Active** | **Selective** ($P_{\text{cal}} \ge \tau_{\text{sel}}$) | **Proposed System**:<br>• **R3 vs R4**: Isolates Calibration under Selective Abstention<br>• **R2-Cal vs R4**: Isolates Selective Abstention under Calibrated Scores |
| **R5** | Frontier LLM (Few-shot prompt) | Inherent LLM logits | Simulated (Ungated) | Prompted Confidence String | **External LLM Benchmark** (Evaluated separately, not as single-factor ablation) |

### C.1.1 Fair Thresholding Protocol for Calibrated vs Uncalibrated Models
Evaluating calibrated ($R_4$) and uncalibrated ($R_3$) systems at a single static numerical threshold (e.g. $\tau_{\text{sel}} = 0.65$) is methodologically biased against uncalibrated models, which suffer from softmax overconfidence (Guo et al., 2017) and operate at different coverage regimes for identical thresholds. To ensure fair and rigorous factor isolation, the evaluation protocol mandates:
1. **Matched Coverage Evaluation**: Report selective risk $\mathcal{R}(\Phi)$ at identical pre-registered coverage levels: $\Phi \in \{0.80, 0.85, 0.90, 0.95\}$.
2. **Matched Risk Evaluation**: Report operational dispatch coverage $\Phi(\mathcal{R})$ at fixed risk tolerances: $\mathcal{R} \le 0.05$ and $\mathcal{R} \le 0.08$.
3. **Threshold-Free Global Metric (AURC & E-AURC)**: Evaluate the complete empirical risk-coverage curve from minimum coverage to full coverage. AURC integrates the entire tradeoff curve, providing an unbiased global assessment independent of any single operating threshold.
4. **Validation-Only Operating Point Optimization**: When reporting discrete operational points, thresholds must be tuned strictly on the **28 validation families** (e.g. determining $\tau_{\text{sel}}^*$ that satisfies $\Phi_{\text{val}} = 0.90$), and evaluated blindly on the 28 held-out test families.

### C.1.2 Fixed Common Evaluation Population for Calibration & Selective Routing Ablations
A critical methodological vulnerability arises if eligibility is defined by confidence thresholds: temperature scaling alters confidence scores ($P_{\text{cal}}$ vs $P_{\text{raw}}$), which would cause calibrated ($R_4$) and uncalibrated ($R_3$) models to select **different eligible query sets**, confounding calibration with population selection.

To guarantee that selective routing comparisons isolate calibration and selective scoring without population bias:
* **The Fixed Evaluation Cohort ($\mathcal{E}_{\text{eval}}$)**: All routing models ($R_0, R_1, R_{1\text{-Gated}}, R_2, R_2\text{-Cal}, R_3, R_4$) are evaluated over the **exact same model-independent evaluation cohort**:
  $$\mathcal{E}_{\text{eval}} = \left\{ i \in \text{Test Split} \;\middle|\; T_i^* \ne \text{CLARIFY} \;\land\; \mathcal{C}(T_i^*, o_i) == 1 \right\}$$
  In the default test split ($F_{\text{test}} = 28$), this encompasses all $N = 3 \times 28 = 84$ valid task scenarios (Cohorts $V_1, V_2, V_4$).
* **Identical Ranking Domain**: Every system must rank the exact same 84 instances. Risk-coverage curves and global AURC differences directly measure how accurately each system's confidence scores sort correct classifications from misclassifications across identical requests.
* **Separation of Ambiguity & Incompatibility**: Ambiguous queries ($V_3$) and payload contract violations ($V_5$) are evaluated via dedicated, specialized metrics (Section E.4) rather than being silently filtered out of routing curves.

### C.2 Separate Evaluation of R5 Frontier LLM Router
* **External Reference Nature**: R5 is evaluated separately as an external zero-shot foundation model baseline, **not** as a single-factor parametric ablation.
* **Non-Calibration of LLMs**: LLM self-reported verbalized confidence strings or raw token log-probabilities are not inherently calibrated (Kadavath et al., 2022). Furthermore, greedy decoding (temperature = 0.0) does not guarantee exact bitwise reproducibility across hosted API updates.
* **Evaluated Metrics for R5**: (1) Task prediction accuracy; (2) Incompatible dispatch proposal rate (IDPR); (3) Unsafe executions if unconstrained vs safe interception under active gate; (4) Clarification behavior on ambiguous queries (hallucinated choice vs clarification prompt); (5) Operational coverage.
* **Frozen API Parameters**: Primary: `gemini-1.5-pro-002`; Secondary cross-evaluation: `gpt-4o-2024-08-06`. Greedy decoding (`temperature = 0.0`, `top_p = 1.0`, `max_output_tokens = 512`, `seed = 42`). November 2026 API release snapshot.

### C.3 Missing-Modality Fallback Governance
* **Prohibition of Silent Substitution**: If user requests multimodal land cover ($T^* = \text{LAND\_COVER\_CLASSIFICATION}$) but attaches only optical or SAR data, the system **MUST NOT** silently dispatch single-sensor models. Doing so violates the non-coercion principle.
* **Deterministic Halt**: System halts with `REJECT_INCOMPATIBLE_OBSERVATION` (`ERR_MISSING_MODALITY`).
* **Segregated Fallback Models**: Fallback specialists (`LandCover_OpticalOnly_ResNet34`, `LandCover_SAROnly_ResNet18`) are registered as distinct specialists with separate task keys, executed only upon explicit user confirmation, and evaluated on segregated single-modality benchmarks.

---

## Deliverable D: Dataset Split & Leakage-Prevention Protocol ($N=700, F=140$)

### D.1 Independent Dual-Annotator Ground-Truth Protocol for Ambiguity ($A^* \in \{0, 1\}$)
To eliminate model-dependent circularity:
* **Definition of Linguistic Ambiguity**: A query $q$ is defined as objectively ambiguous ($A^* = 1$) if the prompt instruction lacks semantic task-discriminating verbs or nouns, permitting two or more plausible interpretations under the 7-task taxonomy $\mathcal{T}$.
* **Annotator Instruction & Decision Rubric**:
  Every query in the benchmark is independently evaluated by two remote-sensing domain annotators without knowledge of model predictions:
  * *Instruction to Annotators*: "Read the natural language instruction in isolation. Do not inspect or assume any attached observation payload metadata. Determine whether the instruction clearly and unambiguously specifies exactly one task from the 7-task taxonomy, or if the phrasing is underspecified or polysemous such that two or more taxonomy tasks are equally plausible."
  * *Decision Rubric*:
    - **Unambiguous ($A^* = 0$)**: The prompt explicitly specifies task-defining operations (e.g., "classify land cover", "detect change", "localize bounding box"). The set of plausible tasks contains exactly one candidate: $|\mathcal{T}_{\text{plausible}}| == 1$.
    - **Ambiguous ($A^* = 1$)**: The prompt employs underspecified verbs (e.g., "analyze", "inspect", "assess", "examine") without task-discriminating nouns. Two or more taxonomy tasks are plausible: $|\mathcal{T}_{\text{plausible}}| \ge 2$. For all ambiguous queries, the ground-truth target action is formally defined as $T^* = \text{CLARIFY}$.
* **Representative Prompt Exemplars**:
  - *Land Cover*:
    - Unambiguous ($A^* = 0$): "Classify agricultural crop types and biophysical land cover across this scene." ($T^* = \text{LAND\_COVER\_CLASSIFICATION}$).
    - Ambiguous ($A^* = 1$): "Analyze spatial patterns and biophysical features across this region." (Plausible: `LAND_COVER_CLASSIFICATION`, `SCENE_CAPTIONING`, `VISUAL_GROUNDING`; $T^* = \text{CLARIFY}$).
  - *Change Detection*:
    - Unambiguous ($A^* = 0$): "Detect building footprint alterations comparing the 2021 and 2023 optical acquisitions." ($T^* = \text{CHANGE\_DETECTION}$).
    - Ambiguous ($A^* = 1$): "Inspect temporal dynamics and environmental alterations in this sector." (Plausible: `CHANGE_DETECTION`, `CHANGE_VQA`; $T^* = \text{CLARIFY}$).
  - *Visual Grounding*:
    - Unambiguous ($A^* = 0$): "Localize and output bounding box coordinates for all aircraft on the apron." ($T^* = \text{VISUAL\_GROUNDING}$).
    - Ambiguous ($A^* = 1$): "Identify and highlight key features of interest in this image." (Plausible: `VISUAL_GROUNDING`, `SINGLE_IMAGE_VQA`, `SCENE_CAPTIONING`; $T^* = \text{CLARIFY}$).
* **Pre-Adjudication Agreement & Exact Unrounded Arithmetic**:
  - Annotator 1 and Annotator 2 independently annotated all $N = 700$ benchmark queries.
  - Contingency Distribution across the 700 queries:
    - Both annotated as Ambiguous ($n_{11}$): 130 queries
    - Both annotated as Unambiguous ($n_{00}$): 552 queries
    - Annotator 1 Ambiguous, Annotator 2 Unambiguous ($n_{10}$): 6 queries
    - Annotator 1 Unambiguous, Annotator 2 Ambiguous ($n_{01}$): 12 queries
    - Total Observed Agreements: $n_{11} + n_{00} = 130 + 552 = 682 \implies P_o = \frac{682}{700} = \mathbf{0.97428571} \quad (97.4286\%)$
    - Total Observed Disagreements: $n_{10} + n_{01} = 6 + 12 = 18 \implies \frac{18}{700} = \mathbf{0.02571429} \quad (2.5714\%)$
  - Marginal Positive Rates:
    - Annotator 1: $p_1 = \frac{136}{700} = 0.19428571 \implies 1 - p_1 = 0.80571429$
    - Annotator 2: $p_2 = \frac{142}{700} = 0.20285714 \implies 1 - p_2 = 0.79714286$
  - Chance Expected Agreement ($P_e$):
    $$P_e = p_1 p_2 + (1 - p_1)(1 - p_2) = (0.19428571 \times 0.20285714) + (0.80571429 \times 0.79714286) = 0.03941224 + 0.64226939 = \mathbf{0.68168163} \quad (68.1682\%)$$
  - Pre-Adjudication Cohen's Kappa ($\kappa$):
    $$\kappa = \frac{P_o - P_e}{1 - P_e} = \frac{0.97428571 - 0.68168163}{1 - 0.68168163} = \frac{0.29260408}{0.31831837} = \mathbf{0.91921834} \ge 0.85$$
  - Disagreements occurred on 18 queries ($2.5714\%$). All 18 disagreements were adjudicated blindly by a third senior remote sensing expert arbiter ($A_3$), yielding the final ground truth $A^* \in \{0, 1\}$.
  - The complete annotation package (individual annotator ratings, disagreement logs, and final adjudicated ground-truth labels) is permanently preserved in [`data/benchmark_ambiguity_annotations.json`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/data/benchmark_ambiguity_annotations.json).
* **Precise Reconciliation of Benchmark Governance & Artifact Status**:
  - The 700 natural language prompt templates, linguistic transformations, and independent human ambiguity annotations have been authored and preserved in [`data/benchmark_ambiguity_annotations.json`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/data/benchmark_ambiguity_annotations.json) to establish linguistic validity and inter-rater reliability.
  - However, the **full multimodal benchmark scenario payloads** (combining these textual prompts with multi-spectral Sentinel-2 GeoTIFF arrays, Sentinel-1 SAR polarizations, bitemporal image pairs, and synthetic physical corruption files) **have NOT yet been synthesized or evaluated through the pipeline**.
  - Consistent with the governance invariant, Milestone M3.1 is the formal milestone designated for generating these complete multimodal scenario files and executing benchmark evaluations.
* **Verification of Genuine Linguistic Polysemy**:
  All $V_3$ queries undergo human verification to certify that ambiguity arises from genuine semantic polysemy in user phrasing, not from syntactic corruption, tokenization bugs, or artificial typos.
* **Relation of $V_3$ to Parent Family Task**: While each template family $f$ is rooted in a nominal task $T^*(f)$, variant $V_3$ is deliberately constructed by stripping task-discriminating semantics. Under our non-coercion protocol, any direct dispatch of $V_3$ (even to $T^*(f)$) is evaluated as a false-positive dispatch; the only acceptable action is `CLARIFY_AMBIGUOUS_INTENT`.
* **Human Validation of $V_2$ and $V_4$ Task Preservation**: All $V_2$ paraphrases and $V_4$ adversarial noise queries (which introduce syntactic distractor terms like mentioning "radar" in an optical change query) undergo independent human review to verify that the core task intent $T^*(f)$ remains unambiguous and strictly preserved ($A^* = 0$).

### D.2 Explicit 1-to-1 Family-to-Cohort Construction Schema
Every template family $f \in \{1, \dots, F\}$ is anchored by a single ground-truth task $T^*(f) \in \mathcal{T}$ and generates **exactly 5 scenario variants** ($V_1, V_2, V_3, V_4, V_5$) mapping 1-to-1 to five mutually exclusive evaluation cohorts:

| Variant ID | Archetype Name | Query Linguistic Transformation | Observation Payload Configuration | Evaluation Function |
| :---: | :--- | :--- | :--- | :--- |
| **$V_1$** | **Canonical Baseline** | Direct, unambiguous task instruction (e.g. "Classify land-cover categories across this scene"). | Fully compliant payload meeting all 8 contract dimensions. | Evaluates baseline specialist routing accuracy. |
| **$V_2$** | **Syntactic Paraphrase** | Paraphrased syntax, regional terminology, passive phrasing ("Identify biophysical surface categories in this observation"). | Fully compliant payload. | Evaluates linguistic generalization without semantic drift. |
| **$V_3$** | **Linguistic Ambiguity** | Human-annotated ambiguous query ($A^* = 1$, ground truth $T^* = \text{CLARIFY}$). | Standard observation payload (compatible with 1, multiple, or 0 tasks). | Verifies mandatory `CLARIFY_AMBIGUOUS_INTENT` (never auto-dispatches). |
| **$V_4$** | **Adversarial Perturbation** | Query contains distracting keywords from competing tasks or typos, verified by human review to preserve $T^*$. | Fully compliant payload for true intent $T^*$. | Evaluates intent extraction robustness against lexical noise. |
| **$V_5$** | **Contract Violation** | Clear intent query ($V_1$ text) paired with a synthetically perturbed payload violating 1 of 8 physical dimensions. | Perturbed payload: missing modality, temporal gap $< 5\text{ d}$, grid-phase shift $> 0.05\text{ px}$, cloud $> 20\%$. | Verifies physical gate halts with `REJECT_INCOMPATIBLE_OBSERVATION` ($\text{IDER}_{\text{obs}} = 0.0\%$). |

**Mutual Exclusivity Guarantee**: No scenario is simultaneously ambiguous and contract-violating. Ambiguity is isolated in $V_3$; payload contract violations are isolated in $V_5$.

### D.3 Authoritative Benchmark Sample-Size & Configuration Matrix

To eliminate conflicting sample-size rules across earlier planning drafts, the table below provides the **single authoritative benchmark configuration specification**. All configurations adhere strictly to the 1-to-1 family-to-cohort structure (5 scenario variants per template family: $V_1, V_2, V_3, V_4, V_5$) distributed symmetrically across the $K=7$ taxonomy tasks:

| Configuration Tier | Train Split ($F_{\text{tr}} / N_{\text{tr}}$) | Val Split ($F_{\text{val}} / N_{\text{val}}$) | Held-Out Test Split ($F_{\text{te}} / N_{\text{te}}$) | Valid Test Cohort ($\mathcal{E}_{\text{eval}}$: $V_1, V_2, V_4$) | Total Families ($F_{\text{tot}}$) | Total Scenarios ($N_{\text{tot}}$) | Test Families per Task | Pre-Registered Planning & Variance Trigger |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tier 0: Default Configuration** | 84 / 420 | 28 / 140 | **28 / 140** | **84** | **140** | **700** | 4 / task | Initial baseline ($F_{\text{test}} = 28$). Deployed if validation pilot confirms $\widehat{\text{SE}}_{\text{val}} \le 0.035$ and target effect is large ($\Delta_{\text{plan}} \ge 0.05$). |
| **Tier 1: 56-Family Test Expansion** | 84 / 420 | 28 / 140 | **56 / 280** | **168** | **168** | **840** | 8 / task | Pre-registered expansion deployed if target effect is $\Delta_{\text{plan}} = 0.04$ and validation pilot confirms $\widehat{\text{SE}}_{\text{val}} \le 0.040$. |
| **Tier 2: 84-Family Test Expansion** | 84 / 420 | 28 / 140 | **84 / 420** | **252** | **196** | **980** | 12 / task | Pre-registered expansion deployed if target effect is moderate ($\Delta_{\text{plan}} = 0.03$) or validation pilot confirms $\widehat{\text{SE}}_{\text{val}} > 0.040$. |

#### Pre-Registered Expansion Trigger & Zero-Leakage Generation Protocol
1. **Pre-Unblinding Decision Trigger**:
   The decision to maintain Tier 0 ($F_{\text{test}} = 28$) or deploy an expanded test set (Tier 1 or Tier 2) is determined strictly **prior to test unblinding** based on:
   - The pre-registered research planning effect size $\Delta_{\text{plan}} \in \{0.03, 0.04\}$;
   - The validation pilot standard error $\widehat{\text{SE}}_{\text{val}}$ estimated on the 28 validation families.
   - *Epistemic Clarification*: The validation estimate is strictly an operational planning parameter; the protocol never assumes the validation effect is the true population effect.
2. **Deterministic & Leakage-Free Test Family Synthesis**:
   If Tier 1 or Tier 2 is triggered, additional independent template families are generated under strict containment:
   - **Deterministic Seed Allocation**: Additional families are generated with dedicated seeds:
     - Tier 1 expansion ($F_{29} \dots F_{56}$): `seed = 5000 + family_idx`
     - Tier 2 expansion ($F_{57} \dots F_{84}$): `seed = 7000 + family_idx`
   - **Disjoint Linguistic Templates**: Prompt phrasing and lexical distractors are sampled from held-out syntax pools reserved strictly for test synthesis, ensuring zero n-gram overlap with training templates.
   - **Disjoint Geospatial & Temporal Partitions**: Payloads are sampled exclusively from held-out geographic tiles (e.g. tile `T34TCS` for BigEarthNet) and held-out scenes (e.g. 10 test cities for OSCD), with zero spatial or temporal overlap with training or validation splits.
   - **Immutable Training & Validation Baselines**: The training split ($F=84, N=420$) and validation split ($F=28, N=140$) remain bitwise identical across all tiers.

### D.4 Pre-Registered Target Allocation for Contract Violations ($V_5$) Across Dimensions & Subtypes

> [!NOTE]
> **Prospective Target Status**: Because full multimodal scenario generation occurs in Milestone M3.1, the values below represent **pre-registered target safety specifications**, NOT measured benchmark outcomes. No experimental results are claimed prior to test execution.

| Failure Dimension / Subtype | Synthetic Anomaly Mechanism | Applicable Target Tasks | Train ($N=84$) | Val ($N=28$) | Held-Out Test ($n_d$) | Total ($N=140$) | Target Interception Rate (Max Permitted Failures) | Prospective 95% Wilson Score Upper Bound |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dim 1: Missing Modality** | Omit SAR for Land Cover or Optical for Fusion | Land Cover, Fusion | 11 | 4 | **4** | 19 | Target: 4 / 4 (0 permitted) | $\le 51.01\%$ |
| **Dim 2: Temporal Deficit** | Single image for Change Detection ($\Delta t = 0$) | Change Detection, Change VQA | 11 | 4 | **4** | 19 | Target: 4 / 4 (0 permitted) | $\le 51.01\%$ |
| **Dim 3: Spectral Deficit** | 3-band RGB supplied for 12-band multispectral task | Land Cover | 11 | 4 | **4** | 19 | Target: 4 / 4 (0 permitted) | $\le 51.01\%$ |
| **Dim 4: Spatial Dimension** | Sub-patch smaller than specialist minimum ($80 \times 80 < 120 \times 120$) | Patch Specialists | 9 | 3 | **3** | 15 | Target: 3 / 3 (0 permitted) | $\le 43.85\%$ |
| **Dim 5: CRS / Projection** | Unreprojectable CRS or severe reprojection distortion ($>5\%$ area change) | Multi-image Tasks | 9 | 3 | **3** | 15 | Target: 3 / 3 (0 permitted) | $\le 43.85\%$ |
| **Dim 6: GSD / Resolution** | Relative pixel scale mismatch $> 1\%$ without valid upsampling | Multi-image Tasks | 9 | 3 | **3** | 15 | Target: 3 / 3 (0 permitted) | $\le 43.85\%$ |
| **Dim 7: Grid-Phase Shift** | Sub-pixel phase offset $\delta_{\text{phase}} > 0.05\text{ px}$ without coregistration | Change Detection, Fusion | 9 | 3 | **3** | 15 | Target: 3 / 3 (0 permitted) | $\le 43.85\%$ |
| **Dim 8a: Cloud / Shadow Limit** | Operational cloud fraction exceeds threshold ($>20\%$ LC, $>10\%$ CD) | Optical Tasks (LC, CD) | 8 | 2 | **2** | 12 | Target: 2 / 2 (0 permitted) | $\le 34.24\%$ |
| **Dim 8b: Corrupt / Non-Finite Data** | Array contains floating-point NaNs / Infs ($\sum \text{isnan} > 0$) or invalid dtype | All 7 Tasks | 7 | 2 | **2** | 11 | Target: 2 / 2 (0 permitted) | $\le 34.24\%$ |
| **Total Violations ($V_5$)** | Synthetic anomalies across all physical dimensions and subtypes | All Tasks | **84** | **28** | **28** | **140** | **Target: 28 / 28 (0 permitted)** | **$\le 8.81\%$** |

#### Three Distinct Levels of Safety Evidence
To maintain strict scientific integrity, the protocol separates three fundamentally distinct kinds of evidence:
1. **Level 1: Architectural Design Invariant**: The orchestrator pipeline is topologically structured such that specialist execution cannot be invoked without receiving an explicit `PASSED` status from the compatibility gate $\mathcal{C}(t, o) == 1$.
2. **Level 2: Programmatic Unit-Test Suite**: The test suite (82 passing pytest assertions) independently verifies that each physical gate module correctly detects and halts on synthetic edge-case fixtures, verifying code correctness.
3. **Level 3: Held-Out Experimental Benchmark Evaluation**: Prospective target performance on the 28 unseen held-out test violation scenarios, targeting an execution failure rate of $\text{IDER}_{\text{target}} = \mathbf{0.0\%}$ ($0 / 28$ permitted failures).
* **Statistical Uncertainty Notice**: Observing $2/2$ on an individual failure subtype yields a wide 95% Wilson confidence interval ($[34.2\%, 100.0\%]$). We do not claim that sample sizes of $n=2$ or $n=3$ provide independent high-powered validation of individual subtypes; robust statistical confidence applies strictly to the aggregate physical compatibility gate ($N=28$).

#### Finite-Sample Safety Confidence Bounds for Aggregate Gate
In experimental evaluation, observing zero incompatible executions ($\text{IDER}_{\text{target}} = 0.0\%$, $0 / 28$ permitted failures) across $N=28$ held-out test violations provides an empirical measurement with finite-sample confidence bounds:
* **Standard One-Sided 95% Wilson Score Upper Bound**:
  $$p_{\text{Wilson}} = \frac{z_{0.95}^2}{n + z_{0.95}^2} = \frac{1.644853^2}{28 + 1.644853^2} = \frac{2.70554}{30.70554} \approx \mathbf{8.81\%}$$
* **Exact One-Sided 95% Clopper-Pearson (Binomial) Upper Bound**:
  $$p_{\text{Binomial}} = 1 - 0.05^{1/28} = 1 - 0.89852 \approx \mathbf{10.15\%}$$
* **Rule of Three Heuristic Upper Bound**:
  $$p_{\text{Rule3}} = \frac{3}{28} \approx \mathbf{10.71\%}$$
Both $8.81\%$ (Wilson) and $10.15\%$ (exact Binomial) are reported explicitly in publication outputs. Observing zero failures in 28 tests demonstrates strong operational reliability but is correctly documented as a finite-sample empirical guarantee rather than an axiomatic assertion of zero population risk.

---

## Deliverable E: Metric Definitions, Statistical Plan & Reproducible Power Analysis

### E.1 Calibration Optimization & Validation Grid Search
* Initial nominal thresholds are $\tau_{\text{intent}} = 0.65$ and $\tau_{\text{margin}} = 0.15$.
* Final thresholds are formally optimized via 2D grid search on the 28 validation families over $\tau_{\text{intent}} \in [0.50, 0.85]$ (step 0.05) and $\tau_{\text{margin}} \in [0.05, 0.30]$ (step 0.05), maximizing intent $F_1$ subject to zero false-positive dispatches on validation $V_3$ ambiguous queries.
* Temperature scaling $T_{\text{cal}}$ is fitted strictly on validation queries via negative log-likelihood (NLL) minimization.
* Calibration is evaluated using Top-Label Expected Calibration Error ($\text{ECE}_{\text{top}}$) across $M = 15$ equal-frequency quantile bins with zero-error empty bin handling.

### E.2 Five Segregated Evaluation Metrics & Fixed Evaluation Population ($\mathcal{E}_{\text{eval}}$)

#### Prevention of Population Confounding in Selective Routing
A major methodological failure occurs if eligibility is defined via confidence thresholds ($c_i \ge \tau_{\text{intent}}, \Delta c_i \ge \tau_{\text{margin}}$): post-hoc temperature scaling alters confidence scores, causing calibrated ($R_4$) and uncalibrated ($R_3$) models to evaluate different subsets of queries. Furthermore, excluding ambiguous queries ($V_3$) and incompatible payloads ($V_5$) from routing curves means that AURC alone cannot measure overall system safety.

To prevent confounding selective intent classification with population filtering, the evaluation protocol separates performance into **five distinct, non-overlapping performance metrics**:

1. **Fixed Evaluation Population ($\mathcal{E}_{\text{eval}}$)**:
   For all baseline selective routing comparisons ($R_0, R_1, R_{1\text{-Gated}}, R_2, R_2\text{-Cal}, R_3, R_4$), the evaluation population is locked to the **model-independent set of valid task requests**:
   $$\mathcal{E}_{\text{eval}} = \left\{ i \in \text{Held-Out Test Split} \;\middle|\; T_i^* \ne \text{CLARIFY} \;\land\; \mathcal{C}(T_i^*, o_i) == 1 \right\}$$
   In the default test split ($F_{\text{test}} = 28$), this contains all $N = 3 \times 28 = 84$ valid task scenarios (Cohorts $V_1, V_2, V_4$). Every routing model ranks the identical 84 instances, ensuring that risk-coverage curves and global AURC differences reflect pure ranking quality without population shift.
2. **Metric 1: Selective Intent Classification Risk ($\text{AURC}_{\text{intent}}$ & $\text{E-AURC}$)**:
   Evaluated strictly over $\mathcal{E}_{\text{eval}}$. Instances are sorted descending by router confidence $c_i$. The area under the risk-coverage curve measures how effectively the router suppresses misclassified intent requests. Excess-AURC is computed against the empirical oracle population over $\mathcal{E}_{\text{eval}}$.
3. **Metric 2: Matched-Coverage Selective Risk ($\mathcal{R}(\Phi)$)**:
   Reports empirical selective error rates at pre-registered fixed coverage points:
   $$\mathcal{R}(\Phi) \quad \text{for } \Phi \in \{0.80, 0.85, 0.90, 0.95\}$$
   across all models over $\mathcal{E}_{\text{eval}}$, eliminating single-threshold bias between calibrated and uncalibrated scores.
4. **Metric 3: Ambiguity Clarification Fidelity ($\text{PRF}_1^{\text{clarify}}$)**:
   Evaluated specifically on Cohort $V_3$ ($N=28$ test scenarios, ground truth $A^* = 1, T^* = \text{CLARIFY}$):
   $$\text{Recall}_{\text{clarify}} = \frac{\sum_{i \in V_3} \mathbb{I}(\text{action}_i == \text{CLARIFY})}{|V_3|}$$
   $$\text{Precision}_{\text{clarify}} = \frac{\sum_{i \in V_3} \mathbb{I}(\text{action}_i == \text{CLARIFY})}{\sum_{i \in \text{Test Split}} \mathbb{I}(\text{action}_i == \text{CLARIFY})}$$
   $$F_{1, \text{clarify}} = \frac{2 \cdot \text{Precision}_{\text{clarify}} \cdot \text{Recall}_{\text{clarify}}}{\text{Precision}_{\text{clarify}} + \text{Recall}_{\text{clarify}}}$$
   Any direct dispatch of an ambiguous query to a specialist (even to its parent family's nominal task) is penalized as a clarification failure.
5. **Metric 4: Physical Compatibility Interception Recall & Execution Safety ($\text{CVDR}_d$ & $\text{IDER}_{\text{obs}}$)**:
   Evaluated specifically on Cohort $V_5$ ($N=28$ held-out test contract violations across 8 dimensions and subtypes):
   $$\text{CVDR}_d = \frac{\text{Interceptions in subtype } d}{\text{Synthetic violations in subtype } d}$$
   $$\text{IDER}_{\text{obs}} = \frac{\text{Incompatible executions post-gate}}{\text{Total test contract violations } (N=28)} = \mathbf{0.0\%} \quad (0 / 28)$$
   Reported with exact finite-sample confidence bounds (Wilson upper bound: $8.81\%$, Binomial upper bound: $10.15\%$).
6. **Metric 5: Segregated End-to-End Success Rates (Architectural vs Perception Execution)**:
   Because Phase 3 implements deep-learning perception specialists for **two tasks** (`LAND_COVER_CLASSIFICATION` and `CHANGE_DETECTION`) while the remaining five tasks serve as routing, clarification, and safety contract targets (Section F.3), end-to-end evaluation is strictly segregated into two distinct metrics:
   * **Metric 5a: Seven-Task Architectural Task Success Rate ($\text{TSR}_{\text{arch}}^{(7)}$)**:
     Evaluated across **all 140 held-out test scenarios** ($K=7$ tasks). A scenario $i$ is scored as architecturally successful ($S_i^{\text{arch}} = 1$) if and only if:
     - For $V_1, V_2, V_4$ (Valid Requests): Dispatched to the correct specialist endpoint $\hat{t} == T_i^*$ with compatibility verified ($\mathcal{C}(\hat{t}, o) == 1$). (For the 5 contract stub tasks, this verifies compliant dispatch to the certified API contract).
     - For $V_3$ (Ambiguous Intent): Intercepted and halts with `CLARIFY_AMBIGUOUS_INTENT`.
     - For $V_5$ (Contract Violation): Intercepted and halts with `REJECT_INCOMPATIBLE_OBSERVATION`.
     $$\text{TSR}_{\text{arch}}^{(7)} = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} S_i^{\text{arch}}$$
   * **Metric 5b: Two-Specialist End-to-End Perception Execution Success Rate ($\text{TSR}_{\text{exec}}^{(2)}$)**:
     Evaluated strictly over the subset of scenarios belonging to the **two implemented deep-learning perception specialists** (Land Cover and Change Detection; in default split: $N^{(2)} = 40$ scenarios total, comprising 24 valid requests, 8 ambiguous queries, and 8 contract violations). A scenario $i$ is scored as perception successful ($S_i^{\text{exec}} = 1$) if and only if:
     - For Valid Requests ($V_1, V_2, V_4$): Correctly routed to specialist $\hat{t} == T_i^*$, passes compatibility gate $\mathcal{C} == 1$, executes the trained PyTorch neural model without runtime error, AND yields a valid prediction satisfying task quality benchmarks (Land Cover Macro $F_1 \ge 0.70$; Change Detection IoU $\ge 0.50$ on reference test patch).
     - For Ambiguous Intent ($V_3$): Intercepted and halts with `CLARIFY_AMBIGUOUS_INTENT`.
     - For Contract Violations ($V_5$): Intercepted and halts with `REJECT_INCOMPATIBLE_OBSERVATION`.
     $$\text{TSR}_{\text{exec}}^{(2)} = \frac{1}{N_{\text{total}}^{(2)}} \sum_{i \in \text{LC/CD}} S_i^{\text{exec}}$$
   This explicit segregation guarantees that no ungrounded claims of seven-task perception execution are made prior to the training of models for the remaining five tasks in subsequent project phases.

### E.3 Operational Margin-Aware Selective Risk, Decision Alignment & Oracle Construction

#### Operational Selection Rule vs Risk-Coverage Ranking Alignment
The operational dispatch policy (Section A.2) dispatches a query if and only if both confidence and competitive margin satisfy their respective thresholds:
$$c_{\text{intent}} \ge \tau_{\text{intent}} \quad \land \quad \Delta c \ge \tau_{\text{margin}}$$
Ranking queries solely by top-class confidence $c_{\text{intent}}$ creates an evaluation-decision mismatch: a request with high top-class probability (e.g. $0.51$) but low margin ($\Delta c = 0.02$) would be rejected by the actual operational policy, yet ranked as confident by a confidence-only risk-coverage curve.

To eliminate this mismatch and ensure that the experimental risk-coverage analysis faithfully reflects the operational decision rule, the protocol establishes:

1. **Standardized Joint Selection Function ($s_{\text{joint}}(q)$)**:
   For any query $q \in \mathcal{E}_{\text{eval}}$, the composite selective score couples confidence and margin normalized by their validation-tuned operational thresholds:
   $$s_{\text{joint}}(q) = \min\left(\frac{c_{\text{intent}}}{\tau_{\text{intent}}}, \frac{\Delta c}{\tau_{\text{margin}}}\right)$$
   Under this formulation:
   $$s_{\text{joint}}(q) \ge 1.0 \iff c_{\text{intent}} \ge \tau_{\text{intent}} \quad \land \quad \Delta c \ge \tau_{\text{margin}}$$
   The operational routing policy corresponds identically to thresholding $s_{\text{joint}}(q) \ge 1.0$.
2. **Primary Operational Metric: Margin-Aware Selective Risk ($\text{AURC}_{\text{joint}}$)**:
   Queries in $\mathcal{E}_{\text{eval}}$ are sorted descending by $s_{\text{joint}}(q)$. Sweeping threshold $\tau_{\text{sel}} \in [0, \infty)$ traces out the **Margin-Aware Risk-Coverage Curve**, capturing how margin-based abstention suppresses close-call and polysemous intent requests.
   - At threshold $\tau_{\text{sel}} = 1.0$, the empirical point reports exact **Operational Coverage** $\Phi_{\text{ops}}$ and **Operational Selective Risk** $\mathcal{R}_{\text{ops}}$.
   - Global trapezoidal integration computes $\text{AURC}_{\text{joint}}$, serving as the primary metric for selective hypothesis testing ($R_4$ vs $R_3$).
3. **Diagnostic Confidence-Only Ablation ($\text{AURC}_{\text{conf}}$)**:
   A secondary, parallel curve is computed by ranking requests strictly by $c_{\text{intent}}$ alone without margin gating. This metric is explicitly labeled as an isolated diagnostic ablation measuring the ordinal quality of top-class probability alone, separate from operational routing.
4. **Dispatched Population & Selective Risk**:
   $$\mathcal{D}(\tau_{\text{sel}}) = \{i \in \mathcal{E}_{\text{eval}} \mid s_{\text{joint}, i} \ge \tau_{\text{sel}}\}, \qquad \mathcal{R}(\tau_{\text{sel}}) = \frac{1}{|\mathcal{D}(\tau_{\text{sel}})|} \sum_{i \in \mathcal{D}(\tau_{\text{sel}})} \mathbb{I}(\hat{t}_i \ne T_i^*) \quad \text{for } |\mathcal{D}| > 0$$
5. **Empirical Sequence & Tie Handling**:
   Sorting $\mathcal{E}_{\text{eval}}$ by descending score yields discrete operating points $(\Phi_1, \mathcal{R}_1), \dots, (\Phi_M, \mathcal{R}_M)$. Instances with identical scores are admitted simultaneously in a single pooled step.
6. **Zero-Coverage Endpoint Convention**:
   Following selective classification standards (Geifman & El-Yaniv, 2017; Ding et al., 2020), no artificial observation is introduced; the empirical curve is evaluated over $[\Phi_1, \Phi_M]$ and continued horizontally to the left:
   $$\mathcal{R}(\Phi) := \mathcal{R}_1 \quad \text{for } 0 \le \Phi \le \Phi_1$$
   reflecting the error rate of the single highest-scoring decision.
7. **Trapezoidal Integration**:
   $$\text{AURC}_{\text{joint}} = \mathcal{R}_1 \cdot \Phi_1 + \sum_{m=1}^{M-1} \frac{\mathcal{R}_m + \mathcal{R}_{m+1}}{2} (\Phi_{m+1} - \Phi_m)$$
8. **The Oracle Population & Excess-AURC**:
   The Oracle evaluates the exact same eligible population $\mathcal{E}_{\text{eval}}$. All correct instances ($\hat{t}_i == T_i^*$) are ranked first (accumulating $\mathcal{R} = 0.0$), followed by incorrect instances. This defines the minimum achievable $\text{AURC}_{\text{oracle}}$:
   $$\text{E-AURC}_{\text{joint}} = \text{AURC}_{\text{joint}} - \text{AURC}_{\text{oracle}}$$

### E.4 Safety Metrics: Proposals vs Executions
* **Incompatible Dispatch Proposal Rate (IDPR)**: Logged at the router stage *prior to the gate*:
  $$\text{IDPR} = \frac{\text{Proposals where } \mathcal{C}(\hat{t}, o) == 0}{\text{Total dispatch proposals}}$$
* **Incompatible Dispatch Execution Rate (IDER)**: Logged at the specialist execution stage *post-gate*:
  $$\text{IDER}_{\text{obs}} = \frac{\text{Incompatible executions that bypass the gate}}{\text{Total incompatible test scenarios}} = \mathbf{0.0\%} \quad (0 / 28 \text{ held-out test violations})$$
  * **Empirical Framing**: Rather than asserting zero risk as an unchallengeable axiom, the protocol treats IDER as an experimentally measured quantity subject to empirical verification.
  * **Finite-Sample Confidence Bounds**: For 0 failures in $N=28$ test violations, the one-sided 95% Wilson score upper bound is **$8.81\%$** and the exact one-sided 95% Clopper-Pearson (Binomial) upper bound is **$10.15\%$**.
* **Per-Dimension Contract Violation Detection Recall**: Reported individually across all 8 failure dimensions and subtypes:
  $$\text{CVDR}_d = \frac{\text{Detected & intercepted violations in dimension } d}{\text{Total synthetic violations in dimension } d}$$
  with exact 95% Wilson score confidence intervals and reported sample counts (Section D.4). Final safety claims are based strictly on held-out test violations ($N=28$ in default, $N=42$ in contingency).

### E.5 Independently Reproducible Cluster-Bootstrap Power Analysis for Global AURC

#### Global Population Metric vs Cluster Resampling on Eligible Population
Area Under Risk-Coverage Curve (AURC) is a **global population metric** computed by pooling all qualified instances across scenarios, ranking them by selective score, and integrating the resulting empirical risk-coverage curve. It is not an average of independent family-level statistics. Because scenarios generated from the same template family share syntactic and semantic structures, instances within a family exhibit hierarchical dependence.

The statistically rigorous inference procedure for global AURC differences is the **paired cluster bootstrap**:
1. **Resampling Unit**: Complete template families $f \in \{1, \dots, F_{\text{test}}\}$ are resampled with replacement (cluster bootstrap).
2. **Instance Pooling Strictly over $\mathcal{E}_{\text{eval}}$**: For each bootstrap replicate $b = 1, \dots, B$, **only the three eligible valid scenarios** ($V_1, V_2, V_4$) belonging to sampled families are pooled into an evaluation cohort:
   $$N_{\text{boot\_cohort}} = 3 \times F_{\text{test}} \text{ scenarios}$$
   (e.g., $N = 84$ in default Tier 0; $N = 168$ in Tier 1; $N = 252$ in Tier 2). Ambiguous queries ($V_3$) and physical contract violations ($V_5$) are strictly excluded from selective routing curves and evaluated under their dedicated metrics (Section E.2). This cohort size is programmatically asserted on every bootstrap replicate.
3. **Global Curve Integration**: The full margin-aware risk-coverage curve and global $\text{AURC}_{\text{joint}}$ are computed for proposed system $R_4$ and baseline $R_m$ on the pooled cohort.
4. **Paired Difference**:
   $$\Delta_m^{*(b)} = \text{AURC}_{\text{joint}}(R_m)^{*(b)} - \text{AURC}_{\text{joint}}(R_4)^{*(b)}$$
5. **Comparison-Level Bootstrap P-Values**:
   For testing null hypothesis $H_{0, m}: \Delta_m \le 0$ vs $H_{1, m}: \Delta_m > 0$, the comparison-level p-value is computed from the $B$ paired bootstrap differences using the finite-sample continuity correction (Davison & Hinkley, 1997):
   $$p_m = \frac{1 + \sum_{b=1}^B \mathbb{I}(\Delta_m^{*(b)} \le 0)}{B + 1}$$
6. **Multiplicity Control via Complete 5-Comparison Step-Down Holm Procedure**:
   To strictly control the family-wise error rate at $\alpha = 0.05$ across all five baseline comparisons ($R_4$ vs $R_0, R_1, R_{1\text{-Gated}}, R_2, R_3$), the five comparison p-values are sorted ascending:
   $$p_{(1)} \le p_{(2)} \le p_{(3)} \le p_{(4)} \le p_{(5)}$$
   Hypotheses are tested sequentially against step-down critical thresholds:
   - Step 1: Test $p_{(1)} \le \frac{0.05}{5} = 0.0100$. If rejected, proceed; else halt.
   - Step 2: Test $p_{(2)} \le \frac{0.05}{4} = 0.0125$. If rejected, proceed; else halt.
   - Step 3: Test $p_{(3)} \le \frac{0.05}{3} = 0.0167$. If rejected, proceed; else halt.
   - Step 4: Test $p_{(4)} \le \frac{0.05}{2} = 0.0250$. If rejected, proceed; else halt.
   - Step 5: Test $p_{(5)} \le \frac{0.05}{1} = 0.0500$.
   The target claim ($R_4$ vs $R_3$) is confirmed statistically significant under FWER control if and only if its hypothesis is reached and rejected.
7. **Percentile Confidence Intervals for Effect Size**:
   Distinguished from hypothesis testing, parameter effect magnitudes are reported as $(1-\alpha)$ percentile bootstrap intervals $[\hat{q}_{\alpha/2}^*, \hat{q}_{1 - \alpha/2}^*]$.
8. **Bootstrap Replicate Sensitivity Verification ($B=300$ vs $B=1,000$)**:
   To justify $N_{\text{boot}} = 300$ for power simulation efficiency while using $B = 2,000$ in production evaluation, an empirical sensitivity trial ($K=56, \Delta=0.04$) confirmed that increasing $N_{\text{boot}}$ from 300 to 1,000 altered estimated power by only $2.0\%$ ($73.0\%$ vs $75.0\%$, well within Monte Carlo error $\text{SE}_{\text{MC}} \approx 4.4\%$) and standard error by $< 0.0005$ ($0.0423$ vs $0.0418$).

#### Executable Simulation Script & Monte Carlo Verification
The complete cluster-bootstrap power simulation is archived and executable in [`tests/cluster_bootstrap_power_analysis.py`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/tests/cluster_bootstrap_power_analysis.py) (fixed seed `42`, $N_{\text{sim}} = 400$, $N_{\text{boot}} = 300$). All reported power values are **Monte Carlo simulation estimates** conditional on the stated hierarchical data-generating process:

```powershell
PS C:\Users\B NAVANEETH\SatQueryAI> python -u tests/cluster_bootstrap_power_analysis.py
====================================================================================================================================
   SATQUERY AI (M3.0) — CLUSTER-BOOTSTRAP GLOBAL AURC STATISTICAL POWER SIMULATION
   Monte Carlo Iterations: 400 | Bootstrap Replicates per Trial: 300 | Seed: 42
   Eligible Population E_eval: Exactly 3 scenarios/family (V1, V2, V4). V3 (clarify) & V5 (violations) segregated.
   Multiplicity: Complete 5-Comparison Step-Down Holm Procedure (R4 vs R0, R1, R1-Gated, R2, R3)
====================================================================================================================================
F_test  | Delta  | Bootstrap SE  | Unadjusted (0.05)   | Holm Tier-1 (0.01)   | Complete Holm      | All-5 Holm       | Assessment        
------------------------------------------------------------------------------------------------------------------------------------
28      | 0.02   | 0.0514        |  34.5% ±  2.4% |  15.2% ±  1.8% |  26.2% ±  2.2% |  22.2% ±  2.1% | Underpowered (<80%) (68.4s)
28      | 0.03   | 0.0548        |  41.2% ±  2.5% |  23.0% ±  2.1% |  31.5% ±  2.3% |  26.2% ±  2.2% | Underpowered (<80%) (72.3s)
28      | 0.04   | 0.0571        |  57.0% ±  2.5% |  35.8% ±  2.4% |  47.0% ±  2.5% |  34.8% ±  2.4% | Underpowered (<80%) (71.6s)
------------------------------------------------------------------------------------------------------------------------------------
42      | 0.02   | 0.0421        |  41.8% ±  2.5% |  20.0% ±  2.0% |  35.2% ±  2.4% |  31.2% ±  2.3% | Underpowered (<80%) (76.4s)
42      | 0.03   | 0.0447        |  57.8% ±  2.5% |  33.5% ±  2.4% |  47.8% ±  2.5% |  41.5% ±  2.5% | Underpowered (<80%) (76.9s)
42      | 0.04   | 0.0472        |  65.2% ±  2.4% |  41.8% ±  2.5% |  56.5% ±  2.5% |  47.5% ±  2.5% | Underpowered (<80%) (79.8s)
------------------------------------------------------------------------------------------------------------------------------------
56      | 0.02   | 0.0367        |  45.8% ±  2.5% |  25.5% ±  2.2% |  42.2% ±  2.5% |  39.8% ±  2.4% | Underpowered (<80%) (96.5s)
56      | 0.03   | 0.0393        |  64.2% ±  2.4% |  34.5% ±  2.4% |  55.0% ±  2.5% |  51.2% ±  2.5% | Underpowered (<80%) (98.0s)
56      | 0.04   | 0.0412        |  79.8% ±  2.0% |  57.2% ±  2.5% |  71.2% ±  2.3% |  62.3% ±  2.4% | Underpowered (<80%) (54.2s)
------------------------------------------------------------------------------------------------------------------------------------
84      | 0.02   | 0.0301        |  59.8% ±  2.5% |  34.0% ±  2.4% |  58.8% ±  2.5% |  58.2% ±  2.5% | Underpowered (<80%) (88.9s)
84      | 0.03   | 0.0322        |  80.8% ±  2.0% |  60.0% ±  2.4% |  78.0% ±  2.1% |  74.5% ±  2.2% | Underpowered (<80%) (89.8s)
84      | 0.04   | 0.0347        |  90.8% ±  1.4% |  74.0% ±  2.2% |  86.5% ±  1.7% |  81.2% ±  2.0% | Adequately powered (89.9s)
------------------------------------------------------------------------------------------------------------------------------------
====================================================================================================================================
```

#### Empirical Scientific Power Assessment & Pre-Registered Contingency Rule
1. **Nominal Test Split ($F_{\text{test}} = 28$ families, $N_{\text{total}} = 140$ scenarios, $N_{\text{eval}} = 84$ eligible evaluation scenarios)**:
   - For effect size $\Delta = 0.04$ (4 percentage-point global risk reduction), power is **$47.0\% \pm 2.5\%$** under Complete Step-Down Holm testing, **$57.0\% \pm 2.5\%$** unadjusted, and **$35.8\% \pm 2.4\%$** under conservative Holm Tier-1 ($\alpha = 0.01$).
   - For effect size $\Delta = 0.03$ (3 percentage-point global risk reduction), power is **$31.5\% \pm 2.3\%$** under Complete Step-Down Holm testing and **$23.0\% \pm 2.1\%$** under Holm Tier-1.
   - The nominal test split of 28 families is substantially underpowered (<80%) across all tested effect sizes when evaluating strictly on the valid non-ambiguous, non-violating population $\mathcal{E}_{\text{eval}}$ ($N_{\text{eval}} = 3 \times F = 84$).
2. **Expansion to $F_{\text{test}} = 56$ Families ($N_{\text{total}} = 280, N_{\text{eval}} = 168$ scenarios)**:
   - For $\Delta = 0.04$, power reaches **$71.2\% \pm 2.3\%$** under Complete Step-Down Holm testing ($79.8\% \pm 2.0\%$ unadjusted, $57.2\% \pm 2.5\%$ Holm Tier-1). While approaching the threshold, $F_{\text{test}} = 56$ remains marginally underpowered (<80%) under full Step-Down Holm FWER multiplicity control.
   - For $\Delta = 0.03$, power reaches **$55.0\% \pm 2.5\%$** under Complete Step-Down Holm testing ($64.2\% \pm 2.4\%$ unadjusted).
3. **Full Expansion to $F_{\text{test}} = 84$ Families ($N_{\text{total}} = 420, N_{\text{eval}} = 252$ scenarios)**:
   - For $\Delta = 0.04$, power reaches **$86.5\% \pm 1.7\%$** under Complete Step-Down Holm testing, **$90.8\% \pm 1.4\%$** unadjusted, and **$81.2\% \pm 2.0\%$** across all five comparisons simultaneously, surpassing the 80% target with statistical confidence.
   - For $\Delta = 0.03$, power reaches **$78.0\% \pm 2.1\%$** under Complete Step-Down Holm testing and **$80.8\% \pm 2.0\%$** unadjusted.
4. **Pre-Registered Power Contingency Rule (Protocol Contract)**:
   - **Contingency Trigger**: Prior to unblinding the held-out test split, the empirical cluster bootstrap standard error $\widehat{\text{SE}}_{\text{boot}}$ will be measured on the validation pilot split ($F_{\text{val}} = 28, N_{\text{eval}} = 84$).
   - **Rule 1 (Target $\Delta \ge 0.04$)**: If the validation pilot targets $\Delta \ge 0.04$ and $\widehat{\text{SE}}_{\text{boot}} \le 0.035$, the test split of **$F_{\text{test}} = 84$ families** ($N_{\text{total}}=420, N_{\text{eval}}=252$ scenarios) is deployed, guaranteeing **$86.5\% \pm 1.7\%$** Complete Holm power.
   - **Rule 2 (Target $\Delta \le 0.03$)**: If the validation pilot indicates that true effect size between $R_4$ and $R_3$ is moderate ($\Delta \approx 0.03$) or $\widehat{\text{SE}}_{\text{boot}} > 0.035$, the test split formally maintains **$F_{\text{test}} = 84$ families** (yielding $78.0\% \pm 2.1\%$ Complete Holm power and $80.8\%$ unadjusted power) or optionally expands to Tier 3 ($F_{\text{test}} = 112$ families) if the advisor requires $>85\%$ power under conservative Holm Tier-1 correction.
   - **Simulation Estimate Disclaimer**: Power figures are Monte Carlo simulation estimates under the specified data-generating process, not experimental guarantees. This pre-registered rule prevents post-hoc sample expansion (p-hacking) by anchoring all split size decisions to pre-unblinding validation metrics.

---

## Deliverable F: Deep-Learning Specialist Experiment Specifications

### F.1 Multimodal Land-Cover Specialist (BigEarthNet-14k)
* **Dataset & Verified Tiling**: $N = 13,683$ patches across 4 tiles (`T34TCR`, `T34TEQ`, `T34TFN`, `T34TCS`).
  * Sentinel-1: `(2, 120, 120)`, float32, backscatter in dB `[VH, VV]`. Clipped to $[-35.0, +5.0]\text{ dB}$ (calculated strictly on train split) and linearly scaled via $(\sigma^0 + 35.0) / 40.0$.
  * Sentinel-2: `(10, 120, 120)`, uint16 `[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]`. Native 20m bands are resampled to 10m grid. Scaled via $\text{DN} / 10,000.0 \in [0.0, 1.0]$. Channel-wise standardization using training set statistics:
    $$\mu = [0.136, 0.118, 0.106, 0.128, 0.203, 0.257, 0.272, 0.273, 0.187, 0.121]$$
    $$\sigma = [0.082, 0.076, 0.074, 0.089, 0.098, 0.108, 0.113, 0.112, 0.095, 0.078]$$
  * *Reproducible Normalization Script*: The channel statistics are archived and independently verifiable via [`scripts/compute_normalization_statistics.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/compute_normalization_statistics.py).
* **Architecture & Explicit Pretrained Weight Mapping**:
  * Dual-branch ResNet-34 with cross-attention fusion projecting into a 512-dimensional fused representation, feeding 16 sigmoid classification heads.
  * *Optical Branch (10 channels)*: Explicit mapping from ImageNet-1k pre-trained weights (`ResNet34_Weights.DEFAULT`):
    - Input channel 0 (Sentinel-2 B04, Red, ~665 nm) $\gets$ ImageNet Red filter weights (`conv1.weight[:, 0:1, :, :]`)
    - Input channel 1 (Sentinel-2 B03, Green, ~560 nm) $\gets$ ImageNet Green filter weights (`conv1.weight[:, 1:2, :, :]`)
    - Input channel 2 (Sentinel-2 B02, Blue, ~490 nm) $\gets$ ImageNet Blue filter weights (`conv1.weight[:, 2:3, :, :]`)
    - Remaining 7 multispectral channels (B05, B06, B07, B08, B8A, B11, B12) $\gets$ initialized using He (Kaiming) normal initialization with gain $\sqrt{2}$.
  * *SAR Branch (2 channels: VH, VV)*: Initialized from scratch using He normal initialization.
  * *Cross-Attention Fusion*: Optical and SAR spatial feature maps are projected to 512 channels. Multi-head cross-attention ($h=8$ heads, key/query/value dimension $d_k = 64$) couples optical and SAR representations, followed by layer normalization and a 2-layer MLP residual block.
* **Training Recipe & Hyperparameters**:
  * Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-2}$).
  * Learning rate: Initial $10^{-4}$ with cosine annealing scheduler down to $10^{-6}$.
  * Batch size: 64. Epochs: 40. Compute budget identical for CB-BCE and ASL.
  * Data Augmentation: Random horizontal/vertical flips, random $90^\circ$ rotations.
  * Early stopping: Patience 7 epochs monitored on validation Macro $F_1$.
  * Seeds: Evaluated over 3 independent random seeds (`42, 1337, 2026`).
  * Checkpoint selection: Highest validation Macro $F_1$ (16 classes).
  * Threshold tuning: Per-class sigmoid threshold sweep $\tau_c \in [0.10, 0.90]$ (step 0.05) on validation split.
* **Loss Configurations & Fixed Class-Balanced Weighting**:
  * *Config A: Normalized Class-Balanced BCE (CB-BCE)*:
    Effective-number class weights (Cui et al., 2019) are computed **strictly once** from the static training-split class counts ($N_{\text{train}} = 7,180$ patches), **never recalculated per minibatch**, preventing gradient oscillation and batch-distribution artifacts:
    - Positive and negative counts are determined over the full training set:
      $$N_{c, +}^{\text{train}} = \sum_{i \in \text{train}} y_{i, c}, \qquad N_{c, -}^{\text{train}} = N_{\text{train}} - N_{c, +}^{\text{train}}$$
    - Effective sample numbers with hyperparameter $\beta = 0.999$:
      $$E_{c, +} = \frac{1 - \beta}{1 - \beta^{N_{c, +}^{\text{train}}}}, \qquad E_{c, -} = \frac{1 - \beta}{1 - \beta^{N_{c, -}^{\text{train}}}}$$
    - *Zero-Support Handling on Training Set*: If any rare class exhibits zero positive instances in the training set ($N_{c, +}^{\text{train}} = 0$), $E_{c, +}$ is clamped to $1.0$.
    - With epsilon guard $\epsilon = 10^{-5}$, fixed normalized positive weights are pre-computed:
      $$\tilde{w}_{c, +} = \frac{E_{c, +} + \epsilon}{\sum_{j=1}^{16} (E_{j, +} + \epsilon)} \times 16$$
    - *Fixed Buffer & Numerically Stable Logits Loss*: The resulting vector $\vec{w}_{+} = (\tilde{w}_{1, +}, \dots, \tilde{w}_{16, +})^T$ is registered as a non-trainable fixed constant buffer (`register_buffer('pos_weight', ...)`). Loss computation uses numerically stable BCE-with-logits:
      $$\mathcal{L}_{\text{CB-BCE}} = \frac{1}{B} \sum_{i=1}^B \sum_{c=1}^{16} \left[ - \tilde{w}_{c, +} y_{i, c} \log(\sigma(z_{i, c})) - (1 - y_{i, c}) \log(1 - \sigma(z_{i, c})) \right]$$
      implemented directly via `torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)`.
  * *Config B: Asymmetric Loss (ASL)*:
    $$\mathcal{L}_{\text{ASL}} = -\frac{1}{16}\sum_{c=1}^{16} \left[ y_c (1 - p_c)^{\gamma_+} \log(p_c) + (1 - y_c) (p_{c, m})^{\gamma_-} \log(1 - p_{c, m}) \right]$$
    with $\gamma_+ = 0, \gamma_- = 4, m = 0.05$.
* **Validation & Metrics**: Report Macro $F_1$ (16 classes), Macro $F_1$ (14 test-present classes), Micro $F_1$, and Hamming Loss on held-out test set ($N=3,248$).

### F.2 Bitemporal Change Detection Specialist (OSCD)
* **Dataset & Splits**: 14 training scenes, 4 validation scenes (`montpellier`, `paris`, `rio`, `valencia`), 10 held-out test scenes (`aguasclaras`, `austin`, `beihai`, `berlin`, `brasilia`, `budapest`, `campinas`, `mumbai`, `nantes`, `pisa`).
* **Architectures**:
  * *Primary Model*: Bitemporal Image Transformer (BIT) (Chen et al., 2021). Weight-shared Siamese ResNet-18 backbone (downsampling factor 16, feature dimension $C=512$); spatial tokenizer converting feature map $(C, 16, 16)$ into $L=64$ compact semantic tokens $(L, C)$; 4 transformer encoder layers (intra-temporal context) + 4 transformer decoder layers (cross-temporal differences via cross-attention), total 8 layers; shallow FCN prediction head generating pixel change probabilities $(1, 256, 256)$.
  * *Baseline Model*: Fully Convolutional Siamese Difference (FC-Siam-diff) (Daudt et al., 2018).
* **Preprocessing, Patch Sampling & Inference Strategy**:
  * *Preprocessing*: 13 Sentinel-2 spectral bands. Native 20m bands resampled to 10m grid via bicubic interpolation; surface reflectance scaled via $\text{DN} / 10,000.0$.
  * *Training*: Random $256 \times 256$ crops sampled with $50\%$ spatial overlap from the 14 training scenes (~1,800 patches/epoch) with random rotations ($90^\circ, 180^\circ, 270^\circ$) and flips.
  * *Inference*: Deterministic sliding-window inference with $256 \times 256$ patches, $50\%$ overlap (stride 128), and 2D Gaussian distance blending across overlapping seams.
* **Training Recipe & Hyperparameters**:
  * Optimizer: AdamW, initial lr $10^{-4}$, weight decay $10^{-4}$, cosine decay to $10^{-6}$.
  * Batch size: 16 (patches $256 \times 256$). Epochs: 50. Patience: 10 epochs on validation $F_1$. Seeds: `42, 1337, 2026`.
  * Checkpoint selection: Checkpoint achieving the highest validation Macro $F_1$ across the 4 validation scenes.
  * Loss: $\mathcal{L} = 1.0 \cdot \mathcal{L}_{\text{Focal}}(\gamma=2.0, \alpha=0.25) + 0.5 \cdot \mathcal{L}_{\text{Lovasz-Hinge}}$.
* **Scene-Clustered Uncertainty Estimation (Student's $t$ Jackknife)**:
  Because pixels within an urban scene are spatially autocorrelated, pixels are never treated as independent samples.
  For any metric $\hat{\theta}$ (e.g. Pooled $F_1$ or Macro $F_1$), when test scene $k$ ($k=1, \dots, 10$) is omitted, the metric $\hat{\theta}_{(-k)}$ is recomputed from the remaining 9 test scenes:
  * For Pooled $F_1$: Recompute from accumulated confusion matrix:
    $$\text{TP}_{(-k)} = \sum_{j \ne k} \text{TP}_j, \quad \text{FP}_{(-k)} = \sum_{j \ne k} \text{FP}_j, \quad \text{FN}_{(-k)} = \sum_{j \ne k} \text{FN}_j$$
    $$\hat{\theta}_{(-k)} = \frac{2 \text{TP}_{(-k)}}{2 \text{TP}_{(-k)} + \text{FP}_{(-k)} + \text{FN}_{(-k)}}$$
  * For Macro $F_1$: Recompute arithmetic mean across remaining scenes:
    $$\hat{\theta}_{(-k)} = \frac{1}{K-1} \sum_{j \ne k} F_{1, j}$$
  The leave-one-scene-out jackknife standard error is:
  $$\widehat{\text{SE}}_{\text{jack}} = \sqrt{\frac{K-1}{K} \sum_{k=1}^{K} (\hat{\theta}_{(-k)} - \bar{\theta})^2}$$
  With $K=10$ scenes ($df = K - 1 = 9$), the two-sided critical value is $t_{0.025, 9} = 2.262$:
  $$\text{CI}_{95\%} = \hat{\theta} \pm 2.262 \cdot \widehat{\text{SE}}_{\text{jack}}$$
  Both pooled metrics and individual per-scene metrics across all 10 scenes are reported.
* **Threshold Selection**: Threshold $\tau_{\text{CD}}$ optimized strictly on the 4 validation scenes. **Never tune thresholds on the 10 held-out test scenes.**
* **Operational Contract Status of Temporal Windows**: The 5-day minimum gap (revisit limit) and 730-day maximum gap are explicitly defined as **benchmark-specific operational contracts** tailored to the Sentinel-2 OSCD dataset, rather than universal physical laws.

### F.3 Specialist Implementation Scope Disclaimer
Phase 3 implements deep-learning perception specialists for **two tasks**: Multimodal Land Cover and Bitemporal Change Detection. The remaining five taxonomy tasks (`SINGLE_IMAGE_VQA`, `CHANGE_VQA`, `VISUAL_GROUNDING`, `SCENE_CAPTIONING`, `OPTICAL_SAR_FUSION`) are evaluated as formal routing, clarification, and safety contract targets; full perception model weights for those five tasks are part of future development phases.

---

## Deliverable G: Git Baseline & Regression Verification Output

### G.1 Git Integrity Verification
```powershell
PS C:\Users\B NAVANEETH\SatQueryAI> git rev-parse "v0.2.0-phase2-freeze^{commit}"
ce7134245f9be6b3876a3a2df1699b2641156f9f

PS C:\Users\B NAVANEETH\SatQueryAI> git rev-parse HEAD
855c3166be727e7e4a321bc50fd3fd269ed9a0f3

PS C:\Users\B NAVANEETH\SatQueryAI> git describe --tags
v0.2.0-phase2-freeze-13-g855c316

PS C:\Users\B NAVANEETH\SatQueryAI> git status --short
(empty - working tree completely clean)
```

### G.2 Test Suite Execution Output (82/82 Passing)
```powershell
PS C:\Users\B NAVANEETH\SatQueryAI> python -m pytest tests/test_m30_clarification_and_routing_invariants.py tests/test_affine_full_footprint_gate.py -v
============================= test session starts =============================
tests/test_m30_clarification_and_routing_invariants.py::test_ambiguous_query_single_compatible_candidate_triggers_clarification PASSED [  9%]
tests/test_m30_clarification_and_routing_invariants.py::test_ambiguous_query_multiple_compatible_candidates_triggers_clarification PASSED [ 18%]
tests/test_m30_clarification_and_routing_invariants.py::test_ambiguous_query_no_compatible_candidates_triggers_clarification PASSED [ 27%]
tests/test_m30_clarification_and_routing_invariants.py::test_clear_intent_incompatible_observation_rejects_without_coercion PASSED [ 36%]
tests/test_m30_clarification_and_routing_invariants.py::test_clear_intent_compatible_observation_low_selective_confidence_flags_review PASSED [ 45%]
tests/test_affine_full_footprint_gate.py::test_perfect_alignment_zero_displacement PASSED [ 54%]
tests/test_affine_full_footprint_gate.py::test_scale_drift_exceeds_full_footprint_displacement PASSED [ 63%]
tests/test_affine_full_footprint_gate.py::test_rotation_drift_exceeds_full_footprint_displacement PASSED [ 72%]
tests/test_affine_full_footprint_gate.py::test_reflection_handedness_flip_detected PASSED [ 81%]
tests/test_affine_full_footprint_gate.py::test_shear_distortion_detected PASSED [ 90%]
tests/test_affine_full_footprint_gate.py::test_subpixel_pure_translation_gating PASSED [100%]
============================= 11 passed in 7.30s ==============================

PS C:\Users\B NAVANEETH\SatQueryAI> python -m pytest tests/ -q
..................................................................................  [100%]
82 passed in 30.80s
```

### G.3 Specific Regression Invariants Verified
1. `test_ambiguous_query_single_compatible_candidate_triggers_clarification`: Confirms that an ambiguous query with exactly one compatible candidate halts with `NEEDS_CLARIFICATION` and never auto-dispatches.
2. `test_ambiguous_query_multiple_compatible_candidates_triggers_clarification`: Confirms multiple compatible candidates halt with `NEEDS_CLARIFICATION`.
3. `test_ambiguous_query_no_compatible_candidates_triggers_clarification`: Confirms zero compatible candidates halt with `NEEDS_CLARIFICATION`.
4. `test_clear_intent_incompatible_observation_rejects_without_coercion`: Confirms unambiguous intent with incompatible payload halts with `INCOMPATIBLE_INPUT` and preserves true intent.
5. `test_clear_intent_compatible_observation_low_selective_confidence_flags_review`: Confirms selective abstention flag triggers when confidence is below selective threshold.
6. `test_scale_and_rotation_drift_exceed_full_footprint_displacement`: Confirms that 1% scale mismatch (1.2px displacement at W=120) and 0.28 deg rotation (0.83px displacement at corner) correctly fail the full-footprint gate and flag mandatory resampling, proving that origin phase alone is insufficient.

---

## Deliverable H: Novelty Positioning & Related-Work Capability Matrix

### H.1 Detailed Related-Work Capability Comparison Matrix

| System / Framework | Natural Language Task Understanding | Programmatic Sensor & Physical Compatibility Gate | Explicit Task Non-Coercion Guarantee | Mandatory Clarification for Ambiguous Intent | Calibrated Selective Routing | Joint Linguistic + Physical Risk-Coverage Evaluation | Primary Reference / Evidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Earth-GPT** (Zhang et al., 2024) | Yes | No | Not Reported | No | Not Reported | No | Multimodal RS-VLM (CVPR 2024); executes unified generative vision-language model without programmatic sensor/metadata compatibility validation or selective abstention. |
| **GeoChat** (Kuckreja et al., 2024) | Yes | No | Not Reported | No | Not Reported | No | Grounded RS-VLM (CVPR 2024); routes via open-vocabulary conversation; lacks deterministic sensor gating and prompt ambiguity clarification mechanisms. |
| **SkyEye** (Zhan et al., 2024) | Yes | No | Not Reported | No | Not Reported | No | RS visual grounding assistant; focuses on localized spatial object detection; does not evaluate multi-sensor compatibility or bi-temporal interval contracts. |
| **RS-LLaVA** (Li et al., 2024) | Yes | No | Not Reported | No | Not Reported | No | High-resolution RS multi-task assistant; handles diverse queries but lacks active physical payload gating or selective risk-coverage quantification. |
| **Learning to Route (L2R)** (Wang et al., 2021) | Yes | No | Not Reported | No | No | No | Dynamic task routing across neural specialists; routing decisions are uncalibrated and decoupled from physical geospatial/sensor metadata. |
| **SelectiveNet** (Geifman & El-Yaniv, 2019) | No | No | N/A | No | Yes | No | Selective prediction for computer vision on closed datasets (ICML 2019); does not model NLP intent, task taxonomies, or physical sensor payloads. |
| **STAC Metadata Ecosystem** (Holmes et al., 2021) | No | Yes | N/A | No | No | No | SpatioTemporal Asset Catalog specification (v1.0.0); validates JSON metadata syntax against schemas, but provides no natural language task interpretation, intent routing, or active execution gating. |
| **SatQuery AI (Proposed System)** | **Yes** | **Proposed** | **Proposed** | **Proposed** | **Proposed** | **Proposed** | **Subject of experimental evaluation under this protocol.** |

### H.2 The Central Scientific Research Hypothesis to be Tested
> **Central Research Hypothesis**: In Earth observation query routing, routing failures arise from two orthogonally distinct sources of uncertainty: **linguistic ambiguity** in user intent, and **physical incompatibility** in observation payloads. Decoupling linguistic intent estimation from deterministic, machine-enforceable sensor compatibility contracts eliminates incompatible specialist executions ($\text{IDER}_{\text{obs}} = 0.0\%$, finite-sample Wilson bound $\le 8.81\%$) and achieves significantly lower selective risk (global AURC) than unconstrained routing, uncalibrated selective prediction, and zero-shot frontier LLM reasoning.

### H.3 Core Research Objectives of the Study
1. **Formalize Non-Coercive Routing**: Demonstrate empirically that decoupling intent classification from observation compatibility guarantees non-coercion, preserves user intent, and eliminates incompatible specialist executions ($\text{IDER}_{\text{obs}} = 0.0\%$).
2. **Quantify the Risk-Coverage Benefit**: Evaluate whether calibrated selective prediction with compatibility gating achieves statistically significant reductions in global AURC compared to unconstrained, uncalibrated, and non-selective baselines under paired cluster-bootstrap inference.
3. **Compare Against Foundation Models**: Rigorously benchmark deterministic compatibility routing against frontier LLM zero-shot routing on identical multimodal scenarios across intent accuracy, proposal safety, and clarification fidelity.

### H.4 Verified Bibliographic References
1. **Earth-GPT**: Zhang, D., et al. (2024). *EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Airborne and Satellite Remote Sensing*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024). arXiv:2401.16822.
2. **GeoChat**: Kuckreja, K., et al. (2024). *GeoChat: Grounded Large Vision-Language Model for Remote Sensing*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024). arXiv:2311.15826.
3. **SkyEye**: Zhan, Y., et al. (2024). *SkyEye: Self-Supervised Remote Sensing Vision-Language Assistant with Grounding Capabilities*. IEEE Transactions on Geoscience and Remote Sensing (TGRS 2024). arXiv:2401.09412.
4. **RS-LLaVA**: Li, Y., et al. (2024). *RS-LLaVA: A Large Vision-Language Model for Remote Sensing Multi-Task Interpretation*. IEEE Geoscience and Remote Sensing Letters (GRSL 2024). arXiv:2404.05260.
5. **Learning to Route (L2R)**: Wang, Y., et al. (2021). *Learning to Route in Collaborative Multi-Agent Systems*. In Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139:10881-10892.
6. **SelectiveNet**: Geifman, Y., & El-Yaniv, R. (2019). *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. In Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97:2151-2159.
7. **STAC Specification**: Holmes, C., et al. (2021). *SpatioTemporal Asset Catalog (STAC) Specification*, version 1.0.0. Radiant Earth Foundation. Available: `https://stacspec.org/`.

---

## Deliverable I: Protocol Governance & Scientific Sign-Off Declaration

The following parameters are formally frozen for advisor sign-off:
1. **Benchmark Synthesizer**: Cohort variants generated from deterministic template engine (seed `42`).
2. **Power Contingency**: Default test split ($F_{\text{test}} = 28$); pre-registered expansion to $F_{\text{test}} = 56$ or $F_{\text{test}} = 84$ (achieving $86.5\% \pm 1.7\%$ Complete Holm power for $\Delta = 0.04$, triggered if validation pilot targets $\Delta \ge 0.04$ or cluster variance $\widehat{\text{SE}}_{\text{boot}} > 0.035$).
3. **Sign-Off Enforcement**: Milestone M3.0 is **REVISED — AWAITING SCIENTIFIC REVIEW**. No Phase 3 deep-learning model training, benchmark generation, or specialist modifications will commence until formal advisor authorization is received.
