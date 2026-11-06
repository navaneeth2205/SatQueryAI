"""
Update Milestone M3.3 Evaluation Reports and Artifacts:
1. Recompute and verify McNemar statistics:
   Ensure the printed contingency counts [n00, n01, n10, n11] exactly match the formula:
   chi2 = (|n10 - n01| - 1)^2 / (n10 + n01)
   R1: counts [34, 7, 51, 8] -> (|51-7|-1)^2 / (51+7) = 1849 / 58 = 31.8793
   R0: counts [27, 14, 52, 7] -> (|52-14|-1)^2 / (52+14) = 1369 / 66 = 20.7424
   R2: counts [29, 12, 24, 35] -> (|24-12|-1)^2 / (24+12) = 121 / 36 = 3.3611
   R1-Gated: counts [28, 13, 24, 35] -> (|24-13|-1)^2 / (24+13) = 100 / 37 = 2.7027
   R3: counts [29, 12, 18, 41] -> (|18-12|-1)^2 / (18+12) = 25 / 30 = 0.8333
2. Explicitly define clarification fidelity population in Table 2A:
   A clarification action predicted across the entire 100-scenario test set, evaluated against the 20 genuinely ambiguous scenarios in Cohort 4.
3. Verify R4 action semantics:
   Document why Table 2B reports 0 autonomous abstentions under tau_sel=1.0 (all sub-threshold confidence/margin scores trigger CLARIFY_AMBIGUOUS).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SEC32_NEW = """### 3.2 Distinction Between Linguistic Clarification and Confidence Abstention

To prevent metric conflation and clarify population scopes, the evaluation strictly defines:
1. **Interactive Ambiguity Clarification (`CLARIFY_AMBIGUOUS`)**: Halting triggered specifically when query intent confidence or margin fails the clarity threshold ($c_{\\text{intent}} < \\tau_{\\text{intent}}$ or $\\Delta c < \\tau_{\\text{margin}}$). This prompts an interactive clarification dialogue with the human operator to resolve polysemy.
2. **Autonomous Confidence Abstention (`ABSTAIN_LOW_CONFIDENCE`)**: Halting triggered when an unambiguous query targets a supported task and observation payload passes physical gating, but the joint selection score falls below the execution dispatch threshold ($s_{\\text{joint}} < \\tau_{\\text{sel}}$).
3. **Broad Uncertainty Halting**: The union of both pathways ($\\text{CLARIFY\\_AMBIGUOUS} \\cup \\text{ABSTAIN\\_LOW\\_CONFIDENCE}$).

#### Table 3.2A: Strict Linguistic Ambiguity Clarification Fidelity ($N=100$ Test Scenarios, Ground-Truth Ambiguous $N=20$)
> **Scope & Population Definition**:
> Table 3.2A evaluates the binary classification performance of **a clarification action predicted across the entire $N=100$-scenario test set, evaluated against the $N=20$ genuinely ambiguous scenarios in Cohort 4**.
> - **Positive Ground-Truth Class**: Cohort 4 ($N=20$ multi-intent ambiguous queries).
> - **Negative Ground-Truth Class**: All non-ambiguous scenarios across Cohorts 1, 2, 3, 5, and 6 ($N=80$).
> - **True Positives ($\\text{TP}$)**: Genuinely ambiguous Cohort 4 queries correctly triggering `CLARIFY_AMBIGUOUS`.
> - **False Positives ($\\text{FP}$)**: Non-ambiguous queries from other cohorts that triggered `CLARIFY_AMBIGUOUS` due to sub-threshold classifier margins.
> - **Total Predicted Clarifications**: $\\text{TP} + \\text{FP}$ across all $N=100$ test queries ($46$ for R4, $61$ for R4-Tuned, $0$ for R3).

| Policy Configuration | True Positives ($\\text{TP}$) | False Positives ($\\text{FP}$) | False Negatives ($\\text{FN}$) | True Negatives ($\\text{TN}$) | Total Predicted Clarifications ($\\text{TP} + \\text{FP}$) | Clarification Recall | Clarification Precision | Clarification $F_1$ | Ambiguity False Dispatches |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R0 (Heuristic)** | $0$ | $0$ | $20$ | $80$ | $0$ | $0.00\\%$ | $0.00\\%$ | $0.00\\%$ | $20$ |
| **R1 (Linear Ungated)** | $0$ | $0$ | $20$ | $80$ | $0$ | $0.00\\%$ | $0.00\\%$ | $0.00\\%$ | $20$ |
| **R1-Gated** | $0$ | $0$ | $20$ | $80$ | $0$ | $0.00\\%$ | $0.00\\%$ | $0.00\\%$ | $20$ |
| **R2 / R2-Cal** | $0$ | $0$ | $20$ | $80$ | $0$ | $0.00\\%$ | $0.00\\%$ | $0.00\\%$ | $20$ |
| **R3 (Uncalibrated Sel)** | $0$ | $0$ | $20$ | $80$ | $0$ | $0.00\\%$ | $0.00\\%$ | $0.00\\%$ | $4$ |
| **R4 (Proposed Frozen)** | **$10$** | **$36$** | **$10$** | **$44$** | **$46$** | **$50.00\\%$** | **$21.74\\%$** | **$30.30\\%$** | **$10$** |
| **R4-Tuned (Grid Ablation)** | **$12$** | **$49$** | **$8$** | **$31$** | **$61$** | **$60.00\\%$** | **$19.67\\%$** | **$29.63\\%$** | **$8$** |
| **R5 (Frontier LLM)** | $0$ | $0$ | $20$ | $80$ | $0$ | $0.00\\%$ | $0.00\\%$ | $0.00\\%$ | $20$ |

*Note on R3*: Uncalibrated selective baseline R3 possesses no interactive clarification dialogue mechanism. It emits zero `CLARIFY_AMBIGUOUS` actions; all 56 of its halts are generic confidence abstentions (`ABSTAIN_LOW_CONFIDENCE`).

#### Table 3.2B: Autonomous Confidence Abstention Profile (`ABSTAIN_LOW_CONFIDENCE`)
Evaluates frequency and cohort provenance of generic confidence abstentions across all $N=100$ test scenarios:

| Policy Configuration | Total Abstentions | Cohort 1 ($N=25$) | Cohort 2 ($N=20$) | Cohort 3 ($N=15$) | Cohort 4 ($N=20$) | Cohort 5 ($N=10$) | Cohort 6 ($N=10$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R0 / R1 / R1-Gated / R2** | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ |
| **R3 (Uncalibrated Sel)** | **$56$** | $18$ | $2$ | $0$ | $16$ | $10$ | $10$ |
| **R4 (Proposed Frozen)** | **$0$** | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ |
| **R4-Tuned (Grid Ablation)** | **$0$** | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ |
| **R5 (Frontier LLM)** | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ | $0$ |

> **Implementation Verification of R4's Zero Autonomous Abstentions**:
> The zero count for R4 in Table 3.2B is mathematically and programmatically verified from the underlying dispatch architecture in `satquery/router.py` and `scripts/run_m33_routing_experiments.py` (`predict_r4_proposed`):
> 1. In R4, the joint selection score is $s_{\\text{joint}} = \\min(c_{\\text{intent}} / \\tau_{\\text{intent}}, \\Delta c / \\tau_{\\text{margin}})$.
> 2. By definition, $s_{\\text{joint}} < 1.0 \\iff (c_{\\text{intent}} < \\tau_{\\text{intent}} \\lor \\Delta c < \\tau_{\\text{margin}})$, which triggers the condition `is_ambiguous = True`.
> 3. Under operating policy $\\tau_{\\text{sel}} = 1.0$, the decision logic branches on `if is_ambiguous:` first. Consequently, all low-confidence/low-margin outcomes take the interactive clarification pathway (`CLARIFY_AMBIGUOUS`) rather than silently dropping the request.
> 4. The branch `elif s_joint < tau_sel:` is reached only if `not is_ambiguous` (which requires $s_{\\text{joint}} \\ge 1.0$), making `s_joint < 1.0` logically unreachable under $\\tau_{\\text{sel}} = 1.0$. (Autonomous abstention activates only under custom operating policies where $\\tau_{\\text{sel}} > 1.0$).

#### Table 3.2C: Combined Broad Uncertainty Halting Profile (`CLARIFY` $\\cup$ `ABSTAIN`)
Evaluates the broad capability of the system to intercept uncertainty through either clarification or abstention:

| Policy Configuration | True Positives ($\\text{TP}$) | False Positives ($\\text{FP}$) | False Negatives ($\\text{FN}$) | True Negatives ($\\text{TN}$) | Total Predicted Halts ($\\text{TP} + \\text{FP}$) | Uncertainty Recall | Uncertainty Precision | Uncertainty $F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R0 / R1 / R1-Gated / R2** | $0$ | $0$ | $20$ | $80$ | $0$ | $0.00\\%$ | $0.00\\%$ | $0.00\\%$ |
| **R3 (Uncalibrated Sel)** | $16$ | $40$ | $4$ | $40$ | $56$ | **$80.00\\%$** | **$28.57\\%$** | **$42.11\\%$** |
| **R4 (Proposed Frozen)** | **$10$** | **$36$** | **$10$** | **$44$** | **$46$** | **$50.00\\%$** | **$21.74\\%$** | **$30.30\\%$** |
| **R4-Tuned (Grid Ablation)** | **$12$** | **$49$** | **$8$** | **$31$** | **$61$** | **$60.00\\%$** | **$19.67\\%$** | **$29.63\\%$** |

#### Audited Cohort Arithmetic for False-Positive Halts
Exact scenario-level audit of the $N=100$ held-out test scenarios resolves the cohort distribution of false-positive halts ($\\text{FP} \\le \\text{Cohort Size}$):

1. **R4-Frozen False-Positive Clarifications ($\\text{FP} = 36$)**:
   - **Cohort 1 (Supported Valid, $N=25$)**: $14$ queries halted for clarification due to tight decision margins ($14/25 = 56.0\\%$).
   - **Cohort 2 (Incompatible Inputs, $N=20$)**: $7$ queries triggered linguistic clarification before physical execution ($7/20 = 35.0\\%$).
   - **Cohort 3 (Missing Modalities, $N=15$)**: $1$ query triggered linguistic clarification ($1/15 = 6.67\\%$).
   - **Cohort 5 (Unsupported Tasks, $N=10$)**: $6$ queries halted for clarification due to low out-of-domain router confidence ($6/10 = 60.0\\%$).
   - **Cohort 6 (Out-of-Distribution, $N=10$)**: $8$ queries halted for clarification ($8/10 = 80.0\\% \\le 10$).  
   *Total*: $14 + 7 + 1 + 6 + 8 = 36$. (Resolves previous draft typographical error claiming 14 in Cohort 6).

2. **R3 False-Positive Uncertainty Halts ($\\text{FP} = 40$)**:
   - **Cohort 1**: $18 / 25$ ($72.0\\%$).
   - **Cohort 2**: $2 / 20$ ($10.0\\%$).
   - **Cohort 3**: $0 / 15$ ($0.0\\%$).
   - **Cohort 5**: $10 / 10$ ($100.0\\%$).
   - **Cohort 6**: $10 / 10$ ($100.0\\%$).  
   *Total Non-Ambiguous Abstentions*: $18 + 2 + 0 + 10 + 10 = 40$. Added to the $16$ true positive abstentions in Cohort 4, this equals exactly $56$ total abstentions across the test set."""

SEC4_NOTE_NEW = """> [!NOTE]
> **Reconciling R3's Denominators: Test-Wide Abstentions ($N=56$) vs Execution Cohort $\\mathcal{E}_{\\text{eval}}$ ($N=25$)**:
> - **$N=56$ Compatible Queries Abstaining**: Across the entire $N=100$ test set, R3 halted on 56 queries due to uncalibrated confidence $< 0.65$. Specifically, of these 56 abstentions, 18 were in Cohort 1, 2 in Cohort 2, 16 in Cohort 4, 10 in Cohort 5, and 10 in Cohort 6.
> - **$N=25$ Valid Specialists Evaluation Population ($\\mathcal{E}_{\\text{eval}}$)**: Cohort 1 comprises the 25 valid, in-domain requests evaluated for perceptual execution $\\text{TSR}_{\\text{exec}}^{(2)}$ across our two trained specialists (Land Cover and Change Detection). Of these 25 valid scenarios, R3 abstained on 18 ($72.0\\%$) and misrouted 7 ($28.0\\%$) to unsupported task slots. Consequently, R3 executed exactly zero valid specialist tasks ($0/25 = 0.0\\%$).

> [!NOTE]
> **Contextualizing R5's $92.0\\%$ Execution Rate vs R4 ($32.0\\%$)**:
> R5's $92.0\\%$ perception execution rate is **not a mark of superior performance**. R5 operates without physical observation awareness: it executes indiscriminately on corrupt inputs, mismatched grids, and cloud-contaminated scenes, suffering a catastrophic $100.0\\%$ Incompatible Dispatch Error Rate ($\\text{IDER}_{\\text{obs}}$). In contrast, R4's $32.0\\%$ execution rate reflects strict quality acceptance and selective filtering: it dispatches only when inputs are verified physically compatible and linguistic confidence exceeds operational bounds."""

SEC52_NEW = """### 5.2 Paired Cluster Bootstrap & McNemar Tests for 7-Task Architectural Success ($\\text{TSR}_{\\text{arch}}^{(7)}$)
To rigorously evaluate the macro architectural success rate difference between R4 and the baselines across all $N=100$ held-out test scenarios, we reconstructed scenario-level success indicators and conducted:
1. **Paired Cluster Bootstrap ($B=1,000$, Resampling 24 Template Family Clusters)**: Computes cluster-robust 95% confidence intervals on the risk difference $\\Delta \\text{TSR}_{\\text{arch}} = \\text{TSR}_{\\text{arch}}(R_4) - \\text{TSR}_{\\text{arch}}(R_m)$.
2. **McNemar Paired Test with Edwards Continuity Correction**: Evaluates discordant pairs ($n_{01}$ and $n_{10}$) where one model succeeded and the other failed:
   $$\\chi^2 = \\frac{(|n_{10} - n_{01}| - 1)^2}{n_{10} + n_{01}}, \\qquad p_{\\text{raw}} = P(\\chi_1^2 \\ge \\text{stat})$$
3. **Step-Down Holm Multiplicity Control**: Controls the family-wise error rate across the $m=5$ paired architectural contrasts at $\\alpha = 0.05$:

| Contrast ($R_4$ vs Baseline) | $R_4$ $\\text{TSR}_{\\text{arch}}$ | Baseline $\\text{TSR}_{\\text{arch}}$ | Risk Difference ($\\Delta \\text{TSR}$) | Bootstrap Standard Error | 95% Clustered CI | Contingency $[n_{00}, n_{01}, n_{10}, n_{11}]$ | McNemar $\\chi^2$ Formula | McNemar $\\chi^2$ | Raw $p$-value | Holm Multiplier | Step-Down Holm $p$-value | Confirmatory Rejection ($\\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R4 vs R1 (Linear Ungated)** | $59.00\\%$ | $15.00\\%$ | **$+44.00\\%$** | $10.14\\%$ | $[+24.00\\%, +64.36\\%]$ | $[34, 7, 51, 8]$ | $\\frac{(|51-7|-1)^2}{51+7} = \\frac{1849}{58}$ | **$31.879$** | $1.64 \\times 10^{-8}$ | $5$ | **$8.20 \\times 10^{-8}$** | **REJECTED ($H_0$ Rejected)** |
| **R4 vs R0 (Heuristic)** | $59.00\\%$ | $21.00\\%$ | **$+38.00\\%$** | $12.74\\%$ | $[+11.88\\%, +61.11\\%]$ | $[27, 14, 52, 7]$ | $\\frac{(|52-14|-1)^2}{52+14} = \\frac{1369}{66}$ | **$20.742$** | $5.25 \\times 10^{-6}$ | $4$ | **$2.10 \\times 10^{-5}$** | **REJECTED ($H_0$ Rejected)** |
| **R4 vs R2 (Gated Non-Sel)** | $59.00\\%$ | $47.00\\%$ | $+12.00\\%$ | $8.77\\%$ | $[-3.48\\%, +30.77\\%]$ | $[29, 12, 24, 35]$ | $\\frac{(|24-12|-1)^2}{24+12} = \\frac{121}{36}$ | **$3.361$** | $0.0668$ | $3$ | $0.2003$ | Not Rejected ($p_{\\text{Holm}} > 0.05$) |
| **R4 vs R1-Gated** | $59.00\\%$ | $48.00\\%$ | $+11.00\\%$ | $8.77\\%$ | $[-6.25\\%, +28.77\\%]$ | $[28, 13, 24, 35]$ | $\\frac{(|24-13|-1)^2}{24+13} = \\frac{100}{37}$ | **$2.703$** | $0.1002$ | $2$ | $0.2004$ | Not Rejected ($p_{\\text{Holm}} > 0.05$) |
| **R4 vs R3 (Uncalibrated Sel)** | $59.00\\%$ | $53.00\\%$ | $+6.00\\%$ | $8.23\\%$ | $[-10.38\\%, +21.11\\%]$ | $[29, 12, 18, 41]$ | $\\frac{(|18-12|-1)^2}{18+12} = \\frac{25}{30}$ | **$0.833$** | $0.3613$ | $1$ | $0.3613$ | Not Rejected ($p_{\\text{Holm}} > 0.05$) |

**Assessment of Architectural Improvement & Multiplicity**:
1. **R4 vs Unconstrained Baselines (R0, R1)**: The architectural advantage of R4 is decisive and survives rigorous Step-Down Holm multiplicity correction ($p_{\\text{Holm}} < 10^{-7}$ vs R1; $p_{\\text{Holm}} < 10^{-4}$ vs R0). The 95% cluster-bootstrap confidence interval ($[+24.00\\%, +64.36\\%]$) excludes zero, statistically confirming that physical compatibility gating and selective uncertainty handling decisively outperform unconstrained routing.
2. **R4 vs Active-Gated & Selective Baselines (R1-Gated, R2, R3)**: While R4 achieves the highest point estimate for macro architectural success ($59.00\\%$, $+12.0\\%$ over R2 and $+6.0\\%$ over R3), **these differences do not survive multiplicity correction** ($p_{\\text{Holm}} = 0.2003$ for R2; $p_{\\text{Holm}} = 0.3613$ for R3), and their 95% clustered bootstrap intervals include zero. SatQuery AI does not claim certified superiority over R2 or R3 on $\\text{TSR}_{\\text{arch}}^{(7)}$, but reports them as consistent sample-level observations."""


def update_report(file_path: Path):
    content = file_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # Replace Section 3.2
    start_sec32 = content.find("### 3.2 Full Confusion Matrix for Ambiguity Clarification Fidelity")
    if start_sec32 == -1:
        start_sec32 = content.find("### 3.2 Distinction Between Linguistic Clarification")
    if start_sec32 != -1:
        end_sec32 = content.find("\n---\n\n## 4. Segregated End-to-End Success Rates", start_sec32)
        if end_sec32 != -1:
            content = content[:start_sec32] + SEC32_NEW + "\n\n" + content[end_sec32:]
            print(f"Updated Section 3.2 in {file_path.name}")
        else:
            print(f"Could not find end of Section 3.2 in {file_path.name}")
    else:
        print(f"Could not find start of Section 3.2 in {file_path.name}")

    # Replace Contextualizing note in Section 4
    start_note = content.find("> [!NOTE]\n> **Contextualizing R5's $92.0\\%")
    if start_note == -1:
        start_note = content.find("> [!NOTE]\n> **Contextualizing R5's $92.0%")
    if start_note == -1:
        start_note = content.find("> [!NOTE]\n> **Reconciling R3's Denominators")
    if start_note != -1:
        end_note = content.find("\n---\n\n## 5. Statistical Inference", start_note)
        if end_note != -1:
            content = content[:start_note] + SEC4_NOTE_NEW + "\n\n" + content[end_note:]
            print(f"Updated Section 4 Notes in {file_path.name}")
        else:
            print(f"Could not find end of Section 4 Note in {file_path.name}")
    else:
        print(f"Could not find Section 4 Note in {file_path.name}")

    # Replace Section 5.2
    start_sec52 = content.find("### 5.2 Paired Cluster Bootstrap & McNemar Tests for 7-Task Architectural Success")
    if start_sec52 != -1:
        end_sec52 = content.find("\n---\n\n## 6. Deliverable Manifests", start_sec52)
        if end_sec52 != -1:
            content = content[:start_sec52] + SEC52_NEW + "\n\n" + content[end_sec52:]
            print(f"Updated Section 5.2 in {file_path.name}")
        else:
            print(f"Could not find end of Section 5.2 in {file_path.name}")
    else:
        print(f"Could not find start of Section 5.2 in {file_path.name}")

    # Clarify AURC negative result language in Section 5.1
    old_aurc_summary = "**Assessment**: Every observed $\\hat{\\Delta}_{\\text{AURC}} \\le 0$ and all Holm-adjusted $p$-values equal $1.0000$. This confirms statistically that R4 did **not** achieve superior selective risk or lower AURC on $\\mathcal{E}_{\\text{eval}}$."
    new_aurc_summary = "**Assessment**: Every observed $\\hat{\\Delta}_{\\text{AURC}} \\le 0$ and all Holm-adjusted $p$-values equal $1.0000$. This demonstrates that the proposed system **failed to establish an AURC advantage or selective risk reduction** on $\\mathcal{E}_{\\text{eval}}$. In accordance with rigorous scientific reporting, this negative result is preserved: R4 does not outperform the baselines on selective risk over this sample."
    if old_aurc_summary in content:
        content = content.replace(old_aurc_summary, new_aurc_summary)
        print(f"Updated AURC language in Section 5.1 in {file_path.name}")

    file_path.write_text(content, encoding="utf-8")
    print(f"Successfully wrote updated content to {file_path.name}")


if __name__ == "__main__":
    report_path = REPO_ROOT / "PHASE3_M33_ROUTING_EVALUATION_REPORT.md"
    brain_path = Path("C:/Users/B NAVANEETH/.gemini/antigravity/brain/89b5d681-0c9a-403e-9edc-f4bd3eee0aba/phase3_m33_routing_evaluation_report.md")

    update_report(report_path)
    if brain_path.exists():
        update_report(brain_path)
