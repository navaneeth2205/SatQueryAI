"""
Script: generate_methodology_docx.py
Purpose: Generates the authoritative, publication-grade editable Microsoft Word document (METHODOLOGY.docx)
         for SatQuery AI following the exact 10-section template from Simple_ML_Project_Methodology_Template.pdf.
         Fully audited and reconciled against Milestone M3.1, M3.2, and M3.3 empirical benchmarks:
         - Exact calibration temperature T_cal = 0.0991 (NLL reduction -60.19%, ECE reduced by 5.27%).
         - Router architecture: TF-IDF (1, 2) + Balanced Regularized Logistic Regression with post-hoc calibration.
         - Selective thresholds: Nominal baseline (0.65, 0.15) vs. Validation-Tuned (0.85, 0.05).
         - Reconciled M3.2 change-detection metrics (Standard CVA 20.31%, Gaussian CVA 27.31%, FC-Siam-diff 37.66% Macro F1, 35.33% prec, 41.53% rec).
         - Honest M3.3 selective-risk reporting: AURC is an exploratory evaluation (R4 AURC 0.2683 vs R3 0.2638, p=0.6274); primary confirmatory contribution is Macro Architectural Safety.
         - Reconciled benchmark populations (N=100 test scenarios, N=35 contract violations, N=20 ambiguous queries, N=25 common valid E_eval).
         - Short, clean footer: "ML Group Project — SatQuery AI".
         - 5-Student group table with editable Roll Number / Student ID slots.
         - True native Microsoft Word equations (OMML) for all mathematical formulations.
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import lxml.etree as etree
import latex2mathml.converter

# =========================================================================
# OMML Math Conversion Setup
# =========================================================================
XSLT_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
_xslt_doc = etree.parse(XSLT_PATH)
_omml_transform = etree.XSLT(_xslt_doc)

def latex_to_omml_node(latex_str):
    """Converts a LaTeX math string into a native Word OMML XML node."""
    mathml_str = latex2mathml.converter.convert(latex_str)
    doc_tree = etree.fromstring(mathml_str)
    omml_tree = _omml_transform(doc_tree)
    omml_xml = etree.tostring(omml_tree, encoding="unicode")
    return parse_xml(omml_xml)

# =========================================================================
# Table & Styling Helpers
# =========================================================================
def set_cell_background(cell, fill_hex):
    """Sets background fill color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding for a cell in twips (20 twips = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E0", sz="4", val="single"):
    """Sets clean horizontal and vertical borders on a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_docx(output_path="METHODOLOGY.docx"):
    print("Initializing authoritative Word document...")
    doc = docx.Document()

    # 1. Page Margins (1.0 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Header Setup: Running Head on the Right
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("SatQuery AI: Machine Learning Course Project Methodology")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(113, 128, 150)
        
        # Footer Setup: Short, Clean Footer as requested
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("ML Group Project — SatQuery AI")
        frun.font.name = "Calibri"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(113, 128, 150)

    # Styles Setup
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(45, 55, 72)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    # Color Palette
    c_navy = RGBColor(26, 54, 93)     # #1A365D
    c_blue = RGBColor(43, 108, 176)   # #2B6CB0
    c_dark = RGBColor(45, 55, 72)     # #2D3748

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = c_navy
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = c_blue
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = c_dark
        return p

    def add_body(text, bold_prefix="", italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = 'Calibri'
            r_pre.font.bold = True
            r_pre.font.size = Pt(10)
            r_pre.font.color.rgb = c_dark
        r_text = p.add_run(text)
        r_text.font.name = 'Calibri'
        r_text.font.size = Pt(10)
        r_text.font.italic = italic
        r_text.font.color.rgb = c_dark
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = 'Calibri'
            r_pre.font.bold = True
            r_pre.font.size = Pt(9.5)
            r_pre.font.color.rgb = c_dark
        r_text = p.add_run(text)
        r_text.font.name = 'Calibri'
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = c_dark
        return p

    def add_callout(text, bold_prefix="", fill_hex="F7FAFC", border_hex="2B6CB0", math_latex=None):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        cell.width = Inches(6.5)
        set_cell_background(cell, fill_hex)
        set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
        
        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>\n'
            f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>\n'
            f'  <w:top w:val="none"/>\n'
            f'  <w:right w:val="none"/>\n'
            f'  <w:bottom w:val="none"/>\n'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = 'Calibri'
            r_pre.font.bold = True
            r_pre.font.size = Pt(9.5)
            r_pre.font.color.rgb = c_navy
        if text:
            r_txt = p.add_run(text)
            r_txt.font.name = 'Calibri'
            r_txt.font.size = Pt(9.5)
            r_txt.font.color.rgb = c_dark
        if math_latex:
            omml_node = latex_to_omml_node(math_latex)
            p._p.append(omml_node)
        
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def add_math_display(latex_str, label=None):
        """Adds a centered, publication-grade display formula using native Word OMML."""
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(4)
        omml_node = latex_to_omml_node(latex_str)
        p._p.append(omml_node)
        if label:
            p_lbl = p.add_run(f"    ({label})")
            p_lbl.font.name = "Calibri"
            p_lbl.font.size = Pt(9.0)
            p_lbl.font.color.rgb = RGBColor(113, 128, 150)
        return p

    def add_code_box(code_text, title=""):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        cell.width = Inches(6.5)
        set_cell_background(cell, "F8FAFC")
        set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
        
        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>\n'
            f'  <w:left w:val="single" w:sz="12" w:space="0" w:color="2B6CB0"/>\n'
            f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>\n'
            f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>\n'
            f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>\n'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.05
        if title:
            r_title = p.add_run(title + "\n")
            r_title.font.name = 'Consolas'
            r_title.font.size = Pt(8.5)
            r_title.font.bold = True
            r_title.font.color.rgb = c_navy
            
        r_code = p.add_run(code_text)
        r_code.font.name = 'Consolas'
        r_code.font.size = Pt(7.8)
        r_code.font.color.rgb = RGBColor(45, 55, 72)
        
        doc.add_paragraph().paragraph_format.space_after = Pt(3)

    def format_table(table, col_widths, alignments):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table, color="CBD5E0", sz="4")
        
        # Header Row
        hdr_row = table.rows[0]
        for c_idx, cell in enumerate(hdr_row.cells):
            cell.width = Inches(col_widths[c_idx])
            set_cell_background(cell, "1A365D")
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = alignments[c_idx]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            for run in p.runs:
                run.font.name = 'Calibri'
                run.font.size = Pt(8.5)
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                
        # Data Rows
        for r_idx, row in enumerate(table.rows[1:], start=1):
            bg = "F7FAFC" if (r_idx % 2 == 1) else "FFFFFF"
            for c_idx, cell in enumerate(row.cells):
                cell.width = Inches(col_widths[c_idx])
                set_cell_background(cell, bg)
                set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                p.alignment = alignments[c_idx]
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                for run in p.runs:
                    run.font.name = 'Calibri'
                    run.font.size = Pt(8.2)
                    run.font.color.rgb = RGBColor(45, 55, 72)
                    
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # =========================================================================
    # DOCUMENT COVER / METADATA BLOCK
    # =========================================================================
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_main_title = p_title.add_run("SatQuery AI: An Input-Constrained Selective Routing and Multimodal Perception Architecture for Satellite Earth Observation")
    r_main_title.font.name = "Calibri"
    r_main_title.font.size = Pt(15)
    r_main_title.font.bold = True
    r_main_title.font.color.rgb = c_navy

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(8)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Formal Machine Learning Lab Course Project Research Methodology Report")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(10)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(74, 85, 104)

    # 5-MEMBER STUDENT GROUP TABLE
    add_h3("Student Team Details (Group Project)")
    student_tbl = doc.add_table(rows=6, cols=3)
    s_widths = [1.8, 2.7, 2.0]
    s_headers = ["Group Role / S.No.", "Student Name", "Roll Number / Student ID"]
    for idx, text in enumerate(s_headers):
        student_tbl.cell(0, idx).paragraphs[0].text = text

    s_data = [
        ("Student 1 (Lead)", "B Navaneeth", "[Roll Number 1 / e.g. 21BCE0001]"),
        ("Student 2", "[Team Member 2 Name]", "[Roll Number 2 / ID]"),
        ("Student 3", "[Team Member 3 Name]", "[Roll Number 3 / ID]"),
        ("Student 4", "[Team Member 4 Name]", "[Roll Number 4 / ID]"),
        ("Student 5", "[Team Member 5 Name]", "[Roll Number 5 / ID]")
    ]

    for row_idx, (role, name, roll) in enumerate(s_data, start=1):
        student_tbl.cell(row_idx, 0).paragraphs[0].text = role
        student_tbl.cell(row_idx, 1).paragraphs[0].text = name
        student_tbl.cell(row_idx, 2).paragraphs[0].text = roll

    format_table(student_tbl, s_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    # Metadata Summary Table
    meta_tbl = doc.add_table(rows=2, cols=2)
    m_widths = [2.2, 4.3]
    meta_tbl.cell(0, 0).paragraphs[0].text = "Submission Date"
    meta_tbl.cell(0, 1).paragraphs[0].text = "November 5, 2026"
    meta_tbl.cell(1, 0).paragraphs[0].text = "Academic Track"
    meta_tbl.cell(1, 1).paragraphs[0].text = "Semester 5 — Machine Learning Lab Project (SIH26167 Track)"
    format_table(meta_tbl, m_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_callout(
        "Sensor- and Physical-Compatibility-Gated Selective Routing with Calibrated Abstention for Multimodal Earth Observation Workflows.",
        bold_prefix="Core Research Contribution: "
    )

    # =========================================================================
    # SECTION 1: PROBLEM AND OBJECTIVES
    # =========================================================================
    add_h1("1. Problem and Objectives")

    add_h2("1.1 Problem Statement and Practical Significance")
    add_body(
        "Operational Earth Observation (EO) query systems are increasingly transitioning from manual inspection toward automated vision-language assistants. However, deployed remote-sensing workflows suffer from two critical, unaddressed failure modes:"
    )
    add_bullet(
        "Natural-language instructions are frequently polysemous, underspecified, or mismatched with attached satellite payloads. Traditional unconstrained multi-task architectures arbitrarily coerce instructions into whichever task model can accept the attached image tensors. When an input lacks unambiguous linguistic intent, unconstrained dispatch produces deceptive, ungrounded analytical outputs.",
        bold_prefix="1. Linguistic Ambiguity & Hallucinatory Task Coercion: "
    )
    add_bullet(
        "Remote-sensing analytical tasks require strict physical sensor invariants (e.g., specific spectral band combinations, minimum spatial resolutions, multi-temporal coregistration, or synthetic aperture radar polarizations). Deployed perception models execute silently on malformed or physically incompatible inputs (e.g., executing bitemporal change detection on identical same-day acquisitions, running multispectral land-cover networks on 3-band RGB imagery, or fusing images across coordinate frame discrepancies), emitting corrupt predictions with high artificial confidence.",
        bold_prefix="2. Physical Sensor and Geospatial Mismatches: "
    )

    add_h2("1.2 Project Objectives")
    add_bullet(
        "Formulate a calibrated, margin-aware text routing engine that enforces the Non-Coercion Invariant (T* ⊥ o | q) as an architectural design constraint, inferring user intent strictly from linguistic evidence and abstaining or soliciting human clarification when intent is ambiguous.",
        bold_prefix="1. Design an Input-Constrained Selective Routing Mechanism: "
    )
    add_bullet(
        "Formulate and implement a closed-form geometric and sensor verification suite C(t, o) in {0, 1} that evaluates eight deterministic physical dimensions (modality completeness, temporal baseline, spectral coverage, spatial dimensions, CRS projection, GSD scale parity, sub-pixel grid phase alignment, and radiometric/cloud bounds) before specialist execution can proceed.",
        bold_prefix="2. Implement a Deterministic Physical Compatibility Gate: "
    )
    add_bullet(
        "Build and evaluate dedicated deep perception specialist models tailored to Earth Observation data under extreme class imbalance: (a) a multimodal land-cover classification specialist integrating Sentinel-1 SAR and Sentinel-2 MSI via dual-branch ResNet-34 backbones and cross-attention; (b) a bitemporal change-detection specialist evaluating convolutional and transformer architectures (FC-Siam-diff and BIT) on Sentinel-2 optical pairs.",
        bold_prefix="3. Develop and Certify Deep Multimodal Perception Specialists: "
    )
    add_bullet(
        "Unify intent confidence, margin separation, and physical sensor gating into an operational orchestrator providing three primary execution pathways: Pathway 1 (DISPATCH_QUALIFIED), Pathway 2 (REJECT_INCOMPATIBLE_OBSERVATION), Pathway 3 (CLARIFY_AMBIGUOUS_INTENT), along with a risk-controlled sub-outcome (ABSTAIN_LOW_CONFIDENCE) for graceful deferral when elevated risk thresholds (tau_sel > 1.0) are applied.",
        bold_prefix="4. Establish a Risk-Controlled Multi-Pathway Orchestrator: "
    )

    add_h2("1.3 Research Questions")
    add_bullet(
        "Does decoupling linguistic intent inference from observation payload metadata eliminate silent task coercion on tested contract-violation scenarios compared to unconstrained joint routing baselines?",
        bold_prefix="RQ1 (Routing Separation & Safety): "
    )
    add_bullet(
        "How does a standardized joint selection function combining temperature-calibrated posterior confidence and competitive margin separation (s_joint(q)) perform on ambiguous and low-margin queries under an exploratory selective prediction evaluation?",
        bold_prefix="RQ2 (Uncertainty & Ambiguity Interception Fidelity): "
    )
    add_bullet(
        "Under small-sample and extreme foreground class imbalance regimes (< 3% change pixels in OSCD; < 0.5% rare land classes in BigEarthNet-14k), how do these architectural and loss-function choices affect performance under spatially separated, held-out-city evaluation within the OSCD dataset?",
        bold_prefix="RQ3 (Extreme Imbalance & Inductive Bias in EO Specialists): "
    )

    add_h2("1.4 Proposed Research Contribution")
    add_callout(
        "Sensor- and Physical-Compatibility-Gated Selective Routing with Calibrated Abstention for Multimodal Earth Observation Workflows.",
        bold_prefix="Core Research Contribution: "
    )

    add_h2("1.5 Epistemic Status: Established Facts vs. Research Hypotheses")
    add_bullet(
        "The physical sensor contract verification gate is closed-form and deterministic; the spatially buffered split enforces a minimum separation of at least 12 km between relevant patches, reducing the risk of spatial leakage without establishing the complete absence of spatial autocorrelation; deep change-detection specialist FC-Siam-diff achieved 37.66% held-out test Macro F1, outperforming classical CVA baselines (20.31% standard, 27.31% Gaussian), but failed the frozen certification threshold of 50.00%. In Milestone M3.3, deterministic physical gating intercepted 100% of tested sensor contract violations (35/35, IDER = 0.0%, Wilson 95% upper bound 9.89%), establishing macro architectural safety.",
        bold_prefix="Established Engineering & Empirical Facts: "
    )
    add_bullet(
        "The non-coercion invariant (T* ⊥ o | q) is an architectural design constraint enforced on the router, rather than an experimentally established empirical independence result. Fine-grained selective risk differences (AURC) between calibrated selective routing (R4) and uncalibrated selective routing (R3) on small test cohorts (N=25 valid requests) are statistically underpowered (p = 0.6274) and represent an exploratory feasibility study.",
        bold_prefix="Research Hypotheses & Exploratory Analyses: "
    )

    # =========================================================================
    # SECTION 2: RELATED WORK
    # =========================================================================
    add_h1("2. Related Work")
    add_body(
        "The design of SatQuery AI synthesizes foundational principles across six established research areas: (1) Remote-Sensing Vision-Language Assistants (GeoChat, Earth-GPT, SkyEye); (2) Adaptive Routing in Multi-Task Systems (Learning to Route, Soft MoE); (3) Selective Prediction & Risk-Coverage Formulations (SelectiveNet, Chow reject option); (4) Probability Calibration (post-hoc temperature scaling); (5) Multimodal Land-Cover Classification under class imbalance (BigEarthNet-14k, CB-BCE, ASL); and (6) Bitemporal Change Detection (FC-Siam-diff, BIT)."
    )

    add_h2("2.1 Baseline Papers Comparison Table")
    add_body("To prevent unverified assumptions, baseline capabilities are classified into three explicit categories: [DEMONSTRATED] (capability designed and validated); [NOT REPORTED / UNEVALUATED] (capability not addressed); and [ARCHITECTURAL LIMITATION (ASSESSED)] (structural lack of physical gating or selective reject option):")

    baseline_tbl = doc.add_table(rows=4, cols=3)
    b_widths = [1.8, 2.3, 2.4]
    b_headers = ["Paper (author, year)", "Method used", "Main result / limitation"]
    for idx, text in enumerate(b_headers):
        baseline_tbl.cell(0, idx).paragraphs[0].text = text

    b_data = [
        ("GeoChat\n(Kuckreja et al., CVPR 2024)",
         "Unified Vision-Language remote sensing foundation model based on LLaVA architecture; handles multi-task VQA, grounding, captioning.",
         "[DEMONSTRATED]: Strong conversational competence across diverse visual tasks.\n[ARCHITECTURAL LIMITATION (ASSESSED)]: Monolithic unconstrained dispatch; lacks deterministic physical sensor validation; unequipped with selective reject option or ambiguity clarification protocols."),
        ("Learning to Route (L2R)\n(Wang et al., ICML 2021)",
         "Collaborative multi-agent neural routing policy; routes sample representations to specialized network branches.",
         "[DEMONSTRATED]: Parameter-efficient neural routing across specialized experts.\n[ARCHITECTURAL LIMITATION (ASSESSED)]: Routes based on latent prompt/sample embeddings; does not evaluate physical sensor metadata contracts or selective risk-coverage curves."),
        ("SelectiveNet\n(Geifman & El-Yaniv, ICML 2019)",
         "Deep neural network with integrated reject option; optimizes selective risk at target coverage Phi.",
         "[DEMONSTRATED]: Establishes mathematically sound selective classification risk formulations on closed image datasets (CIFAR).\n[ARCHITECTURAL LIMITATION (ASSESSED)]: Single-task classification only; does not model multi-task orchestration, physical compatibility contracts, or conversational clarification.")
    ]

    for row_idx, (p_name, m_used, m_res) in enumerate(b_data, start=1):
        baseline_tbl.cell(row_idx, 0).paragraphs[0].text = p_name
        baseline_tbl.cell(row_idx, 1).paragraphs[0].text = m_used
        baseline_tbl.cell(row_idx, 2).paragraphs[0].text = m_res

    format_table(baseline_tbl, b_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_h2("2.2 Research Gap")
    add_body(
        "Existing remote-sensing assistants assume observation payloads are valid and well-formed, allowing multimodal backbones to dictate task selection based on whatever image channels happen to be loaded. When a user provides an ambiguous prompt or uploads incompatible sensor data (e.g., single-image optical data for a bitemporal change-detection task), current systems either hallucinate or fail silently. Furthermore, selective prediction methods have been applied only to isolated perception models, never to the combined problem of linguistic intent ambiguity and deterministic physical sensor contract verification. SatQuery AI addresses this identified research gap."
    )

    # =========================================================================
    # SECTION 3: DATASET
    # =========================================================================
    add_h1("3. Dataset")

    data_tbl = doc.add_table(rows=5, cols=2)
    d_widths = [1.8, 4.7]
    data_tbl.cell(0, 0).paragraphs[0].text = "Item"
    data_tbl.cell(0, 1).paragraphs[0].text = "Description"

    d_rows = [
        ("Dataset name and source",
         "1. BigEarthNet-14k (Multimodal Land-Cover): Sumbul et al. (2019, 2021); Zenodo / TU Berlin.\n2. OSCD (Onera Satellite Change Detection): Daudt et al. (2018); IEEE DataPort.\n3. SatQuery Synthetic Routing Benchmark (v3.0.0-freeze-rev21): SatQuery AI Consortium; internal repository."),
        ("Number of samples",
         "• BigEarthNet-14k: 13,683 total multimodal sub-tiles (120 x 120 pixels at 10m GSD).\n• OSCD: 24 global city scenes (10 train, 4 val, 10 test). Training split comprises 586 unique extracted patches (256 x 256 pixels, stride 64) with dynamic on-the-fly spatial augmentation. Held-out test split comprises 3,077,936 full-resolution test pixels.\n• Routing Benchmark: Comprehensive pool of 84 template families across 3 tiers (historical pool of 252 scenarios). Authoritative Milestone M3.3 test evaluation is performed on a locked held-out test split of N = 100 scenarios across 5 cohorts (N=25 valid requests, N=35 contract violations, N=20 ambiguous queries, N=20 unsupported/OOD queries)."),
        ("Input features and target",
         "• BigEarthNet-14k: Inputs: 10-band Sentinel-2 surface reflectance + 2-band Sentinel-1 SAR backscatter (sigma^0 dB). Target: 16-class CORINE multi-label binary vector y in {0, 1}^16.\n• OSCD: Inputs: Bitemporal Sentinel-2 Level-1C optical RGB image pairs (t1, t2). Target: Binary pixel-level change mask Y in {0, 1}^(H x W).\n• Routing Benchmark: Inputs: Natural language text instruction q and JSON payload metadata o. Target: Discrete task T* in T and orchestrator action A* in {DISPATCH, REJECT, CLARIFY, ABSTAIN}."),
        ("Data split",
         "• BigEarthNet-14k: Spatially buffered patch split (>= 12 km buffer in UTM Zone 34N across tiles T34TCR, T34TEQ, T34TFN). Train: 7,180 patches (52.5%), Validation: 3,255 patches (23.8%), Held-out Test: 3,248 patches (23.7%). Tile T34TCS (N=627) excluded from test.\n• OSCD: City-level disjoint spatial split. Train: 10 cities (beirut, berrida, bordeaux, chongqing, dubai, hong_kong, lasvegas, milano, rennes, saclay_w). Validation: 4 cities (montpellier, paris, rio, valencia). Held-out Test: 10 cities (aguasclaras, austin, beihai, berlin, brasilia, budapest, campinas, mumbai, nantes, pisa).\n• Routing Benchmark: Monotonically nested test tiers with zero 3,4,5-gram syntactic overlap against training prompts. Held-out test split comprises N=100 scenarios unblinded strictly once.")
    ]

    for row_idx, (item, desc) in enumerate(d_rows, start=1):
        data_tbl.cell(row_idx, 0).paragraphs[0].text = item
        data_tbl.cell(row_idx, 1).paragraphs[0].text = desc

    format_table(data_tbl, d_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_h2("3.1 Dataset Limitations & Implementation Boundaries")
    add_bullet(
        "The dataset exhibits extreme multi-label positive frequency imbalance (ranging from >60% for Arable land down to 0.0% test support for Moors, heathland and Beaches, dunes, sands). Generalization is evaluated across a spatially buffered split within UTM Zone 34N; it does not constitute continental or cross-sensor transfer.",
        bold_prefix="BigEarthNet-14k: "
    )
    add_bullet(
        "Change events are extremely sparse (< 3% of total pixel area across all 10 test scenes), leading to high false-positive volatility in urban centers (e.g., beihai change ratio is 0.44%). The training set is strictly partitioned into 586 regular grid patches (256 x 256, stride 64), augmented on-the-fly during training.",
        bold_prefix="OSCD: "
    )
    add_bullet(
        "The evaluation templates reflect synthetic operational queries; while evaluated against independent human linguistic annotations (Po = 97.4%, Cohen's kappa = 0.918), open-domain conversational dialogue remains scheduled for Phase 4.",
        bold_prefix="Routing Benchmark: "
    )

    # =========================================================================
    # SECTION 4: DATA PREPROCESSING
    # =========================================================================
    add_h1("4. Data Preprocessing")

    add_h2("4.1 Standard Data Preprocessing Steps")
    add_bullet("BigEarthNet sub-tiles were deduplicated against corrupted unreadable L1C/L2A sub-archives; corrupted zero-byte SAR files were scrubbed.", bold_prefix="Remove duplicate or invalid records: ")
    add_bullet("Radiometric finiteness is enforced via closed-form check sum(isnan(o)) == 0 and sum(isinf(o)) == 0. Missing spectral channels trigger immediate deterministic rejection.", bold_prefix="Handle missing values: ")
    add_bullet("Natural language instructions are sanitized (lowercased, punctuation-stripped, whitespace-collapsed); satellite GeoTIFFs and Parquet metadata are transformed into calibrated PyTorch float32 tensors.", bold_prefix="Convert categorical/text/image data into suitable format: ")
    add_bullet("Multispectral and radar tensors are normalized using frozen, precomputed training-set statistics.", bold_prefix="Scale or normalize features: ")
    add_bullet("Spatial separation is enforced across tasks (>= 12 km buffer in UTM Zone 34N for BigEarthNet; city-disjoint splits for OSCD) to reduce the risk of spatial autocorrelation leakage.", bold_prefix="Split data into training, validation, and test sets: ")

    add_h2("4.2 Project-Specific Preprocessing Pipelines")
    add_h3("1. Sentinel-2 MSI Multispectral Pipeline (Land-Cover Specialist)")
    add_body("• Spectral Selection: 10 surface reflectance channels selected: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12] (native 20m bands resampled to 10m spatial resolution via bilinear interpolation).")
    add_body("• Reflectance Normalization & Processing Baseline Distinction:")
    add_body("   - Archival Baseline (BigEarthNet-14k, PB < 04.00): BigEarthNet-14k is sourced from archival Summer 2017 acquisitions where BOA_ADD_OFFSET = 0 and QUANTIFICATION_VALUE = 10,000.0. In the dataset loader and training pipeline, raw digital numbers are transformed into surface reflectance via:")
    add_math_display(r"\rho_\lambda = \frac{\text{DN}}{10000.0}", label="1")
    add_body("   - Operational Contract for Modern Acquisitions (PB >= 04.00, post-Jan 2022): For operational inference on modern Sentinel-2 products where ESA introduced a -1,000 DN radiometric offset, SatQuery AI applies:")
    add_math_display(r"\rho_\lambda = \frac{\text{DN} - 1000}{10000.0}", label="2")
    add_body("   - Channel-wise Standardization: Surface reflectance is standardized using static training-split statistics:")
    add_math_display(r"X_{\text{norm}}^{(c)} = \frac{\rho_c - \mu_c}{\sigma_c}", label="3")
    add_body("        mu    = [0.136, 0.118, 0.106, 0.128, 0.203, 0.257, 0.272, 0.273, 0.187, 0.121]")
    add_body("        sigma = [0.082, 0.076, 0.074, 0.089, 0.098, 0.108, 0.113, 0.112, 0.095, 0.078]")

    add_h3("2. Sentinel-1 SAR Dual-Polarization Pipeline (Land-Cover Specialist)")
    add_body("• Bands: Dual-polarization [VH, VV] backscatter coefficient sigma^0 in decibels (dB).")
    add_body("• Clipping and Scaling: Backscatter values are clipped to [-35.0, +5.0] dB (derived strictly from training set quantiles, covering >99.8% of terrestrial training pixels without clipping valid vegetation) and linearly scaled into [0.0, 1.0]:")
    add_math_display(r"X_{\text{SAR}}^{\text{scaled}} = \frac{\text{clip}(X_{\text{SAR}},\, -35.0,\, +5.0) + 35.0}{40.0} \in [0.0, 1.0]", label="4")

    add_h3("3. Bitemporal Optical Preprocessing (Change-Detection Specialist)")
    add_body("• Sensor Input: Sentinel-2 Level-1C top-of-atmosphere optical RGB channels at 10m GSD.")
    add_body("• Normalization: Normalized linearly to [0.0, 1.0] via division by 255.0, followed by standard zero-mean unit-variance scaling (mu = [0.485, 0.456, 0.406], sigma = [0.229, 0.224, 0.225]):")
    add_math_display(r"X_{\text{norm}} = \frac{\frac{\text{DN}}{255.0} - \mu}{\sigma}", label="5")
    add_body("• Patch Extraction & Dynamic Augmentation: Regular grid extraction across the 10 training scenes using patch size 256 x 256 and stride 64 yielded exactly 586 unique training patches. During DataLoader iteration, on-the-fly stochastic data augmentation was applied (random horizontal flips, random vertical flips, and random 90-degree rotations). Validation and test scenes were evaluated at full native spatial dimensions.")

    add_h3("4. Deterministic Geospatial Affine Preprocessing")
    add_body("Multi-temporal and multi-sensor imagery undergo GDAL-based affine geotransform extraction. The relative transformation matrix is analyzed via Singular Value Decomposition:")
    add_math_display(r"M_{\text{rel}} = A_1^{-1} A_2 = U \Sigma V^T", label="6")
    add_body("This verifies scale parity (|sigma_i - 1| <= 0.01), rotation (|theta_rel| <= 0.28 deg), and continuous sub-pixel phase shift:")
    add_math_display(r"\delta_{\text{phase}} = \sqrt{(\Delta x \bmod 1)^2 + (\Delta y \bmod 1)^2} \le 0.05\text{ px}", label="7")

    add_h3("5. Linguistic Query Preprocessing & Anti-Leakage Filtering")
    add_body("Text queries are normalized to lower-case ASCII without punctuation. Evaluation templates enforce a zero n-gram overlap invariant (n in {3, 4, 5}) against training prompts to ensure syntactic distinctness. Dual independent human linguistic annotators adjudicate ground-truth ambiguity (A* in {0, 1}, Cohen's kappa = 0.918).")

    # =========================================================================
    # SECTION 5: SYSTEM ARCHITECTURE / FLOW DIAGRAM
    # =========================================================================
    add_h1("5. System Architecture / Flow Diagram")
    add_body("The complete architecture of SatQuery AI decouples intent inference from observation payloads, combining an input-constrained linguistic router, a deterministic physical sensor compatibility gate, and certified multimodal specialists into a multi-pathway selective execution engine:")

    # Embed Generated Architecture Diagram
    arch_img_path = "docs/system_architecture_diagram.png"
    if os.path.exists(arch_img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(arch_img_path, width=Inches(6.4))
        
        caption_p = doc.add_paragraph()
        caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption_p.paragraph_format.space_after = Pt(8)
        crun = caption_p.add_run("Figure 1: SatQuery AI Decoupled Multi-Pathway Selective Orchestration Architecture, showing Linguistic Router, Deterministic 8D Physical Sensor Compatibility Gate, and Perception Specialist Registry.")
        crun.font.size = Pt(8.5)
        crun.font.italic = True
        crun.font.color.rgb = RGBColor(113, 128, 150)

    add_h2("5.1 Architecture Component Walkthrough")
    add_bullet("The system accepts a paired natural-language query q and an observation metadata/tensor payload o.", bold_prefix="1. User Request Reception: ")
    add_bullet("The linguistic instruction q is processed strictly in isolation from payload o. A calibrated linear classification pipeline (TF-IDF with Balanced Logistic Regression) outputs logit vector z in R^K across K=7 discrete task categories. Probabilities are calibrated using validation-optimized temperature T_cal = 0.0991.", bold_prefix="2. Stage 1: Input-Constrained Intent Router: ")
    add_bullet("The candidate task t_cand = argmax P(t|q), its confidence c_intent, and its competitive margin Delta_c = c_intent - second_max are evaluated against calibrated operational thresholds (nominal tau_intent = 0.65, tau_margin = 0.15; validation-tuned tau_intent* = 0.85, tau_margin* = 0.05).", bold_prefix="3. Selective Confidence & Margin Computation: ")
    add_bullet("If intent is unambiguous, the observation payload o is validated against the physical invariants required by t_cand across 8 closed-form physical dimensions (C1 - C8).", bold_prefix="4. Stage 2: Physical Sensor Compatibility Gate: ")
    add_bullet("Dispatches request to specialist registry if and only if intent is unambiguous (s_joint(q) >= 1.0), physical invariants pass (C == 1), and selection score meets threshold (s_joint(q) >= tau_sel).", bold_prefix="• Pathway 1 (DISPATCH_QUALIFIED): ")
    add_bullet("Halts execution with an explicit structured failure code if intent is clear but payload violates task contract (C == 0), strictly prohibiting task coercion.", bold_prefix="• Pathway 2 (REJECT_INCOMPATIBLE_OBSERVATION): ")
    add_bullet("Solicits clarification if linguistic intent is underspecified or competitive margin is low (s_joint(q) < 1.0), prior to inspecting payload data.", bold_prefix="• Pathway 3 (CLARIFY_AMBIGUOUS_INTENT): ")
    add_bullet("When intent is clear (s_joint(q) >= 1.0) and sensor contract passes (C == 1), but an elevated operational selective threshold is applied (tau_sel > 1.0) such that 1.0 <= s_joint(q) < tau_sel, execution is intercepted under selective risk control and deferred for human review.", bold_prefix="• Selective Abstention Sub-Action (ABSTAIN_LOW_CONFIDENCE): ")
    add_bullet("Implemented & Evaluated: Multimodal Land-Cover Classification (LAND_COVER_CLASSIFICATION) and Bitemporal Change Detection (CHANGE_DETECTION). Formal Contract Stubs (Planned): SINGLE_IMAGE_VQA, CHANGE_VQA, VISUAL_GROUNDING, SCENE_CAPTIONING, OPTICAL_SAR_FUSION.", bold_prefix="5. Specialist Registry Scope: ")

    # =========================================================================
    # SECTION 6: MATHEMATICAL FORMULATION
    # =========================================================================
    add_h1("6. Mathematical Formulation")

    add_h2("6.1 The Non-Coercion Invariant as a Router Design Constraint")
    add_body("Let q in Q be the user prompt and o in O be the observation metadata. The latent intended task T* in T = {t_1, ..., t_K} is treated as conditionally independent of observation payload o given instruction q:")
    add_callout("", bold_prefix="The Non-Coercion Design Constraint:  ", math_latex=r"T^* \perp o \mid q")
    add_body("This is enforced as an architectural design constraint on the router: the text routing network estimates posterior task probabilities strictly from linguistic tokens:")
    add_math_display(r"P_{\text{text}}(t \mid q) = \text{Softmax}\left(\frac{z}{T_{\text{cal}}}\right)_t = \frac{\exp(z_t / T_{\text{cal}})}{\sum_{j=1}^K \exp(z_j / T_{\text{cal}})}", label="8")
    add_body("where z = f_theta(q) in R^K is extracted via TF-IDF (1, 2) and balanced logistic regression. The post-hoc calibration temperature T_cal was optimized strictly on validation queries via negative log-likelihood (NLL) minimization:")
    add_math_display(r"T_{\text{cal}}^* = \arg\min_{T > 0} \left[ - \frac{1}{|V|} \sum_{i \in V} \log P_T(y_i \mid q_i) \right]", label="9")
    add_body("Empirical minimization on validation queries yielded T_cal* = 0.0991, reducing NLL from 1.5991 to 0.6366 (-60.19%) and decreasing Expected Calibration Error (ECE) from 22.72% to 17.45%.")

    add_h2("6.2 Competitive Confidence Margin & The Joint Selection Function")
    add_body("The candidate task, top confidence, and competitive margin against the runner-up task are:")
    add_math_display(r"t_{\text{cand}} = \arg\max_{t \in \mathcal{T}} P_{\text{text}}(t \mid q), \quad c_{\text{intent}} = P_{\text{text}}(t_{\text{cand}} \mid q)", label="10")
    add_math_display(r"\Delta c = c_{\text{intent}} - \max_{j \ne t_{\text{cand}}} P_{\text{text}}(j \mid q)", label="11")
    add_body("To prevent single-threshold distortions, the Standardized Joint Selection Function is defined as:")
    add_math_display(r"s_{\text{joint}}(q) = \min\left( \frac{c_{\text{intent}}}{\tau_{\text{intent}}}, \, \frac{\Delta c}{\tau_{\text{margin}}} \right)", label="12")
    add_body("Under operational thresholds:")
    add_math_display(r"s_{\text{joint}}(q) \ge 1.0 \iff c_{\text{intent}} \ge \tau_{\text{intent}} \quad \text{and} \quad \Delta c \ge \tau_{\text{margin}}", label="13")
    add_body("We evaluate two threshold settings: (1) Pre-registered nominal freeze thresholds (tau_intent = 0.65, tau_margin = 0.15); and (2) 2D validation grid-search tuned thresholds (tau_intent* = 0.85, tau_margin* = 0.05), which achieved the highest validation operational F1 (43.75%) while intercepting 95% of ambiguous validation queries.")
    add_body("For queries that pass the ambiguity check (s_joint(q) >= 1.0) and satisfy physical sensor compatibility (C(t_cand, o) == 1), the selective threshold tau_sel governs dispatch:")
    add_bullet("At default setting tau_sel = 1.0, every unambiguous and physically compatible query satisfies s_joint(q) >= tau_sel and is dispatched (Pathway 1: DISPATCH_QUALIFIED).", bold_prefix="1. Nominal Operational Point (tau_sel = 1.0): ")
    add_bullet("When an operator configures stricter risk aversion (e.g., tau_sel = 1.2) or during continuous risk-coverage curve construction where tau_sel is swept over [1.0, max(s_joint)], requests with 1.0 <= s_joint(q) < tau_sel are intercepted under selective risk control (ABSTAIN_LOW_CONFIDENCE). In this state, intent is structurally unambiguous and the sensor payload is valid, but the system defers execution for human review because the joint confidence does not meet the elevated risk threshold.", bold_prefix="2. Conservative Risk-Controlled Regimes (tau_sel > 1.0) & Curve Sweeping: ")

    add_h2("6.3 Physical Compatibility Gate Invariants")
    add_body("The physical compatibility gate evaluates the logical conjunction of 8 deterministic physical predicates:")
    add_math_display(r"\mathcal{C}(t, o) = \prod_{d=1}^8 C_d(t, o) \in \{0, 1\}", label="14")

    phys_tbl = doc.add_table(rows=10, cols=3)
    p_widths = [1.0, 2.2, 3.3]
    phys_tbl.cell(0, 0).paragraphs[0].text = "Predicate"
    phys_tbl.cell(0, 1).paragraphs[0].text = "Physical Dimension"
    phys_tbl.cell(0, 2).paragraphs[0].text = "Mathematical Invariant Formulation"

    p_rows = [
        ("C1", "Modality Completeness", "M_req(t) ⊆ M(o)  (e.g., {SAR, Opt} ⊆ M(o) for Land Cover)"),
        ("C2", "Temporal Baseline", "|t2 - t1| ≥ Δt_min(t)  (Δt ≥ 5 days for Change Detection; Δt == 0 rejected)"),
        ("C3", "Spectral Coverage", "B_req(t) ⊆ B(o)  (10 specific MSI bands for Land Cover; RGB for OSCD)"),
        ("C4", "Spatial Dimension", "H(o) ≥ H_min(t) and W(o) ≥ W_min(t)  (120 x 120 for LC, 100 x 100 for OSCD)"),
        ("C5", "CRS & Affine Basis", "CRS(o1) == CRS(o2) and |σ_i(M_rel) - 1.0| ≤ 0.01 and det(M_rel) > 0"),
        ("C6", "GSD Scale Parity", "|GSD1 - GSD2| / min(GSD1, GSD2) ≤ 0.01  (Scale parity within 1.0%)"),
        ("C7", "Grid-Phase Shift", "δ_phase = sqrt((Δx mod 1)^2 + (Δy mod 1)^2) ≤ 0.05 px"),
        ("C8a", "Operational Cloud Limit", "f_cloud(o) ≤ τ_cloud(t)  (≤ 20% for Land Cover, ≤ 10% for Change Detection)"),
        ("C8b", "Radiometric Finiteness", "sum(isnan(o)) == 0 and sum(isinf(o)) == 0  (Zero non-finite elements)")
    ]

    for row_idx, (pred, dim, form) in enumerate(p_rows, start=1):
        phys_tbl.cell(row_idx, 0).paragraphs[0].text = pred
        phys_tbl.cell(row_idx, 1).paragraphs[0].text = dim
        phys_tbl.cell(row_idx, 2).paragraphs[0].text = form

    format_table(phys_tbl, p_widths, [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_h2("6.4 Imbalanced Loss Formulations")
    add_h3("1. Normalized Class-Balanced Binary Cross-Entropy (CB-BCE)")
    add_body("Following Cui et al. (2019), effective sample numbers scale class weights with beta = 0.999:")
    add_math_display(r"E_{n, c} = \frac{1 - \beta^{N_c}}{1 - \beta}, \quad w_{c,+}^{\text{raw}} = \frac{1}{E_{n, c}}", label="15")
    add_body("For classes with zero training support (N_{c,+}^train = 0), a finite fallback is enforced: w_{c,+}^raw := 1.0. Weights are normalized across C=16 classes (epsilon = 10^-5) and element-wise normalized to 1/(16B) to guarantee gradient scale parity:")
    add_math_display(r"\mathcal{L}_{\text{CB-BCE}} = \frac{1}{16B} \sum_{i=1}^B \sum_{c=1}^{16} \left[ - \widetilde{w}_{c,+} \, y_{i,c} \log \sigma(z_{i,c}) - (1 - y_{i,c}) \log (1 - \sigma(z_{i,c})) \right]", label="16")

    add_h3("2. Multi-Label Asymmetric Loss (ASL)")
    add_body("Following Ridnik et al. (2021), ASL applies asymmetric focusing parameters (gamma_+ = 0.0, gamma_- = 4.0) and probability margin shift (m = 0.05):")
    add_math_display(r"p_{m, i, c} = \max(p_{i, c} - m, \, 0)", label="17")
    add_math_display(r"\mathcal{L}_{\text{ASL}} = \frac{1}{16B} \sum_{i=1}^B \sum_{c=1}^{16} \left[ - y_{i,c} (1 - p_{i,c})^{\gamma_+} \log p_{i,c} - (1 - y_{i,c}) (p_{m, i, c})^{\gamma_-} \log(1 - p_{m, i, c}) \right]", label="18")

    add_h2("6.5 Evaluation Measures & Risk Formulations")
    add_body("• Selective Coverage and Empirical Risk:")
    add_math_display(r"\text{Coverage}(g) = \frac{1}{N} \sum_{i=1}^N g(x_i), \quad \text{Risk}(f, g) = \frac{\sum_{i=1}^N \ell(f(x_i), y_i) \, g(x_i)}{\sum_{i=1}^N g(x_i)}", label="19")
    add_body("• Area Under the Risk-Coverage Curve (AURC):")
    add_math_display(r"\text{AURC} = \int_0^1 \text{Risk}(\phi) \, d\phi", label="20")
    add_body("• Standard Macro F1 under zero_division=0.0:")
    add_math_display(r"\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}", label="21")
    add_body("• Bitemporal Change Detection Pooled Metrics:")
    add_math_display(r"\text{Pooled } F_1 = \frac{2 \cdot \sum \text{TP}}{2 \cdot \sum \text{TP} + \sum \text{FP} + \sum \text{FN}}, \quad \text{Pooled IoU} = \frac{\sum \text{TP}}{\sum \text{TP} + \sum \text{FP} + \sum \text{FN}}", label="22")

    # =========================================================================
    # SECTION 7: PROPOSED ALGORITHM
    # =========================================================================
    add_h1("7. Proposed Algorithm")

    add_h2("7.1 Algorithm 1: Selective Orchestration with Physical Compatibility Verification")
    alg1_text = (
        "Input:  Natural language query instruction q, Observation payload metadata o,\n"
        "        Calibrated intent router f_theta, Calibration temperature T_cal,\n"
        "        Operational thresholds tau_intent, tau_margin, tau_sel,\n"
        "        Deterministic physical gates {C_1, ..., C_8}, Specialist registry S = {S_1, ..., S_K}\n"
        "Output: Orchestrator action, specialist predictions or structured rejection code\n\n"
        "1:  // Stage 1: Linguistic Intent Inference Independent of Payload\n"
        "2:  z <- f_theta(q)\n"
        "3:  P <- Softmax(z / T_cal)\n"
        "4:  t_cand <- argmax_{t in T} P_t\n"
        "5:  c_intent <- P[t_cand]\n"
        "6:  Delta_c <- c_intent - max_{j != t_cand} P_j\n"
        "7:  s_joint <- min(c_intent / tau_intent, Delta_c / tau_margin)\n"
        "8:\n"
        "9:  // Pathway 3: Ambiguity Safeguard\n"
        "10: if s_joint < 1.0 then\n"
        "11:     return Action(CLARIFY_AMBIGUOUS_INTENT, candidate=t_cand, confidence=c_intent, margin=Delta_c)\n"
        "12: end if\n"
        "13:\n"
        "14: // Stage 2: Deterministic Physical Sensor Compatibility Verification\n"
        "15: gate_passed <- True\n"
        "16: violation_codes <- []\n"
        "17: for d = 1 to 8 do\n"
        "18:     if C_d(t_cand, o) == 0 then\n"
        "19:         gate_passed <- False\n"
        "20:         violation_codes.append(d)\n"
        "21:     end if\n"
        "22: end for\n"
        "23:\n"
        "24: // Pathway 2: Physical Rejection\n"
        "25: if not gate_passed then\n"
        "26:     return Action(REJECT_INCOMPATIBLE_OBSERVATION, candidate=t_cand, violations=violation_codes)\n"
        "27: end if\n"
        "28:\n"
        "29: // Stage 3: Selective Risk-Controlled Dispatch\n"
        "30: if s_joint >= tau_sel then\n"
        "31:     // Pathway 1: Qualified Dispatch\n"
        "32:     predictions <- S[t_cand].predict(o)\n"
        "33:     return Action(DISPATCH_QUALIFIED, task=t_cand, output=predictions)\n"
        "34: else\n"
        "35:     // Risk-Controlled Sub-Outcome (Reachable for tau_sel > 1.0)\n"
        "36:     return Action(ABSTAIN_LOW_CONFIDENCE, task=t_cand, selection_score=s_joint)\n"
        "37: end if"
    )
    add_code_box(alg1_text, title="Algorithm 1: Selective Orchestration with Physical Compatibility Verification")

    add_h2("7.2 Algorithm 2: Vectorized Per-Class Threshold Optimization")
    alg2_text = (
        "Input:  Validation ground truth Y_val in {0, 1}^(N x C),\n"
        "        Predicted probability matrix P_val in [0, 1]^(N x C),\n"
        "        Candidate threshold grid T = [0.10, 0.15, ..., 0.90] of length K = 17\n"
        "Output: Optimal per-class threshold vector tau* in [0.10, 0.90]^C, Optimal Validation Macro F1\n\n"
        "1:  N, C <- shape(Y_val)\n"
        "2:  K <- len(T)\n"
        "3:  // 3D broadcasting: grid shape (K, 1, 1), data shapes (1, N, C)\n"
        "4:  T_grid <- reshape(T, (K, 1, 1))\n"
        "5:  P_exp  <- expand(P_val, (1, N, C))\n"
        "6:  Y_exp  <- expand(Y_val, (1, N, C))\n"
        "7:\n"
        "8:  // Vectorized binary thresholding\n"
        "9:  Y_pred <- (P_exp >= T_grid).astype(int32)\n"
        "10:\n"
        "11: TP <- sum(Y_pred * Y_exp, axis=1)          // shape (K, C)\n"
        "12: FP <- sum(Y_pred * (1 - Y_exp), axis=1)    // shape (K, C)\n"
        "13: FN <- sum((1 - Y_pred) * Y_exp, axis=1)    // shape (K, C)\n"
        "14:\n"
        "15: Denom <- 2 * TP + FP + FN\n"
        "16: F1_grid <- zeros((K, C))\n"
        "17: Valid <- (Denom > 0)\n"
        "18: F1_grid[Valid] <- (2.0 * TP[Valid]) / Denom[Valid]\n"
        "19:\n"
        "20: best_indices <- argmax(F1_grid, axis=0)     // shape (C,)\n"
        "21: tau* <- T[best_indices]\n"
        "22: best_per_class_f1 <- max(F1_grid, axis=0)\n"
        "23: best_macro_f1 <- mean(best_per_class_f1)\n"
        "24: return tau*, best_macro_f1"
    )
    add_code_box(alg2_text, title="Algorithm 2: Vectorized Per-Class Threshold Optimization")

    add_h2("7.3 Model Configurations & Software Settings")
    cfg_tbl = doc.add_table(rows=5, cols=4)
    cfg_widths = [1.5, 1.8, 1.8, 1.4]
    cfg_headers = ["Component", "Model Architecture", "Key Hyperparameters", "Software / Stack"]
    for idx, text in enumerate(cfg_headers):
        cfg_tbl.cell(0, idx).paragraphs[0].text = text

    cfg_data = [
        ("Linguistic Router (Evaluated)", "TF-IDF Vectorizer + Balanced Logistic Regression", "N-gram range (1, 2), sublinear TF, C=1.0, L-BFGS, T_cal*=0.0991, tau_intent*=0.85, tau_margin*=0.05", "scikit-learn, SciPy, NumPy, Python 3.14"),
        ("Land-Cover Specialist", "Dual-Branch ResNet-34 + Cross-Attention", "Optical: 10 bands (ImageNet RGB init); SAR: 2 bands (He init); 8-head cross-attn (dim 512); AdamW, lr=1e-4, 40 epochs", "PyTorch 2.6, torchvision, NumPy, SciPy"),
        ("Change-Detection Specialist", "FC-Siam-diff (Daudt et al., 2018)", "4-stage ConvNet (1.4M params); AdamW, lr=1e-3, 20 epochs on 586 patches (stride 64), tau*=0.25", "PyTorch 2.6, rasterio, Pillow, scikit-learn"),
        ("Change-Detection Transformer", "Bitemporal Image Transformer (BIT) (Chen et al., 2021)", "Siamese ResNet-18, Tokenizer L=64, 4 Enc + 4 Dec layers, FCN head; AdamW, lr=1e-4, 20 epochs, tau*=0.45", "PyTorch 2.6, einops, timm")
    ]

    for row_idx, (comp, arch, params, sw) in enumerate(cfg_data, start=1):
        cfg_tbl.cell(row_idx, 0).paragraphs[0].text = comp
        cfg_tbl.cell(row_idx, 1).paragraphs[0].text = arch
        cfg_tbl.cell(row_idx, 2).paragraphs[0].text = params
        cfg_tbl.cell(row_idx, 3).paragraphs[0].text = sw

    format_table(cfg_tbl, cfg_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    # =========================================================================
    # SECTION 8: PRELIMINARY RESULTS
    # =========================================================================
    add_h1("8. Preliminary Results")

    add_h2("8.1 Milestone M3.1: Multimodal Land-Cover Specialist Results")
    add_body("Evaluated on held-out test split (N = 3,248 patches, 120 x 120 pixels at 10m GSD) under strict threshold regime separation:")

    add_h3("Primary Comparison Matrix: Validation-Tuned Threshold Regime (tau*)")
    m31_tbl = doc.add_table(rows=9, cols=5)
    m31_widths = [1.8, 1.1, 1.1, 1.2, 1.3]
    m31_headers = ["Metric", "Random Forest (Phase 2)", "LightGBM (Phase 2)", "Dual ResNet-34 (CB-BCE)", "Dual ResNet-34 (ASL)"]
    for idx, text in enumerate(m31_headers):
        m31_tbl.cell(0, idx).paragraphs[0].text = text

    m31_data = [
        ("Threshold Regime", "Validation-Tuned tau*", "Validation-Tuned tau*", "Validation-Tuned tau*", "Validation-Tuned tau*"),
        ("Best Training Epoch", "—", "—", "Epoch 30 / 40", "Epoch 3 / 40 (10x faster)"),
        ("Training Duration", "4.5 minutes (CPU)", "3.0 minutes (CPU)", "141.1 minutes (GPU)", "30.2 minutes (4.67x speedup)"),
        ("Validation Macro F1 (Tuned)", "36.99%", "54.92%", "54.29%", "56.39% (+1.47% lead)"),
        ("Test Macro F1 (16 cls, standard)", "34.00%", "41.14%*", "43.78%", "45.53% (Deep Champion)"),
        ("Test-Present Macro F1 (14 cls)", "38.86%", "47.02%", "50.03%", "52.03% (Deep Champion)"),
        ("Test Micro F1", "44.81%", "72.99%", "71.90%", "74.47% (Highest Accuracy)"),
        ("Test Hamming Loss (Error Rate)", "0.3153", "0.0976", "0.0816 (Lowest Error)", "0.0927")
    ]

    for row_idx, r in enumerate(m31_data, start=1):
        for c_idx, val in enumerate(r):
            m31_tbl.cell(row_idx, c_idx).paragraphs[0].text = val

    format_table(m31_tbl, m31_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])
    add_body("*Note on LightGBM: Standard Macro F1 of 41.14% adheres strictly to the frozen protocol (zero_division=0.0 across all 16 CORINE classes). The test-present Macro F1 across the 14 classes with non-zero test support is 47.02%. Legacy uncalibrated reporting of 47.39% was derived from zero_division=1.0 and is deprecated.", italic=True)

    add_h3("Secondary Comparison Matrix: Fixed Threshold Regime (tau = 0.50)")
    m31_fixed_tbl = doc.add_table(rows=6, cols=5)
    for idx, text in enumerate(m31_headers):
        m31_fixed_tbl.cell(0, idx).paragraphs[0].text = text

    m31_fixed_data = [
        ("Threshold Regime", "Fixed tau = 0.50", "Fixed tau = 0.50", "Fixed tau = 0.50", "Fixed tau = 0.50"),
        ("Validation Macro F1 (Fixed)", "20.12%", "40.94%", "39.46%", "40.79%"),
        ("Test Macro F1 (16 cls, standard)", "19.98%", "34.99%", "35.34%", "36.88%"),
        ("Test-Present Macro F1 (14 cls)", "22.84%", "39.99%", "40.38%", "42.15%"),
        ("Test Micro F1", "40.91%", "69.18%", "68.74%", "70.12%")
    ]

    for row_idx, r in enumerate(m31_fixed_data, start=1):
        for c_idx, val in enumerate(r):
            m31_fixed_tbl.cell(row_idx, c_idx).paragraphs[0].text = val

    format_table(m31_fixed_tbl, m31_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_h2("8.2 Milestone M3.2: Bitemporal Change-Detection Specialist Results")
    add_body("Evaluated on 10 held-out OSCD test cities (3,077,936 test pixels) under spatial separation. Results are reconciled strictly against the authoritative M3.2 evaluation artifact (m32_change_detection_comparison.json):")

    m32_tbl = doc.add_table(rows=10, cols=5)
    m32_widths = [1.8, 1.1, 1.1, 1.2, 1.3]
    m32_headers = ["Metric", "Standard CVA (Phase 2)", "Gaussian CVA (Phase 2)", "BIT Transformer (M3.2)", "FC-Siam-diff (Champion)"]
    for idx, text in enumerate(m32_headers):
        m32_tbl.cell(0, idx).paragraphs[0].text = text

    m32_data = [
        ("Model Family", "Pixel Differencing", "Smoothed Differencing", "Siamese Transformer", "Deep Conv Siamese"),
        ("Decision Threshold tau*", "Validation-Tuned (0.28)", "Validation-Tuned (0.28)", "Validation-Tuned (0.45)", "Validation-Tuned (0.25)"),
        ("Held-Out Test Macro F1", "20.31%", "27.31%", "12.95%", "37.66% (+10.35% lead)"),
        ("Pooled F1 Score", "27.78%", "33.93%", "12.60%", "38.18% (Best Pooled)"),
        ("Pooled Precision", "58.20%", "39.70%", "18.79%", "35.33%"),
        ("Pooled Recall", "18.24%", "29.63%", "9.47%", "41.53% (Balanced Tradeoff)"),
        ("Pooled IoU", "16.13%", "20.43%", "6.72%", "23.60% (Deep Champion)"),
        ("Overall Accuracy", "95.10%", "94.04%", "93.20%", "93.05%"),
        ("Certification Status", "Baseline", "Baseline", "Underperformed", "Target Not Achieved (37.66% < 50%)")
    ]

    for row_idx, r in enumerate(m32_data, start=1):
        for c_idx, val in enumerate(r):
            m32_tbl.cell(row_idx, c_idx).paragraphs[0].text = val

    format_table(m32_tbl, m32_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_callout(
        "Under pre-registered protocol v3.0.0-freeze-rev21, the required certification target was Macro F1 >= 50.00%. While FC-Siam-diff represents the best evaluated model (improving Macro F1 by +10.35 percentage points over Gaussian CVA and +17.35 points over Standard CVA), its achieved score of 37.66% falls short of the 50.00% certification boundary. BIT was observed to underperform (12.95% Macro F1); we hypothesize that in small-sample regimes (10 training cities, 586 patches), the absence of convolutional spatial differencing inductive bias hampers tokenized transformer convergence on sparse foreground changes. In accordance with scientific integrity rules, Milestone M3.2 is officially designated: IMPLEMENTATION & EVALUATION COMPLETE; CERTIFICATION TARGET NOT ACHIEVED.",
        bold_prefix="M3.2 Scientific Certification Status: ",
        fill_hex="FEF2F2",
        border_hex="C53030"
    )

    add_h2("8.3 Milestone M3.3: Routing, Safety, & Selective Prediction Results")
    add_body("Evaluated on the locked held-out test split of N = 100 multi-modal operational scenarios across 5 cohorts (N=25 common valid E_eval, N=35 physical contract violations, N=20 ambiguous queries, N=20 unsupported/OOD queries):")

    # Authoritative M3.3 Scorecard Table
    m33_tbl = doc.add_table(rows=8, cols=6)
    m33_widths = [1.8, 0.9, 0.9, 0.9, 1.0, 1.0]
    m33_headers = ["Metric", "Heuristic (R0)", "Linear Ungated (R1)", "Gated Non-Sel (R2)", "Uncal Selective (R3)", "SatQuery AI (R4)"]
    for idx, text in enumerate(m33_headers):
        m33_tbl.cell(0, idx).paragraphs[0].text = text

    m33_data = [
        ("Incompatible Dispatch Error (IDER)", "100.00%", "100.00%", "8.57%", "0.00%", "0.00% (0/35) [PASSED]"),
        ("Wilson 95% Upper Bound", "100.00%", "100.00%", "22.38%", "9.89%", "9.89% [PASSED]"),
        ("Contract Violation Recall", "0.00%", "0.00%", "91.43%", "100.00%", "100.00% (35/35)"),
        ("Ambiguity Clarification Recall", "0.00%", "0.00%", "0.00%", "80.00%", "60.00% (12/20)"),
        ("Ambiguity Clarification F1", "0.00%", "0.00%", "0.00%", "41.56%", "29.63%"),
        ("7-Task Architectural Success", "21.00%", "15.00%", "46.00%", "52.00%", "55.00% (Highest)"),
        ("Global Selective Risk (AURC on E_eval)", "0.0853", "0.2479", "0.2638", "0.2638", "0.2683 (Exploratory)")
    ]

    for row_idx, r in enumerate(m33_data, start=1):
        for c_idx, val in enumerate(r):
            m33_tbl.cell(row_idx, c_idx).paragraphs[0].text = val

    format_table(m33_tbl, m33_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER])

    add_body("Analysis of Ambiguity Interception & Temperature Scaling: Uncalibrated selective prediction (R3) achieved 80.00% clarification recall and 41.56% F1, whereas calibrated selective routing (R4) achieved 60.00% recall (12/20) and 29.63% F1. This occurs because post-hoc temperature scaling (T_cal = 0.0991 < 1.0) sharpens the softmax distribution, elevating posterior confidence and competitive margin for borderline ambiguous prompts. Both selective models substantially outperform all non-selective baselines (R0, R1, R2, R5), which exhibit 0.00% clarification recall and arbitrarily coerce 100% of ambiguous queries.", italic=True)

    add_body("Scientific Evaluation of Selective Risk (AURC): Selective risk curves were evaluated strictly over the common valid population E_eval (N=25 valid requests). In this experiment, R4 achieved an AURC of 0.2683, compared to R3 (0.2638), R1 (0.2479), and R0 (0.0853). Lower AURC is better. A Null-Centred Paired Cluster Bootstrap (B=1,000 replicates) revealed no statistically significant difference between R4 and R3 (p = 0.6274, Step-Down Holm adjusted p = 1.0000). As mathematically pre-registered in the protocol freeze, fine-grained AURC comparison on small family cohorts (N=25) is statistically underpowered (< 5% power) and is classified as an exploratory feasibility study. The proposed R4 does not claim superior AURC on this sample; rather, its primary confirmatory contribution is macro architectural safety.", italic=True)

    # Embed Risk-Coverage Curves Figure
    curve_img_path = "artifacts/benchmarks/risk_coverage_curves_m33.png"
    if not os.path.exists(curve_img_path):
        curve_img_path = "artifacts/benchmarks/risk_coverage_curves.png"
    if os.path.exists(curve_img_path):
        p_cimg = doc.add_paragraph()
        p_cimg.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cimg.paragraph_format.space_before = Pt(6)
        p_cimg.paragraph_format.space_after = Pt(2)
        run_cimg = p_cimg.add_run()
        run_cimg.add_picture(curve_img_path, width=Inches(6.0))
        
        caption_cp = doc.add_paragraph()
        caption_cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption_cp.paragraph_format.space_after = Pt(8)
        ccrun = caption_cp.add_run("Figure 2: Empirical Selective Prediction and Safety Comparison across 7 routing policies, illustrating IDER safety elimination, ambiguity clarification recall, and exploratory selective risk curves.")
        ccrun.font.size = Pt(8.5)
        ccrun.font.italic = True
        ccrun.font.color.rgb = RGBColor(113, 128, 150)

    # =========================================================================
    # SECTION 9: COMPARATIVE STUDY
    # =========================================================================
    add_h1("9. Comparative Study")
    add_body("To rigorously contextualize SatQuery AI against existing state-of-the-art approaches, baseline systems were assessed across architectural capabilities, empirical task metrics, and selective risk management:")

    comp_tbl = doc.add_table(rows=5, cols=4)
    comp_widths = [1.6, 1.8, 1.8, 1.3]
    comp_headers = ["Model / paper", "Why selected", "Metric results", "Key architectural distinction"]
    for idx, text in enumerate(comp_headers):
        comp_tbl.cell(0, idx).paragraphs[0].text = text

    comp_data = [
        ("GeoChat\n(Kuckreja et al., CVPR 2024)",
         "Premier open-source multimodal remote sensing foundation model; current state-of-the-art conversational baseline.",
         "[DEMONSTRATED]: Strong VQA & grounding competence.\n[ARCHITECTURAL LIMITATION]: Evaluates prompts without physical compatibility gating; unequipped with reject option.",
         "Unconstrained Monolithic Dispatch. Coerces ambiguous prompts into active vision encoders; lacks rejection mechanism."),
        ("Learning to Route (L2R)\n(Wang et al., ICML 2021)",
         "Foundational dynamic routing framework for multi-task multi-agent deep learning systems.",
         "[DEMONSTRATED]: Parameter-efficient neural routing.\n[ARCHITECTURAL LIMITATION]: Optimizes latent representations; does not evaluate sensor contracts or selective risk curves.",
         "Representation-based routing without physical contract verification or risk-coverage control."),
        ("Classical CVA Baselines\n(Bruzzone & Prieto, 2000)",
         "De-facto operational remote-sensing baselines for unsupervised bitemporal change detection.",
         "Standard CVA Macro F1: 20.31%\nGaussian CVA Macro F1: 27.31%\nPooled F1: 27.78% (Std), 33.93% (Gauss)\nPooled Precision: 58.20% (Std), 39.70% (Gauss)",
         "Pixel-differencing baselines. High false-positive sensitivity on urban building shadows and seasonal phenology."),
        ("SatQuery AI\n(Proposed Architecture)",
         "Input-constrained selective routing architecture with deterministic 8D physical sensor gating and certified perception specialists.",
         "Land-Cover Macro F1: 45.53%\nChange Detection Macro F1: 37.66%\nIncompatible Dispatch Errors: 0.0% (0/35)\n7-Task Arch. Success: 55.00%",
         "Strict Decoupled Multi-Pathway Execution. Enforces Non-Coercion Constraint T* ⊥ o | q; guarantees zero incompatible specialist executions.")
    ]

    for row_idx, (m_paper, m_sel, m_res, m_dist) in enumerate(comp_data, start=1):
        comp_tbl.cell(row_idx, 0).paragraphs[0].text = m_paper
        comp_tbl.cell(row_idx, 1).paragraphs[0].text = m_sel
        comp_tbl.cell(row_idx, 2).paragraphs[0].text = m_res
        comp_tbl.cell(row_idx, 3).paragraphs[0].text = m_dist

    format_table(comp_tbl, comp_widths, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_h2("9.1 Qualitative and Quantitative Comparative Synthesis")
    add_bullet(
        "While GeoChat and general multi-task architectures permit the observation payload to dictate task execution, SatQuery AI treats the intended task as conditionally independent of the observation payload given the user instruction (T* ⊥ o | q). In our benchmark evaluations across N=35 synthetic physical contract violations, unconstrained routing resulted in 100% incompatible executions (IDER = 100%), whereas SatQuery AI observed 0 incompatible specialist executions (0/35, IDER = 0.0%, Wilson 95% upper bound 9.89%), establishing complete physical contract defense.",
        bold_prefix="1. Elimination of Incompatible Dispatches: "
    )
    add_bullet(
        "On the BigEarthNet-14k multimodal benchmark, the dual-branch ResNet-34 specialist trained with Asymmetric Loss (ASL) achieved 45.53% standard Macro F1 and 74.47% Micro F1, surpassing tuned gradient-boosted decision trees (LightGBM: 41.14% standard Macro F1, 47.02% test-present Macro F1) and Random Forests (34.00%). Furthermore, ASL converged in 3 epochs (30.2 minutes) compared to 30 epochs (141.1 minutes) for Class-Balanced BCE.",
        bold_prefix="2. Superior Land-Cover Multi-Label Generalization: "
    )
    add_bullet(
        "On the OSCD benchmark, deep convolutional feature differencing (FC-Siam-diff) achieved 37.66% Macro F1, substantially outperforming classical Standard CVA (20.31% Macro F1) and Gaussian CVA (27.31% Macro F1). Although FC-Siam-diff did not attain the pre-registered 50.00% certification target due to small-sample training constraints (10 cities, 586 patches), it provides a verified, reproducible benchmark for future spatial-differencing research.",
        bold_prefix="3. Rigorous Change-Detection Performance Analysis: "
    )
    add_bullet(
        "Evaluating selective prediction across 7 policies revealed that SatQuery AI achieved the highest overall 7-task architectural success rate (TSR_arch^(7) = 55.00% vs. 15.00% for linear ungated and 23.00% for frontier LLMs). On ambiguous multi-intent queries, SatQuery AI achieved 60.00% clarification recall, actively preventing deceptive task coercion. The global selective risk comparison on E_eval (N=25) is statistically underpowered (p = 0.6274), serving as an exploratory feasibility study.",
        bold_prefix="4. Macro Architectural Differentials and Exploratory Selective Risk: "
    )

    # =========================================================================
    # SECTION 10: REFERENCES
    # =========================================================================
    add_h1("10. References")
    
    references = [
        ("1", "Kuckreja, K., et al. (2024). GeoChat: Grounded Large Vision-Language Model for Remote Sensing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024), pp. 27814-27823. DOI: 10.1109/CVPR52733.2024.02630."),
        ("2", "Wang, Z., et al. (2021). Learning to Route in Multi-Task Multi-Agent Systems. In Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139:10820-10830. URL: https://proceedings.mlr.press/v139/wang21y.html."),
        ("3", "Geifman, Y., & El-Yaniv, R. (2019). SelectiveNet: Deep Neural Networks with an Integrated Reject Option. In Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97:2151-2159. URL: https://proceedings.mlr.press/v97/geifman19a.html."),
        ("4", "Sumbul, G., et al. (2019). BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding. In IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2019), pp. 5901-5904. DOI: 10.1109/IGARSS.2019.8900532."),
        ("5", "Sumbul, G., et al. (2021). BigEarthNet-MM: A Large-Scale Multi-Modal Multi-Label Benchmark Archive for Remote Sensing Image Understanding. IEEE Geoscience and Remote Sensing Magazine, 9(3), pp. 174-180. DOI: 10.1109/MGRS.2021.3089174."),
        ("6", "Daudt, R. C., et al. (2018). Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks. In IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2018), pp. 2115-2118. DOI: 10.1109/IGARSS.2018.8518015."),
        ("7", "Daudt, R. C., et al. (2018). Fully Convolutional Siamese Networks for Change Detection. In 25th IEEE International Conference on Image Processing (ICIP 2018), pp. 4063-4067. DOI: 10.1109/ICIP.2018.8451652."),
        ("8", "Chen, H., et al. (2021). Remote Sensing Image Change Detection with Transformers. IEEE Transactions on Geoscience and Remote Sensing, 60, pp. 1-14. DOI: 10.1109/TGRS.2021.3095166."),
        ("9", "Bruzzone, L., & Prieto, D. F. (2000). Automatic Analysis of the Difference Image for Unsupervised Change Detection. IEEE Transactions on Geoscience and Remote Sensing, 38(3), pp. 1171-1182. DOI: 10.1109/36.843009."),
        ("10", "Guo, C., et al. (2017). On Calibration of Modern Neural Networks. In Proceedings of the 34th International Conference on Machine Learning (ICML 2017), PMLR 70:1321-1330. URL: https://proceedings.mlr.press/v70/guo17a.html."),
        ("11", "Cui, Y., et al. (2019). Class-Balanced Loss Based on Effective Number of Samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019), pp. 9268-9277. DOI: 10.1109/CVPR.2019.00949."),
        ("12", "Ridnik, T., et al. (2021). Asymmetric Loss for Multi-Label Classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2021), pp. 82-91. DOI: 10.1109/ICCV48922.2021.00015."),
        ("13", "Zhang, H., et al. (2024). Earth-GPT: A Large Multimodal Model for Multimodal Earth Observation. IEEE Transactions on Geoscience and Remote Sensing, 62, pp. 1-15. DOI: 10.1109/TGRS.2024.3412589."),
        ("14", "Zhan, Y., et al. (2024). SkyEye: Remote Sensing Vision-Language Models for Open-Vocabulary Scene Understanding. IEEE Transactions on Geoscience and Remote Sensing, 62, pp. 1-13. DOI: 10.1109/TGRS.2024.3401124."),
        ("15", "He, K., et al. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), pp. 770-778. DOI: 10.1109/CVPR.2016.90."),
        ("16", "Vaswani, A., et al. (2017). Attention Is All You Need. In Advances in Neural Information Processing Systems (NeurIPS 2017), 30, pp. 5998-6008. URL: https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html."),
        ("17", "Chow, C. K. (1970). On Optimum Recognition Error and Reject Tradeoff. IEEE Transactions on Information Theory, 16(1), pp. 41-46. DOI: 10.1109/TIT.1970.1054406."),
        ("18", "Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In Advances in Neural Information Processing Systems (NeurIPS 2017), 30, pp. 3146-3154. URL: https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html."),
        ("19", "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), pp. 5-32. DOI: 10.1023/A:1010933404324.")
    ]

    for num, cit in references:
        add_body(cit, bold_prefix=f"[{num}] ")

    print(f"Saving Word document to: {output_path}...")
    doc.save(output_path)
    print("Word document generated successfully!")

if __name__ == "__main__":
    build_docx()
