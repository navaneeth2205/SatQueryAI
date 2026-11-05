"""
Script: generate_methodology_pdf.py
Author: Lead Research Verification Agent
Purpose: Generates a professionally styled, publication-grade academic PDF
         from METHODOLOGY.md following the 10-section structure.
"""

import sys
import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "SatQuery AI: Research Methodology Report")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "Machine Learning Course Project")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, page_text)
        self.drawString(54, 32, "CONFIDENTIAL & AUTHORITATIVE — Frozen Protocol v3.0.0-freeze-rev21")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 44, letter[0] - 54, 44)
        
        self.restoreState()

def clean_text(text):
    """Clean markdown formatting for ReportLab Paragraphs."""
    # Convert <br> to <br/>
    text = re.sub(r'<br\s*/?>', '<br/>', text, flags=re.IGNORECASE)
    
    # Handle &amp; and ampersands
    text = re.sub(r'&(?!amp;|lt;|gt;|quot;|#\d+;)', '&amp;', text)
    
    # Convert math symbols to unicode / readable text
    text = text.replace(r"\perp", "⟂")
    text = text.replace(r"\mid", "|")
    text = text.replace(r"\ge", "≥")
    text = text.replace(r"\le", "≤")
    text = text.replace(r"\in", "∈")
    text = text.replace(r"\times", "×")
    text = text.replace(r"\approx", "≈")
    text = text.replace(r"\to", "→")
    text = text.replace(r"\prod", "∏")
    text = text.replace(r"\sum", "∑")
    text = text.replace(r"\Delta", "Δ")
    text = text.replace(r"\sigma", "σ")
    text = text.replace(r"\mu", "μ")
    text = text.replace(r"\tau", "τ")
    text = text.replace(r"\theta", "θ")
    text = text.replace(r"\Phi", "Φ")
    text = text.replace(r"\mathcal{C}", "C")
    text = text.replace(r"\mathcal{T}", "T")
    text = text.replace(r"\mathcal{O}", "O")
    text = text.replace(r"\mathcal{Q}", "Q")
    text = text.replace(r"\mathcal{M}", "M")
    text = text.replace(r"\mathcal{B}", "B")
    text = text.replace(r"\mathcal{R}", "R")
    text = text.replace(r"\mathbb{R}", "R")
    text = text.replace(r"\mathbf{M}", "M")
    text = text.replace(r"\mathbf{A}", "A")
    text = text.replace(r"\mathbf{U}", "U")
    text = text.replace(r"\mathbf{V}", "V")
    text = text.replace(r"\mathbf{\Sigma}", "Σ")
    text = text.replace(r"\text{", "").replace(r"\operatorname{", "")
    text = text.replace(r"\hat{", "").replace(r"\tilde{", "")
    text = text.replace(r"\vec{", "")
    text = text.replace("$", "")
    
    # Convert markdown links [Text](URL) -> Text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Bold & Italic markdown conversion
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.*?)`', r'<font face="Courier" color="#C53030"><b>\1</b></font>', text)
    return text

def build_pdf(md_path, pdf_path):
    print(f"Reading {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Academic Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=6,
        alignment=1 # Center
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2B6CB0'),
        spaceAfter=14,
        alignment=1
    )
    
    h1_style = ParagraphStyle(
        'AcademicH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'AcademicH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2C5282'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'AcademicH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#2D3748'),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'AcademicBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=5
    )
    
    bullet_style = ParagraphStyle(
        'AcademicBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'CalloutBody',
        parent=body_style,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1A202C'),
        fontName='Helvetica-Oblique'
    )
    
    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.0,
        leading=9.5,
        textColor=colors.HexColor('#1A202C')
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.0,
        leading=9.0,
        textColor=colors.HexColor('#1A202C')
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold'
    )

    story = []
    
    lines = md_content.split("\n")
    i = 0
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Code block handling
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "<br/>".join([clean_text(cl).replace(" ", "&nbsp;") for cl in code_lines])
                code_p = Paragraph(code_text, code_style)
                code_table = Table([[code_p]], colWidths=[letter[0] - 108])
                code_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                    ('TOPPADDING', (0,0), (-1,-1), 4),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                    ('LEFTPADDING', (0,0), (-1,-1), 8),
                    ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ]))
                story.append(code_table)
                story.append(Spacer(1, 4))
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue
            
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
            
        # Table handling
        if stripped.startswith("|") and stripped.endswith("|"):
            # Check if separator row like | :--- | :---: |
            if re.match(r'^\|[\s:-|]+\|$', stripped):
                i += 1
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            table_rows.append(cells)
            in_table = True
            i += 1
            continue
        else:
            if in_table:
                # Flush table
                in_table = False
                num_cols = max(len(r) for r in table_rows)
                # Compute col widths to fit page width (letter width - 108 = 504 pt)
                total_w = letter[0] - 108
                col_w = total_w / num_cols
                
                # Custom column widths if known
                if num_cols == 2:
                    col_widths = [130, 374]
                elif num_cols == 3:
                    col_widths = [110, 180, 214]
                elif num_cols == 4:
                    col_widths = [110, 130, 150, 114]
                elif num_cols == 5:
                    col_widths = [84, 100, 100, 110, 110]
                elif num_cols == 6:
                    col_widths = [94, 82, 82, 82, 82, 82]
                elif num_cols == 7:
                    col_widths = [90, 69, 69, 69, 69, 69, 69]
                elif num_cols == 10:
                    col_widths = [64, 45, 45, 50, 50, 50, 50, 50, 50, 50]
                elif num_cols == 13:
                    col_widths = [20, 94, 40, 40, 35, 40, 35, 40, 40, 40, 40, 40, 40]
                elif num_cols == 14:
                    col_widths = [20, 84, 40, 40, 35, 40, 35, 40, 35, 35, 35, 40, 40, 35]
                else:
                    col_widths = [col_w] * num_cols
                    
                table_data = []
                for row_idx, r in enumerate(table_rows):
                    row_cells = []
                    # Pad cells if needed
                    while len(r) < num_cols:
                        r.append("")
                    for col_idx, cell_str in enumerate(r):
                        cleaned = clean_text(cell_str)
                        if row_idx == 0:
                            p = Paragraph(f"<b>{cleaned}</b>", table_header_style)
                        else:
                            if "PASS" in cleaned or "Best" in cleaned or "0.0816" in cleaned or "37.66%" in cleaned:
                                p = Paragraph(cleaned, table_cell_bold)
                            else:
                                p = Paragraph(cleaned, table_cell_style)
                        row_cells.append(p)
                    table_data.append(row_cells)
                    
                t = Table(table_data, colWidths=col_widths, repeatRows=1)
                t_style = [
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
                    ('TOPPADDING', (0,0), (-1,-1), 2.5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
                    ('LEFTPADDING', (0,0), (-1,-1), 3),
                    ('RIGHTPADDING', (0,0), (-1,-1), 3),
                ]
                # Alternating row background
                for r_i in range(1, len(table_data)):
                    if r_i % 2 == 1:
                        t_style.append(('BACKGROUND', (0, r_i), (-1, r_i), colors.HexColor('#FFFFFF')))
                    else:
                        t_style.append(('BACKGROUND', (0, r_i), (-1, r_i), colors.HexColor('#F7FAFC')))
                t.setStyle(TableStyle(t_style))
                story.append(t)
                story.append(Spacer(1, 5))
                table_rows = []

        # Empty lines
        if not stripped:
            i += 1
            continue
            
        # Headers
        if stripped.startswith("# MACHINE LEARNING"):
            p = Paragraph("<b>MACHINE LEARNING COURSE PROJECT</b>", subtitle_style)
            story.append(p)
        elif stripped.startswith("## Project Title:"):
            title_text = stripped.replace("## Project Title:", "").strip()
            p = Paragraph(title_text, title_style)
            story.append(p)
            story.append(Spacer(1, 4))
        elif stripped.startswith("## "):
            sec_title = stripped.replace("## ", "").strip()
            # Clean section title
            p = Paragraph(clean_text(sec_title), h1_style)
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2B6CB0'), spaceBefore=2, spaceAfter=4))
            story.append(p)
        elif stripped.startswith("### "):
            sub_title = stripped.replace("### ", "").strip()
            p = Paragraph(clean_text(sub_title), h2_style)
            story.append(p)
        elif stripped.startswith("#### "):
            sub_title = stripped.replace("#### ", "").strip()
            p = Paragraph(clean_text(sub_title), h3_style)
            story.append(p)
        elif stripped.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceBefore=4, spaceAfter=4))
        elif stripped.startswith("> "):
            callout_text = clean_text(stripped[2:])
            callout_p = Paragraph(callout_text, callout_style)
            callout_t = Table([[callout_p]], colWidths=[letter[0] - 108])
            callout_t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EDF2F7')),
                ('LINELEFT', (0,0), (0,-1), 3.0, colors.HexColor('#3182CE')),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(callout_t)
            story.append(Spacer(1, 3))
        elif stripped.startswith("* ") or stripped.startswith("- "):
            bullet_text = "• " + clean_text(stripped[2:])
            p = Paragraph(bullet_text, bullet_style)
            story.append(p)
        elif re.match(r'^\d+\.\s+', stripped):
            num_match = re.match(r'^(\d+\.)\s+(.*)', stripped)
            prefix = num_match.group(1)
            content = num_match.group(2)
            p = Paragraph(f"<b>{prefix}</b> {clean_text(content)}", bullet_style)
            story.append(p)
        else:
            p = Paragraph(clean_text(stripped), body_style)
            story.append(p)

        i += 1
        
    print(f"Building document to {pdf_path} with {len(story)} flowable elements...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF build complete successfully!")

if __name__ == "__main__":
    md_file = r"c:\Users\B NAVANEETH\SatQueryAI\METHODOLOGY.md"
    pdf_file = r"c:\Users\B NAVANEETH\SatQueryAI\METHODOLOGY.pdf"
    build_pdf(md_file, pdf_file)
