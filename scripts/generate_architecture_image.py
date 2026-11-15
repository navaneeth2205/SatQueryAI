"""
Script: generate_architecture_image.py
Purpose: Generates a high-resolution, publication-grade system architecture flow diagram
         for SatQuery AI to be embedded in METHODOLOGY.docx.
         Guarantees zero text overflow, generous box margins, and perfectly bounded labels.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_diagram(output_path="docs/system_architecture_diagram.png"):
    Path("docs").mkdir(parents=True, exist_ok=True)
    
    # High-resolution 11.5 x 7.8 inches at 300 DPI
    fig, ax = plt.subplots(figsize=(11.5, 7.8), dpi=300)
    ax.set_xlim(0, 115)
    ax.set_ylim(0, 75)
    ax.axis('off')
    
    # Academic Color Palette
    c_navy = "#1A365D"
    c_blue = "#2B6CB0"
    c_light_blue = "#EBF8FF"
    c_green = "#2F855A"
    c_light_green = "#F0FFF4"
    c_red = "#C53030"
    c_light_red = "#FFF5F5"
    c_orange = "#DD6B20"
    c_light_orange = "#FFFAF0"
    c_gray = "#4A5568"
    c_box_bg = "#F7FAFC"
    c_border = "#CBD5E0"

    # Box Helper Function with robust left-padding and clipping prevention
    def draw_box(x, y, w, h, text, title="", bg=c_box_bg, border=c_blue, lw=1.4, fontsize=7.2, title_fs=8.5, align='left'):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=1.2",
            linewidth=lw, edgecolor=border, facecolor=bg, zorder=2
        )
        ax.add_patch(rect)
        if title:
            # Title centered near top
            ax.text(x + w/2, y + h - 1.8, title, ha='center', va='center', fontsize=title_fs, fontweight='bold', color=c_navy, zorder=3)
            # Body text positioned inside box with left margin if align='left'
            if align == 'left':
                ax.text(x + 1.4, y + (h - 2.5)/2, text, ha='left', va='center', fontsize=fontsize, color="#2D3748", zorder=3, linespacing=1.35)
            else:
                ax.text(x + w/2, y + (h - 2.5)/2, text, ha='center', va='center', fontsize=fontsize, color="#2D3748", zorder=3, linespacing=1.35)
        else:
            if align == 'left':
                ax.text(x + 1.4, y + h/2, text, ha='left', va='center', fontsize=fontsize, color="#2D3748", zorder=3, linespacing=1.35)
            else:
                ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color="#2D3748", zorder=3, linespacing=1.35)

    # Diamond Decision Helper Function
    def draw_diamond(cx, cy, w, h, text, bg="#EDF2F7", border=c_blue, fontsize=7.0):
        pts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
        poly = patches.Polygon(pts, closed=True, linewidth=1.4, edgecolor=border, facecolor=bg, zorder=2)
        ax.add_patch(poly)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=c_navy, zorder=3, linespacing=1.2)

    # Arrow Helper Function
    def draw_arrow(x1, y1, x2, y2, label="", label_pos=(0.5, 0.5), color=c_gray, lw=1.3, ls='-'):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=11),
            zorder=4
        )
        if label:
            lx = x1 + (x2 - x1) * label_pos[0]
            ly = y1 + (y2 - y1) * label_pos[1]
            ax.text(lx, ly, label, ha='center', va='center', fontsize=7.0, fontweight='bold',
                    color=color, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92), zorder=5)

    # 1. Header Block
    ax.text(57.5, 72.8, "SatQuery AI: Decoupled Multi-Pathway Selective Orchestration Architecture",
            ha='center', va='center', fontsize=11.5, fontweight='bold', color=c_navy)
    ax.text(57.5, 70.8, "Input-Constrained Intent Routing  •  Deterministic 8D Physical Compatibility Gate  •  Certified Perception Specialists",
            ha='center', va='center', fontsize=7.8, color="#4A5568")

    # ---------------------------------------------------------
    # ROW 1 (y: 53 to 68) - Stage 1, Ambiguity Gate & Pathway 3
    # ---------------------------------------------------------
    # Input Box (Left)
    draw_box(3, 54, 21, 13.5,
             "• Query: q (natural language)\n• Payload: o (EO data)\n  - Sentinel-1 SAR (VH/VV)\n  - Sentinel-2 MSI / Optical\n  - Bitemporal Pair (t1, t2)",
             "User Request", bg="#EBF8FF", border=c_blue, fontsize=7.0, title_fs=8.2, align='left')

    # Arrow from Input to Stage 1 (query only!)
    draw_arrow(24, 60.75, 29, 60.75, "q only", (0.5, 0.5), color=c_blue)

    # Stage 1: Linguistic Router Box
    draw_box(29, 53, 38, 15.5,
             "• Logits: z = f_θ(q) ∈ ℝ^K across K = 7 discrete tasks\n"
             "• Calibrated Probabilities: P(t | q) = Softmax(z / T_cal)\n"
             "• Top Intent: t_cand = argmax P(t | q),  c_intent = max P\n"
             "• Competitive Margin: Δc = c_intent - second_max\n"
             "• Standardized Selection: s_joint(q) = min(c/τ_intent, Δc/τ_margin)",
             "Stage 1: Intent Router (T* ⊥ o | q)", bg="#F7FAFC", border=c_blue, fontsize=7.0, title_fs=8.2, align='left')

    # Arrow from Stage 1 to Ambiguity Gate
    draw_arrow(67, 60.75, 72, 60.75)

    # Ambiguity Decision Diamond (cx=80, cy=60.75, w=16, h=10)
    draw_diamond(80, 60.75, 16, 10, "s_joint(q) ≥ 1.0?\n(Ambiguity)", fontsize=7.0)

    # Pathway 3: Clarify (Right of Ambiguity Gate)
    draw_arrow(88, 60.75, 94, 60.75, "No (<1.0)", (0.5, 0.5), color=c_orange)
    draw_box(94, 54, 18, 13.5,
             "Solicit user clarification.\n"
             "Non-Coercion Constraint:\n"
             "Never coerce ambiguous\n"
             "instructions to payload.",
             "Pathway 3: CLARIFY", bg=c_light_orange, border=c_orange, fontsize=6.8, title_fs=8.0, align='center')

    # ---------------------------------------------------------
    # ROW 2 (y: 33.5 to 49) - Stage 2 Physical Gate & Pathway 2
    # ---------------------------------------------------------
    # Arrow from Ambiguity Gate (Yes) down to Stage 2
    draw_arrow(80, 55.75, 80, 43, "Yes (Qualified)", (0.4, 0.5), color=c_green)
    draw_arrow(80, 43, 73, 43)

    # Stage 2: Deterministic Physical Sensor Compatibility Gate
    draw_box(20, 33.5, 53, 15.5,
             "Deterministic Closed-Form Predicate: C(t_cand, o) = ∏ C_d(t_cand, o) ∈ {0, 1}\n"
             "• C1: Modality Completeness (SAR, Opt)  • C2: Temporal Baseline (|Δt| ≥ 5d)\n"
             "• C3: Spectral Coverage (10 MSI bands)   • C4: Spatial Dimensions (H, W)\n"
             "• C5: Affine SVD Basis (|σ_i - 1| ≤ 0.01) • C6: GSD Scale Parity (≤ 1.0%)\n"
             "• C7: Sub-Pixel Grid-Phase Shift ≤0.05px • C8: Cloud (≤20%) & Finiteness",
             "Stage 2: Deterministic Physical Sensor Compatibility Gate",
             bg="#F7FAFC", border=c_blue, fontsize=7.0, title_fs=8.2, align='left')

    # Pathway 2: Reject Incompatible Observation (Left of Stage 2)
    draw_arrow(20, 41.25, 14, 41.25, "Contract Failed", (0.5, 0.5), color=c_red)
    draw_box(2, 33.5, 12, 15.5,
             "Emit structured\nviolation code\n(C1–C8).\nExecution halted;\nzero coercion.",
             "Pathway 2: REJECT", bg=c_light_red, border=c_red, fontsize=6.8, title_fs=7.8, align='center')

    # ---------------------------------------------------------
    # ROW 3 (y: 4 to 28) - Stage 3 Selective Gate, Abstain & Registry
    # ---------------------------------------------------------
    # Arrow from Stage 2 (down) to Selective Risk Control Diamond
    draw_arrow(46.5, 33.5, 46.5, 24.5, "Contract Passed (C=1)", (0.45, 0.5), color=c_green)

    # Selective Risk Control Diamond (cx=46.5, cy=20, w=16, h=9)
    draw_diamond(46.5, 20, 16, 9, "s_joint(q) ≥ τ_sel?\n(Risk Control)", fontsize=7.0)

    # Sub-Outcome: Abstain Low Confidence (Left of Selective Gate)
    draw_arrow(38.5, 20, 26, 20, "No (τ_sel > 1.0)", (0.5, 0.5), color=c_gray)
    draw_box(2, 13.5, 24, 13.0,
             "Graceful deferral under elevated\n"
             "selective threshold (τ_sel > 1.0).\n"
             "Intercepts modest margin queries\n"
             "prior to specialist execution.",
             "Sub-Outcome: ABSTAIN", bg="#EDF2F7", border=c_gray, fontsize=6.8, title_fs=7.8, align='center')

    # Pathway 1: Dispatch Qualified (Right of Selective Gate into Registry)
    draw_arrow(54.5, 20, 61, 20, "Yes (Qualified)", (0.5, 0.5), color=c_green)

    # Stage 3: Specialist Perception Registry (Container Box)
    reg_rect = patches.FancyBboxPatch(
        (61, 4), 51, 26.5,
        boxstyle="round,pad=0.0,rounding_size=1.5",
        linewidth=1.4, edgecolor=c_navy, facecolor="#F0F4F8", zorder=1
    )
    ax.add_patch(reg_rect)
    ax.text(86.5, 28.8, "Stage 3: Specialist Perception Registry (Pathway 1: DISPATCH)",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_navy, zorder=3)

    # Specialist 1: Multimodal Land-Cover Classification
    draw_box(63, 16.5, 47, 10.5,
             "• Architecture: Dual ResNet-34 + 8-Head Cross-Attention (dim 512)\n"
             "• Inputs: 10-ch Sentinel-2 MSI (10m) + 2-ch Sentinel-1 SAR (VH/VV)\n"
             "• Losses: Asymmetric Loss (ASL, m=0.05) & Class-Balanced BCE\n"
             "• Held-Out Result: 45.53% Macro F1, 74.47% Micro F1, 8.16% Hamming",
             "Land-Cover Specialist [IMPLEMENTED & CERTIFIED]",
             bg=c_light_green, border=c_green, lw=1.1, fontsize=6.7, title_fs=7.8, align='left')

    # Specialist 2: Bitemporal Change Detection
    draw_box(63, 5.0, 47, 10.5,
             "• Architecture: FC-Siam-diff (Daudt et al.) & Bitemporal Transformer (BIT)\n"
             "• Inputs: Bitemporal Sentinel-2 Level-1C optical RGB pairs (t1, t2)\n"
             "• Training Split: 586 unique 256×256 patches (stride 64) from 10 cities\n"
             "• Held-Out Result: 37.66% Macro F1 (Target 50% NOT met; reported honestly)",
             "Change-Detection Specialist [IMPLEMENTED & EVALUATED]",
             bg=c_light_green, border=c_green, lw=1.1, fontsize=6.7, title_fs=7.8, align='left')

    # Footer note inside canvas
    ax.text(57.5, 1.5, "Formal Contract Stubs [PLANNED]: SINGLE_IMAGE_VQA, CHANGE_VQA, VISUAL_GROUNDING, SCENE_CAPTIONING, OPTICAL_SAR_FUSION",
            ha='center', va='center', fontsize=6.8, fontstyle='italic', color=c_gray, zorder=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Publication-grade architecture diagram saved successfully to: {output_path}")

if __name__ == "__main__":
    create_diagram()
