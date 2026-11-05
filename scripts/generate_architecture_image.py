"""
Script: generate_architecture_image.py
Purpose: Generates a high-resolution, publication-grade system architecture flow diagram
         for SatQuery AI to be embedded in METHODOLOGY.docx.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_diagram(output_path="docs/system_architecture_diagram.png"):
    Path("docs").mkdir(parents=True, exist_ok=True)
    
    # 12 x 8 inches at 300 DPI
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 75)
    ax.axis('off')
    
    # Colors
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

    # Box Helper Function
    def draw_box(x, y, w, h, text, title="", bg=c_box_bg, border=c_blue, lw=1.5, fontsize=8.5, title_fs=9.5, align='center'):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            linewidth=lw, edgecolor=border, facecolor=bg, zorder=2
        )
        ax.add_patch(rect)
        if title:
            ax.text(x + w/2, y + h - 2.2, title, ha='center', va='center', fontsize=title_fs, fontweight='bold', color=c_navy, zorder=3)
            ax.text(x + w/2, y + (h - 2.5)/2, text, ha=align, va='center', fontsize=fontsize, color="#2D3748", zorder=3)
        else:
            ax.text(x + w/2, y + h/2, text, ha=align, va='center', fontsize=fontsize, color="#2D3748", zorder=3)

    # Diamond Helper Function
    def draw_diamond(cx, cy, w, h, text, bg="#EDF2F7", border=c_blue):
        pts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
        poly = patches.Polygon(pts, closed=True, linewidth=1.5, edgecolor=border, facecolor=bg, zorder=2)
        ax.add_patch(poly)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=8, fontweight='bold', color=c_navy, zorder=3)

    # Arrow Helper Function
    def draw_arrow(x1, y1, x2, y2, label="", label_pos=(0.5, 0.5), color=c_gray, lw=1.5, ls='-'):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=12),
            zorder=4
        )
        if label:
            lx = x1 + (x2 - x1) * label_pos[0]
            ly = y1 + (y2 - y1) * label_pos[1]
            ax.text(lx, ly, label, ha='center', va='center', fontsize=7.5, fontweight='bold',
                    color=color, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=5)

    # 1. Top Title
    ax.text(55, 72.5, "SatQuery AI: Decoupled Multi-Pathway Selective Orchestration Architecture",
            ha='center', va='center', fontsize=12, fontweight='bold', color=c_navy)

    # 2. Input Box (Left)
    draw_box(4, 58, 22, 9, "Instruction: q\nPayload: o (Metadata/Tensors)", "User Request", bg="#EBF8FF", border=c_blue)

    # 3. Stage 1: Linguistic Router
    draw_box(33, 57, 32, 11, "Intent Logits: z = f_θ(q)\nP(t|q) = Softmax(z / T_cal)\nc_intent = max P, Δc = margin\ns_joint(q) = min(c/τ_intent, Δc/τ_margin)",
             "Stage 1: Intent Router (T* _|_ o | q)", bg="#F7FAFC", border=c_blue)
    draw_arrow(26, 62.5, 33, 62.5)

    # 4. Ambiguity Decision Diamond
    draw_diamond(76, 62.5, 16, 9, "s_joint(q) ≥ 1.0?\n(Ambiguity Check)")
    draw_arrow(65, 62.5, 68, 62.5)

    # Pathway 3: Clarify (Branch Up/Right)
    draw_box(89, 58, 18, 9, "Prompt user to clarify;\nTask coercion\nstrictly prohibited.", "Pathway 3: CLARIFY",
             bg=c_light_orange, border=c_orange)
    draw_arrow(76, 67, 76, 62.5 + 4.5, "")
    draw_arrow(84, 62.5, 89, 62.5, "No (Ambiguous)", (0.5, 0.5), color=c_orange)

    # 5. Stage 2: Deterministic Physical Sensor Gate (Down from Ambiguity Diamond)
    draw_arrow(76, 58, 76, 49, "Yes (Qualified)", (0.5, 0.5), color=c_green)
    
    draw_box(16, 38, 48, 11,
             "C(t_intent, o) = ∏ C_d(t_intent, o)  [d=1..8]\n"
             "• C1: Modality Completeness  • C2: Temporal Baseline\n"
             "• C3: Spectral Coverage      • C4: Spatial Dimensions\n"
             "• C5: Affine SVD Basis Check • C6: GSD Scale Parity\n"
             "• C7: Grid-Phase Shift ≤0.05px • C8: Cloud & Finiteness",
             "Stage 2: Deterministic Physical Sensor Gate", bg="#F7FAFC", border=c_blue)

    draw_arrow(76, 43.5, 64, 43.5)

    # Pathway 2: Reject Incompatible Observation
    draw_box(4, 25, 20, 9, "Explicit error code;\nNever coerce to\nalternative task.", "Pathway 2: REJECT",
             bg=c_light_red, border=c_red)
    draw_arrow(16, 43.5, 14, 43.5)
    draw_arrow(14, 43.5, 14, 34, "Contract Failed", (0.5, 0.5), color=c_red)

    # 6. Stage 3: Selective Risk Threshold Diamond
    draw_diamond(40, 29.5, 18, 8, "s_joint(q) ≥ τ_sel?\n(Risk Control)")
    draw_arrow(40, 38, 40, 33.5, "Contract Passed", (0.5, 0.5), color=c_green)

    # Sub-Outcome: Abstain Low Confidence
    draw_box(4, 13, 20, 9, "Graceful deferral;\nIntercepts modest\nmargin under τ_sel > 1.", "Sub-Outcome: ABSTAIN",
             bg="#EDF2F7", border=c_gray)
    draw_arrow(31, 29.5, 24, 29.5)
    draw_arrow(24, 29.5, 24, 17.5, "No (τ_sel > 1)", (0.5, 0.5), color=c_gray)
    draw_arrow(24, 17.5, 24, 17.5)

    # Pathway 1: Dispatch Qualified
    draw_arrow(49, 29.5, 58, 29.5, "Yes (Qualified)", (0.5, 0.5), color=c_green)

    # 7. Specialist Registry Box (Bottom Right)
    reg_rect = patches.FancyBboxPatch(
        (58, 7), 50, 32,
        boxstyle="round,pad=0.5,rounding_size=2.0",
        linewidth=1.5, edgecolor=c_navy, facecolor="#F0F4F8", zorder=1
    )
    ax.add_patch(reg_rect)
    ax.text(83, 36.8, "Stage 3: Specialist Perception Registry (Pathway 1: DISPATCH)",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_navy, zorder=3)

    # Implemented Specialists (Green Border)
    draw_box(60, 22.5, 46, 12,
             "Dual-Branch ResNet-34 + 8-Head Cross-Attention\n"
             "• 10-ch Optical (ImageNet RGB init) + 2-ch SAR (He init)\n"
             "• Multi-Label Losses: ASL (m=0.05) & CB-BCE (β=0.999)\n"
             "• Result: 45.53% Macro F1, 74.47% Micro F1, 8.16% Hamming",
             "Land-Cover Specialist [IMPLEMENTED & EVALUATED]",
             bg=c_light_green, border=c_green, lw=1.2, fontsize=7.5, title_fs=8.5, align='left')

    draw_box(60, 9, 46, 12,
             "FC-Siam-diff & Bitemporal Image Transformer (BIT)\n"
             "• Bitemporal Sentinel-2 RGB (586 train patches, stride 64)\n"
             "• FC-Siam-diff Result: 37.66% Macro F1, 38.18% Pooled F1\n"
             "• Status: Implemented & Evaluated; 50% Certification NOT Met",
             "Change-Detection Specialist [IMPLEMENTED & EVALUATED]",
             bg=c_light_green, border=c_green, lw=1.2, fontsize=7.5, title_fs=8.5, align='left')

    # Planned Tasks Footer
    ax.text(83, 6, "Formal Contract Targets [PLANNED]: VQA, Change VQA, Grounding, Captioning, Fusion",
            ha='center', va='center', fontsize=7.5, fontstyle='italic', color=c_gray, zorder=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram saved successfully to: {output_path}")

if __name__ == "__main__":
    create_diagram()
