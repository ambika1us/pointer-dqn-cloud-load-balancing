# scripts/generate_architecture_diagram.py
"""
Generate a system architecture block diagram for the paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from config import RESULTS_DIR


def draw_box(ax, x, y, w, h, text, color="#1f77b4", fontsize=8):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="black",
        facecolor=color,
        alpha=0.85,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold",
        color="white" if color != "#ffffff" else "black",
        wrap=True,
    )


def draw_arrow(ax, x1, y1, x2, y2, label=None, color="black"):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=14,
        linewidth=1.2, color=color,
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + 0.015, label,
            ha="center", va="bottom", fontsize=7,
            style="italic", color=color,
        )


def main():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Colors
    c_data = "#2ca02c"
    c_env = "#1f77b4"
    c_dqn = "#9467bd"
    c_out = "#ff7f0e"

    # ---------- Data pipeline (left) ----------
    draw_box(ax, 0.02, 0.70, 0.18, 0.14, "Google Borg\n2019 Trace", c_data)
    draw_box(ax, 0.02, 0.48, 0.18, 0.14, "Alibaba\n2018 Trace", c_data)
    draw_box(ax, 0.02, 0.26, 0.18, 0.14, "Preprocessing\n+ Normalization", c_data)

    draw_arrow(ax, 0.11, 0.70, 0.11, 0.62)   # Google → Preprocessing
    draw_arrow(ax, 0.11, 0.48, 0.11, 0.40)   # Alibaba → Preprocessing
    draw_arrow(ax, 0.20, 0.33, 0.28, 0.33)   # Preprocessing → Env

    # ---------- Environment (center) ----------
    draw_box(ax, 0.30, 0.20, 0.24, 0.55, "RL Environment\n\n• 50 VMs\n• M/M/1 response\n• Time-scaled decay\n• Single-objective reward", c_env, fontsize=7.5)

    # ---------- DQN (right) ----------
    draw_box(ax, 0.62, 0.42, 0.22, 0.20, "Pointer-style\nDQN Agent", c_dqn, fontsize=8)
    draw_box(ax, 0.62, 0.20, 0.22, 0.14, "Action:\nVM selection", c_dqn, fontsize=7.5)

    draw_arrow(ax, 0.54, 0.55, 0.62, 0.52, "State\n(50 loads + task)")
    draw_arrow(ax, 0.73, 0.42, 0.73, 0.34, "Q(s, a)")
    draw_arrow(ax, 0.62, 0.27, 0.54, 0.30, "reward")

    # ---------- Output (far right) ----------
    draw_box(ax, 0.87, 0.42, 0.11, 0.20, "Metrics\n\n• Reward\n• Response\n• Imbalance", c_out, fontsize=7)

    draw_arrow(ax, 0.84, 0.50, 0.87, 0.50)

    # ---------- Title ----------
    ax.text(
        0.5, 0.97,
        "System Architecture: Pointer-Style DQN for Cloud Load Balancing",
        ha="center", va="top", fontsize=10, fontweight="bold",
    )

    # ---------- Legend ----------
    legend_elements = [
        mpatches.Patch(color=c_data, label="Data pipeline"),
        mpatches.Patch(color=c_env, label="Environment"),
        mpatches.Patch(color=c_dqn, label="Agent"),
        mpatches.Patch(color=c_out, label="Metrics"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower center",
        ncol=4, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    out = RESULTS_DIR / "system_architecture.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      -> {out}")


if __name__ == "__main__":
    main()