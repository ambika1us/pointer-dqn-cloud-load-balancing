# scripts/generate_environment_figure.py
"""
Generate a figure showing one step of the cloud load-balancing
environment loop. No title inside the figure — the caption in
the paper provides the title.

Version 3: title removed, layout shifted up.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from config import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, title, lines, color="#e8f0fe", border="#1f77b4",
             title_fontsize=10, body_fontsize=9):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008",
        linewidth=1.4,
        edgecolor=border,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.028, title,
            ha="center", va="top", fontsize=title_fontsize,
            fontweight="bold", color=border)
    if lines:
        body = "\n".join(lines)
        ax.text(x + w / 2, y + h - 0.062, body,
                ha="center", va="top", fontsize=body_fontsize, color="black",
                linespacing=1.55)


def draw_arrow(ax, x1, y1, x2, y2, label=None):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=16,
        linewidth=1.4, color="black",
    )
    ax.add_patch(arrow)
    if label:
        ax.text((x1 + x2) / 2 + 0.012, (y1 + y2) / 2, label,
                ha="left", va="center", fontsize=8, style="italic", color="#333333")


def main():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # No title — the caption below the figure provides the text

    bw = 0.86
    bx = 0.07
    bh = 0.10

    # Box 1: Incoming task
    draw_box(ax, bx, 0.885, bw, bh,
             "Incoming Task at Step $t$",
             ["Task features $x_t = [c_t,\\ m_t,\\ d_t,\\ q_t]$",
              "$c_t$: CPU request      $m_t$: memory request",
              "$d_t$: duration          $q_t$: queue length"])

    draw_arrow(ax, 0.5, 0.885, 0.5, 0.865)

    # Box 2: Environment state
    draw_box(ax, bx, 0.745, bw, bh,
             "Environment State $s_t$",
             ["$s_t = (x_t,\\ \\{l_1(t),\\ l_2(t),\\ \\ldots,\\ l_N(t)\\})$",
              "Current candidate VM loads",
              "VM$_1$: $l_1(t)$     VM$_2$: $l_2(t)$     $\\ldots$     VM$_N$: $l_N(t)$"],
             color="#fff4e5", border="#ff7f0e")

    draw_arrow(ax, 0.5, 0.745, 0.5, 0.725,
               "observation $s_t$")

    # Box 3: Pointer-DQN agent
    draw_box(ax, bx, 0.590, bw, 0.115,
             "Pointer-DQN Scheduling Agent",
             ["Shared candidate scoring:",
              "$Q(s_t, 1),\\ Q(s_t, 2),\\ \\ldots,\\ Q(s_t, N)$",
              "",
              "Selected action:  $a_t = \\arg\\max_{i}\\ Q(s_t, i)$"],
             color="#ede7f6", border="#9467bd")

    draw_arrow(ax, 0.5, 0.590, 0.5, 0.570,
               "selected VM $a_t$")

    # Box 4: Task assignment and reward
    draw_box(ax, bx, 0.405, bw, 0.145,
             "Task Assignment and Reward",
             ["Assign task $t$ to VM$_{a_t}$",
              "",
              "Pre-assignment response time:",
              "$R_{a_t}(t) = d_t\\ /\\ \\max(1 - l_{a_t}(t),\\ 0.2)$",
              "$R_{a_t}(t) = \\min\\{R_{a_t}(t),\\ 5 d_t\\}$",
              "Reward:  $r_t = -R_{a_t}(t)\\ /\\ (5 d_t)$"],
             color="#e8f5e9", border="#2ca02c")

    draw_arrow(ax, 0.5, 0.405, 0.5, 0.385,
               "reward $r_t$")

    # Box 5: VM load-state update
    draw_box(ax, bx, 0.215, bw, 0.150,
             "VM Load-State Update",
             ["Selected VM:",
              "$l_{a_t}(t+1) = \\min\\{1,\\ l_{a_t}(t) + c_t\\}$",
              "",
              "Non-selected VMs:",
              "$l_j(t+1) = l_j(t),\\ \\ j \\neq a_t$",
              "",
              "Next state: $s_{t+1}$"],
             color="#fce4ec", border="#d62728")

    draw_arrow(ax, 0.5, 0.215, 0.5, 0.195,
               "transition $(s_t, a_t, r_t, s_{t+1})$")

    # Box 6: Replay buffer / training
    draw_box(ax, bx, 0.055, bw, 0.120,
             "Replay Buffer / Training",
             ["Recent-biased sampling",
              "Target-network update",
              "Mini-batch gradient descent"],
             color="#e0f2f1", border="#17becf")

    plt.tight_layout()
    out = FIG_DIR / "fig_environment_step.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      -> {out}")


if __name__ == "__main__":
    main()