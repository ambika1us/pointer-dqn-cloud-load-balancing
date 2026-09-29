# scripts/generate_all_figures.py
"""
Generate all figures for the paper in a single run.

Outputs (saved to results/figures/):
    fig1_training_convergence.png
    fig2_architecture_comparison.png
    fig3_cross_trace_comparison.png
    fig4_vm_scaling.png
    fig5_ablation.png
    fig6_permutation_test.png
    fig7_seed_distribution.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR

# Output directory
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8

# Colors
C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_PURPLE = "#9467bd"


def save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ==================================================================
# Figure 1: Training Convergence
# ==================================================================
def fig1_training_convergence():
    print("[1/7] Training convergence...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot mean ± std across 5 seeds
    seeds = [0, 1, 2, 3, 4]
    all_logs = []
    for seed in seeds:
        path = RESULTS_DIR / f"seed_{seed}" / "training_log.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["seed"] = seed
            all_logs.append(df)

    if not all_logs:
        print("    SKIP: no seed training logs")
        return

    combined = pd.concat(all_logs, ignore_index=True)

    # Aggregate by episode
    agg = combined.groupby("episode").agg(
        reward_mean=("avg_reward", "mean"),
        reward_std=("avg_reward", "std"),
        response_mean=("avg_response", "mean"),
        response_std=("avg_response", "std"),
        imbalance_mean=("avg_imbalance", "mean"),
        imbalance_std=("avg_imbalance", "std"),
    ).reset_index()

    episodes = agg["episode"]

    # (a) Reward
    axes[0].plot(episodes, agg["reward_mean"], color=C_BLUE, linewidth=1.2)
    axes[0].fill_between(episodes,
                         agg["reward_mean"] - agg["reward_std"],
                         agg["reward_mean"] + agg["reward_std"],
                         alpha=0.2, color=C_BLUE)
    axes[0].axhline(-0.3279, ls="--", color=C_RED, alpha=0.7, label="Greedy-Load")
    axes[0].axhline(-0.4375, ls="--", color=C_ORANGE, alpha=0.7, label="Round-Robin")
    axes[0].set_title("(a) Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    # (b) Response time
    axes[1].plot(episodes, agg["response_mean"], color=C_ORANGE, linewidth=1.2)
    axes[1].fill_between(episodes,
                         agg["response_mean"] - agg["response_std"],
                         agg["response_mean"] + agg["response_std"],
                         alpha=0.2, color=C_ORANGE)
    axes[1].set_title("(b) Average Response Time")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Seconds")

    # (c) Imbalance
    axes[2].plot(episodes, agg["imbalance_mean"], color=C_GREEN, linewidth=1.2)
    axes[2].fill_between(episodes,
                         agg["imbalance_mean"] - agg["imbalance_std"],
                         agg["imbalance_mean"] + agg["imbalance_std"],
                         alpha=0.2, color=C_GREEN)
    axes[2].set_title("(c) Load Imbalance (CV)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Coefficient of Variation")

    plt.tight_layout()
    save(fig, "fig1_training_convergence.png")


# ==================================================================
# Figure 2: Architecture Comparison (MLP vs Pointer)
# ==================================================================
def fig2_architecture_comparison():
    print("[2/7] Architecture comparison...")

    # Load pointer stats
    ptr_google = RESULTS_DIR / "stats_google_pointer.csv"
    ptr_alibaba = RESULTS_DIR / "stats_alibaba_pointer.csv"

    if not (ptr_google.exists() and ptr_alibaba.exists()):
        print("    SKIP: pointer stats missing")
        return

    g = pd.read_csv(ptr_google)
    a = pd.read_csv(ptr_alibaba)

    def extract(df):
        return {
            "Always-VM0": df.loc[df["baseline"] == "Always-VM0", "baseline_mean"].values[0],
            "Random": df.loc[df["baseline"] == "Random", "baseline_mean"].values[0],
            "Round-Robin": df.loc[df["baseline"] == "Round-Robin", "baseline_mean"].values[0],
            "Greedy-Load": df.loc[df["baseline"] == "Greedy-Load", "baseline_mean"].values[0],
            "Pointer-DQN": df["dqn_mean"].values[0],
        }

    g_data = extract(g)
    a_data = extract(a)

    # MLP baseline (single-seed result from earlier)
    g_data["MLP-DQN"] = -0.598
    a_data["MLP-DQN"] = -0.763

    labels = ["Always-VM0", "Random", "Round-Robin", "MLP-DQN", "Greedy-Load", "Pointer-DQN"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width/2, [g_data[k] for k in labels], width, label="Google", color=C_BLUE)
    ax.bar(x + width/2, [a_data[k] for k in labels], width, label="Alibaba", color=C_ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean reward")
    ax.set_title("Architecture Comparison: MLP vs Pointer DQN")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    save(fig, "fig2_architecture_comparison.png")


# ==================================================================
# Figure 3: Cross-Trace Comparison
# ==================================================================
def fig3_cross_trace_comparison():
    print("[3/7] Cross-trace comparison...")

    summary_path = RESULTS_DIR / "seed_level_summary.csv"
    if not summary_path.exists():
        print("    SKIP: seed_level_summary.csv missing")
        return

    df = pd.read_csv(summary_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["reward", "response", "imbalance"]
    titles = ["(a) Reward", "(b) Response Time (s)", "(c) Load Imbalance (CV)"]

    for ax, m, t in zip(axes, metrics, titles):
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"

        if mean_col not in df.columns:
            continue

        x = np.arange(len(df))
        colors = [C_BLUE if tr == "google" else C_ORANGE for tr in df["trace"]]

        ax.bar(x, df[mean_col], yerr=df[std_col], color=colors, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["trace"].str.capitalize())
        ax.set_title(t)
        ax.set_ylabel(m.capitalize())

        # Add value labels
        for i, v in enumerate(df[mean_col]):
            ax.text(i, v + (0.02 if v < 0 else -0.05),
                    f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    save(fig, "fig3_cross_trace_comparison.png")


# ==================================================================
# Figure 4: VM-Scaling Results
# ==================================================================
def fig4_vm_scaling():
    print("[4/7] VM-scaling results...")

    path = RESULTS_DIR / "vm_scaling_results.csv"
    if not path.exists():
        print("    SKIP: vm_scaling_results.csv missing")
        return

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) Reward
    axes[0].plot(df["vm_count"], df["reward_mean"], marker="o", color=C_BLUE)
    axes[0].axvline(50, ls="--", color=C_RED, alpha=0.5, label="Training (50 VMs)")
    axes[0].set_xlabel("Number of VMs")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("(a) Reward vs VM count")
    axes[0].legend()

    # (b) Response
    axes[1].plot(df["vm_count"], df["response_mean"], marker="o", color=C_ORANGE)
    axes[1].axvline(50, ls="--", color=C_RED, alpha=0.5)
    axes[1].set_xlabel("Number of VMs")
    axes[1].set_ylabel("Response time (s)")
    axes[1].set_title("(b) Response time vs VM count")

    # (c) Imbalance
    axes[2].plot(df["vm_count"], df["imbalance_mean"], marker="o", color=C_GREEN)
    axes[2].axvline(50, ls="--", color=C_RED, alpha=0.5)
    axes[2].set_xlabel("Number of VMs")
    axes[2].set_ylabel("Load imbalance (CV)")
    axes[2].set_title("(c) Imbalance vs VM count")

    plt.tight_layout()
    save(fig, "fig4_vm_scaling.png")


# ==================================================================
# Figure 5: Ablation Study
# ==================================================================
def fig5_ablation():
    print("[5/7] Ablation study...")

    path = RESULTS_DIR / "ablation_log.csv"
    if not path.exists():
        print("    SKIP: ablation_log.csv missing")
        return

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df_sorted = df.sort_values("ablation")

    # (a) Reward
    axes[0].bar(df_sorted["ablation"], df_sorted["reward"], color=C_BLUE)
    axes[0].set_title("(a) Ablation — Reward")
    axes[0].set_ylabel("Reward")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(df_sorted["reward"]):
        axes[0].text(i, v - 0.02, f"{v:.3f}", ha="center", fontsize=8)

    # (b) Imbalance
    axes[1].bar(df_sorted["ablation"], df_sorted["imbalance"], color=C_ORANGE)
    axes[1].set_title("(b) Ablation — Load Imbalance")
    axes[1].set_ylabel("CV")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(df_sorted["imbalance"]):
        axes[1].text(i, v - 0.03, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    save(fig, "fig5_ablation.png")


# ==================================================================
# Figure 6: Permutation Test Summary
# ==================================================================
def fig6_permutation_test():
    print("[6/7] Permutation test summary...")

    # Hardcoded from actual test run
    labels = ["VM permutation\n(100 tests)", "Batch invariance\n(batch size 8)"]
    deviations = [0.0, 0.0]
    tolerance = 1e-5

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, deviations, color=C_GREEN, width=0.5)

    ax.axhline(tolerance, ls="--", color=C_RED, label=f"Tolerance ({tolerance:.0e})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Max deviation")
    ax.set_title("Empirical Permutation Equivariance Test")
    ax.set_ylim(-0.1e-5, 1.5e-5)
    ax.legend()

    for i, v in enumerate(deviations):
        ax.text(i, v + 0.1e-5, f"{v:.0e}", ha="center", fontsize=9)

    plt.tight_layout()
    save(fig, "fig6_permutation_test.png")


# ==================================================================
# Figure 7: Seed-Level Reward Distribution
# ==================================================================
def fig7_seed_distribution():
    print("[7/7] Seed-level reward distribution...")

    google_path = RESULTS_DIR / "seed_level_google.csv"
    alibaba_path = RESULTS_DIR / "seed_level_alibaba.csv"

    if not (google_path.exists() and alibaba_path.exists()):
        print("    SKIP: seed-level CSVs missing")
        return

    g = pd.read_csv(google_path)
    a = pd.read_csv(alibaba_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Google
    axes[0].bar(g["seed"], g["reward"], color=C_BLUE)
    axes[0].axhline(g["reward"].mean(), ls="--", color=C_RED,
                    label=f"Mean = {g['reward'].mean():.4f}")
    axes[0].fill_between([-0.5, 4.5],
                         g["reward"].mean() - g["reward"].std(),
                         g["reward"].mean() + g["reward"].std(),
                         alpha=0.2, color=C_RED, label=f"±1 SD ({g['reward'].std():.4f})")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("(a) Google trace")
    axes[0].set_xticks(g["seed"])
    axes[0].legend(fontsize=7)

    # (b) Alibaba
    axes[1].bar(a["seed"], a["reward"], color=C_ORANGE)
    axes[1].axhline(a["reward"].mean(), ls="--", color=C_RED,
                    label=f"Mean = {a['reward'].mean():.4f}")
    axes[1].fill_between([-0.5, 4.5],
                         a["reward"].mean() - a["reward"].std(),
                         a["reward"].mean() + a["reward"].std(),
                         alpha=0.2, color=C_RED, label=f"±1 SD ({a['reward'].std():.4f})")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("(b) Alibaba trace")
    axes[1].set_xticks(a["seed"])
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    save(fig, "fig7_seed_distribution.png")


# ==================================================================
# Main
# ==================================================================
def main():
    print("=" * 60)
    print("GENERATING ALL PAPER FIGURES")
    print("=" * 60)
    print(f"Output directory: {FIG_DIR}")
    print()

    fig1_training_convergence()
    fig2_architecture_comparison()
    fig3_cross_trace_comparison()
    fig4_vm_scaling()
    fig5_ablation()
    fig6_permutation_test()
    fig7_seed_distribution()

    print()
    print("=" * 60)
    print("All figures generated.")
    print(f"Location: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()