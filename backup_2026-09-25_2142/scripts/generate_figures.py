# scripts/generate_figures.py
"""
Generate all paper figures and tables from saved experiment logs.

Outputs (saved to results/):
    - training_convergence.png     (Figure 6.1: DQN training curves)
    - cross_trace_comparison.png   (Figure 6.2: Google vs Alibaba metrics)
    - reward_component_breakdown.png (Figure 6.3: reward vs baselines)
    - table_6_3_transfer_gap.csv   (Table 6.3: transfer gap summary)
    - table_6_3_transfer_gap.txt   (same, human-readable)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    RESULTS_DIR,
    TRAIN_LOG_PATH,
    EVAL_LOG_PATH,
)

sns.set_style("whitegrid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Figure 6.1 — Training Convergence
# ============================================================
def generate_training_convergence():
    print("[fig] Training convergence...")
    log = pd.read_csv(TRAIN_LOG_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) Reward
    axes[0].plot(log["episode"], log["avg_reward"], color="#1f77b4", linewidth=1.2)
    axes[0].axhline(-0.3314, ls="--", color="red", label="Greedy-Load")
    axes[0].axhline(-0.4375, ls="--", color="orange", label="Round-Robin")
    axes[0].set_title("(a) Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend(fontsize=9)

    # (b) Response Time
    axes[1].plot(log["episode"], log["avg_response"], color="#ff7f0e", linewidth=1.2)
    axes[1].set_title("(b) Average Response Time")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Seconds")

    # (c) Imbalance
    axes[2].plot(log["episode"], log["avg_imbalance"], color="#2ca02c", linewidth=1.2)
    axes[2].set_title("(c) Load Imbalance (CV)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Coefficient of Variation")

    plt.tight_layout()
    out = RESULTS_DIR / "training_convergence.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      -> {out}")


# ============================================================
# Figure 6.2 — Cross-Trace Comparison
# ============================================================
def generate_cross_trace_comparison():
    print("[fig] Cross-trace comparison...")
    df = pd.read_csv(EVAL_LOG_PATH)
    summary = df.groupby("trace")[["avg_response", "avg_imbalance", "avg_reward"]].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["avg_response", "avg_imbalance", "avg_reward"]
    titles = ["(a) Avg Response Time (s)", "(b) Avg Load Imbalance (CV)", "(c) Avg Reward"]
    colors = ["#1f77b4", "#ff7f0e"]

    for ax, m, t in zip(axes, metrics, titles):
        summary[m].plot(kind="bar", ax=ax, color=colors)
        ax.set_title(t)
        ax.set_xlabel("")
        ax.set_xticklabels(summary.index, rotation=0)
        for i, v in enumerate(summary[m].values):
            ax.text(i, v * 0.95 if v < 0 else v * 0.95,
                    f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "cross_trace_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      -> {out}")


# ============================================================
# Figure 6.3 — Reward vs Baselines (bar chart)
# ============================================================
def generate_reward_vs_baselines():
    print("[fig] Reward vs baselines...")
    baselines = {
        "Always-VM0": -0.9989,
        "Random": -0.4965,
        "Round-Robin": -0.4375,
        "Greedy-Load": -0.3354,
    }
    log = pd.read_csv(TRAIN_LOG_PATH)
    baselines["DQN"] = log["avg_reward"].min()

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(baselines.keys())
    values = list(baselines.values())
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    ax.bar(names, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v - 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Reward (higher is better)")
    ax.set_title("Policy Comparison — Google Trace")
    plt.tight_layout()
    out = RESULTS_DIR / "reward_vs_baselines.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      -> {out}")


# ============================================================
# Table 6.3 — Transfer Gap Summary
# ============================================================
def generate_transfer_gap_table():
    print("[tbl] Transfer gap table...")
    df = pd.read_csv(EVAL_LOG_PATH)
    summary = df.groupby("trace")[["avg_response", "avg_imbalance", "avg_reward"]].mean()

    g = summary.loc["google"]
    a = summary.loc["alibaba"]
    gap = (a - g) / g.abs() * 100

    table = pd.DataFrame({
        "Metric": ["Response Time (s)", "Load Imbalance (CV)", "Reward"],
        "Google": [g["avg_response"], g["avg_imbalance"], g["avg_reward"]],
        "Alibaba": [a["avg_response"], a["avg_imbalance"], a["avg_reward"]],
        "Gap (%)": [gap["avg_response"], gap["avg_imbalance"], gap["avg_reward"]],
    })

    csv_out = RESULTS_DIR / "table_6_3_transfer_gap.csv"
    txt_out = RESULTS_DIR / "table_6_3_transfer_gap.txt"
    table.to_csv(csv_out, index=False)

    with open(txt_out, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TABLE 6.3: CROSS-TRACE TRANSFER RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(table.to_string(index=False))
        f.write("\n" + "=" * 70 + "\n")

    print(f"      -> {csv_out}")
    print(f"      -> {txt_out}")
    print()
    print(table.to_string(index=False))

def generate_ablation_figure():
    import pandas as pd
    import matplotlib.pyplot as plt
    from config import RESULTS_DIR

    log_path = RESULTS_DIR / "ablation_log.csv"
    if not log_path.exists():
        print(f"[fig] Skipped (missing {log_path})")
        return

    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Reward
    df_sorted = df.sort_values("ablation")
    axes[0].bar(df_sorted["ablation"], df_sorted["reward"], color="#1f77b4")
    axes[0].set_title("(a) Ablation — Reward")
    axes[0].set_ylabel("Reward")
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(df_sorted["reward"]):
        axes[0].text(i, v - 0.02, f"{v:.3f}", ha="center", fontsize=8)

    # Imbalance
    axes[1].bar(df_sorted["ablation"], df_sorted["imbalance"], color="#ff7f0e")
    axes[1].set_title("(b) Ablation — Load Imbalance")
    axes[1].set_ylabel("CV")
    axes[1].tick_params(axis="x", rotation=30)
    for i, v in enumerate(df_sorted["imbalance"]):
        axes[1].text(i, v - 0.03, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "ablation_results.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      -> {out}")

def generate_stats_figure():
    """Figure: reward comparison with confidence intervals."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from config import RESULTS_DIR

    g = pd.read_csv(RESULTS_DIR / "stats_google.csv")
    a = pd.read_csv(RESULTS_DIR / "stats_alibaba.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, title in [(axes[0], g, "(a) Google trace"),
                          (axes[1], a, "(b) Alibaba trace")]:
        labels = ["Always-VM0", "Random", "Round-Robin", "Greedy-Load", "DQN"]
        means = [df.loc[df["baseline"] == b, "baseline_mean"].values[0]
                 if b != "DQN" else df["dqn_mean"].values[0]
                 for b in labels]
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
        ax.bar(labels, means, color=colors)
        ax.set_ylabel("Mean reward")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(means):
            ax.text(i, v - 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = RESULTS_DIR / "stats_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      -> {out}")

def generate_architecture_comparison():
    """Compare MLP DQN, Pointer DQN, and baselines (hardcoded MLP baselines)."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from config import RESULTS_DIR

    # Load pointer stats
    ptr_g = RESULTS_DIR / "stats_google_pointer.csv"
    ptr_a = RESULTS_DIR / "stats_alibaba_pointer.csv"

    if not (ptr_g.exists() and ptr_a.exists()):
        print("[fig] Skipped — need stats_google_pointer.csv and stats_alibaba_pointer.csv")
        return

    df_g = pd.read_csv(ptr_g)
    df_a = pd.read_csv(ptr_a)

    # Extract baseline + pointer DQN means
    def extract(df):
        return {
            "Always-VM0": df.loc[df["baseline"] == "Always-VM0", "baseline_mean"].values[0],
            "Random": df.loc[df["baseline"] == "Random", "baseline_mean"].values[0],
            "Round-Robin": df.loc[df["baseline"] == "Round-Robin", "baseline_mean"].values[0],
            "Greedy-Load": df.loc[df["baseline"] == "Greedy-Load", "baseline_mean"].values[0],
            "Pointer DQN": df["dqn_mean"].values[0],
        }

    g = extract(df_g)
    a = extract(df_a)

    # Hardcoded MLP results from earlier experiments
    g["MLP DQN"] = -0.598
    a["MLP DQN"] = -0.763

    labels = ["Always-VM0", "Random", "Round-Robin", "MLP DQN", "Greedy-Load", "Pointer DQN"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, [g[k] for k in labels], width, label="Google", color="#1f77b4")
    ax.bar(x + width/2, [a[k] for k in labels], width, label="Alibaba", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean reward")
    ax.set_title("Architecture Comparison: MLP vs Pointer DQN")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    out = RESULTS_DIR / "architecture_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      -> {out}")
    
    
# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("GENERATING PAPER FIGURES AND TABLES")
    print("=" * 60)
    generate_training_convergence()
    generate_cross_trace_comparison()
    generate_reward_vs_baselines()
    generate_transfer_gap_table()
    generate_ablation_figure()
    generate_stats_figure()
    generate_architecture_comparison()
    print("\nAll artifacts written to:", RESULTS_DIR)


if __name__ == "__main__":
    main()