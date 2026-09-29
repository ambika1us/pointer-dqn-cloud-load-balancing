# scripts/generate_vm_scaling_figure.py
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR

def main():
    df = pd.read_csv(RESULTS_DIR / "vm_scaling_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["vm_count"], df["reward_mean"], marker="o", color="#1f77b4")
    axes[0].axvline(50, ls="--", color="red", alpha=0.5, label="Training (50 VMs)")
    axes[0].set_xlabel("Number of VMs")
    axes[0].set_ylabel("Mean reward")
    axes[0].set_title("(a) Reward vs VM count")
    axes[0].legend()

    axes[1].plot(df["vm_count"], df["response_mean"], marker="o", color="#ff7f0e")
    axes[1].axvline(50, ls="--", color="red", alpha=0.5)
    axes[1].set_xlabel("Number of VMs")
    axes[1].set_ylabel("Response time (s)")
    axes[1].set_title("(b) Response time vs VM count")

    axes[2].plot(df["vm_count"], df["imbalance_mean"], marker="o", color="#2ca02c")
    axes[2].axvline(50, ls="--", color="red", alpha=0.5)
    axes[2].set_xlabel("Number of VMs")
    axes[2].set_ylabel("Load imbalance (CV)")
    axes[2].set_title("(c) Imbalance vs VM count")

    plt.tight_layout()
    out = RESULTS_DIR / "vm_scaling_figure.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()