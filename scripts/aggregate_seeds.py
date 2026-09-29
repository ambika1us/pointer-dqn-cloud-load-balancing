# scripts/aggregate_seeds.py
"""
Aggregate evaluation results across all trained seeds.
Produces seed-level statistics for the paper.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = Path("results")


def load_seed_results() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        eval_path = RESULTS_DIR / f"seed_{seed}" / "evaluation_log.csv"
        if not eval_path.exists():
            print(f"  WARNING: missing {eval_path}")
            continue
        df = pd.read_csv(eval_path)
        df["seed"] = seed
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No seed evaluation logs found")
    return pd.concat(rows, ignore_index=True)


def aggregate_trace(df: pd.DataFrame, trace_name: str) -> pd.DataFrame:
    sub = df[df["trace"] == trace_name]

    # Per seed: mean across evaluation runs
    per_seed = sub.groupby("seed").agg(
        reward=("avg_reward", "mean"),
        response=("avg_response", "mean"),
        imbalance=("avg_imbalance", "mean"),
    )

    n = len(per_seed)
    summary = {
        "trace": trace_name,
        "seeds": n,
        "reward_mean": per_seed["reward"].mean(),
        "reward_std": per_seed["reward"].std(),
        "response_mean": per_seed["response"].mean(),
        "response_std": per_seed["response"].std(),
        "imbalance_mean": per_seed["imbalance"].mean(),
        "imbalance_std": per_seed["imbalance"].std(),
    }

    # 95% CI for reward
    if n >= 2:
        ci_lo, ci_hi = stats.t.interval(
            0.95, df=n - 1,
            loc=per_seed["reward"].mean(),
            scale=stats.sem(per_seed["reward"]),
        )
        summary["reward_ci_lo"] = ci_lo
        summary["reward_ci_hi"] = ci_hi

    print()
    print("=" * 70)
    print(f"{trace_name.upper()} TRACE — SEED-LEVEL RESULTS")
    print("=" * 70)
    print(per_seed.to_string())
    print()
    print(f"Reward:    {summary['reward_mean']:+.4f} ± {summary['reward_std']:.4f}")
    print(f"Response:  {summary['response_mean']:.2f} ± {summary['response_std']:.2f} s")
    print(f"Imbalance: {summary['imbalance_mean']:.4f} ± {summary['imbalance_std']:.4f}")
    if "reward_ci_lo" in summary:
        print(f"Reward 95% CI: [{summary['reward_ci_lo']:+.4f}, {summary['reward_ci_hi']:+.4f}]")

    return summary, per_seed


def main():
    df = load_seed_results()
    print(f"Loaded {len(df)} records across {df['seed'].nunique()} seeds")

    google_summary, google_per_seed = aggregate_trace(df, "google")
    alibaba_summary, alibaba_per_seed = aggregate_trace(df, "alibaba")

    # Save summary
    summary_df = pd.DataFrame([google_summary, alibaba_summary])
    summary_df.to_csv(RESULTS_DIR / "seed_level_summary.csv", index=False)

    google_per_seed.to_csv(RESULTS_DIR / "seed_level_google.csv")
    alibaba_per_seed.to_csv(RESULTS_DIR / "seed_level_alibaba.csv")

    print()
    print("Saved:")
    print(f"  {RESULTS_DIR / 'seed_level_summary.csv'}")
    print(f"  {RESULTS_DIR / 'seed_level_google.csv'}")
    print(f"  {RESULTS_DIR / 'seed_level_alibaba.csv'}")


if __name__ == "__main__":
    main()