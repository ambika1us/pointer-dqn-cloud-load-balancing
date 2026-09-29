# scripts/paired_statistical_validation.py
"""
Paired statistical validation of DQN vs baselines.

For each (trace, baseline) pair, compute:
    - Mean paired difference
    - 95% bootstrap CI for the difference
    - Wilcoxon signed-rank test p-value
    - Cliff's delta effect size
    - Holm-Bonferroni corrected p-value across all 4 tests
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR


SEEDS = [0, 1, 2, 3, 4]
TRACES = ["google", "alibaba"]
BASELINES = ["Greedy-Load", "Round-Robin"]


def load_dqn_rewards(trace):
    """Load per-episode rewards from each seed's evaluation log."""
    rewards = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"seed_{seed}" / "evaluation_log.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        sub = df[df["trace"] == trace]
        rewards.extend(sub["avg_reward"].tolist())
    return np.array(rewards)


def load_baseline_rewards(trace, baseline_name):
    """Baseline rewards come from the statistical validation CSV."""
    path = RESULTS_DIR / f"stats_{trace}_pointer.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    row = df[df["baseline"] == baseline_name]
    if len(row) == 0:
        raise ValueError(f"No {baseline_name} row in {path}")
    return row.iloc[0]


def paired_analysis(dqn_rewards, baseline_mean, baseline_name):
    """Perform a paired analysis using per-episode DQN rewards against the
    baseline mean (which is deterministic for these baselines)."""
    # Since baselines are deterministic, the paired difference per episode is
    # simply the DQN reward minus the baseline mean reward.
    paired_diff = dqn_rewards - baseline_mean

    n = len(paired_diff)
    mean_diff = float(np.mean(paired_diff))

    # Bootstrap 95% CI for the mean difference
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(10000):
        sample = rng.choice(paired_diff, size=n, replace=True)
        boot.append(np.mean(sample))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # Wilcoxon signed-rank test (H0: median difference = 0)
    # Skip if all differences are identical (deterministic baseline vs.
    # constant DQN reward)
    if np.allclose(paired_diff, paired_diff[0]):
        w_stat, p_value = np.nan, 1.0
    else:
        try:
            w_stat, p_value = stats.wilcoxon(paired_diff, alternative="two-sided")
        except Exception:
            w_stat, p_value = np.nan, np.nan

    # Cliff's delta (paired form: fraction of pairs where DQN > baseline)
    delta = float(np.mean(np.sign(paired_diff)))

    return {
        "baseline": baseline_name,
        "mean_diff": mean_diff,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p_value),
        "cliffs_delta": delta,
        "n_pairs": n,
    }


def holm_bonferroni(p_values):
    """Return corrected p-values using the Holm-Bonferroni method."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    corrected = np.zeros_like(sorted_p)
    running_max = 0.0
    for i, p in enumerate(sorted_p):
        adjusted = min(p * (m - i), 1.0)
        running_max = max(running_max, adjusted)
        corrected[i] = running_max
    # Reorder back
    out = np.zeros_like(corrected)
    out[sorted_indices] = corrected
    return out


def main():
    rows = []
    p_values_raw = []

    for trace in TRACES:
        dqn = load_dqn_rewards(trace)
        if len(dqn) == 0:
            print(f"No DQN rewards found for {trace}")
            continue

        for baseline in BASELINES:
            meta = load_baseline_rewards(trace, baseline)
            baseline_mean = float(meta["baseline_mean"])
            result = paired_analysis(dqn, baseline_mean, baseline)
            result["trace"] = trace
            rows.append(result)
            p_values_raw.append(result["p_value"])

    # Apply Holm-Bonferroni correction
    corrected = holm_bonferroni(p_values_raw)
    for row, pc in zip(rows, corrected):
        row["p_corrected"] = float(pc)

    # Build and save table
    df = pd.DataFrame(rows)[[
        "trace", "baseline", "mean_diff",
        "ci_lo", "ci_hi", "p_value", "p_corrected",
        "cliffs_delta", "n_pairs",
    ]]

    print("=" * 100)
    print("PAIRED STATISTICAL VALIDATION")
    print("=" * 100)
    print(df.to_string(index=False))

    out_path = RESULTS_DIR / "paired_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()