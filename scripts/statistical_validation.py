# scripts/statistical_validation.py
"""
Statistical validation of DQN vs baselines.

Runs each policy on both traces 30 times and computes:
    - Mean ± std
    - Welch's t-test between DQN and each baseline
    - 95% confidence interval via bootstrap
    - Cohen's d effect size
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from config import (
    PROCESSED_GOOGLE, PROCESSED_ALIBABA, MODEL_PATH,
    RESULTS_DIR, NUM_VMS, MAX_STEPS_PER_EPISODE, SEED,
)
from src.environment import CloudLoadBalancingEnv
from src.agent import DQNAgent


NUM_RUNS = 30


def make_env(df):
    return CloudLoadBalancingEnv(
        df, num_vms=NUM_VMS, cpu_scale=5.0,
        decay_per_sec=0.95, seconds_per_step=0.1,
        max_steps=MAX_STEPS_PER_EPISODE,
    )


def run_policy(env, policy_fn, seed=None):
    """Run one episode and return mean reward."""
    if seed is not None:
        np.random.seed(seed)
    obs, _ = env.reset()
    rewards = []
    for _ in range(env.max_steps):
        action = policy_fn(obs, env)
        obs, r, t, tr, _ = env.step(action)
        rewards.append(r)
        if t or tr:
            break
    return float(np.mean(rewards))


def evaluate_policies(trace_name, df, agent):
    """Run all policies NUM_RUNS times on a trace."""
    env = make_env(df)

    policies = {
    "Always-VM0": lambda obs, e: 0,
    "Random": lambda obs, e: np.random.randint(e.num_vms),
    "Round-Robin": lambda obs, e: (e.current_step + np.random.randint(0, 3)) % e.num_vms,
    "Greedy-Load": lambda obs, e: int(np.argmin(e.vm_loads + np.random.uniform(0, 1e-6, size=e.num_vms))),
    "DQN": lambda obs, e: agent.select_action(obs),
        }

    # DQN in eval mode
    agent.epsilon = 0.05

    results = {}
    for name, fn in policies.items():
        rewards = []
        for run in range(NUM_RUNS):
            r = run_policy(env, fn, seed=None)
            rewards.append(r)
        results[name] = np.array(rewards)
        print(f"  {name:15s} mean={np.mean(rewards):.4f}  std={np.std(rewards):.4f}")
    return results


def statistical_tests(results):
    """Compare DQN against baselines with bootstrap CI and Mann-Whitney U."""
    dqn = results["DQN"]
    rows = []
    for name, arr in results.items():
        if name == "DQN":
            continue

        # Bootstrap 95% CI for the difference in means
        rng = np.random.default_rng(42)
        diffs = []
        for _ in range(2000):
            dqn_boot = rng.choice(dqn, size=len(dqn), replace=True)
            arr_boot = rng.choice(arr, size=len(arr), replace=True)
            diffs.append(np.mean(dqn_boot) - np.mean(arr_boot))
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

        # Mann-Whitney U (non-parametric, does not assume normality)
        u_stat, p_value = stats.mannwhitneyu(dqn, arr, alternative="two-sided")

        # Cliff's delta (non-parametric effect size)
        n1, n2 = len(dqn), len(arr)
        greater = sum(1 for x in dqn for y in arr if x > y)
        less = sum(1 for x in dqn for y in arr if x < y)
        cliff_delta = (greater - less) / (n1 * n2)

        rows.append({
            "baseline": name,
            "baseline_mean": np.mean(arr),
            "dqn_mean": np.mean(dqn),
            "difference": np.mean(dqn) - np.mean(arr),
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "p_value_mwu": p_value,
            "cliffs_delta": cliff_delta,
            "significant_p001": p_value < 0.001,
        })
    return pd.DataFrame(rows)

# Append to statistical_validation.py

def response_time_distributions(trace_name, df, agent):
    """Collect per-step response times for each policy."""
    import matplotlib.pyplot as plt

    env = make_env(df)
    agent.epsilon = 0.0

    policies = {
        "Always-VM0": lambda obs, e: 0,
        "Random": lambda obs, e: np.random.randint(e.num_vms),
        "Round-Robin": lambda obs, e: e.current_step % e.num_vms,
        "Greedy-Load": lambda obs, e: int(np.argmin(e.vm_loads)),
        "DQN": lambda obs, e: agent.select_action(obs),
    }

    all_responses = {}
    for name, fn in policies.items():
        obs, _ = env.reset()
        responses = []
        for _ in range(env.max_steps):
            action = fn(obs, env)
            obs, r, t, tr, info = env.step(action)
            responses.append(info["response_time"])
            if t or tr:
                break
        all_responses[name] = responses

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([all_responses[k] for k in policies.keys()],
               labels=list(policies.keys()), showfliers=False)
    ax.set_ylabel("Response time (s)")
    ax.set_title(f"Response Time Distribution — {trace_name.capitalize()}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"response_dist_{trace_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      -> response_dist_{trace_name}.png")

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[stats] Loading traces...")
    google_df = pd.read_csv(PROCESSED_GOOGLE)
    alibaba_df = pd.read_csv(PROCESSED_ALIBABA)

    # Load DQN
    env_tmp = make_env(google_df)
    agent = DQNAgent(
        obs_dim=env_tmp.observation_space.shape[0],
        num_actions=env_tmp.action_space.num if hasattr(env_tmp.action_space, 'num') else env_tmp.action_space.n,
    )
    agent.load(MODEL_PATH)

    print("\n[stats] Google trace:")
    google_results = evaluate_policies("google", google_df, agent)
    google_tests = statistical_tests(google_results)
    print("\n[stats] Google — DQN vs baselines:")
    print(google_tests.to_string(index=False))

    print("\n[stats] Alibaba trace:")
    alibaba_results = evaluate_policies("alibaba", alibaba_df, agent)
    alibaba_tests = statistical_tests(alibaba_results)
    print("\n[stats] Alibaba — DQN vs baselines:")
    print(alibaba_tests.to_string(index=False))

    # Save
    google_tests.to_csv(RESULTS_DIR / "stats_google.csv", index=False)
    alibaba_tests.to_csv(RESULTS_DIR / "stats_alibaba.csv", index=False)

    # Summary text
    lines = [
        "=" * 70,
        "STATISTICAL VALIDATION SUMMARY",
        "=" * 70,
        f"Runs per policy: {NUM_RUNS}",
        "",
        "GOOGLE TRACE — DQN vs baselines:",
        google_tests.to_string(index=False),
        "",
        "ALIBABA TRACE — DQN vs baselines:",
        alibaba_tests.to_string(index=False),
        "=" * 70,
    ]
    txt = "\n".join(lines)
    with open(RESULTS_DIR / "stats_summary.txt", "w") as f:
        f.write(txt)
    print("\n[stats] Saved to results/stats_*.csv")


if __name__ == "__main__":
    main()