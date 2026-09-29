# scripts/ablation_experiment.py
"""
Ablation experiment:
    Test how sensitive the trained policy is to specific features of
    the evaluation trace. We manipulate the Alibaba trace in three ways
    and measure the impact on the trained policy.

Ablations:
    A. Duration shuffle      - shuffle task durations randomly
    B. Poisson arrivals      - replace inter-arrival times with Poisson
    C. Feature ablation      - zero out task feature dimension
    D. VM count scaling      - evaluate at 25 / 50 / 100 VMs

Output:
    results/ablation_log.csv
    results/ablation_summary.txt
"""

import numpy as np
import pandas as pd
import torch

from config import (
    PROCESSED_ALIBABA,
    PROCESSED_GOOGLE,
    MODEL_PATH,
    RESULTS_DIR,
    MAX_STEPS_PER_EPISODE,
    SEED,
)

from src.environment import CloudLoadBalancingEnv
from src.agent import DQNAgent


NUM_EVAL_RUNS = 3


def make_env(df, num_vms):
    return CloudLoadBalancingEnv(
        df,
        num_vms=num_vms,
        cpu_scale=5.0,
        decay_per_sec=0.95,
        seconds_per_step=0.1,
        max_steps=MAX_STEPS_PER_EPISODE,
    )


def evaluate(env, agent):
    rewards, responses, imbalances = [], [], []
    for _ in range(NUM_EVAL_RUNS):
        obs, _ = env.reset()
        r_buf, resp_buf, imb_buf = [], [], []
        for _ in range(env.max_steps):
            action = agent.select_action(obs)
            obs, r, t, tr, info = env.step(action)
            r_buf.append(r)
            resp_buf.append(info["response_time"])
            imb_buf.append(info["imbalance"])
            if t or tr:
                break
        rewards.append(np.mean(r_buf))
        responses.append(np.mean(resp_buf))
        imbalances.append(np.mean(imb_buf))
    return {
        "reward": float(np.mean(rewards)),
        "response": float(np.mean(responses)),
        "imbalance": float(np.mean(imbalances)),
    }


# ============================================================
# Ablation manipulations
# ============================================================

def shuffle_durations(df):
    """Randomly permute durations across tasks."""
    df = df.copy()
    rng = np.random.default_rng(SEED)
    df["duration"] = rng.permutation(df["duration"].values)
    return df


def poisson_arrivals(df):
    """Replace inter-arrival times with Poisson at the same mean rate."""
    df = df.copy().sort_values("time").reset_index(drop=True)
    if len(df) < 2:
        return df
    inter = np.diff(df["time"].values)
    mean_iat = max(np.mean(inter), 1e-3)
    rng = np.random.default_rng(SEED)
    new_iat = rng.exponential(mean_iat, size=len(df) - 1)
    df["time"] = np.concatenate([[0.0], np.cumsum(new_iat)])
    return df


def zero_task_features(df):
    """Remove task-level features (cpu, memory, duration) — leaves only VM state."""
    df = df.copy()
    df["cpu"] = 0.5   # constant
    df["memory"] = 0.5
    df["duration"] = 60.0
    return df


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[ablation] Loading traces...")
    google_df = pd.read_csv(PROCESSED_GOOGLE)
    alibaba_df = pd.read_csv(PROCESSED_ALIBABA)

    agent_env = make_env(google_df, num_vms=50)
    agent = DQNAgent(
        obs_dim=agent_env.observation_space.shape[0],
        num_actions=agent_env.action_space.n,
    )
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0

    results = []

    # --- Baseline: unmodified Alibaba ---
    print("\n[ablation] Baseline: unmodified Alibaba (50 VMs)")
    env = make_env(alibaba_df, num_vms=50)
    m = evaluate(env, agent)
    m.update({"ablation": "baseline", "num_vms": 50})
    results.append(m)
    print(f"  reward={m['reward']:.4f}  response={m['response']:.2f}  imb={m['imbalance']:.4f}")

    # --- A. Duration shuffle ---
    print("\n[ablation] A: Duration shuffle")
    df_ablated = shuffle_durations(alibaba_df)
    env = make_env(df_ablated, num_vms=50)
    m = evaluate(env, agent)
    m.update({"ablation": "duration_shuffle", "num_vms": 50})
    results.append(m)
    print(f"  reward={m['reward']:.4f}  response={m['response']:.2f}  imb={m['imbalance']:.4f}")

    # --- B. Poisson arrivals ---
    print("\n[ablation] B: Poisson arrivals")
    df_ablated = poisson_arrivals(alibaba_df)
    env = make_env(df_ablated, num_vms=50)
    m = evaluate(env, agent)
    m.update({"ablation": "poisson_arrivals", "num_vms": 50})
    results.append(m)
    print(f"  reward={m['reward']:.4f}  response={m['response']:.2f}  imb={m['imbalance']:.4f}")

    # --- C. Zero task features ---
    print("\n[ablation] C: Zero task features")
    df_ablated = zero_task_features(alibaba_df)
    env = make_env(df_ablated, num_vms=50)
    m = evaluate(env, agent)
    m.update({"ablation": "zero_task_features", "num_vms": 50})
    results.append(m)
    print(f"  reward={m['reward']:.4f}  response={m['response']:.2f}  imb={m['imbalance']:.4f}")

    

    # --- Save ---
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "ablation_log.csv", index=False)

    # --- Summary text ---
    lines = [
        "=" * 70,
        "ABLATION EXPERIMENT SUMMARY",
        "=" * 70,
        df.to_string(index=False),
        "=" * 70,
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(RESULTS_DIR / "ablation_summary.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()