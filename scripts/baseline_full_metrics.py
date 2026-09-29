# scripts/baseline_full_metrics.py
"""
Compute reward, response time, and imbalance for all baselines
on the Google trace with 30 evaluation episodes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import PROCESSED_GOOGLE, NUM_VMS, SEED
from src.environment import CloudLoadBalancingEnv


def run_policy(env, policy_fn, num_episodes=30, seed=SEED):
    rewards, responses, imbalances = [], [], []
    for ep in range(num_episodes):
        np.random.seed(seed + ep)
        obs, _ = env.reset()
        ep_r, ep_resp, ep_imb = [], [], []
        for _ in range(env.max_steps):
            action = policy_fn(obs, env)
            obs, r, term, trunc, info = env.step(action)
            ep_r.append(r)
            ep_resp.append(info["response_time"])
            ep_imb.append(info["imbalance"])
            if term or trunc:
                break
        rewards.append(np.mean(ep_r))
        responses.append(np.mean(ep_resp))
        imbalances.append(np.mean(ep_imb))
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "response_mean": float(np.mean(responses)),
        "response_std": float(np.std(responses)),
        "imbalance_mean": float(np.mean(imbalances)),
        "imbalance_std": float(np.std(imbalances)),
    }


def main():
    df = pd.read_csv(PROCESSED_GOOGLE)
    env = CloudLoadBalancingEnv(df, num_vms=NUM_VMS, cpu_scale=5.0,
                                 decay_per_sec=0.95, seconds_per_step=0.1,
                                 max_steps=2000)

    policies = {
        "Always-VM0":  lambda obs, e: 0,
        "Random":      lambda obs, e: np.random.randint(e.num_vms),
        "Round-Robin": lambda obs, e: e.current_step % e.num_vms,
        "Greedy-Load": lambda obs, e: int(np.argmin(e.vm_loads)),
    }

    print("=" * 80)
    print("BASELINE FULL METRICS — Google Trace")
    print("=" * 80)
    print(f"{'Policy':<15} {'Reward':>18} {'Response':>18} {'Imbalance':>18}")
    print("-" * 80)

    results = {}
    for name, fn in policies.items():
        m = run_policy(env, fn)
        results[name] = m
        print(f"{name:<15} "
              f"{m['reward_mean']:>+10.4f}±{m['reward_std']:<5.4f} "
              f"{m['response_mean']:>10.2f}±{m['response_std']:<5.2f} "
              f"{m['imbalance_mean']:>10.4f}±{m['imbalance_std']:<5.4f}")

    # Save for LaTeX
    pd.DataFrame(results).T.to_csv("results/baseline_google_full.csv")
    print()
    print("Saved: results/baseline_google_full.csv")


if __name__ == "__main__":
    main()