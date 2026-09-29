# scripts/vm_scaling_test.py
"""
VM-scaling experiment: evaluate the same trained pointer DQN at
different VM counts without retraining.

This tests the paper's claim that the pointer architecture has
no fixed output head and can be applied to any cluster size.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pandas as pd

from config import (
    PROCESSED_GOOGLE,
    MODEL_PATH,
    RESULTS_DIR,
    SEED,
)

from src.environment import CloudLoadBalancingEnv
from src.agent_v2 import PointerQNetworkV2


VM_COUNTS = [20, 50, 75, 100]
EPISODES_PER_VM = 10
MAX_STEPS = 2000


def evaluate_at_vm_count(net, df, num_vms, num_episodes=EPISODES_PER_VM, seed=SEED):
    """
    Evaluate the network at a given VM count.

    Returns:
        dict with mean/std of reward, response, imbalance.
    """
    env = CloudLoadBalancingEnv(
        df,
        num_vms=num_vms,
        cpu_scale=5.0,
        decay_per_sec=0.95,
        seconds_per_step=0.1,
        max_steps=MAX_STEPS,
    )

    rewards = []
    responses = []
    imbalances = []

    net.eval()

    for ep in range(num_episodes):
        np.random.seed(seed + ep)
        torch.manual_seed(seed + ep)

        obs, _ = env.reset()
        ep_rewards = []
        ep_responses = []
        ep_imbalances = []

        for _ in range(env.max_steps):
            vm_loads = obs[:num_vms]
            task_feats = obs[num_vms:]

            with torch.no_grad():
                vm_t = torch.tensor(vm_loads, dtype=torch.float32).unsqueeze(0)
                task_t = torch.tensor(task_feats, dtype=torch.float32).unsqueeze(0)
                q_vals = net(vm_t, task_t)
                action = int(torch.argmax(q_vals, dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            ep_rewards.append(reward)
            ep_responses.append(info["response_time"])
            ep_imbalances.append(info["imbalance"])

            if terminated or truncated:
                break

        rewards.append(float(np.mean(ep_rewards)))
        responses.append(float(np.mean(ep_responses)))
        imbalances.append(float(np.mean(ep_imbalances)))

    return {
        "vm_count": num_vms,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "response_mean": float(np.mean(responses)),
        "response_std": float(np.std(responses)),
        "imbalance_mean": float(np.mean(imbalances)),
        "imbalance_std": float(np.std(imbalances)),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("VM-SCALING EXPERIMENT")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"VM counts tested: {VM_COUNTS}")
    print(f"Episodes per VM count: {EPISODES_PER_VM}")
    print()

    # Load frozen model
    net = PointerQNetworkV2(task_feat_dim=4, hidden=96)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    print("Model loaded successfully.")
    print()

    # Load trace (only need task features)
    df = pd.read_csv(PROCESSED_GOOGLE)
    print(f"Trace: {len(df)} tasks from Google Borg")
    print()

    # Evaluate at each VM count
    results = []
    for vm_count in VM_COUNTS:
        print(f"Evaluating at {vm_count} VMs...")
        r = evaluate_at_vm_count(net, df, vm_count)
        results.append(r)
        print(f"  reward={r['reward_mean']:+.4f} ± {r['reward_std']:.4f}  "
              f"response={r['response_mean']:.2f} ± {r['response_std']:.2f}  "
              f"imbalance={r['imbalance_mean']:.4f} ± {r['imbalance_std']:.4f}")
        print()

    # Save results
    df_out = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "vm_scaling_results.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    print()

    # Print summary table
    print("=" * 70)
    print("VM-SCALING SUMMARY")
    print("=" * 70)
    print(f"{'VMs':>5}  {'Reward':>12}  {'Response (s)':>15}  {'Imbalance':>15}")
    print("-" * 70)
    for r in results:
        print(f"{r['vm_count']:>5}  "
              f"{r['reward_mean']:>+8.4f}±{r['reward_std']:.3f}  "
              f"{r['response_mean']:>10.2f}±{r['response_std']:.2f}  "
              f"{r['imbalance_mean']:>10.4f}±{r['imbalance_std']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()