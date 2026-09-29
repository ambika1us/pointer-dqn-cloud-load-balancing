# scripts/vm_scaling_multiseed.py
"""
Run VM-scaling evaluation across multiple seeds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from config import PROCESSED_GOOGLE, RESULTS_DIR
from src.environment import CloudLoadBalancingEnv
from src.agent_v2 import PointerQNetworkV2


VM_COUNTS = [20, 50, 75, 100]
SEEDS = [0, 1, 2, 3, 4]


def evaluate_at_vm(vm_count, model_path, seed):
    df = pd.read_csv(PROCESSED_GOOGLE)
    env = CloudLoadBalancingEnv(df, num_vms=vm_count, cpu_scale=5.0,
                                 decay_per_sec=0.95, seconds_per_step=0.1,
                                 max_steps=2000)
    net = PointerQNetworkV2(task_feat_dim=4, hidden=96)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    np.random.seed(seed)
    rewards, responses, imbalances = [], [], []
    for _ in range(10):
        obs, _ = env.reset()
        r_buf, resp_buf, imb_buf = [], [], []
        for _ in range(env.max_steps):
            vm_loads = obs[:vm_count]
            task_feats = obs[vm_count:]
            with torch.no_grad():
                vm_t = torch.tensor(vm_loads, dtype=torch.float32).unsqueeze(0)
                task_t = torch.tensor(task_feats, dtype=torch.float32).unsqueeze(0)
                q_vals = net(vm_t, task_t)
                action = int(torch.argmax(q_vals, dim=1).item())
            obs, r, term, trunc, info = env.step(action)
            r_buf.append(r)
            resp_buf.append(info["response_time"])
            imb_buf.append(info["imbalance"])
            if term or trunc:
                break
        rewards.append(np.mean(r_buf))
        responses.append(np.mean(resp_buf))
        imbalances.append(np.mean(imb_buf))
    return {
    "reward": (float(np.mean(rewards)), float(np.std(rewards))),
    "response": (float(np.mean(responses)), float(np.std(responses))),
    "imbalance": (float(np.mean(imbalances)), float(np.std(imbalances))),
}


if __name__ == "__main__":
    import json

    results = []
    for vm in VM_COUNTS:
        seed_rewards = []
        seed_responses = []
        seed_imbalances = []

        for seed in SEEDS:
            model_path = f"results/seed_{seed}/dqn_google.pt"
            metrics = evaluate_at_vm(vm, model_path, seed)

            seed_rewards.append(metrics["reward"][0])
            seed_responses.append(metrics["response"][0])
            seed_imbalances.append(metrics["imbalance"][0])

        row = {
            "vms": vm,
            "reward_mean": float(np.mean(seed_rewards)),
            "reward_std": float(np.std(seed_rewards)),
            "response_mean": float(np.mean(seed_responses)),
            "response_std": float(np.std(seed_responses)),
            "imbalance_mean": float(np.mean(seed_imbalances)),
            "imbalance_std": float(np.std(seed_imbalances)),
        }
        results.append(row)

        print(f"VMs={vm:3d}: "
              f"reward={row['reward_mean']:+.4f}±{row['reward_std']:.4f}  "
              f"response={row['response_mean']:.2f}±{row['response_std']:.2f}s  "
              f"imbalance={row['imbalance_mean']:.4f}±{row['imbalance_std']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "vm_scaling_multiseed.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'vm_scaling_multiseed.csv'}")