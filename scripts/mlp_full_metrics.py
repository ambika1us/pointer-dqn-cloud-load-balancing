# scripts/mlp_full_metrics.py
"""
Compute reward, response, and imbalance for the MLP-DQN model
on both Google and Alibaba traces.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from config import (
    PROCESSED_GOOGLE,
    PROCESSED_ALIBABA,
    NUM_VMS,
    MAX_STEPS_PER_EPISODE,
    SEED,
)

from src.environment import CloudLoadBalancingEnv


# Import MLP model (old architecture)
# You need to have the MLP class defined somewhere
# If not, use the current agent's PointerQNetwork and load MLP weights
from src.agent_v2 import PointerQNetworkV2


def evaluate(model_path, trace_name, df):
    """Run evaluation with the given model."""
    env = CloudLoadBalancingEnv(
        df, num_vms=NUM_VMS, cpu_scale=5.0,
        decay_per_sec=0.95, seconds_per_step=0.1,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    # Load model (adjust based on whether it is MLP or Pointer)
    # If MLP, you need the old agent.py — see note below
    model = PointerQNetworkV2(task_feat_dim=4, hidden=96)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    rewards, responses, imbalances = [], [], []
    for ep in range(30):
        np.random.seed(SEED + ep)
        obs, _ = env.reset()
        ep_r, ep_resp, ep_imb = [], [], []
        for _ in range(env.max_steps):
            vm_loads = obs[:NUM_VMS]
            task_feats = obs[NUM_VMS:]
            with torch.no_grad():
                vm_t = torch.tensor(vm_loads, dtype=torch.float32).unsqueeze(0)
                task_t = torch.tensor(task_feats, dtype=torch.float32).unsqueeze(0)
                q_vals = model(vm_t, task_t)
                action = int(torch.argmax(q_vals, dim=1).item())

            obs, r, term, trunc, info = env.step(action)
            ep_r.append(r)
            ep_resp.append(info["response_time"])
            ep_imb.append(info["imbalance"])
            if term or trunc:
                break
        rewards.append(np.mean(ep_r))
        responses.append(np.mean(ep_resp))
        imbalances.append(np.mean(ep_imb))

    print(f"\n=== {trace_name} ===")
    print(f"Reward:    {np.mean(rewards):+.4f} ± {np.std(rewards):.4f}")
    print(f"Response:  {np.mean(responses):.2f} ± {np.std(responses):.2f} s")
    print(f"Imbalance: {np.mean(imbalances):.4f} ± {np.std(imbalances):.4f}")

    return {
        "reward": (np.mean(rewards), np.std(rewards)),
        "response": (np.mean(responses), np.std(responses)),
        "imbalance": (np.mean(imbalances), np.std(imbalances)),
    }


def main():
    google_df = pd.read_csv(PROCESSED_GOOGLE)
    alibaba_df = pd.read_csv(PROCESSED_ALIBABA)

    # Evaluate MLP on Google
    print("=" * 60)
    print("MLP-DQN evaluation")
    print("=" * 60)
    evaluate("results/dqn_google_mlp.pt", "Google (MLP)", google_df)
    evaluate("results/dqn_google_mlp.pt", "Alibaba (MLP)", alibaba_df)


if __name__ == "__main__":
    main()