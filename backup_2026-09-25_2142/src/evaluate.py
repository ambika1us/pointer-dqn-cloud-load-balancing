# src/evaluate.py
import numpy as np
import pandas as pd
import torch

from config import (
    PROCESSED_GOOGLE,
    PROCESSED_ALIBABA,
    MODEL_PATH,
    EVAL_LOG_PATH,
    RESULTS_DIR,
    NUM_VMS,
    MAX_STEPS_PER_EPISODE,
    SEED,
)

from src.environment import CloudLoadBalancingEnv
from src.agent import DQNAgent


def run_episode(env, agent):
    obs, _ = env.reset()
    response_times = []
    imbalances = []
    rewards = []

    for _ in range(env.max_steps):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        response_times.append(info["response_time"])
        imbalances.append(info["imbalance"])
        rewards.append(reward)
        if terminated or truncated:
            break

    return {
        "avg_response": float(np.mean(response_times)),
        "avg_imbalance": float(np.mean(imbalances)),
        "avg_reward": float(np.mean(rewards)),
        "p95_response": float(np.percentile(response_times, 95)),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[eval] Loading processed Google trace...")
    google_df = pd.read_csv(PROCESSED_GOOGLE)

    print("[eval] Loading processed Alibaba trace (if available)...")
    try:
        alibaba_df = pd.read_csv(PROCESSED_ALIBABA)
        has_alibaba = True
    except FileNotFoundError:
        print("[eval] Alibaba trace not found — running Google-only evaluation")
        alibaba_df = None
        has_alibaba = False

    google_env = CloudLoadBalancingEnv(google_df, num_vms=NUM_VMS,cpu_scale=5.0,decay_per_sec=0.95,seconds_per_step=0.1,max_steps=MAX_STEPS_PER_EPISODE,)
    agent = DQNAgent(
        obs_dim=google_env.observation_space.shape[0],
        num_actions=google_env.action_space.n,
    )
    agent.load(MODEL_PATH)

    results = []

    print("[eval] Evaluating on Google trace...")
    for run in range(5):
        metrics = run_episode(google_env, agent)
        metrics.update({"trace": "google", "run": run})
        results.append(metrics)

    if has_alibaba:
        alibaba_env = CloudLoadBalancingEnv(alibaba_df, num_vms=NUM_VMS)
        print("[eval] Evaluating on Alibaba trace (zero-shot transfer)...")
        for run in range(5):
            metrics = run_episode(alibaba_env, agent)
            metrics.update({"trace": "alibaba", "run": run})
            results.append(metrics)

    df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EVAL_LOG_PATH, index=False)

    print("\n[eval] Summary:")
    print(df.groupby("trace")[["avg_response", "avg_imbalance", "avg_reward"]].mean())
    print(f"\n[eval] Full results -> {EVAL_LOG_PATH}")


if __name__ == "__main__":
    main()