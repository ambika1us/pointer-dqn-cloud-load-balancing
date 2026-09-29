# scripts/finetune_experiment.py
"""
Fine-tuning experiment:
    1. Load the Google-trained DQN
    2. Measure zero-shot performance on Alibaba
    3. Continue training on Alibaba for N episodes
    4. Re-measure performance on Alibaba
    5. Report the improvement

Output:
    results/finetune_log.csv   (per-episode reward during fine-tuning)
    results/finetune_summary.txt
"""

import numpy as np
import pandas as pd
import torch

from config import (
    PROCESSED_GOOGLE,
    PROCESSED_ALIBABA,
    MODEL_PATH,
    RESULTS_DIR,
    NUM_VMS,
    MAX_STEPS_PER_EPISODE,
    SEED,
)

from src.environment import CloudLoadBalancingEnv
from src.agent import DQNAgent


FINETUNE_EPISODES = 10
NUM_EVAL_RUNS = 3
FINETUNE_EPSILON = 0.3       # moderate exploration during fine-tuning


def make_env(df):
    return CloudLoadBalancingEnv(
        df,
        num_vms=NUM_VMS,
        cpu_scale=5.0,
        decay_per_sec=0.95,
        seconds_per_step=0.1,
        max_steps=MAX_STEPS_PER_EPISODE,
    )


def evaluate(env, agent, num_runs=3):
    rewards, responses, imbalances = [], [], []
    for _ in range(num_runs):
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


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[finetune] Loading traces...")
    google_df = pd.read_csv(PROCESSED_GOOGLE)
    alibaba_df = pd.read_csv(PROCESSED_ALIBABA)

    env_google = make_env(google_df)
    env_alibaba = make_env(alibaba_df)

    # Load trained agent
    print(f"[finetune] Loading model from {MODEL_PATH}")
    agent = DQNAgent(
        obs_dim=env_google.observation_space.shape[0],
        num_actions=env_google.action_space.n,
    )
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0   # pure exploitation for baseline eval

    # --- Baseline (zero-shot) on Alibaba ---
    print("[finetune] Baseline zero-shot evaluation on Alibaba...")
    baseline_alibaba = evaluate(env_alibaba, agent, NUM_EVAL_RUNS)
    baseline_google = evaluate(env_google, agent, NUM_EVAL_RUNS)
    print(f"           Google:  reward={baseline_google['reward']:.4f}  "
          f"response={baseline_google['response']:.2f}")
    print(f"           Alibaba: reward={baseline_alibaba['reward']:.4f}  "
          f"response={baseline_alibaba['response']:.2f}")

    # --- Fine-tune on Alibaba ---
    print(f"[finetune] Fine-tuning for {FINETUNE_EPISODES} episodes...")
    agent.epsilon = FINETUNE_EPSILON
    log_rows = []
    for ep in range(FINETUNE_EPISODES):
        obs, _ = env_alibaba.reset()
        ep_reward, ep_resp, ep_imb = [], [], []
        for _ in range(env_alibaba.max_steps):
            action = agent.select_action(obs)
            next_obs, r, t, tr, info = env_alibaba.step(action)
            agent.store(obs, action, r, next_obs, t or tr)
            agent.train_step()
            obs = next_obs
            ep_reward.append(r)
            ep_resp.append(info["response_time"])
            ep_imb.append(info["imbalance"])
            if t or tr:
                break
        agent.decay_epsilon()
        log_rows.append({
            "episode": ep,
            "reward": float(np.mean(ep_reward)),
            "response": float(np.mean(ep_resp)),
            "imbalance": float(np.mean(ep_imb)),
            "epsilon": agent.epsilon,
        })
        print(f"  [ft ep {ep+1:2d}] reward={np.mean(ep_reward):.4f} "
              f"resp={np.mean(ep_resp):.2f} imb={np.mean(ep_imb):.4f}")

    # --- Re-evaluate on Alibaba after fine-tuning ---
    print("[finetune] Post-fine-tune evaluation on Alibaba...")
    agent.epsilon = 0.0
    after_alibaba = evaluate(env_alibaba, agent, NUM_EVAL_RUNS)
    print(f"           Alibaba after: reward={after_alibaba['reward']:.4f}  "
          f"response={after_alibaba['response']:.2f}")

    # --- Save ---
    pd.DataFrame(log_rows).to_csv(RESULTS_DIR / "finetune_log.csv", index=False)

    reward_improvement = (after_alibaba["reward"] - baseline_alibaba["reward"]) / abs(baseline_alibaba["reward"]) * 100
    response_improvement = (baseline_alibaba["response"] - after_alibaba["response"]) / baseline_alibaba["response"] * 100

    summary_lines = [
        "=" * 70,
        "FINE-TUNING EXPERIMENT SUMMARY",
        "=" * 70,
        f"Fine-tuning episodes: {FINETUNE_EPISODES}",
        "",
        "BEFORE FINE-TUNING (zero-shot Alibaba):",
        f"  Reward:        {baseline_alibaba['reward']:.4f}",
        f"  Response time: {baseline_alibaba['response']:.2f} sec",
        f"  Imbalance:     {baseline_alibaba['imbalance']:.4f}",
        "",
        "AFTER FINE-TUNING (Alibaba):",
        f"  Reward:        {after_alibaba['reward']:.4f}",
        f"  Response time: {after_alibaba['response']:.2f} sec",
        f"  Imbalance:     {after_alibaba['imbalance']:.4f}",
        "",
        f"Reward improvement:   {reward_improvement:+.1f}%",
        f"Response improvement: {response_improvement:+.1f}%",
        "=" * 70,
    ]
    summary_txt = "\n".join(summary_lines)
    print("\n" + summary_txt)

    with open(RESULTS_DIR / "finetune_summary.txt", "w") as f:
        f.write(summary_txt)


if __name__ == "__main__":
    main()