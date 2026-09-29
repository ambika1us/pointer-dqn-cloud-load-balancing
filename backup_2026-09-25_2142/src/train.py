# src/train.py
import time
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from config import (
    PROCESSED_GOOGLE,
    MODEL_PATH,
    TRAIN_LOG_PATH,
    RESULTS_DIR,
    NUM_EPISODES,
    NUM_VMS,
    SEED,
    LOG_INTERVAL,
    MAX_STEPS_PER_EPISODE,
)

from src.environment import CloudLoadBalancingEnv
from src.agent import DQNAgent
torch.set_num_threads(6)

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[train] Loading processed Google trace...")
    df = pd.read_csv(PROCESSED_GOOGLE)
    print(f"[train] Loaded {len(df)} tasks")

    env = CloudLoadBalancingEnv(
        df,
        num_vms=NUM_VMS,
        cpu_scale=5.0,
        decay_per_sec=0.95,
        seconds_per_step=0.1,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = DQNAgent(obs_dim=obs_dim, num_actions=num_actions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_rows = []

    print(f"[train] Starting training for {NUM_EPISODES} episodes...")
    for episode in tqdm(range(NUM_EPISODES), desc="Training"):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_response = 0.0
        episode_imbalance = 0.0
        losses = []

        for step in range(env.max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            episode_reward += reward
            episode_response += info["response_time"]
            episode_imbalance += info["imbalance"]

            if done:
                break

        agent.decay_epsilon()

        avg_reward = episode_reward / (step + 1)
        avg_response = episode_response / (step + 1)
        avg_imbalance = episode_imbalance / (step + 1)
        avg_loss = float(np.mean(losses)) if losses else 0.0

        log_rows.append({
            "episode": episode,
            "avg_reward": avg_reward,
            "avg_response": avg_response,
            "avg_imbalance": avg_imbalance,
            "avg_loss": avg_loss,
            "epsilon": agent.epsilon,
        })

        if (episode + 1) % LOG_INTERVAL == 0:
            print(
                f"[ep {episode + 1:4d}] "
                f"reward={avg_reward:+.4f} "
                f"resp={avg_response:7.2f} "
                f"imb={avg_imbalance:.4f} "
                f"eps={agent.epsilon:.4f} "
                f"loss={avg_loss:.4f}"
            )

    print("[train] Saving model...")
    agent.save(MODEL_PATH)

    print("[train] Saving training log...")
    pd.DataFrame(log_rows).to_csv(TRAIN_LOG_PATH, index=False)

    print(f"[train] Done. Model -> {MODEL_PATH}, log -> {TRAIN_LOG_PATH}")


if __name__ == "__main__":
    main()