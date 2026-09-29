# scripts/run_all_seeds.py
"""
Run training + evaluation for all seeds sequentially.
Designed to be launched overnight.

Each seed:
    - trains 200 episodes
    - evaluates on Google + Alibaba
    - writes to results/seed_{seed}/
"""
import os
import subprocess
import sys
from pathlib import Path

SEEDS = [0, 1, 2, 3, 4]   # 5 seeds


def run_seed(seed: int) -> bool:
    print(f"\n{'=' * 70}")
    print(f"SEED {seed}")
    print(f"{'=' * 70}\n")

    env = os.environ.copy()
    env["RL_SEED"] = str(seed)

    project_root = Path(__file__).resolve().parent.parent

    # Train
    print(f"[seed {seed}] Training...")
    result = subprocess.run(
        [sys.executable, "-m", "src.train"],
        env=env,
        cwd=project_root,
    )
    if result.returncode != 0:
        print(f"[seed {seed}] Training FAILED")
        return False

    # Evaluate
    print(f"[seed {seed}] Evaluating...")
    result = subprocess.run(
        [sys.executable, "-m", "src.evaluate"],
        env=env,
        cwd=project_root,
    )
    if result.returncode != 0:
        print(f"[seed {seed}] Evaluation FAILED")
        return False

    return True


def main():
    completed = []
    failed = []

    for seed in SEEDS:
        try:
            if run_seed(seed):
                completed.append(seed)
                print(f"\n[seed {seed}] COMPLETE\n")
            else:
                failed.append(seed)
        except KeyboardInterrupt:
            print(f"\nInterrupted at seed {seed}")
            break

    print(f"\n{'=' * 70}")
    print("BATCH SUMMARY")
    print(f"{'=' * 70}")
    print(f"Completed seeds: {completed}")
    print(f"Failed seeds:    {failed}")
    print(f"Results saved to: results/seed_*/")


if __name__ == "__main__":
    main()