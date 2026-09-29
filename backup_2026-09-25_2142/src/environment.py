# src/environment.py
"""
Cloud VM load balancing environment with:
  - Time-scaled load dynamics (scales from 4 to 100+ VMs)
  - M/M/1 response time capped at 5x task duration
  - Single-objective reward for stable learning
  - Imbalance reported in info dict for logging
  - All rewards bounded in [-1, 0]
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config import NUM_VMS, MAX_QUEUE


class CloudLoadBalancingEnv(gym.Env):
    """
    Observation (dim = num_vms + 4):
        [vm_loads (num_vms), task_cpu, task_mem, task_duration_norm, queue_norm]

    Action:
        Discrete(num_vms) — which VM to route the next task to

    Reward (in [-1, 0]):
        -norm_response, where norm_response ∈ [0, 1]

    Dynamics:
        - Assigning a task adds cpu*scale to the VM's load
        - Load decays by decay_per_sec ** seconds_per_step every step
        - Response time follows M/M/1, capped at 5x baseline
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace_df: pd.DataFrame,
        num_vms: int = NUM_VMS,
        max_queue: int = MAX_QUEUE,
        max_steps: int = 2000,
        decay_per_sec: float = 0.95,
        seconds_per_step: float = 0.1,
        cpu_scale: float = 5.0,
    ):
        super().__init__()
        self.trace = trace_df.reset_index(drop=True)
        self.num_vms = num_vms
        self.max_queue = max_queue
        self.max_steps = min(max_steps, len(self.trace) - 1)

        # Time-scaled decay
        self.decay_factor = decay_per_sec ** seconds_per_step
        self.cpu_scale = cpu_scale

        # Spaces
        self.action_space = spaces.Discrete(num_vms)
        obs_dim = num_vms + 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vm_loads = np.zeros(self.num_vms, dtype=np.float32)
        self.queue_length = 0
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        task = self.trace.iloc[self.current_step]
        task_feats = np.array(
            [
                float(task["cpu"]),
                float(task["memory"]),
                min(float(task["duration"]) / 300.0, 1.0),
                min(self.queue_length / self.max_queue, 1.0),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([self.vm_loads, task_feats])
        return obs.astype(np.float32)

    def step(self, action):
        task = self.trace.iloc[self.current_step]
        cpu = float(task["cpu"]) * self.cpu_scale
        duration = float(task["duration"])

        vm = int(action)
        load_before = float(self.vm_loads[vm])

        # --- M/M/1 response time, capped at 5x baseline ---
        raw_service = duration * (1.0 / max(1.0 - load_before, 0.2))
        response_time = min(raw_service, 5.0 * duration)

        # --- Time-scaled decay applied to ALL VMs ---
        self.vm_loads *= self.decay_factor

        # --- Add new task load to the selected VM ---
        self.vm_loads[vm] = min(1.0, self.vm_loads[vm] + cpu)

        # --- Imbalance (coefficient of variation), reported in info ---
        mean_load = float(np.mean(self.vm_loads)) + 1e-6
        imbalance = min(float(np.std(self.vm_loads) / mean_load), 1.0)

        # --- Normalized response time ---
        reference = duration * 1.0
        norm_response = min(response_time / (reference * 5.0), 1.0)

        # --- Single-objective reward in [-1, 0] ---
        #reward = -norm_response
        reward = -(0.7 * norm_response + 0.3 * imbalance)
        reward = max(-1.0, min(0.0, reward))

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, {
            "response_time": response_time,
            "imbalance": imbalance,
            "vm_load_mean": float(np.mean(self.vm_loads)),
            "vm_load_max": float(np.max(self.vm_loads)),
        }