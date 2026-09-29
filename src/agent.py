# src/agent.py
"""
Pointer-style DQN agent.

Q-network architecture:
    - Shared task encoder (MLP over task features)
    - Shared VM encoder (MLP over each VM's load)
    - Shared scorer (MLP over concatenated [vm_emb, task_emb] -> scalar Q)

This makes Q(s, a) permutation-invariant over VMs: the Q-value of assigning
the current task to VM i depends only on VM i's load and the task features,
not on the position of VM i in the observation vector.

Other components are the same as before:
    - Huber loss (SmoothL1Loss)
    - Recent-biased replay sampling
    - Soft target updates (tau=0.01)
    - Gradient clipping (max_norm=1.0)
    - Target clamp [-100, 0]
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    LEARNING_RATE,
    GAMMA,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    BATCH_SIZE,
    REPLAY_BUFFER,
    HIDDEN_DIM,
)


class PointerQNetwork(nn.Module):
    """
    Permutation-invariant Q-network.

    Input: obs = [vm_loads (num_vms), task_features (task_feat_dim)]
    Output: q_per_vm (num_vms) — Q-value of assigning the current task to each VM
    """

    def __init__(self, num_vms: int, task_feat_dim: int = 4, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.num_vms = num_vms
        self.task_feat_dim = task_feat_dim

        # Shared task encoder
        self.task_net = nn.Sequential(
            nn.Linear(task_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # Shared per-VM encoder (input is a single scalar: VM load)
        self.vm_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # Shared scorer
        self.score_net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        # obs shape: (batch, num_vms + task_feat_dim)
        vm_loads = obs[:, : self.num_vms].unsqueeze(-1)       # (batch, num_vms, 1)
        task_feats = obs[:, self.num_vms:]                     # (batch, task_feat_dim)

        # Per-VM embeddings — shared vm_net applied to each VM independently
        vm_emb = self.vm_net(vm_loads)                         # (batch, num_vms, hidden)

        # Task embedding — shared task_net
        task_emb = self.task_net(task_feats)                   # (batch, hidden)

        # Broadcast task embedding to each VM and concatenate
        task_exp = task_emb.unsqueeze(1).expand(-1, self.num_vms, -1)
        combined = torch.cat([vm_emb, task_exp], dim=-1)       # (batch, num_vms, 2*hidden)

        # Score each VM
        q_per_vm = self.score_net(combined).squeeze(-1)        # (batch, num_vms)
        return q_per_vm


class DQNAgent:
    """DQN agent with pointer-style Q-network."""

    def __init__(self, obs_dim: int, num_actions: int, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.device = torch.device(device)

        # Infer task_feat_dim from obs_dim and num_actions
        # obs_dim = num_actions + task_feat_dim
        self.task_feat_dim = obs_dim - num_actions

        # Online and target networks
        self.q_net = PointerQNetwork(
            num_vms=num_actions, task_feat_dim=self.task_feat_dim
        ).to(self.device)
        self.target_net = PointerQNetwork(
            num_vms=num_actions, task_feat_dim=self.task_feat_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.replay = deque(maxlen=REPLAY_BUFFER)

        # Exploration
        self.epsilon = EPSILON_START

        # Soft target update
        self.tau = 0.01

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q_net(obs_t)
            return int(torch.argmax(q_vals, dim=1).item())

    def store(self, obs, action, reward, next_obs, done):
        self.replay.append((obs, action, reward, next_obs, done))

    def train_step(self):
        if len(self.replay) < BATCH_SIZE:
            return None

        # Recent-biased sampling
        replay_list = list(self.replay)
        n = len(replay_list)
        weights = np.linspace(0.5, 1.0, n)
        weights = weights / weights.sum()
        indices = np.random.choice(n, size=BATCH_SIZE, replace=False, p=weights)
        batch = [replay_list[i] for i in indices]

        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a) — gather only the chosen action
        q_all = self.q_net(obs)                                  # (batch, num_actions)
        q_values = q_all.gather(1, actions)                      # (batch, 1)

        # Q-target
        with torch.no_grad():
            next_q_all = self.target_net(next_obs)               # (batch, num_actions)
            next_q = next_q_all.max(dim=1, keepdim=True)[0]      # (batch, 1)
            target = rewards + GAMMA * next_q * (1.0 - dones)
            target = torch.clamp(target, min=-100.0, max=0.0)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft target update
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(),
                                            self.q_net.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = EPSILON_END