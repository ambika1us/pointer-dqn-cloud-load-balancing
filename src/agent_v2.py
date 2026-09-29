# src/agent_v2.py
"""
VM-count-agnostic Pointer-style Q-network.

Differences from src/agent.py:
    - forward() accepts vm_loads and task_feats as separate tensors
    - Network does not store num_vms as a fixed attribute
    - Network supports evaluation at different VM counts without retraining

The layer architecture is identical to the original PointerQNetwork,
so existing trained weights load without modification.
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


class PointerQNetworkV2(nn.Module):
    """
    Permutation-equivariant Q-network that scales to any number of VMs.

    Architecture:
        task_net:  MLP(task_feat_dim -> hidden -> hidden)
        vm_net:    MLP(1 -> hidden -> hidden)
        score_net: MLP(2*hidden -> hidden -> 1)

    Forward:
        Input: vm_loads (batch, num_vms), task_feats (batch, task_feat_dim)
        Output: Q-values (batch, num_vms)
    """

    def __init__(self, task_feat_dim: int = 4, hidden: int = HIDDEN_DIM):
        super().__init__()
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

    def forward(self, vm_loads: torch.Tensor, task_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vm_loads:   (batch, num_vms)    normalized VM loads in [0, 1]
            task_feats: (batch, task_feat_dim)

        Returns:
            q_values:   (batch, num_vms)
        """
        # Add trailing dim for vm_net: (batch, num_vms, 1)
        vm_loads = vm_loads.unsqueeze(-1)

        # Encode each VM independently
        vm_emb = self.vm_net(vm_loads)                       # (batch, num_vms, hidden)

        # Encode the task
        task_emb = self.task_net(task_feats)                 # (batch, hidden)

        # Broadcast task embedding across all VMs
        num_vms = vm_emb.size(1)
        task_exp = task_emb.unsqueeze(1).expand(-1, num_vms, -1)   # (batch, num_vms, hidden)

        # Concatenate per-VM and task embeddings
        combined = torch.cat([vm_emb, task_exp], dim=-1)     # (batch, num_vms, 2*hidden)

        # Score each VM
        q_per_vm = self.score_net(combined).squeeze(-1)      # (batch, num_vms)

        return q_per_vm


class DQNAgentV2:
    """
    DQN agent wrapper compatible with PointerQNetworkV2.

    Observation convention:
        obs is a flat vector [vm_loads..., task_feats...]
        The agent splits obs internally using self.num_vms and self.task_feat_dim.

    num_vms is passed at init for the training environment, but the network
    itself does not depend on it — different VM counts can be evaluated
    with the same loaded weights.
    """

    def __init__(self, num_vms: int, task_feat_dim: int = 4, device: str = "cpu"):
        self.num_vms = num_vms
        self.task_feat_dim = task_feat_dim
        self.device = torch.device(device)

        self.q_net = PointerQNetworkV2(
            task_feat_dim=task_feat_dim, hidden=HIDDEN_DIM
        ).to(self.device)
        self.target_net = PointerQNetworkV2(
            task_feat_dim=task_feat_dim, hidden=HIDDEN_DIM
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = deque(maxlen=REPLAY_BUFFER)
        self.epsilon = EPSILON_START
        self.tau = 0.01

    def _split_obs(self, obs: np.ndarray):
        """Split flat observation into VM loads and task features."""
        vm_loads = obs[: self.num_vms]
        task_feats = obs[self.num_vms :]
        return vm_loads, task_feats

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_vms - 1)

        vm_loads, task_feats = self._split_obs(obs)
        with torch.no_grad():
            vm_t = torch.tensor(vm_loads, dtype=torch.float32, device=self.device).unsqueeze(0)
            task_t = torch.tensor(task_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q_net(vm_t, task_t)
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

        # Split observations
        vm_obs = torch.tensor(
            np.array([o[: self.num_vms] for o in obs]),
            dtype=torch.float32, device=self.device,
        )
        task_obs = torch.tensor(
            np.array([o[self.num_vms :] for o in obs]),
            dtype=torch.float32, device=self.device,
        )
        vm_next = torch.tensor(
            np.array([o[: self.num_vms] for o in next_obs]),
            dtype=torch.float32, device=self.device,
        )
        task_next = torch.tensor(
            np.array([o[self.num_vms :] for o in next_obs]),
            dtype=torch.float32, device=self.device,
        )

        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q(s, a)
        q_all = self.q_net(vm_obs, task_obs)
        q_values = q_all.gather(1, actions)

        # Target
        with torch.no_grad():
            next_q_all = self.target_net(vm_next, task_next)
            next_q = next_q_all.max(dim=1, keepdim=True)[0]
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
        """Load weights from a model saved by the original PointerQNetwork.

        The layer names and shapes are identical, so load_state_dict works
        without modification.
        """
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = EPSILON_END