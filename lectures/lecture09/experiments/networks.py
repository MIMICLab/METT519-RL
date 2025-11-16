"""Neural network modules shared across Lecture 9 experiments."""
from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared-body actor-critic network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value.squeeze(-1)
