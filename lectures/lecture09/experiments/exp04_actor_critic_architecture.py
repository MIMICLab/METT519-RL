#!/usr/bin/env python3
"""Lecture 9 Experiment 04: Actor-Critic network architecture."""
from __future__ import annotations

import torch

from helpers import get_device, set_seed
from networks import ActorCritic


def main() -> None:
    print("=" * 60)
    print("Experiment 04: Actor-Critic Architecture")
    print("=" * 60)

    set_seed(7)
    device = get_device()

    obs_dim, act_dim = 4, 2
    model = ActorCritic(obs_dim, act_dim).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    actor_params = sum(p.numel() for p in model.actor_head.parameters())
    critic_params = sum(p.numel() for p in model.critic_head.parameters())

    print(f"Total parameters: {total_params}")
    print(f"Actor head parameters:  {actor_params}")
    print(f"Critic head parameters: {critic_params}")

    dummy_obs = torch.randn(5, obs_dim, device=device)
    logits, values = model(dummy_obs)
    print(f"\nLogits shape: {logits.shape}, Values shape: {values.shape}")
    print(f"Sample logits[0]: {logits[0].tolist()}")
    print(f"Sample value[0]:  {values[0].item():.4f}")

    print("\nShared-body actor-critic is ready for use in the upcoming A2C experiments.\n")


if __name__ == "__main__":
    main()
