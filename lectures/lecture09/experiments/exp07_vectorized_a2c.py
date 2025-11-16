#!/usr/bin/env python3
"""Lecture 9 Experiment 07: Vectorised Advantage Actor-Critic (A2C)."""
from __future__ import annotations

from collections import deque

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from helpers import FIGURES_DIR, get_device, set_seed, to_device
from networks import ActorCritic

GAMMA = 0.99
LAM = 0.95
NUM_ENVS = 4
ROLLOUT = 32
UPDATES = 30
LR = 3e-4


def compute_gae(rewards, values, dones, gamma, lam):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        mask = 1.0 - dones[step]
        delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advantages[step] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def main() -> None:
    print("=" * 60)
    print("Experiment 07: Vectorised A2C (CartPole)")
    print("=" * 60)

    set_seed(0)
    device = get_device()

    env = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(NUM_ENVS)])
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    model = ActorCritic(obs_dim=env.single_observation_space.shape[0], act_dim=env.single_action_space.n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_returns = deque(maxlen=100)
    mean_returns = []

    running_returns = np.zeros(NUM_ENVS)

    for update in range(1, UPDATES + 1):
        obs_batch = []
        actions_batch = []
        logprob_batch = []
        value_batch = []
        reward_batch = []
        done_batch = []

        for _ in range(ROLLOUT):
            logits, values = model(obs)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()

            next_obs, rewards, terminations, truncations, infos = env.step(actions.cpu().numpy())
            done = np.logical_or(terminations, truncations)

            obs_batch.append(obs)
            actions_batch.append(actions)
            logprob_batch.append(dist.log_prob(actions))
            value_batch.append(values)
            reward_batch.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            done_batch.append(torch.tensor(done, dtype=torch.float32, device=device))

            running_returns += rewards
            completed = np.where(done)[0]
            for idx in completed:
                episode_returns.append(running_returns[idx])
                running_returns[idx] = 0.0

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, next_values = model(obs)

        value_batch.append(next_values)

        rewards = torch.stack(reward_batch)
        dones = torch.stack(done_batch)
        values = torch.stack(value_batch)
        log_probs = torch.stack(logprob_batch)
        actions_stack = torch.stack(actions_batch)
        obs_stack = torch.stack(obs_batch)

        advantages, returns = compute_gae(rewards, values, dones, GAMMA, LAM)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits, _ = model(obs_stack.view(-1, obs_stack.shape[-1]))
        dist = torch.distributions.Categorical(logits=logits)
        flat_log_probs = dist.log_prob(actions_stack.view(-1))
        entropy = dist.entropy().mean()

        policy_loss = -(flat_log_probs * advantages.view(-1)).mean()
        value_loss = F.mse_loss(values[:-1].view(-1), returns.view(-1))
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        mean_returns.append(mean_return)
        print(f"Update {update:02d}/{UPDATES} | Loss={loss.item():.3f} | Mean return (100 ep)={mean_return:.1f}")

    env.close()

    # Save curve for slides
    curve_path = FIGURES_DIR / "a2c_vectorized_learning_curve.png"
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3.5))
    plt.plot(mean_returns, marker="o", linewidth=1.5)
    plt.xlabel("Update")
    plt.ylabel("Mean return (last 100 episodes)")
    plt.title("Vectorised A2C on CartPole (4 envs, 30 updates)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved learning curve to {curve_path}\n")


if __name__ == "__main__":
    main()
