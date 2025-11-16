#!/usr/bin/env python3
"""Lecture 9 Experiment 09: Integrated A2C smoke test."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from helpers import FIGURES_DIR, get_device, set_seed
from networks import ActorCritic

GAMMA = 0.99
LAM = 0.95
NUM_ENVS = 8
ROLLOUT = 32
UPDATES = 40
LR = 2.5e-4
ENTROPY_COEF = 1e-3


def compute_gae(rewards, values, dones):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        mask = 1.0 - dones[step]
        delta = rewards[step] + GAMMA * values[step + 1] * mask - values[step]
        gae = delta + GAMMA * LAM * mask * gae
        advantages[step] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def rollout_and_update(model, optimizer, obs, env):
    device = obs.device
    obs_batch = []
    actions_batch = []
    logprob_batch = []
    value_batch = []
    reward_batch = []
    done_batch = []
    episodic_returns = []
    running_returns = np.zeros(env.num_envs)

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
        for idx, done_flag in enumerate(done):
            if done_flag:
                episodic_returns.append(running_returns[idx])
                running_returns[idx] = 0.0

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    with torch.no_grad():
        _, next_values = model(obs)
    value_batch.append(next_values)

    rewards = torch.stack(reward_batch)
    dones = torch.stack(done_batch)
    values = torch.stack(value_batch)

    advantages, returns = compute_gae(rewards, values, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    flat_obs = torch.cat(obs_batch)
    flat_actions = torch.cat(actions_batch)
    flat_log_probs = torch.cat(logprob_batch)
    logits, value_preds = model(flat_obs)
    dist = torch.distributions.Categorical(logits=logits)
    entropy = dist.entropy().mean()

    policy_loss = -(dist.log_prob(flat_actions) * advantages.view(-1)).mean()
    value_loss = F.mse_loss(value_preds, returns.view(-1))
    loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return obs, episodic_returns, loss.item()


def train(seed: int = 0) -> Dict[str, list[float]]:
    set_seed(seed)
    device = get_device()
    env = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(NUM_ENVS)])
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    model = ActorCritic(
        obs_dim=env.single_observation_space.shape[0],
        act_dim=env.single_action_space.n,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    mean_returns = []
    losses = []
    recent_returns: list[float] = []

    for update in range(1, UPDATES + 1):
        obs, episodic_returns, loss = rollout_and_update(model, optimizer, obs, env)
        losses.append(loss)
        recent_returns.extend(episodic_returns)
        mean_return = float(np.mean(recent_returns[-100:])) if recent_returns else float("nan")
        mean_returns.append(mean_return)
        print(f"Update {update:02d}/{UPDATES} | Loss={loss:.3f} | Mean return={mean_return:.1f}")

    env.close()
    return {"returns": mean_returns, "losses": losses}


def main() -> None:
    print("=" * 60)
    print("Experiment 09: Integrated Advantage Actor-Critic")
    print("=" * 60)

    stats = train(seed=123)

    curve_path = FIGURES_DIR / "a2c_integrated_learning.png"
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 3.5))
    plt.subplot(1, 2, 1)
    plt.plot(stats["returns"], color="tab:blue")
    plt.title("Mean return (last 100 episodes)")
    plt.xlabel("Update")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(stats["losses"], color="tab:orange")
    plt.title("Training loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved integrated training curves to {curve_path}")

    summary_path = FIGURES_DIR / "a2c_integrated_summary.json"
    summary = {
        "final_return": stats["returns"][-1],
        "best_return": max(stats["returns"]),
        "final_loss": stats["losses"][-1],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}\n")


if __name__ == "__main__":
    main()
