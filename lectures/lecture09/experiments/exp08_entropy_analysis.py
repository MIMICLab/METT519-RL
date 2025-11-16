#!/usr/bin/env python3
"""Lecture 9 Experiment 08: Entropy coefficient sensitivity for A2C."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from helpers import FIGURES_DIR, get_device, set_seed
from networks import ActorCritic

GAMMA = 0.99
LAM = 0.95
NUM_ENVS = 4
ROLLOUT = 16
UPDATES = 20
LR = 3e-4
BETAS = [0.0, 1e-3, 5e-3, 1e-2]


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


def run_training(entropy_coef: float) -> list[float]:
    set_seed(123)
    device = get_device()
    env = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(NUM_ENVS)])
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    model = ActorCritic(obs_dim=env.single_observation_space.shape[0], act_dim=env.single_action_space.n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    running_returns: list[float] = []
    episode_returns: list[float] = []
    running_returns = np.zeros(NUM_ENVS)
    mean_history: list[float] = []

    for _ in range(UPDATES):
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
            for idx, done_flag in enumerate(done):
                if done_flag:
                    episode_returns.append(running_returns[idx])
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
        logits, _ = model(flat_obs)
        actions = torch.cat(actions_batch)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages.view(-1)).mean()
        value_loss = F.mse_loss(values[:-1].view(-1), returns.view(-1))
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        mean_history.append(float(np.mean(episode_returns[-20:])) if episode_returns else float("nan"))

    env.close()
    return mean_history


def main() -> None:
    print("=" * 60)
    print("Experiment 08: Entropy Sensitivity")
    print("=" * 60)

    results = {}
    for beta in BETAS:
        curve = run_training(beta)
        final = curve[-1]
        results[beta] = {"curve": curve, "final": final}
        print(f"β={beta:.4f} → final mean return {final:.1f}")

    import matplotlib.pyplot as plt

    fig_path = FIGURES_DIR / "a2c_entropy_sweep.png"
    plt.figure(figsize=(6.5, 3.5))
    for beta, data in results.items():
        plt.plot(data["curve"], label=f"β={beta}")
    plt.xlabel("Update")
    plt.ylabel("Mean return (last 20 episodes)")
    plt.title("Entropy coefficient sweep (CartPole, A2C)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved entropy sweep curve to {fig_path}\n")


if __name__ == "__main__":
    main()
