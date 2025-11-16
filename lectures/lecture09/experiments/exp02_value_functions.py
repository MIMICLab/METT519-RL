#!/usr/bin/env python3
"""Lecture 9 Experiment 02: Monte Carlo value estimates."""
from __future__ import annotations

import statistics
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from helpers import FIGURES_DIR, get_device, set_seed

GAMMA = 0.99
EPISODES = 25


def collect_returns(env: gym.Env, episodes: int) -> list[list[float]]:
    """Collect discounted returns G_t for each episode using a random policy."""
    all_returns: list[list[float]] = []
    for ep in range(episodes):
        obs, _ = env.reset()
        rewards: list[float] = []
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
        # discounted returns
        g_list: list[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = r + GAMMA * g
            g_list.insert(0, g)
        all_returns.append(g_list)
    return all_returns


def pad_and_average(sequences: list[list[float]]) -> np.ndarray:
    max_len = max(len(seq) for seq in sequences)
    padded = np.zeros((len(sequences), max_len))
    mask = np.zeros_like(padded)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
        mask[i, : len(seq)] = 1.0
    avg = np.sum(padded, axis=0) / np.clip(mask.sum(axis=0), a_min=1.0, a_max=None)
    return avg


def main() -> None:
    print("=" * 60)
    print("Experiment 02: Monte Carlo Value Estimates")
    print("=" * 60)

    set_seed(42)
    env = gym.make("CartPole-v1")
    returns = collect_returns(env, EPISODES)
    env.close()

    total_returns = [episode_returns[0] for episode_returns in returns]
    mean_return = statistics.mean(total_returns)
    std_return = statistics.pstdev(total_returns)

    print(f"Collected episodes: {EPISODES}")
    print(f"Average episode return (random policy): {mean_return:.2f} ± {std_return:.2f}")

    avg_return_by_t = pad_and_average(returns)
    value_table = [(t, avg_return_by_t[t]) for t in range(len(avg_return_by_t))]
    print("\nEstimated V(s_t) for first five time steps:")
    for t, val in value_table[:5]:
        print(f"  t={t:2d} → V≈{val:.3f}")

    # Save numeric summary for slides
    summary_path = FIGURES_DIR / "lecture09_exp02_value_summary.json"
    summary_path.write_text(
        f"{{\"mean_return\": {mean_return:.4f}, \"std_return\": {std_return:.4f},"
        f" \"value_at_t0\": {avg_return_by_t[0]:.4f}}}\n"
    )

    # Plot value estimate over time
    fig_path = FIGURES_DIR / "value_estimate_random_policy.png"
    plt.figure(figsize=(6, 3.5))
    plt.plot(avg_return_by_t, marker="o", linewidth=1.5)
    plt.title("Monte Carlo estimate of V(s_t) under random policy")
    plt.xlabel("Time step t")
    plt.ylabel("Estimated V(s_t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved value profile to {fig_path}")

    print("\nTakeaway: even a random policy exhibits rapidly decaying returns,\n"
          "highlighting why actor-critic methods update online instead of waiting\n"
          "for full-episode Monte Carlo targets.\n")


if __name__ == "__main__":
    main()
