#!/usr/bin/env python3
"""Lecture 9 Experiment 06: Generalised Advantage Estimation (GAE)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from helpers import FIGURES_DIR, set_seed

GAMMA = 0.99
LAMBDAS = [0.0, 0.5, 0.95, 0.99]


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    adv = np.zeros_like(rewards)
    gae_acc = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae_acc = delta + gamma * lam * gae_acc
        adv[t] = gae_acc
    return adv


def main() -> None:
    set_seed(123)
    horizon = 30
    rewards = np.random.normal(loc=1.0, scale=0.3, size=horizon)
    values = np.concatenate([np.linspace(0.6, 0.0, num=horizon), np.array([0.0])])

    advantages = {lam: gae(rewards, values, GAMMA, lam) for lam in LAMBDAS}

    print("=" * 60)
    print("Experiment 06: Generalized Advantage Estimation")
    print("=" * 60)
    for lam, adv in advantages.items():
        print(f"λ={lam:4.2f} → mean={adv.mean():+.4f}, std={adv.std():.4f}")

    fig_path = FIGURES_DIR / "gae_lambda_profiles.png"
    plt.figure(figsize=(6.5, 3.5))
    for lam, adv in advantages.items():
        plt.plot(adv, label=f"λ={lam}")
    plt.title("GAE advantages for different λ")
    plt.xlabel("Time step t")
    plt.ylabel("Advantage A_t")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved advantage profiles to {fig_path}\n")


if __name__ == "__main__":
    main()
