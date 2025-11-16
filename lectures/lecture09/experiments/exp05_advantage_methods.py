#!/usr/bin/env python3
"""Lecture 9 Experiment 05: Advantage computation methods."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from helpers import FIGURES_DIR

GAMMA = 0.99
LAMBDA = 0.95


def n_step_returns(rewards: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    T = len(rewards)
    returns = np.zeros(T)
    for t in range(T):
        g = 0.0
        discount = 1.0
        for k in range(n):
            idx = min(t + k, T - 1)
            g += discount * rewards[idx]
            discount *= GAMMA
            if idx == T - 1:
                break
        if t + n < T:
            g += discount * values[t + n]
        returns[t] = g
    return returns


def generalized_advantage_estimation(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else 0.0) - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


@dataclass
class AdvantageSummary:
    method: str
    mean: float
    std: float
    first_value: float


def main() -> None:
    np.random.seed(123)
    T = 20
    rewards = np.random.normal(loc=1.0, scale=0.2, size=T)
    values = np.linspace(0.5, 0.0, num=T)

    one_step = n_step_returns(rewards, values, n=1) - values
    three_step = n_step_returns(rewards, values, n=3) - values
    returns = n_step_returns(rewards, values, n=T) - values
    gae = generalized_advantage_estimation(rewards, np.append(values, 0.0), GAMMA, LAMBDA)

    summaries = [
        AdvantageSummary("1-step TD", float(one_step.mean()), float(one_step.std()), float(one_step[0])),
        AdvantageSummary("3-step TD", float(three_step.mean()), float(three_step.std()), float(three_step[0])),
        AdvantageSummary("Monte Carlo", float(returns.mean()), float(returns.std()), float(returns[0])),
        AdvantageSummary("GAE (λ=0.95)", float(gae.mean()), float(gae.std()), float(gae[0])),
    ]

    print("=" * 60)
    print("Experiment 05: Advantage Computation Methods")
    print("=" * 60)
    print(f"Synthetic trajectory length: {T}")
    for summary in summaries:
        print(
            f"{summary.method:14s} → mean={summary.mean:+.4f}, std={summary.std:.4f}, first step={summary.first_value:+.4f}"
        )

    out_path = FIGURES_DIR / "lecture09_exp05_advantage_summary.json"
    out_data = {s.method: {"mean": s.mean, "std": s.std, "first": s.first_value} for s in summaries}
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nStored summary to {out_path}\n")


if __name__ == "__main__":
    main()
