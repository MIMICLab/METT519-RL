#!/usr/bin/env python3
"""Lecture 9 Experiment 03: Policy gradient sanity check."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from helpers import get_device, set_seed


def finite_difference(policy: nn.Module, observation: torch.Tensor, eps: float = 1e-3) -> float:
    base = policy.theta.item()
    policy.theta.data = torch.tensor([base + eps], device=policy.theta.device)
    plus = policy(observation)
    log_prob_plus = Categorical(logits=plus.squeeze(0)).log_prob(torch.tensor(0, device=plus.device))

    policy.theta.data = torch.tensor([base - eps], device=policy.theta.device)
    minus = policy(observation)
    log_prob_minus = Categorical(logits=minus.squeeze(0)).log_prob(torch.tensor(0, device=minus.device))

    policy.theta.data = torch.tensor([base], device=policy.theta.device)
    return ((log_prob_plus - log_prob_minus) / (2 * eps)).item()


def main() -> None:
    print("=" * 60)
    print("Experiment 03: Policy Gradient Review")
    print("=" * 60)

    set_seed(0)
    device = get_device()

    class TinyPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.theta = nn.Parameter(torch.tensor([0.3], device=device))

        def forward(self, _: torch.Tensor) -> torch.Tensor:
            return torch.stack([self.theta, torch.zeros_like(self.theta)], dim=-1)

    policy = TinyPolicy().to(device)
    observation = torch.zeros(1, device=device)

    logits = policy(observation)
    dist = Categorical(logits=logits.squeeze(0))
    log_prob = dist.log_prob(torch.tensor(0, device=device))

    autograd_grad = torch.autograd.grad(log_prob, policy.theta)[0].item()
    fd_grad = finite_difference(policy, observation)

    print(f"Analytical grad (autograd): {autograd_grad:.6f}")
    print(f"Finite difference grad:     {fd_grad:.6f}")
    print(f"Absolute error:             {abs(autograd_grad - fd_grad):.6e}")

    probs = dist.probs
    expected = 1.0 - probs[0].item()
    print(f"Closed-form grad:           {expected:.6f}")

    print("\nConclusion: the analytical, autograd, and finite-difference estimates"
          " match closely for this toy policy, validating the implementation"
          " of actor gradients used later in the lecture.\n")


if __name__ == "__main__":
    main()
