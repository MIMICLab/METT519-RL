#!/usr/bin/env python3
"""
RL2025 - Lecture 2: Experiment 03 - Autograd Graph and Gradient Checking

Explores detach, no_grad, custom autograd Function, and numeric gradient check.

Learning objectives:
- Manage computation graphs and gradient flow safely
- Implement and sanity-check custom gradients

Prerequisites: exp02_tensors_autograd.py
"""

import os, random, numpy as np, torch
from torch.autograd import Function

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

setup_seed(42)


class Square(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * input
    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return 2 * input * grad_output


def numeric_grad(f, x, eps=1e-5):
    with torch.no_grad():
        x1 = x.clone(); x1[0] += eps
        x2 = x.clone(); x2[0] -= eps
        return (f(x1) - f(x2)) / (2*eps)


def main():
    print("="*50)
    print("Experiment 03: Autograd Graph and Gradient Checking")
    print("="*50)

    # Detach and no_grad
    x = torch.randn(4, requires_grad=True)
    y = (x*x).sum()
    y.backward()
    with torch.no_grad():
        x -= 0.1 * x.grad
    x = x.detach().requires_grad_(True)

    # Custom function
    t = torch.tensor([3.0], requires_grad=True)
    z = Square.apply(t)
    z.backward()
    assert torch.allclose(t.grad, torch.tensor([6.0]))

    # Numeric gradient check
    u = torch.tensor([1.23], requires_grad=True)
    f = lambda a: (a*a).sum()
    f(u).backward()
    num = numeric_grad(f, u.detach())
    assert abs(u.grad.item() - num.item()) < 1e-2, f"Autograd: {u.grad.item()}, Numeric: {num.item()}"
    print("Gradient check OK")

    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()

