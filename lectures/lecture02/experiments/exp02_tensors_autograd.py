#!/usr/bin/env python3
"""
RL2025 - Lecture 2: Experiment 02 - Tensors and Autograd Basics

Covers tensor creation, views vs copies, broadcasting, and basic autograd.

Learning objectives:
- Use PyTorch tensor APIs correctly and efficiently
- Understand requires_grad, backward, and gradient access

Prerequisites: exp01_setup.py
"""

import os, random, numpy as np, torch

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else (
    'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
setup_seed(42)


def main():
    print("="*50)
    print("Experiment 02: Tensors and Autograd Basics")
    print("="*50)

    # Tensors and views
    a = torch.arange(12.0).reshape(3,4)
    view = a.view(6,2)
    clone = a.clone()
    perm = a.permute(1,0)
    contig = perm.contiguous().view(-1)
    assert view.storage().data_ptr() == a.storage().data_ptr()
    assert clone.storage().data_ptr() != a.storage().data_ptr()

    # Broadcasting
    x = torch.randn(32, 128)
    bias = torch.randn(128)
    y = x + bias
    assert y.shape == (32, 128)

    # Autograd basics
    t = torch.tensor([2.0], requires_grad=True)
    f = t**3 + 2*t
    f.backward()
    assert torch.allclose(t.grad, torch.tensor([14.0]))
    print("Autograd OK: df/dt at t=2 is 14")

    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()

