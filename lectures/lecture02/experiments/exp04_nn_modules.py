#!/usr/bin/env python3
"""
RL2025 - Lecture 2: Experiment 04 - Neural Network Modules

Builds a simple MLP and verifies forward pass shapes on synthetic data.

Learning objectives:
- Implement nn.Module and forward method
- Verify tensor shapes and logits usage

Prerequisites: exp03_autograd_graph.py
"""

import os, random, numpy as np, torch
import torch.nn as nn

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

setup_seed(42)


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)


def main():
    print("="*50)
    print("Experiment 04: Neural Network Modules")
    print("="*50)
    model = MLP(in_dim=2, hidden=32, out_dim=2)
    x = torch.randn(5, 2)
    logits = model(x)  # [B, C]
    assert logits.shape == (5, 2)
    print("Forward shape OK:", logits.shape)
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()

