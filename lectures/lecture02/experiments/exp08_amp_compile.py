#!/usr/bin/env python3
"""
RL2025 - Lecture 2: Experiment 08 - AMP and torch.compile

Demonstrates enabling Automatic Mixed Precision (AMP) and torch.compile
for potential speedups. Falls back gracefully if unsupported.

Learning objectives:
- Use GradScaler/autocast for mixed precision
- Apply torch.compile when available

Prerequisites: exp07_training_pipeline.py
"""

import os, time, random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_data(n=6000):
    X = torch.randn(n, 2)
    y = (X[:,0] * 0.7 + X[:,1] * -0.3 > 0).long()
    return X, y

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_steps(model, loader, optimizer, criterion, steps=50, use_amp=False):
    model.train(); scaler = GradScaler(enabled=use_amp)
    t0 = time.time()
    it = 0
    for x, y in loader:
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        it += 1
        if it >= steps: break
    return time.time() - t0

def main():
    print("="*50)
    print("Experiment 08: AMP and torch.compile")
    print("="*50)
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = make_data(10000)
    ds = TensorDataset(X.to(device), y.to(device))
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # Baseline
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    t_base = train_steps(model, loader, optimizer, criterion, use_amp=False)

    # AMP
    model_amp = MLP().to(device)
    optimizer_amp = optim.Adam(model_amp.parameters(), lr=1e-3)
    t_amp = train_steps(model_amp, loader, optimizer_amp, criterion, use_amp=torch.cuda.is_available())

    # compile
    model_comp = MLP().to(device)
    if hasattr(torch, 'compile'):
        model_comp = torch.compile(model_comp, mode='default')
    optimizer_comp = optim.Adam(model_comp.parameters(), lr=1e-3)
    t_comp = train_steps(model_comp, loader, optimizer_comp, criterion, use_amp=False)

    print(f"Baseline steps time: {t_base:.3f}s")
    print(f"AMP steps time:      {t_amp:.3f}s")
    print(f"compile steps time:  {t_comp:.3f}s")
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()

