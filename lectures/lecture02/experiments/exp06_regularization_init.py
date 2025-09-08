#!/usr/bin/env python3
"""
RL2025 - Lecture 2: Experiment 06 - Initialization and Regularization

Applies Kaiming/Xavier initialization, weight decay, dropout, and batchnorm
to observe effects on simple synthetic classification performance.

Learning objectives:
- Use proper initialization for ReLU networks
- Apply L2, dropout, and batchnorm appropriately

Prerequisites: exp05_losses_optimizers.py
"""

import os, random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_data(n=4000):
    X = torch.randn(n, 2)
    y = (X[:,0] - 0.8*X[:,1] > 0).long()
    return X, y

class RegNet(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); total = 0.0; correct = 0
    for x, y in loader:
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item()*x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
    return total/len(loader.dataset), correct/len(loader.dataset)

def train_epoch(model, loader, opt, crit):
    model.train(); total = 0.0
    for x, y in loader:
        opt.zero_grad(); logits = model(x); loss = crit(logits, y)
        loss.backward(); opt.step(); total += loss.item()*x.size(0)
    return total/len(loader.dataset)

def main():
    print("="*50)
    print("Experiment 06: Initialization and Regularization")
    print("="*50)
    setup_seed(42)
    X, y = make_data(4000)
    ds = TensorDataset(X, y)
    train_loader = DataLoader(ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(ds, batch_size=256)
    crit = nn.CrossEntropyLoss()

    model = RegNet(p=0.5)
    model.apply(init_kaiming)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(8):
        train_epoch(model, train_loader, opt, crit)
    val_loss, val_acc = evaluate(model, val_loader, crit)
    print(f"Val acc with dropout+BN+WD: {val_acc:.3f}")

    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()

