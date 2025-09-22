#!/usr/bin/env python3
"""
RL2025 - Lecture 2: Experiment 07 - Training Pipeline and LR Scheduler

Builds a small training pipeline with DataLoader, early stopping, and a
StepLR scheduler on synthetic data.

Learning objectives:
- Structure a minimal yet robust training pipeline
- Use learning-rate scheduling and early stopping

Prerequisites: exp06_regularization_init.py
"""

import os, random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_data(n=5000):
    X = torch.randn(n, 2)
    y = (X[:,0] + X[:,1] > 0).long()
    return X, y

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); total = 0.0; correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item()*x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
    return total/len(loader.dataset), correct/len(loader.dataset)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train(); total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(); logits = model(x); loss = criterion(logits, y)
        loss.backward(); optimizer.step(); total += loss.item()*x.size(0)
    return total/len(loader.dataset)

def main():
    print("="*50)
    print("Experiment 07: Training Pipeline and LR Scheduler")
    print("="*50)
    setup_seed(42)
    X, y = make_data(5000)
    n_train = int(0.8 * len(X))
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], y[n_train:])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val = float('inf'); patience = 3; wait = 0
    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch:02d} | val_loss={val_loss:.4f} acc={val_acc:.3f}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss; wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()
