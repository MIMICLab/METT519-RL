#!/usr/bin/env python3
"""
RL2025 - Lecture 2: Experiment 05 - Training Loop, Losses, and Optimizers

Implements a basic training loop on a synthetic classification task
using CrossEntropyLoss and SGD/Adam. Demonstrates train/eval modes.

Learning objectives:
- Write canonical train/eval loops with PyTorch
- Compare SGD vs Adam on simple data

Prerequisites: exp04_nn_modules.py
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Add deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Proper device selection (CUDA > MPS > CPU)
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
amp_enabled = (device.type == 'cuda')
setup_seed(42)

def make_data(n=2000):
    X = torch.randn(n, 2)
    y = (X[:,0] + 0.5*X[:,1] > 0).long()
    return X, y

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

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); total = 0.0; correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item()*x.size(0)
        pred = logits.argmax(-1)
        correct += (pred==y).sum().item()
    return total/len(loader.dataset), correct/len(loader.dataset)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

def main():
    print("="*50)
    print("Experiment 05: Training Loop, Losses, and Optimizers")
    print("="*50)
    print(f"Using device: {device}")
    print(f"AMP enabled: {amp_enabled}")

    X, y = make_data(4000)
    n_train = int(0.8*len(X))
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], y[n_train:])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = MLP().to(device)  # CRITICAL: Move model to device
    criterion = nn.CrossEntropyLoss()

    for name, optimizer in {
        'SGD': optim.SGD(model.parameters(), lr=1e-2),
        'Adam': optim.Adam(model.parameters(), lr=1e-3)
    }.items():
        # Reinit weights to compare fairly
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        model.apply(init)

        for epoch in range(10):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"{name} -> Val acc: {val_acc:.3f}")

    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()

