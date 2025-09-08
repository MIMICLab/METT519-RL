#!/usr/bin/env python3
"""
RL2025 - Lecture 2: Experiment 09 - Integrated Pipeline Test

End-to-end training on a synthetic classification dataset that exercises
data pipeline, model, optimizer, scheduler, AMP (if available), and
checkpointing. Serves as a smoke test for the full stack.

Learning objectives:
- Integrate components into a reliable training script
- Produce a reproducible result with saved artifacts

Prerequisites: exp01-exp08
"""

import os, json, random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_data(n=10000):
    X = torch.randn(n, 2)
    y = ((X[:,0]**2 - 0.8*X[:,1]) > 0).long()
    return X, y

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

def train(model, loaders, optimizer, scheduler, criterion, device, epochs=15):
    scaler = GradScaler(enabled=(device.type=='cuda'))
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = float('inf'); best_state = None
    for epoch in range(epochs):
        model.train(); total = 0.0
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type=='cuda')):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item()*x.size(0)
        train_loss = total/len(loaders['train'].dataset)
        # Eval
        model.eval(); vt = 0.0; correct = 0
        with torch.no_grad():
            for x, y in loaders['val']:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                vt += loss.item()*x.size(0)
                pred = logits.argmax(-1)
                correct += (pred==y).sum().item()
        val_loss = vt/len(loaders['val'].dataset)
        val_acc = correct/len(loaders['val'].dataset)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch:02d} | train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.3f}")
        if val_loss < best:
            best = val_loss
            best_state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
    return history, best_state

def main():
    print("="*50)
    print("Experiment 09: Integrated Pipeline Test")
    print("="*50)
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else (
        'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
    print("Device:", device)
    X, y = make_data(12000)
    n_train = int(0.8*len(X))
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], y[n_train:])
    loaders = {
        'train': DataLoader(train_ds, batch_size=256, shuffle=True),
        'val': DataLoader(val_ds, batch_size=512)
    }
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history, best_state = train(model, loaders, optimizer, scheduler, criterion, device, epochs=12)

    os.makedirs('runs', exist_ok=True)
    with open('runs/lec2_history.json', 'w') as f:
        json.dump(history, f)
    if best_state is not None:
        torch.save(best_state, 'runs/lec2_checkpoint.pt')
    assert history['val_acc'][-1] >= 0.80 or max(history['val_acc']) >= 0.80, "Expected >=80% on synthetic data"
    print("Integration test passed. Checkpoints and history saved under runs/")

if __name__ == "__main__":
    main()

