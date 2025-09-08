#!/usr/bin/env python3
"""
RL2025 - Lecture 2: Experiment 01 - Setup and Sanity

Verifies PyTorch installation, device selection, seeding, and basic tensor ops.

Learning objectives:
- Confirm environment is ready (CUDA/MPS/CPU)
- Practice tensor creation, reshaping, and broadcasting

Prerequisites: PyTorch 2.x installed from Lecture 1
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
amp_enabled = torch.cuda.is_available()
setup_seed(42)


def main():
    print("="*50)
    print("Experiment 01: Setup and Sanity")
    print("="*50)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}")
    print(f"AMP enabled: {amp_enabled}")

    # Tensor sanity checks
    x = torch.arange(12).reshape(3,4).to(device)
    y = torch.ones((4,), device=device)
    z = x + y  # broadcasting
    assert z.shape == (3,4)
    print("Broadcasting OK, shape:", z.shape)

    # Reproducibility check
    a = torch.randn(3, 3)
    setup_seed(42)
    b = torch.randn(3, 3)
    setup_seed(42)
    c = torch.randn(3, 3)
    assert torch.allclose(b, c), "Seeding not deterministic"
    print("Seeding OK")

    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()

