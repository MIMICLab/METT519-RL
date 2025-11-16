#!/usr/bin/env python3
"""RL2025 - Lecture 9 Experiment 01: Environment Setup & Verification.

Learning objectives:
- Confirm required packages (PyTorch, Gymnasium) and their versions
- Validate vectorised environment support for Actor-Critic experiments
- Warm up device detection utilities shared across later experiments
- Demonstrate reproducible seeding for CPU/CUDA/MPS backends

Prerequisites: Python 3.10+, PyTorch 2.x, Gymnasium 0.29+
"""

from __future__ import annotations

import json
import sys
from pprint import pprint

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch

from helpers import FIGURES_DIR, get_device, set_seed


def main() -> None:
    print("=" * 60)
    print("Experiment 01: Lecture 9 Setup Check")
    print("=" * 60)

    # 1. Python and package versions
    print("\n1. Version check")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Gymnasium: {gym.__version__}")

    # 2. Device availability
    device = get_device()
    print("\n2. Device detection")
    print(f"   Selected device: {device}")
    if device.type == "cuda":
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    if hasattr(torch.backends, "mps"):
        print(f"   MPS available: {torch.backends.mps.is_available()}")

    # 3. Reproducibility smoke test
    print("\n3. Reproducibility check (seed=2024)")
    set_seed(2024)
    torch_tensor = torch.randn(3, 3)
    print(f"   Torch tensor mean: {torch_tensor.mean().item():.4f}")

    # 4. Vectorised environment support (CartPole)
    print("\n4. Vector environment probe (CartPole-v1, n=4)")
    env = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])
    obs, _ = env.reset(seed=2024)
    print(f"   Observation shape: {obs.shape}")
    actions = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(actions)
    print(f"   Reward sample: {reward}")
    print(f"   Terminated flags: {terminated}")
    env.close()

    # 5. Configuration snapshot for reports
    snapshot = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "device": str(device),
    }
    snapshot_path = FIGURES_DIR / "lecture09_exp01_env_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2))
    print(f"\nSaved version snapshot to {snapshot_path}")

    print("\nSetup check complete. Ready for Actor-Critic experiments!\n")


if __name__ == "__main__":
    main()
