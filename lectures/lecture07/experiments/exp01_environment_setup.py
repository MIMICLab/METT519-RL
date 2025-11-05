#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 01 - Environment Setup and Verification

This experiment verifies that all dependencies are properly installed and
that the CartPole-v1 environment works correctly with deterministic seeding.

Learning objectives:
- Verify PyTorch, Gymnasium, and other dependencies
- Test device selection (CUDA > MPS > CPU)
- Confirm reproducible environment behavior with seeding
- Check CartPole-v1 observation and action spaces

Prerequisites: PyTorch 2.x, Gymnasium, NumPy
"""

import os
import sys
import random
import numpy as np
import torch

def setup_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# Proper device selection (CUDA > MPS > CPU)
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
amp_enabled = torch.cuda.is_available()

def main():
    print("="*50)
    print("Experiment 01: Environment Setup and Verification")
    print("="*50)
    
    # 1. Check Python version
    print("\n1. Python Version Check:")
    python_version = sys.version
    print(f"   Python: {python_version.split()[0]}")
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    print("   [OK] Python 3.10+ detected")
    
    # 2. Check PyTorch
    print("\n2. PyTorch Installation:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"   Selected device: {device}")
    print(f"   AMP enabled: {amp_enabled}")
    
    # 3. Check NumPy
    print("\n3. NumPy Installation:")
    print(f"   Version: {np.__version__}")
    
    # 4. Check Gymnasium
    print("\n4. Gymnasium Installation:")
    try:
        import gymnasium as gym
        print(f"   Version: {gym.__version__}")
    except ImportError:
        print("   [ERROR] Gymnasium not installed!")
        print("   Run: pip install gymnasium")
        return
    
    # 5. Test CartPole-v1 environment
    print("\n5. CartPole-v1 Environment Test:")
    try:
        env = gym.make("CartPole-v1")
        print(f"   Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Observation shape: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space}")
        print(f"   Number of actions: {env.action_space.n}")
    except Exception as e:
        print(f"   [ERROR] Failed to create environment: {e}")
        return
    
    # 6. Test deterministic seeding
    print("\n6. Reproducibility Test:")
    setup_seed(42)
    
    # First run
    obs1, info1 = env.reset(seed=42)
    actions = []
    observations = []
    for _ in range(5):
        action = env.action_space.sample()
        actions.append(action)
        obs, _, _, _, _ = env.step(action)
        observations.append(obs.copy())
    
    # Second run with same seed
    obs2, info2 = env.reset(seed=42)
    for i, action in enumerate(actions):
        obs, _, _, _, _ = env.step(action)
        if not np.allclose(obs, observations[i]):
            print("   [WARNING] Non-deterministic behavior detected!")
            break
    else:
        print("   [OK] Deterministic behavior confirmed")
    
    print(f"   Initial observation (seed=42): {obs1}")
    
    # 7. Test tensor operations
    print("\n7. Tensor Operations Test:")
    setup_seed(42)
    test_tensor = torch.randn(4, 128, device=device)
    print(f"   Created tensor shape: {test_tensor.shape}")
    print(f"   Tensor device: {test_tensor.device}")
    print(f"   Tensor mean: {test_tensor.mean().item():.4f}")
    print(f"   Tensor std: {test_tensor.std().item():.4f}")
    
    # 8. Memory info
    print("\n8. System Resources:")
    if torch.cuda.is_available():
        print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"   GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("   Running on CPU")
    
    # 9. Check for TensorBoard
    print("\n9. Optional Dependencies:")
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("   [OK] TensorBoard available")
    except ImportError:
        print("   [INFO] TensorBoard not installed (optional)")
        print("   Install with: pip install tensorboard")
    
    print("\n" + "="*50)
    print("Environment setup verified successfully!")
    print("Ready for DQN experiments")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
