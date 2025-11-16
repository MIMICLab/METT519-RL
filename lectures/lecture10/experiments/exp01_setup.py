#!/usr/bin/env python3
"""
RL2025 - Lecture 10: Experiment 01 - Environment Setup and Verification

This experiment verifies that all required packages are installed and working
correctly for PPO implementation. Tests basic PyTorch operations, Gymnasium
environments, and TensorBoard logging.

Learning objectives:
- Verify PyTorch 2.x installation and device selection
- Test CartPole-v1 environment functionality  
- Confirm TensorBoard logging capability
- Validate vectorized environment creation

Prerequisites: Fresh Python environment with gymnasium, torch, tensorboard
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Proper device selection (CUDA > MPS > CPU)
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
amp_enabled = torch.cuda.is_available()
setup_seed(42)

import sys
import json
import time
from typing import Dict, Any

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

def print_system_info() -> Dict[str, Any]:
    """Print comprehensive system information for reproducibility."""
    info = {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "selected_device": str(device),
        "amp_enabled": amp_enabled,
        "gymnasium_version": gym.__version__,
        "numpy_version": np.__version__,
    }
    
    print("System Information:")
    print(json.dumps(info, indent=2))
    return info

def test_pytorch_operations():
    """Test basic PyTorch operations on the selected device."""
    print(f"\nTesting PyTorch operations on device: {device}")
    
    # Test tensor creation and operations
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = torch.mm(x, y)
    
    print(f"Matrix multiplication test: {z.shape}")
    print(f"Mean value: {z.mean().item():.4f}")
    
    # Test automatic differentiation
    x.requires_grad_(True)
    loss = (x ** 2).sum()
    loss.backward()
    print(f"Gradient computation test: grad norm = {x.grad.norm().item():.4f}")
    
    # Test mixed precision if available
    if amp_enabled and device.type == 'cuda':
        with torch.autocast(device_type='cuda'):
            z_amp = torch.mm(x, y)
        print(f"AMP test passed: {z_amp.dtype}")
    
    print("✓ PyTorch operations test passed")

def test_cartpole_environment():
    """Test CartPole-v1 environment functionality."""
    print("\nTesting CartPole-v1 environment...")
    
    # Single environment test
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test episode interaction
    total_reward = 0
    steps = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    
    env.close()
    print(f"Episode test: {steps} steps, total reward: {total_reward}")
    print("✓ CartPole environment test passed")

def test_vectorized_environments():
    """Test vectorized environment creation."""
    print("\nTesting vectorized environments...")
    
    def make_env(env_id: str, seed: int, idx: int):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed + idx)
            return env
        return thunk
    
    # Create vectorized environments
    num_envs = 4
    envs = gym.vector.SyncVectorEnv([
        make_env("CartPole-v1", 42, i) for i in range(num_envs)
    ])
    
    obs = envs.reset(seed=42)[0]  # New gymnasium API returns (obs, info)
    print(f"Vectorized observation shape: {obs.shape}")
    print(f"Number of environments: {num_envs}")
    
    # Test vectorized step
    actions = np.random.randint(0, 2, size=num_envs)
    obs, rewards, terminated, truncated, infos = envs.step(actions)
    print(f"Step results - obs: {obs.shape}, rewards: {rewards.shape}")
    
    envs.close()
    print("✓ Vectorized environments test passed")

def test_tensorboard_logging():
    """Test TensorBoard logging functionality."""
    print("\nTesting TensorBoard logging...")
    
    # Create a test log directory
    log_dir = f"runs/test_setup_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Log some test metrics
    for i in range(10):
        writer.add_scalar("test/loss", 1.0 / (i + 1), i)
        writer.add_scalar("test/accuracy", i / 10.0, i)
    
    # Log a histogram
    x = torch.randn(100)
    writer.add_histogram("test/weights", x, 0)
    
    writer.close()
    
    # Verify files were created
    if os.path.exists(log_dir):
        files = os.listdir(log_dir)
        print(f"TensorBoard files created: {len(files)} files")
        print("✓ TensorBoard logging test passed")
    else:
        print("✗ TensorBoard logging test failed")

def test_neural_network_creation():
    """Test basic neural network creation for actor-critic."""
    print("\nTesting neural network creation...")
    
    import torch.nn as nn
    
    class SimpleActorCritic(nn.Module):
        def __init__(self, obs_dim=4, act_dim=2, hidden_dim=128):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden_dim, act_dim)
            self.critic = nn.Linear(hidden_dim, 1)
        
        def forward(self, x):
            shared_features = self.shared(x)
            return self.actor(shared_features), self.critic(shared_features)
    
    # Test network creation and forward pass
    net = SimpleActorCritic().to(device)
    test_obs = torch.randn(8, 4, device=device)  # Batch of 8 observations
    
    logits, values = net(test_obs)
    print(f"Network output shapes - logits: {logits.shape}, values: {values.shape}")
    
    # Test parameter count
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("✓ Neural network creation test passed")

def run_comprehensive_setup_check():
    """Run all setup verification tests."""
    print("="*60)
    print("PPO Implementation Setup Verification")
    print("="*60)
    
    try:
        # System info
        system_info = print_system_info()
        
        # Core tests
        test_pytorch_operations()
        test_cartpole_environment()
        test_vectorized_environments()
        test_tensorboard_logging()
        test_neural_network_creation()
        
        print("\n" + "="*60)
        print("✓ ALL SETUP TESTS PASSED")
        print("✓ Ready for PPO implementation!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Setup test failed: {e}")
        print("Please check your environment setup.")
        return False

def main():
    success = run_comprehensive_setup_check()
    
    if success:
        print("\nNext steps:")
        print("1. Proceed to exp02_policy_gradient_basics.py")
        print("2. Launch TensorBoard: tensorboard --logdir runs")
        print("3. Check GPU memory if using CUDA")
    else:
        print("\nSetup issues detected. Please resolve before continuing.")

if __name__ == "__main__":
    main()