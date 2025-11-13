#!/usr/bin/env python3
"""
RL2025 - Lecture 8: Experiment 01 - Environment Setup & Policy Gradient Verification

This experiment verifies that all required libraries are installed and demonstrates
the basic setup for policy gradient methods.

Learning objectives:
- Verify PyTorch and Gymnasium installation
- Test GPU/CPU device selection
- Understand stochastic policy representation
- Verify environment interaction loop

Prerequisites: Python 3.9+, pip install torch gymnasium tensorboard
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import sys

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

def main():
    print("="*50)
    print("Experiment 01: Setup Verification")
    print("="*50)
    
    # 1. Library versions
    print("\n1. Library Versions:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Gymnasium: {gym.__version__}")
    print(f"   NumPy: {np.__version__}")
    
    # 2. Device information
    print("\n2. Device Information:")
    print(f"   Device: {device}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Device: {torch.cuda.get_device_name()}")
    print(f"   AMP Enabled: {amp_enabled}")
    
    # 3. Create CartPole environment
    print("\n3. Environment Test:")
    env = gym.make('CartPole-v1')
    obs, info = env.reset(seed=42)
    print(f"   Environment: CartPole-v1")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Number of actions: {env.action_space.n}")
    
    # 4. Simple stochastic policy network
    print("\n4. Policy Network Test:")
    
    class SimplePolicy(nn.Module):
        def __init__(self, obs_dim, n_actions):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, n_actions)
            )
        
        def forward(self, x):
            return self.net(x)
    
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = SimplePolicy(obs_dim, n_actions).to(device)
    
    print(f"   Policy architecture created")
    print(f"   Input dimension: {obs_dim}")
    print(f"   Output dimension: {n_actions}")
    print(f"   Total parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # 5. Test forward pass and action sampling
    print("\n5. Action Sampling Test:")
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    print(f"   Observation tensor shape: {obs_tensor.shape}")
    
    with torch.no_grad():
        logits = policy(obs_tensor)
        print(f"   Logits shape: {logits.shape}")
        print(f"   Logits values: {logits.cpu().numpy()}")
        
        # Create categorical distribution
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        print(f"   Sampled action: {action.item()}")
        print(f"   Log probability: {log_prob.item():.4f}")
        print(f"   Entropy: {entropy.item():.4f}")
    
    # 6. Test environment step
    print("\n6. Environment Step Test:")
    next_obs, reward, terminated, truncated, info = env.step(action.item())
    print(f"   Next observation shape: {next_obs.shape}")
    print(f"   Reward: {reward}")
    print(f"   Terminated: {terminated}")
    print(f"   Truncated: {truncated}")
    
    # 7. Test gradient computation
    print("\n7. Gradient Computation Test:")
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # Simulate a policy gradient update
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    logits = policy(obs_tensor)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    
    # Fake advantage for testing
    advantage = torch.tensor([1.0], device=device)
    loss = -(log_prob * advantage).mean()
    
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_gradients = all(p.grad is not None for p in policy.parameters())
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   All parameters have gradients: {has_gradients}")
    
    if has_gradients:
        total_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float('inf'))
        print(f"   Total gradient norm: {total_grad_norm:.6f}")
    
    # 8. Memory and reproducibility test
    print("\n8. Reproducibility Test:")
    setup_seed(42)
    test_tensor = torch.randn(5)
    print(f"   Random tensor with seed 42: {test_tensor.numpy()}")
    
    setup_seed(42)
    test_tensor2 = torch.randn(5)
    print(f"   Same seed produces same tensor: {torch.allclose(test_tensor, test_tensor2)}")
    
    env.close()
    
    print("\n" + "="*50)
    print("Setup verification completed successfully!")
    print("All components ready for policy gradient experiments.")
    print("="*50)

if __name__ == "__main__":
    main()