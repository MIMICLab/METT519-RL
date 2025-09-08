#!/usr/bin/env python3
"""
RL2025 - Lecture 3: Experiment 01 - Environment Setup and Verification

This experiment verifies the Gymnasium installation and basic environment setup
for reinforcement learning, specifically testing CartPole-v1 environment.

Learning objectives:
- Verify Gymnasium installation and API
- Test environment creation and basic interaction
- Understand observation and action spaces
- Validate reproducibility setup

Prerequisites: pip install 'gymnasium[classic-control]' torch
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
import platform
import json
import time
import hashlib

try:
    import gymnasium as gym
except ImportError:
    print("ERROR: Gymnasium not found. Please install with:")
    print("pip install 'gymnasium[classic-control]'")
    sys.exit(1)

def reproducibility_probe():
    """Check system configuration for reproducibility"""
    metadata = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "gymnasium_version": gym.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "device_selected": str(device),
        "amp_enabled": amp_enabled,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
    print("RUN_METADATA::", metadata)
    print("RUN_METADATA_SHA256::", hashlib.sha256(payload).hexdigest())
    return metadata

def test_environment_creation():
    """Test basic environment creation and properties"""
    print("\n--- Testing Environment Creation ---")
    
    # Create CartPole-v1 environment
    env = gym.make("CartPole-v1")
    print(f"Environment: {env.spec.id}")
    print(f"Max episode steps: {env.spec.max_episode_steps}")
    
    # Check observation space
    obs_space = env.observation_space
    print(f"Observation space: {obs_space}")
    print(f"  Shape: {obs_space.shape}")
    print(f"  Low bounds: {obs_space.low}")
    print(f"  High bounds: {obs_space.high}")
    
    # Check action space
    action_space = env.action_space
    print(f"Action space: {action_space}")
    print(f"  Number of actions: {action_space.n}")
    
    env.close()
    return True

def test_environment_interaction():
    """Test basic environment interaction with proper API"""
    print("\n--- Testing Environment Interaction ---")
    
    env = gym.make("CartPole-v1")
    
    # Reset environment with seed
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Info keys: {list(info.keys())}")
    
    # Take a few random steps
    total_reward = 0.0
    for step in range(5):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Next observation: {next_obs}")
        
        if terminated or truncated:
            print("  Episode ended!")
            break
    
    print(f"Total reward: {total_reward}")
    env.close()
    return True

def test_reproducibility():
    """Test that seeded environments produce reproducible results"""
    print("\n--- Testing Reproducibility ---")
    
    def run_episode(seed):
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        
        total_reward = 0.0
        observations = [obs.copy()]
        actions = []
        
        for _ in range(10):  # Fixed number of steps
            action = env.action_space.sample()
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            observations.append(obs.copy())
            
            if terminated or truncated:
                break
        
        env.close()
        return total_reward, observations, actions
    
    # Run same seed twice
    seed = 12345
    setup_seed(seed)
    run1 = run_episode(seed)
    
    setup_seed(seed)
    run2 = run_episode(seed)
    
    # Check reproducibility
    rewards_match = abs(run1[0] - run2[0]) < 1e-6
    obs_match = all(np.allclose(o1, o2) for o1, o2 in zip(run1[1], run2[1]))
    actions_match = run1[2] == run2[2]
    
    print(f"Rewards match: {rewards_match} ({run1[0]} vs {run2[0]})")
    print(f"Observations match: {obs_match}")
    print(f"Actions match: {actions_match}")
    
    reproducible = rewards_match and obs_match and actions_match
    print(f"Reproducibility test: {'PASSED' if reproducible else 'FAILED'}")
    
    return reproducible

def main():
    """Run all setup and verification tests"""
    print("="*50)
    print("Experiment 01: Environment Setup and Verification")
    print("="*50)
    
    # System probe
    metadata = reproducibility_probe()
    
    # Test environment features
    test_results = []
    test_results.append(test_environment_creation())
    test_results.append(test_environment_interaction())
    test_results.append(test_reproducibility())
    
    # Summary
    print("\n" + "="*50)
    print("EXPERIMENT 01 RESULTS")
    print("="*50)
    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Environment creation: {'PASS' if test_results[0] else 'FAIL'}")
    print(f"Environment interaction: {'PASS' if test_results[1] else 'FAIL'}")
    print(f"Reproducibility: {'PASS' if test_results[2] else 'FAIL'}")
    
    all_passed = all(test_results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if not all_passed:
        print("\nPlease fix issues before proceeding to next experiments.")
        return False
    
    print("Experiment 01 completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)