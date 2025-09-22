#!/usr/bin/env python3
"""
RL2025 - Lecture 5: Experiment 01 - Environment Setup and Verification

This experiment verifies the FrozenLake environment setup and basic functionality.

Learning objectives:
- Set up and verify Gymnasium FrozenLake environment
- Understand state/action spaces and transitions
- Test random agent baseline performance

Prerequisites: gymnasium, numpy, torch installed
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

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

def print_environment_info():
    """Display system and library information."""
    print("="*50)
    print("System Information")
    print("="*50)
    print(f"Python version: {np.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Gymnasium version: {gym.__version__}")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

def explore_frozenlake():
    """Explore FrozenLake environment structure."""
    print("="*50)
    print("FrozenLake Environment Details")
    print("="*50)
    
    # Create default 4x4 FrozenLake
    env = gym.make("FrozenLake-v1", render_mode="ansi")
    
    # Get environment details
    print(f"Observation space: {env.observation_space}")
    print(f"Number of states: {env.observation_space.n}")
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    print(f"Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP")
    print()
    
    # Show the map
    print("Default 4x4 Map:")
    print("-" * 20)
    state, info = env.reset(seed=42)
    print(env.render())
    print("Legend: S=Start, F=Frozen, H=Hole, G=Goal")
    print()
    
    # State representation
    print(f"Initial state: {state}")
    print(f"State type: {type(state)}")
    print(f"Info dict: {info}")
    
    env.close()
    return env.observation_space.n, env.action_space.n

def test_random_agent(episodes=100, seed=42):
    """Test random agent performance as baseline."""
    print("="*50)
    print("Random Agent Baseline (100 episodes)")
    print("="*50)
    
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    successes = 0
    total_steps = 0
    returns = []
    
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        episode_return = 0
        steps = 0
        
        while not done:
            # Random action selection
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            steps += 1
            state = next_state
            
            # Prevent infinite loops
            if steps > 100:
                break
        
        if episode_return > 0:
            successes += 1
        total_steps += steps
        returns.append(episode_return)
    
    env.close()
    
    avg_return = np.mean(returns)
    success_rate = successes / episodes
    avg_steps = total_steps / episodes
    
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average return: {avg_return:.3f}")
    print(f"Average steps per episode: {avg_steps:.1f}")
    print()
    
    return success_rate, avg_return

def test_slippery_dynamics():
    """Compare deterministic vs slippery dynamics."""
    print("="*50)
    print("Slippery vs Deterministic Dynamics")
    print("="*50)
    
    for is_slippery in [False, True]:
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
        
        print(f"\nis_slippery={is_slippery}:")
        print("-" * 30)
        
        # Test action outcomes from state 0
        state, _ = env.reset(seed=42)
        
        # Try moving RIGHT (action=2) multiple times
        action = 2  # RIGHT
        outcomes = {}
        
        for trial in range(10):
            env.reset(seed=42)
            next_state, _, _, _, _ = env.step(action)
            outcomes[next_state] = outcomes.get(next_state, 0) + 1
        
        print(f"From state 0, action RIGHT repeated 10 times:")
        for s, count in sorted(outcomes.items()):
            print(f"  -> State {s}: {count} times")
        
        env.close()

def create_custom_maps():
    """Create custom maps with different difficulty levels."""
    print("="*50)
    print("Custom Map Generation")
    print("="*50)
    
    sizes = [4, 8]
    hole_probs = [0.1, 0.2, 0.3]
    
    for size in sizes:
        for hole_prob in hole_probs:
            # Generate random map
            safe_prob = 1.0 - hole_prob
            desc = generate_random_map(size=size, p=safe_prob, seed=42)
            
            print(f"\nMap {size}x{size}, hole_prob={hole_prob}:")
            print("-" * 30)
            
            # Create environment with custom map
            env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
            env.reset(seed=42)
            
            # Count holes
            flat_map = ''.join(desc)
            num_holes = flat_map.count('H')
            total_tiles = size * size
            actual_hole_ratio = num_holes / total_tiles
            
            print(f"Total tiles: {total_tiles}")
            print(f"Number of holes: {num_holes}")
            print(f"Actual hole ratio: {actual_hole_ratio:.2%}")
            
            # Show small maps
            if size == 4:
                print("Map layout:")
                for row in desc:
                    print("  " + row)
            
            env.close()

def main():
    print("="*50)
    print("Experiment 01: FrozenLake Setup Verification")
    print("="*50)
    
    # 1. System information
    print_environment_info()
    
    # 2. Explore environment
    n_states, n_actions = explore_frozenlake()
    
    # 3. Random baseline
    success_rate, avg_return = test_random_agent()
    
    # 4. Test slippery dynamics
    test_slippery_dynamics()
    
    # 5. Custom maps
    create_custom_maps()
    
    print("="*50)
    print("Setup Verification Summary")
    print("="*50)
    print(f"Environment: FrozenLake-v1")
    print(f"State space size: {n_states}")
    print(f"Action space size: {n_actions}")
    print(f"Random agent success rate: {success_rate:.2%}")
    print(f"Device for Q-learning: {device}")
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()