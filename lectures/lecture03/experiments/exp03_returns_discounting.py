#!/usr/bin/env python3
"""
RL2025 - Lecture 3: Experiment 03 - Returns and Discounting

This experiment demonstrates the concept of returns (cumulative rewards) and 
the effect of discount factors on future reward valuation.

Learning objectives:
- Understand returns and discounted returns
- Implement return calculation with different discount factors
- Analyze the effect of gamma on return values
- Visualize the impact of discounting on episode evaluation

Prerequisites: Experiments 01-02 completed successfully
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

import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json

def make_env(env_id: str = "CartPole-v1", seed: int = 42) -> gym.Env:
    """Create and initialize environment with proper seeding"""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def collect_episode_rewards(env: gym.Env, policy_fn, max_steps: int = 500) -> List[float]:
    """
    Collect rewards from a single episode
    
    Args:
        env: Gymnasium environment
        policy_fn: Function that maps observation to action
        max_steps: Maximum episode length
    
    Returns:
        List of rewards received during the episode
    """
    obs, _ = env.reset()
    rewards = []
    
    for step in range(max_steps):
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        obs = next_obs
        
        if terminated or truncated:
            break
    
    return rewards

def random_policy(obs: np.ndarray) -> int:
    """Random policy: uniformly sample from action space"""
    return np.random.randint(0, 2)

def calculate_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Calculate discounted returns from rewards using the standard RL formula:
    G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k+1}
    
    Args:
        rewards: List of rewards [r_1, r_2, ..., r_T]
        gamma: Discount factor
    
    Returns:
        List of returns [G_0, G_1, ..., G_{T-1}]
    """
    returns = []
    T = len(rewards)
    
    for t in range(T):
        G_t = 0.0
        for k in range(T - t):
            G_t += (gamma ** k) * rewards[t + k]
        returns.append(G_t)
    
    return returns

def calculate_returns_efficient(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Efficient backward calculation of returns:
    G_t = r_{t+1} + gamma * G_{t+1}
    
    Args:
        rewards: List of rewards [r_1, r_2, ..., r_T]
        gamma: Discount factor
    
    Returns:
        List of returns [G_0, G_1, ..., G_{T-1}]
    """
    returns = []
    G = 0.0
    
    # Calculate backwards from the end
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)
    
    # Reverse to get forward order
    return list(reversed(returns))

def analyze_discounting_effect():
    """Analyze how different gamma values affect return calculation"""
    print("\n--- Discounting Effect Analysis ---")
    
    # Create a synthetic reward sequence for clear demonstration
    rewards = [1.0, 2.0, 3.0, 1.0, 5.0]  # Simple reward sequence
    gamma_values = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    
    print(f"Reward sequence: {rewards}")
    print(f"Episode length: {len(rewards)} steps")
    print()
    
    results = {}
    
    for gamma in gamma_values:
        returns = calculate_returns(rewards, gamma)
        returns_efficient = calculate_returns_efficient(rewards, gamma)
        
        # Verify both methods give same results
        assert np.allclose(returns, returns_efficient), f"Method mismatch for gamma={gamma}"
        
        results[gamma] = returns
        
        print(f"Gamma = {gamma:.2f}:")
        print(f"  Returns: {[f'{r:.3f}' for r in returns]}")
        print(f"  G_0 (total return): {returns[0]:.3f}")
        print()
    
    return results, rewards

def collect_episode_analysis():
    """Collect real episode data and analyze returns"""
    print("\n--- Real Episode Analysis ---")
    
    env = make_env("CartPole-v1", seed=42)
    
    # Collect a few episodes
    num_episodes = 3
    gamma_values = [0.9, 0.99, 1.0]
    
    episode_data = []
    
    for ep in range(num_episodes):
        env.reset(seed=42 + ep)
        rewards = collect_episode_rewards(env, random_policy)
        
        episode_info = {
            'episode': ep + 1,
            'rewards': rewards,
            'length': len(rewards),
            'undiscounted_return': sum(rewards)
        }
        
        # Calculate returns for different gamma values
        for gamma in gamma_values:
            returns = calculate_returns_efficient(rewards, gamma)
            episode_info[f'G_0_gamma_{gamma}'] = returns[0]
            episode_info[f'returns_gamma_{gamma}'] = returns
        
        episode_data.append(episode_info)
        
        print(f"Episode {ep + 1}:")
        print(f"  Length: {len(rewards)} steps")
        print(f"  Undiscounted return (gamma=1.0): {sum(rewards):.1f}")
        for gamma in gamma_values:
            if gamma != 1.0:
                returns = calculate_returns_efficient(rewards, gamma)
                print(f"  Discounted return (gamma={gamma}): {returns[0]:.3f}")
        print()
    
    env.close()
    return episode_data

def plot_discounting_comparison(results: Dict[float, List[float]], 
                              rewards: List[float], 
                              save_path: str = None):
    """Visualize the effect of different discount factors"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Effect of Discount Factor on Returns', fontsize=14)
    
    time_steps = list(range(len(rewards)))
    
    # Plot 1: Rewards over time
    axes[0, 0].bar(time_steps, rewards, alpha=0.7, color='gray', label='Rewards')
    axes[0, 0].set_title('Reward Sequence')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Returns for different gamma values
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'black']
    for i, (gamma, returns) in enumerate(results.items()):
        axes[0, 1].plot(time_steps, returns, 'o-', color=colors[i], 
                       label=f'γ={gamma}', linewidth=2, markersize=6)
    
    axes[0, 1].set_title('Returns G_t for Different Discount Factors')
    axes[0, 1].set_xlabel('Time Step t')
    axes[0, 1].set_ylabel('Return G_t')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Initial return (G_0) vs gamma
    gamma_vals = list(results.keys())
    g0_vals = [results[gamma][0] for gamma in gamma_vals]
    
    axes[1, 0].plot(gamma_vals, g0_vals, 'bo-', linewidth=2, markersize=8)
    axes[1, 0].set_title('Initial Return G_0 vs Discount Factor')
    axes[1, 0].set_xlabel('Discount Factor γ')
    axes[1, 0].set_ylabel('Initial Return G_0')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Discount weights over time
    max_len = len(rewards)
    time_steps_extended = list(range(max_len))
    
    for gamma in [0.5, 0.9, 0.99]:
        weights = [gamma ** k for k in time_steps_extended]
        axes[1, 1].plot(time_steps_extended, weights, 'o-', 
                       label=f'γ={gamma}', linewidth=2, markersize=4)
    
    axes[1, 1].set_title('Discount Weights γ^k')
    axes[1, 1].set_xlabel('Time Step k')
    axes[1, 1].set_ylabel('Weight γ^k')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Discounting analysis plot saved to: {save_path}")
    else:
        plt.show()

def main():
    """Run returns and discounting experiment"""
    print("="*50)
    print("Experiment 03: Returns and Discounting")
    print("="*50)
    
    # Demonstrate discounting effect with synthetic data
    results, rewards = analyze_discounting_effect()
    
    # Analyze real episode data
    episode_data = collect_episode_analysis()
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    # Save synthetic analysis
    with open("results/discounting_analysis.json", "w") as f:
        # Convert keys to strings for JSON serialization
        json_results = {str(k): v for k, v in results.items()}
        analysis_data = {
            'synthetic_rewards': rewards,
            'gamma_analysis': json_results
        }
        json.dump(analysis_data, f, indent=2)
    
    # Save episode data
    with open("results/episode_returns.json", "w") as f:
        json.dump(episode_data, f, indent=2)
    
    print("Results saved to:")
    print("  - results/discounting_analysis.json")
    print("  - results/episode_returns.json")
    
    # Create visualizations
    try:
        plot_discounting_comparison(results, rewards, "results/discounting_comparison.png")
    except ImportError:
        print("Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("KEY INSIGHTS:")
    print("="*50)
    print("1. Lower gamma values emphasize immediate rewards")
    print("2. Higher gamma values consider long-term consequences")
    print("3. gamma=1.0 gives undiscounted (total) return")
    print("4. gamma=0.0 only considers immediate reward")
    print("5. Choice of gamma affects policy evaluation significantly")
    
    print("\nExperiment 03 completed successfully!")
    return True

if __name__ == "__main__":
    main()