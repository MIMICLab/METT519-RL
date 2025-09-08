#!/usr/bin/env python3
"""
RL2025 - Lecture 3: Experiment 02 - RL Basics and Agent-Environment Interaction

This experiment demonstrates the fundamental RL concepts: agent-environment loop,
states, actions, rewards, and episodes using CartPole-v1.

Learning objectives:
- Understand the agent-environment interaction cycle
- Implement basic episode collection
- Analyze reward structure and episode termination
- Visualize episode trajectories

Prerequisites: Experiment 01 completed successfully
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
from typing import List, Tuple, Dict, Any
import json

class Episode:
    """Container for episode data"""
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.terminated: bool = False
        self.truncated: bool = False
        self.total_reward: float = 0.0
        self.length: int = 0

def make_env(env_id: str = "CartPole-v1", seed: int = 42) -> gym.Env:
    """Create and initialize environment with proper seeding"""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def collect_episode(env: gym.Env, policy_fn, max_steps: int = 500) -> Episode:
    """
    Collect a single episode using the given policy function
    
    Args:
        env: Gymnasium environment
        policy_fn: Function that maps observation to action
        max_steps: Maximum episode length
    
    Returns:
        Episode object with complete trajectory
    """
    episode = Episode()
    
    # Reset environment
    obs, info = env.reset()
    episode.observations.append(obs.copy())
    
    step = 0
    while step < max_steps:
        # Policy decides action
        action = policy_fn(obs)
        episode.actions.append(action)
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode.rewards.append(float(reward))
        episode.observations.append(next_obs.copy())
        
        # Update for next iteration
        obs = next_obs
        step += 1
        
        # Check episode termination
        if terminated or truncated:
            episode.terminated = terminated
            episode.truncated = truncated
            break
    
    episode.total_reward = sum(episode.rewards)
    episode.length = len(episode.rewards)
    
    return episode

def random_policy(obs: np.ndarray) -> int:
    """Random policy: uniformly sample from action space"""
    return np.random.randint(0, 2)

def analyze_episode(episode: Episode, episode_num: int = None) -> Dict[str, Any]:
    """Analyze a single episode and extract key statistics"""
    
    # Basic statistics
    stats = {
        'episode_num': episode_num,
        'total_reward': episode.total_reward,
        'length': episode.length,
        'terminated': episode.terminated,
        'truncated': episode.truncated,
        'avg_reward_per_step': episode.total_reward / max(episode.length, 1),
    }
    
    # Observation analysis
    if episode.observations:
        obs_array = np.array(episode.observations[:-1])  # Exclude final obs
        stats.update({
            'avg_position': float(np.mean(obs_array[:, 0])),
            'avg_velocity': float(np.mean(obs_array[:, 1])),
            'avg_angle': float(np.mean(obs_array[:, 2])),
            'avg_angular_velocity': float(np.mean(obs_array[:, 3])),
            'max_abs_angle': float(np.max(np.abs(obs_array[:, 2]))),
            'max_abs_position': float(np.max(np.abs(obs_array[:, 0]))),
        })
    
    # Action analysis
    if episode.actions:
        action_counts = np.bincount(episode.actions, minlength=2)
        stats.update({
            'action_0_count': int(action_counts[0]),
            'action_1_count': int(action_counts[1]),
            'action_balance': float(action_counts[1] / max(len(episode.actions), 1)),
        })
    
    return stats

def collect_multiple_episodes(env_id: str = "CartPole-v1", 
                            num_episodes: int = 10, 
                            seed: int = 42) -> List[Episode]:
    """Collect multiple episodes for analysis"""
    
    setup_seed(seed)
    env = make_env(env_id, seed)
    episodes = []
    
    print(f"Collecting {num_episodes} episodes with random policy...")
    
    for ep in range(num_episodes):
        # Use different seed for each episode to get variety
        env.reset(seed=seed + ep)
        episode = collect_episode(env, random_policy)
        episodes.append(episode)
        
        print(f"Episode {ep + 1}: reward={episode.total_reward}, length={episode.length}, "
              f"terminated={episode.terminated}, truncated={episode.truncated}")
    
    env.close()
    return episodes

def plot_episode_analysis(episodes: List[Episode], save_path: str = None):
    """Create visualizations of episode statistics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Episode Analysis - Random Policy on CartPole-v1', fontsize=14)
    
    # Episode rewards
    rewards = [ep.total_reward for ep in episodes]
    axes[0, 0].bar(range(len(rewards)), rewards, alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    lengths = [ep.length for ep in episodes]
    axes[0, 1].bar(range(len(lengths)), lengths, alpha=0.7, color='orange')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length (steps)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 0].hist(rewards, bins=10, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Total Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Termination analysis
    terminated_count = sum(1 for ep in episodes if ep.terminated)
    truncated_count = sum(1 for ep in episodes if ep.truncated)
    categories = ['Terminated', 'Truncated']
    counts = [terminated_count, truncated_count]
    
    axes[1, 1].bar(categories, counts, alpha=0.7, color=['red', 'blue'])
    axes[1, 1].set_title('Episode Termination Types')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    """Run RL basics experiment"""
    print("="*50)
    print("Experiment 02: RL Basics and Agent-Environment Interaction")
    print("="*50)
    
    # Collect episodes
    episodes = collect_multiple_episodes(
        env_id="CartPole-v1",
        num_episodes=20,
        seed=42
    )
    
    # Analyze episodes
    print("\n--- Episode Analysis ---")
    all_stats = []
    for i, episode in enumerate(episodes):
        stats = analyze_episode(episode, i + 1)
        all_stats.append(stats)
    
    # Summary statistics
    rewards = [stats['total_reward'] for stats in all_stats]
    lengths = [stats['length'] for stats in all_stats]
    terminated_count = sum(1 for stats in all_stats if stats['terminated'])
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"Episodes collected: {len(episodes)}")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"Min reward: {np.min(rewards):.0f}, Max reward: {np.max(rewards):.0f}")
    print(f"Min length: {np.min(lengths):.0f}, Max length: {np.max(lengths):.0f}")
    print(f"Terminated episodes: {terminated_count}/{len(episodes)}")
    print(f"Truncated episodes: {len(episodes) - terminated_count}/{len(episodes)}")
    
    # Save detailed stats
    os.makedirs("results", exist_ok=True)
    with open("results/episode_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("\nDetailed statistics saved to: results/episode_stats.json")
    
    # Create visualizations
    try:
        plot_episode_analysis(episodes, "results/episode_analysis.png")
    except ImportError:
        print("Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\nExperiment 02 completed successfully!")
    return True

if __name__ == "__main__":
    main()