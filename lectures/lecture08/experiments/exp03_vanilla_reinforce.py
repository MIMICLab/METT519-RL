#!/usr/bin/env python3
"""
RL2025 - Lecture 8: Experiment 03 - Vanilla REINFORCE Implementation

This experiment implements the basic REINFORCE algorithm without any
variance reduction techniques.

Learning objectives:
- Implement episode collection
- Calculate returns (full episode)
- Apply policy gradient updates
- Observe high variance in vanilla REINFORCE

Prerequisites: Completed exp01 and exp02
"""

# PyTorch 2.x Standard Practice Header
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

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

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR = Path(os.environ.get("LECTURE08_FIGURES_DIR", DEFAULT_FIGURES_DIR))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)

def collect_episode(env, policy, max_steps=500):
    """Collect one episode of experience."""
    obs, _ = env.reset()
    states = []
    actions = []
    rewards = []
    log_probs = []
    
    for _ in range(max_steps):
        # Convert observation to tensor
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)  # [1, obs_dim]
        
        # Get action from policy (keep grad for REINFORCE)
        logits = policy(state)  # [1, n_actions]
        
        dist = Categorical(logits=logits)
        action = dist.sample()  # [1]
        log_prob = dist.log_prob(action)  # [1]
        
        # Take action in environment
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        
        # Store experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        
        obs = next_obs
        
        if terminated or truncated:
            break
    
    return states, actions, rewards, log_probs

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns for all timesteps."""
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

def vanilla_reinforce(env, policy, optimizer, n_episodes=100, gamma=0.99):
    """Train policy using vanilla REINFORCE."""
    episode_returns = []
    episode_lengths = []
    losses = []
    
    for episode in range(n_episodes):
        # Collect episode
        states, actions, rewards, log_probs = collect_episode(env, policy)
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize returns (helps with stability)
        # Note: This is not a baseline, just normalization
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.cat(policy_loss).mean()
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Record statistics
        episode_return = sum(rewards)
        episode_returns.append(episode_return)
        episode_lengths.append(len(rewards))
        losses.append(loss.item())
        
        if (episode + 1) % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode+1:3d} | Avg Return: {avg_return:7.2f} | "
                  f"Avg Length: {avg_length:6.1f} | Loss: {loss.item():.4f}")
    
    return episode_returns, episode_lengths, losses

def main():
    print("="*50)
    print("Experiment 03: Vanilla REINFORCE")
    print("="*50)
    
    # Create environment
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    print(f"\nEnvironment: CartPole-v1")
    print(f"Observation dimension: {obs_dim}")
    print(f"Number of actions: {n_actions}")
    
    # Create policy network
    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    
    print(f"\nPolicy Network:")
    print(f"Architecture: {obs_dim} -> 64 -> 64 -> {n_actions}")
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Train with vanilla REINFORCE
    print("\nTraining with Vanilla REINFORCE...")
    print("-" * 50)
    
    episode_returns, episode_lengths, losses = vanilla_reinforce(
        env, policy, optimizer, n_episodes=200, gamma=0.99
    )
    
    # Evaluate final policy
    print("\n" + "="*50)
    print("Final Evaluation (10 episodes):")
    
    eval_returns = []
    for _ in range(10):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = policy(state)
            
            # Use greedy action for evaluation
            action = logits.argmax().item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        eval_returns.append(total_reward)
    
    print(f"Evaluation returns: {eval_returns}")
    print(f"Mean return: {np.mean(eval_returns):.2f} +/- {np.std(eval_returns):.2f}")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode returns
    axes[0, 0].plot(episode_returns, alpha=0.3, color='blue')
    axes[0, 0].plot(np.convolve(episode_returns, np.ones(20)/20, mode='valid'), 
                    color='blue', linewidth=2, label='20-episode moving avg')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, color='green')
    axes[0, 1].plot(np.convolve(episode_lengths, np.ones(20)/20, mode='valid'),
                    color='green', linewidth=2, label='20-episode moving avg')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss
    axes[1, 0].plot(losses, alpha=0.5, color='red')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Variance analysis
    window = 20
    rolling_returns = [episode_returns[i:i+window] for i in range(len(episode_returns)-window+1)]
    variances = [np.var(r) for r in rolling_returns]
    axes[1, 1].plot(variances, color='purple')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_title(f'Return Variance ({window}-episode window)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Vanilla REINFORCE Training Analysis', fontsize=14)
    plt.tight_layout()
    save_path = FIGURES_DIR / 'vanilla_reinforce_training.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved training plots to {save_path}")
    
    # Observations about vanilla REINFORCE
    print("\n" + "="*50)
    print("Key Observations:")
    print("1. High variance in returns (noisy learning curve)")
    print("2. Slow convergence compared to value-based methods")
    print("3. Sensitive to learning rate")
    print("4. Benefits from return normalization")
    print("5. Needs variance reduction for practical use")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
