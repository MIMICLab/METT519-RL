#!/usr/bin/env python3
"""
RL2025 - Lecture 8: Experiment 04 - Reward-to-Go Implementation

This experiment demonstrates the reward-to-go technique, which reduces
variance by only considering future rewards for each action.

Learning objectives:
- Implement reward-to-go calculation
- Compare with full episode returns
- Analyze variance reduction
- Understand why this doesn't bias the gradient

Prerequisites: Completed exp03
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
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
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

def compute_full_returns(rewards, gamma=0.99):
    """Compute full episode return for each timestep (vanilla REINFORCE)."""
    G_0 = sum([gamma**t * r for t, r in enumerate(rewards)])
    # Every action gets credited with the full episode return
    return [G_0] * len(rewards)

def compute_reward_to_go(rewards, gamma=0.99):
    """Compute reward-to-go (future returns) for each timestep."""
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

def train_reinforce(env, policy, optimizer, n_episodes=100, gamma=0.99, use_reward_to_go=True):
    """Train policy using REINFORCE with optional reward-to-go."""
    episode_returns = []
    episode_lengths = []
    gradient_norms = []
    
    for episode in range(n_episodes):
        # Collect episode
        obs, _ = env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        done = False
        while not done:
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            logits = policy(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            obs = next_obs
        
        # Compute returns
        if use_reward_to_go:
            returns = compute_reward_to_go(rewards, gamma)
        else:
            returns = compute_full_returns(rewards, gamma)
        
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute loss
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
        loss = loss / len(log_probs)
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        gradient_norms.append(total_norm.item())
        
        optimizer.step()
        
        # Record statistics
        episode_return = sum(rewards)
        episode_returns.append(episode_return)
        episode_lengths.append(len(rewards))
        
        if (episode + 1) % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            avg_grad_norm = np.mean(gradient_norms[-20:])
            print(f"Episode {episode+1:3d} | Avg Return: {avg_return:7.2f} | "
                  f"Grad Norm: {avg_grad_norm:.4f}")
    
    return episode_returns, episode_lengths, gradient_norms

def compare_variance(env, n_runs=5, n_episodes=100):
    """Compare variance between full returns and reward-to-go."""
    
    full_returns_runs = []
    rtg_returns_runs = []
    
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        
        # Reset seed for fair comparison
        setup_seed(42 + run)
        
        # Train with full returns
        print("Training with full returns...")
        policy_full = PolicyNetwork(env.observation_space.shape[0], 
                                   env.action_space.n).to(device)
        optimizer_full = optim.Adam(policy_full.parameters(), lr=1e-2)
        
        returns_full, _, _ = train_reinforce(
            env, policy_full, optimizer_full, 
            n_episodes=n_episodes, use_reward_to_go=False
        )
        full_returns_runs.append(returns_full)
        
        # Train with reward-to-go
        print("Training with reward-to-go...")
        policy_rtg = PolicyNetwork(env.observation_space.shape[0], 
                                  env.action_space.n).to(device)
        optimizer_rtg = optim.Adam(policy_rtg.parameters(), lr=1e-2)
        
        returns_rtg, _, _ = train_reinforce(
            env, policy_rtg, optimizer_rtg,
            n_episodes=n_episodes, use_reward_to_go=True
        )
        rtg_returns_runs.append(returns_rtg)
    
    return full_returns_runs, rtg_returns_runs

def main():
    print("="*50)
    print("Experiment 04: Reward-to-Go")
    print("="*50)
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    print("\n1. Single Run Comparison")
    print("-" * 30)
    
    # Train with reward-to-go
    setup_seed(42)
    policy_rtg = PolicyNetwork(env.observation_space.shape[0], 
                              env.action_space.n).to(device)
    optimizer_rtg = optim.Adam(policy_rtg.parameters(), lr=1e-2)
    
    print("Training with reward-to-go...")
    returns_rtg, lengths_rtg, grads_rtg = train_reinforce(
        env, policy_rtg, optimizer_rtg, n_episodes=200, use_reward_to_go=True
    )
    
    # Train with full returns
    setup_seed(42)
    policy_full = PolicyNetwork(env.observation_space.shape[0], 
                               env.action_space.n).to(device)
    optimizer_full = optim.Adam(policy_full.parameters(), lr=1e-2)
    
    print("\nTraining with full returns...")
    returns_full, lengths_full, grads_full = train_reinforce(
        env, policy_full, optimizer_full, n_episodes=200, use_reward_to_go=False
    )
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Returns comparison
    axes[0, 0].plot(returns_rtg, alpha=0.3, color='blue', label='Reward-to-go')
    axes[0, 0].plot(np.convolve(returns_rtg, np.ones(20)/20, mode='valid'),
                    color='blue', linewidth=2)
    axes[0, 0].plot(returns_full, alpha=0.3, color='red', label='Full returns')
    axes[0, 0].plot(np.convolve(returns_full, np.ones(20)/20, mode='valid'),
                    color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Episode Returns Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gradient norms
    axes[0, 1].plot(grads_rtg, alpha=0.5, color='blue', label='Reward-to-go')
    axes[0, 1].plot(grads_full, alpha=0.5, color='red', label='Full returns')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_title('Gradient Norms')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Variance over time
    window = 20
    variance_rtg = [np.var(returns_rtg[max(0,i-window):i+1]) 
                   for i in range(len(returns_rtg))]
    variance_full = [np.var(returns_full[max(0,i-window):i+1]) 
                    for i in range(len(returns_full))]
    
    axes[1, 0].plot(variance_rtg, color='blue', label='Reward-to-go', linewidth=2)
    axes[1, 0].plot(variance_full, color='red', label='Full returns', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title(f'Return Variance ({window}-episode window)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning speed comparison
    target_return = 195  # Target performance
    rtg_convergence = next((i for i, r in enumerate(returns_rtg) if r >= target_return), -1)
    full_convergence = next((i for i, r in enumerate(returns_full) if r >= target_return), -1)
    
    axes[1, 1].bar(['Reward-to-go', 'Full returns'], 
                   [rtg_convergence if rtg_convergence != -1 else 200,
                    full_convergence if full_convergence != -1 else 200],
                   color=['blue', 'red'])
    axes[1, 1].set_ylabel('Episodes to reach 195')
    axes[1, 1].set_title('Convergence Speed')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Reward-to-Go vs Full Returns', fontsize=14)
    plt.tight_layout()
    save_path = FIGURES_DIR / 'reward_to_go_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plots to {save_path}")
    
    # Statistical comparison
    print("\n2. Statistical Analysis")
    print("-" * 30)
    
    print("Final 50 episodes statistics:")
    print(f"Reward-to-go:")
    print(f"  Mean: {np.mean(returns_rtg[-50:]):.2f}")
    print(f"  Std:  {np.std(returns_rtg[-50:]):.2f}")
    print(f"Full returns:")
    print(f"  Mean: {np.mean(returns_full[-50:]):.2f}")
    print(f"  Std:  {np.std(returns_full[-50:]):.2f}")
    
    # Variance reduction percentage
    var_rtg = np.var(returns_rtg)
    var_full = np.var(returns_full)
    reduction = (1 - var_rtg/var_full) * 100
    print(f"\nVariance reduction: {reduction:.1f}%")
    
    # Mathematical explanation
    print("\n3. Why Reward-to-Go Works")
    print("-" * 30)
    print("Full returns:    grad = sum_t log pi(a_t|s_t) * G_0")
    print("Reward-to-go:    grad = sum_t log pi(a_t|s_t) * G_t")
    print("")
    print("Key insight: Past rewards R_0...R_{t-1} are independent")
    print("of action a_t, so they only add variance without signal!")
    print("")
    print("Mathematically: E[log pi(a_t) * R_past] = R_past * E[log pi(a_t)] = 0")
    print("(The expected gradient of log probability is zero)")
    
    print("\n" + "="*50)
    print("Reward-to-go demonstration completed!")
    print("Key takeaway: Simple trick, significant variance reduction.")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
