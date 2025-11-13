#!/usr/bin/env python3
"""
RL2025 - Lecture 8: Experiment 06 - Learned Value Function Baseline

This experiment implements a learned value function as baseline,
which is more sophisticated than scalar baselines and leads to
actor-critic methods.

Learning objectives:
- Implement value network training
- Use V(s) as state-dependent baseline
- Compare with EMA baseline
- Understand the path to actor-critic

Prerequisites: Completed exp05
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

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def train_reinforce_with_value_baseline(env, policy, value_net, 
                                       policy_optimizer, value_optimizer,
                                       n_episodes=200, gamma=0.99):
    """Train REINFORCE with learned value function baseline."""
    episode_returns = []
    episode_lengths = []
    value_losses = []
    policy_losses = []
    advantages_history = []
    
    for episode in range(n_episodes):
        # Collect episode
        obs, _ = env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        done = False
        while not done:
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action from policy
            logits = policy(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Get value estimate
            value = value_net(state)
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            obs = next_obs
        
        # Compute returns (reward-to-go)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(device)
        values = torch.cat(values).squeeze()
        
        # Compute advantages: A = G - V(s)
        advantages = returns - values.detach()
        advantages_history.extend(advantages.cpu().numpy().tolist())
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        policy_loss = 0
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss += -log_prob * advantage
        policy_loss = policy_loss / len(log_probs)
        
        # Value loss (MSE between V(s) and returns)
        value_loss = nn.MSELoss()(values, returns)
        
        # Update policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        policy_optimizer.step()
        
        # Update value function
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        value_optimizer.step()
        
        # Record statistics
        episode_return = sum(rewards)
        episode_returns.append(episode_return)
        episode_lengths.append(len(rewards))
        value_losses.append(value_loss.item())
        policy_losses.append(policy_loss.item())
        
        if (episode + 1) % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            avg_value_loss = np.mean(value_losses[-20:])
            print(f"Episode {episode+1:3d} | Return: {avg_return:7.2f} | "
                  f"V Loss: {avg_value_loss:.4f}")
    
    return episode_returns, value_losses, policy_losses, advantages_history

def evaluate_value_function(env, value_net, n_episodes=10):
    """Evaluate how well the value function predicts returns."""
    predictions = []
    actual_returns = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_predictions = []
        episode_rewards = []
        
        done = False
        while not done:
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                value_pred = value_net(state).item()
            
            episode_predictions.append(value_pred)
            
            # Random action for evaluation
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward)
        
        # Compute actual returns
        returns = []
        G = 0
        for reward in reversed(episode_rewards):
            G = reward + 0.99 * G
            returns.insert(0, G)
        
        predictions.extend(episode_predictions)
        actual_returns.extend(returns)
    
    return predictions, actual_returns

def main():
    print("="*50)
    print("Experiment 06: Learned Value Function Baseline")
    print("="*50)
    
    # Create environment
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    print(f"\nEnvironment: CartPole-v1")
    print(f"State dimension: {obs_dim}")
    print(f"Action space: {n_actions}")
    
    # Train with value baseline
    print("\n1. Training with Value Function Baseline")
    print("-" * 40)
    
    setup_seed(42)
    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    
    returns_value, v_losses, p_losses, advantages = train_reinforce_with_value_baseline(
        env, policy, value_net, policy_optimizer, value_optimizer, n_episodes=200
    )
    
    # Train without baseline for comparison
    print("\n2. Training without Baseline (for comparison)")
    print("-" * 40)
    
    setup_seed(42)
    policy_no_baseline = PolicyNetwork(obs_dim, n_actions).to(device)
    optimizer_no_baseline = optim.Adam(policy_no_baseline.parameters(), lr=1e-2)
    
    returns_no_baseline = []
    for episode in range(200):
        obs, _ = env.reset()
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        while not done:
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            logits = policy_no_baseline(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns_tensor = torch.FloatTensor(returns).to(device)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Policy gradient
        loss = 0
        for log_prob, G in zip(episode_log_probs, returns_tensor):
            loss += -log_prob * G
        loss = loss / len(episode_log_probs)
        
        optimizer_no_baseline.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_no_baseline.parameters(), max_norm=1.0)
        optimizer_no_baseline.step()
        
        returns_no_baseline.append(sum(episode_rewards))
        
        if (episode + 1) % 20 == 0:
            avg_return = np.mean(returns_no_baseline[-20:])
            print(f"Episode {episode+1:3d} | Return: {avg_return:7.2f}")
    
    # Evaluate value function quality
    print("\n3. Value Function Quality Assessment")
    print("-" * 40)
    
    predictions, actual = evaluate_value_function(env, value_net, n_episodes=10)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Returns comparison
    axes[0, 0].plot(returns_value, alpha=0.3, color='blue', label='With V(s) baseline')
    axes[0, 0].plot(np.convolve(returns_value, np.ones(20)/20, mode='valid'),
                    color='blue', linewidth=2)
    axes[0, 0].plot(returns_no_baseline, alpha=0.3, color='red', label='No baseline')
    axes[0, 0].plot(np.convolve(returns_no_baseline, np.ones(20)/20, mode='valid'),
                    color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Episode Returns Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Value loss
    axes[0, 1].plot(v_losses, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('Value Function Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Policy loss
    axes[0, 2].plot(p_losses, color='orange', alpha=0.7)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Policy Loss')
    axes[0, 2].set_title('Policy Gradient Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Advantage distribution
    axes[1, 0].hist(advantages, bins=50, alpha=0.7, color='purple', density=True)
    axes[1, 0].set_xlabel('Advantage')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Advantage Distribution')
    axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Value function predictions vs actual
    if predictions and actual:
        axes[1, 1].scatter(actual[:100], predictions[:100], alpha=0.5, s=10)
        axes[1, 1].plot([min(actual), max(actual)], 
                       [min(actual), max(actual)], 
                       'r--', label='Perfect prediction')
        axes[1, 1].set_xlabel('Actual Return')
        axes[1, 1].set_ylabel('Predicted Value')
        axes[1, 1].set_title('Value Function Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Variance comparison
    window = 20
    var_value = [np.var(returns_value[max(0,i-window):i+1]) 
                for i in range(len(returns_value))]
    var_no_baseline = [np.var(returns_no_baseline[max(0,i-window):i+1]) 
                      for i in range(len(returns_no_baseline))]
    
    axes[1, 2].plot(var_value, color='blue', label='With V(s)', linewidth=2)
    axes[1, 2].plot(var_no_baseline, color='red', label='No baseline', linewidth=2)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Variance')
    axes[1, 2].set_title(f'Return Variance ({window}-episode window)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Value Function Baseline Analysis', fontsize=14)
    plt.tight_layout()
    save_path = FIGURES_DIR / 'value_baseline_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved analysis plots to {save_path}")
    
    # Statistical analysis
    print("\n4. Statistical Comparison")
    print("-" * 40)
    
    print("Final 50 episodes:")
    print(f"With V(s) baseline: {np.mean(returns_value[-50:]):.2f} +/- {np.std(returns_value[-50:]):.2f}")
    print(f"No baseline:        {np.mean(returns_no_baseline[-50:]):.2f} +/- {np.std(returns_no_baseline[-50:]):.2f}")
    
    # Variance reduction
    var_reduction = (1 - np.var(returns_value) / np.var(returns_no_baseline)) * 100
    print(f"\nVariance reduction: {var_reduction:.1f}%")
    
    # Path to Actor-Critic
    print("\n5. Connection to Actor-Critic")
    print("-" * 40)
    print("Current approach (REINFORCE with baseline):")
    print("  1. Collect full episode")
    print("  2. Compute returns G_t")
    print("  3. Train V(s) to predict G_t")
    print("  4. Use A_t = G_t - V(s_t) for policy gradient")
    print("")
    print("Actor-Critic (next lecture):")
    print("  1. Take single step")
    print("  2. Use TD error: δ = r + γV(s') - V(s)")
    print("  3. Update policy with δ (one-step advantage)")
    print("  4. Update V(s) with TD learning")
    print("")
    print("Key difference: AC updates online, REINFORCE waits for episode end")
    
    print("\n" + "="*50)
    print("Value baseline demonstration completed!")
    print("This bridges the gap between REINFORCE and Actor-Critic.")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
