#!/usr/bin/env python3
"""
RL2025 - Lecture 8: Experiment 08 - Advanced REINFORCE Techniques

This experiment combines multiple advanced techniques for improved
REINFORCE performance: GAE-style advantages, gradient clipping,
learning rate scheduling, and proper initialization.

Learning objectives:
- Implement multiple episodes per update (batch training)
- Apply advanced normalization techniques
- Use learning rate scheduling
- Combine all improvements for best performance

Prerequisites: Completed exp07
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

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
        
        # Better initialization (orthogonal)
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class RunningMeanStd:
    """Running mean and standard deviation for observation normalization."""
    def __init__(self, shape, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

def collect_batch_episodes(env, policy, n_episodes, obs_normalizer=None):
    """Collect multiple episodes for batch training."""
    all_states = []
    all_actions = []
    all_rewards = []
    all_log_probs = []
    all_returns = []
    all_advantages = []
    episode_returns = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        done = False
        while not done:
            # Normalize observation if normalizer provided
            if obs_normalizer:
                norm_obs = obs_normalizer.normalize(obs)
            else:
                norm_obs = obs
            
            state = torch.FloatTensor(norm_obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = policy(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(norm_obs)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            
            obs = next_obs
        
        # Update observation normalizer
        if obs_normalizer:
            obs_normalizer.update(np.array(states))
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + 0.99 * G
            returns.insert(0, G)
        
        # Store episode data
        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_log_probs.extend(log_probs)
        all_returns.extend(returns)
        
        episode_returns.append(sum(rewards))
    
    return (np.array(all_states), np.array(all_actions), 
            np.array(all_returns), episode_returns)

def advanced_reinforce(env, n_updates=100, episodes_per_update=8,
                      lr_start=3e-3, lr_end=3e-4, 
                      entropy_start=0.01, entropy_end=0.001,
                      use_obs_norm=True):
    """Advanced REINFORCE with all improvements."""
    
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    # Networks
    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    
    # Optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr_start)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_start)
    
    # Learning rate scheduling
    policy_scheduler = CosineAnnealingLR(policy_optimizer, T_max=n_updates, eta_min=lr_end)
    value_scheduler = CosineAnnealingLR(value_optimizer, T_max=n_updates, eta_min=lr_end)
    
    # Observation normalization
    obs_normalizer = RunningMeanStd(obs_dim) if use_obs_norm else None
    
    # Tracking
    all_returns = []
    all_lengths = []
    entropy_values = []
    value_losses = []
    policy_losses = []
    learning_rates = []
    
    for update in range(n_updates):
        # Entropy coefficient scheduling
        entropy_coef = entropy_start + (entropy_end - entropy_start) * (update / n_updates)
        
        # Collect batch of episodes
        states, actions, returns, episode_returns = collect_batch_episodes(
            env, policy, episodes_per_update, obs_normalizer
        )
        
        all_returns.extend(episode_returns)
        all_lengths.append(len(episode_returns))
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # Compute values and advantages
        with torch.no_grad():
            values = value_net(states).squeeze()
        
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Normalize returns for value target
        returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy update
        logits = policy(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        policy_loss = -(log_probs * advantages.detach()).mean()
        total_policy_loss = policy_loss - entropy_coef * entropy
        
        policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        policy_optimizer.step()
        
        # Value update
        values = value_net(states).squeeze()
        value_loss = nn.MSELoss()(values, returns_normalized)
        
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
        value_optimizer.step()
        
        # Update schedulers
        policy_scheduler.step()
        value_scheduler.step()
        
        # Track statistics
        entropy_values.append(entropy.item())
        value_losses.append(value_loss.item())
        policy_losses.append(policy_loss.item())
        learning_rates.append(policy_optimizer.param_groups[0]['lr'])
        
        if (update + 1) % 10 == 0:
            avg_return = np.mean(all_returns[-80:])  # Last 10 updates * 8 episodes
            print(f"Update {update+1:3d} | Return: {avg_return:7.2f} | "
                  f"Entropy: {entropy.item():.4f} | LR: {learning_rates[-1]:.5f}")
    
    return (all_returns, entropy_values, value_losses, 
            policy_losses, learning_rates, policy, value_net)

def compare_techniques(env):
    """Compare basic vs advanced REINFORCE."""
    
    print("\n1. Basic REINFORCE")
    print("-" * 40)
    
    setup_seed(42)
    basic_policy = PolicyNetwork(env.observation_space.shape[0], 
                                 env.action_space.n).to(device)
    basic_optimizer = optim.Adam(basic_policy.parameters(), lr=1e-2)
    
    basic_returns = []
    for episode in range(200):
        obs, _ = env.reset()
        rewards = []
        log_probs = []
        
        done = False
        while not done:
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            logits = basic_policy(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            rewards.append(reward)
            log_probs.append(log_prob)
        
        # Compute returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns_t = torch.FloatTensor(returns).to(device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        
        # Update
        loss = 0
        for log_prob, G in zip(log_probs, returns_t):
            loss += -log_prob * G
        loss = loss / len(log_probs)
        
        basic_optimizer.zero_grad()
        loss.backward()
        basic_optimizer.step()
        
        basic_returns.append(sum(rewards))
        
        if (episode + 1) % 20 == 0:
            avg_return = np.mean(basic_returns[-20:])
            print(f"Episode {episode+1:3d} | Return: {avg_return:7.2f}")
    
    print("\n2. Advanced REINFORCE")
    print("-" * 40)
    
    setup_seed(42)
    advanced_returns, entropies, v_losses, p_losses, lrs, final_policy, final_value = \
        advanced_reinforce(env, n_updates=25, episodes_per_update=8)
    
    return basic_returns, advanced_returns, entropies, lrs

def main():
    print("="*50)
    print("Experiment 08: Advanced REINFORCE Techniques")
    print("="*50)
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Compare techniques
    basic_returns, advanced_returns, entropies, lrs = compare_techniques(env)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Returns comparison
    axes[0, 0].plot(basic_returns, alpha=0.3, color='red', label='Basic')
    axes[0, 0].plot(np.convolve(basic_returns, np.ones(20)/20, mode='valid'),
                    color='red', linewidth=2)
    axes[0, 0].plot(advanced_returns, alpha=0.3, color='blue', label='Advanced')
    axes[0, 0].plot(np.convolve(advanced_returns, np.ones(20)/20, mode='valid'),
                    color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Returns Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample efficiency
    x_basic = np.arange(len(basic_returns))
    x_advanced = np.arange(len(advanced_returns))
    
    axes[0, 1].plot(x_basic, basic_returns, alpha=0.3, color='red')
    ma_basic = np.convolve(basic_returns, np.ones(20)/20, mode='valid')
    axes[0, 1].plot(x_basic[19:], ma_basic, color='red', linewidth=2, label='Basic')
    
    axes[0, 1].plot(x_advanced, advanced_returns, alpha=0.3, color='blue')
    ma_advanced = np.convolve(advanced_returns, np.ones(20)/20, mode='valid')
    axes[0, 1].plot(x_advanced[19:], ma_advanced, color='blue', linewidth=2, label='Advanced')
    
    axes[0, 1].axhline(y=195, color='green', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].set_xlabel('Episodes')
    axes[0, 1].set_ylabel('Return')
    axes[0, 1].set_title('Sample Efficiency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Entropy schedule
    axes[0, 2].plot(entropies, color='purple', linewidth=2)
    axes[0, 2].set_xlabel('Update')
    axes[0, 2].set_ylabel('Entropy')
    axes[0, 2].set_title('Entropy Schedule')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[1, 0].plot(lrs, color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Variance comparison
    window = 20
    var_basic = [np.var(basic_returns[max(0,i-window):i+1]) 
                for i in range(len(basic_returns))]
    var_advanced = [np.var(advanced_returns[max(0,i-window):i+1]) 
                   for i in range(len(advanced_returns))]
    
    axes[1, 1].plot(var_basic, color='red', label='Basic', linewidth=2)
    axes[1, 1].plot(var_advanced, color='blue', label='Advanced', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_title(f'Return Variance ({window}-episode window)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Final performance
    techniques = ['Basic', 'Advanced']
    final_means = [np.mean(basic_returns[-50:]), np.mean(advanced_returns[-50:])]
    final_stds = [np.std(basic_returns[-50:]), np.std(advanced_returns[-50:])]
    
    axes[1, 2].bar(techniques, final_means, yerr=final_stds, 
                  capsize=5, color=['red', 'blue'])
    axes[1, 2].set_ylabel('Mean Return')
    axes[1, 2].set_title('Final Performance (last 50 episodes)')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Advanced REINFORCE Techniques', fontsize=14)
    plt.tight_layout()
    save_path = FIGURES_DIR / 'advanced_reinforce_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plots to {save_path}")
    
    # Summary of techniques
    print("\n3. Techniques Summary")
    print("-" * 50)
    print("Advanced techniques used:")
    print("✓ Batch training (8 episodes per update)")
    print("✓ Value function baseline")
    print("✓ Observation normalization")
    print("✓ Orthogonal initialization")
    print("✓ Learning rate scheduling (cosine annealing)")
    print("✓ Entropy scheduling (decreasing)")
    print("✓ Gradient clipping (max norm 0.5)")
    print("✓ Advantage normalization")
    print("✓ Return normalization for value targets")
    
    # Performance comparison
    print("\n4. Performance Metrics")
    print("-" * 40)
    print(f"Basic REINFORCE:")
    print(f"  Final return: {np.mean(basic_returns[-50:]):.2f} ± {np.std(basic_returns[-50:]):.2f}")
    print(f"  Episodes to 195: {next((i for i, r in enumerate(basic_returns) if r >= 195), 200)}")
    
    print(f"\nAdvanced REINFORCE:")
    print(f"  Final return: {np.mean(advanced_returns[-50:]):.2f} ± {np.std(advanced_returns[-50:]):.2f}")
    print(f"  Episodes to 195: {next((i for i, r in enumerate(advanced_returns) if r >= 195), 200)}")
    
    improvement = (np.mean(advanced_returns[-50:]) / np.mean(basic_returns[-50:]) - 1) * 100
    print(f"\nPerformance improvement: {improvement:.1f}%")
    
    print("\n" + "="*50)
    print("Advanced REINFORCE demonstration completed!")
    print("Combining techniques yields significant improvements.")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
