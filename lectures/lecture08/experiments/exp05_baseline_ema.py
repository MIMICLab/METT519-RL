#!/usr/bin/env python3
"""
RL2025 - Lecture 8: Experiment 05 - Exponential Moving Average Baseline

This experiment implements a simple scalar baseline using exponential
moving average (EMA) of returns to reduce variance.

Learning objectives:
- Implement EMA baseline
- Understand why baselines don't bias gradients
- Compare different EMA decay rates
- Analyze variance reduction effectiveness

Prerequisites: Completed exp04
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

class EMABaseline:
    def __init__(self, alpha=0.01):
        """
        Exponential Moving Average baseline.
        alpha: smoothing factor (0 < alpha <= 1)
        - Small alpha (e.g., 0.01): slow adaptation, stable
        - Large alpha (e.g., 0.1): fast adaptation, responsive
        """
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
    
    def update(self, new_value):
        """Update EMA with new value."""
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * new_value
        return self.value
    
    def get(self):
        """Get current baseline value."""
        return self.value

def train_reinforce_with_baseline(env, policy, optimizer, n_episodes=200, 
                                 gamma=0.99, baseline=None):
    """Train REINFORCE with optional baseline."""
    episode_returns = []
    episode_lengths = []
    advantages_history = []
    baseline_values = []
    
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
        
        # Compute returns (reward-to-go)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        returns_tensor = torch.FloatTensor(returns).to(device)
        
        # Compute advantages
        if baseline is not None:
            # Update baseline with episode return
            episode_return = sum(rewards)
            baseline.update(episode_return)
            baseline_value = baseline.get()
            baseline_values.append(baseline_value)
            
            # Compute advantages: A_t = G_t - b
            # Note: We use episode return as baseline, not per-timestep
            advantages = returns_tensor - baseline_value
        else:
            advantages = returns_tensor
            baseline_values.append(0)
        
        # Store raw advantages for analysis
        advantages_history.extend(advantages.cpu().numpy().tolist())
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute loss
        loss = 0
        for log_prob, advantage in zip(log_probs, advantages):
            loss += -log_prob * advantage
        loss = loss / len(log_probs)
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Record statistics
        episode_return = sum(rewards)
        episode_returns.append(episode_return)
        episode_lengths.append(len(rewards))
        
        if (episode + 1) % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            if baseline is not None:
                print(f"Episode {episode+1:3d} | Avg Return: {avg_return:7.2f} | "
                      f"Baseline: {baseline_value:7.2f}")
            else:
                print(f"Episode {episode+1:3d} | Avg Return: {avg_return:7.2f} | "
                      f"No baseline")
    
    return episode_returns, episode_lengths, advantages_history, baseline_values

def compare_ema_alphas(env, alphas=[0.01, 0.05, 0.1], n_episodes=200):
    """Compare different EMA decay rates."""
    results = {}
    
    for alpha in alphas:
        print(f"\nTraining with EMA baseline (alpha={alpha})...")
        setup_seed(42)
        
        policy = PolicyNetwork(env.observation_space.shape[0], 
                              env.action_space.n).to(device)
        optimizer = optim.Adam(policy.parameters(), lr=1e-2)
        baseline = EMABaseline(alpha=alpha)
        
        returns, lengths, advantages, baselines = train_reinforce_with_baseline(
            env, policy, optimizer, n_episodes=n_episodes, baseline=baseline
        )
        
        results[alpha] = {
            'returns': returns,
            'advantages': advantages,
            'baselines': baselines
        }
    
    # No baseline for comparison
    print(f"\nTraining without baseline...")
    setup_seed(42)
    
    policy = PolicyNetwork(env.observation_space.shape[0], 
                          env.action_space.n).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    
    returns, lengths, advantages, _ = train_reinforce_with_baseline(
        env, policy, optimizer, n_episodes=n_episodes, baseline=None
    )
    
    results['none'] = {
        'returns': returns,
        'advantages': advantages,
        'baselines': [0] * n_episodes
    }
    
    return results

def main():
    print("="*50)
    print("Experiment 05: EMA Baseline")
    print("="*50)
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Compare different alpha values
    print("\n1. Comparing EMA Alpha Values")
    print("-" * 30)
    
    results = compare_ema_alphas(env, alphas=[0.01, 0.05, 0.1], n_episodes=200)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot returns for different alphas
    colors = {'none': 'gray', 0.01: 'blue', 0.05: 'green', 0.1: 'red'}
    
    for key in ['none', 0.01, 0.05, 0.1]:
        label = f'No baseline' if key == 'none' else f'α={key}'
        returns = results[key]['returns']
        
        # Raw returns
        axes[0, 0].plot(returns, alpha=0.3, color=colors[key])
        # Moving average
        if len(returns) >= 20:
            ma = np.convolve(returns, np.ones(20)/20, mode='valid')
            axes[0, 0].plot(range(19, len(returns)), ma, 
                          color=colors[key], linewidth=2, label=label)
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot baselines
    for alpha in [0.01, 0.05, 0.1]:
        baselines = results[alpha]['baselines']
        axes[0, 1].plot(baselines, color=colors[alpha], 
                       linewidth=2, label=f'α={alpha}')
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Baseline Value')
    axes[0, 1].set_title('EMA Baseline Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Variance analysis
    window = 20
    for key in ['none', 0.01, 0.05, 0.1]:
        returns = results[key]['returns']
        variance = [np.var(returns[max(0,i-window):i+1]) 
                   for i in range(len(returns))]
        label = f'No baseline' if key == 'none' else f'α={key}'
        axes[0, 2].plot(variance, color=colors[key], 
                       linewidth=2, label=label)
    
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Variance')
    axes[0, 2].set_title(f'Return Variance ({window}-episode window)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Advantage distributions
    for i, key in enumerate(['none', 0.01, 0.05, 0.1]):
        ax_idx = i if i < 3 else 2
        if i < 3:
            advantages = results[key]['advantages']
            axes[1, ax_idx].hist(advantages, bins=50, alpha=0.7, 
                                color=colors[key], density=True)
            axes[1, ax_idx].set_xlabel('Advantage')
            axes[1, ax_idx].set_ylabel('Density')
            title = 'No baseline' if key == 'none' else f'α={key}'
            axes[1, ax_idx].set_title(f'Advantage Distribution ({title})')
            axes[1, ax_idx].grid(True, alpha=0.3)
            
            # Add mean and std
            mean = np.mean(advantages)
            std = np.std(advantages)
            axes[1, ax_idx].axvline(mean, color='black', linestyle='--', 
                                   label=f'μ={mean:.2f}\nσ={std:.2f}')
            axes[1, ax_idx].legend()
    
    plt.suptitle('EMA Baseline Analysis', fontsize=14)
    plt.tight_layout()
    save_path = FIGURES_DIR / 'ema_baseline_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plots to {save_path}")
    
    # Statistical comparison
    print("\n2. Statistical Analysis")
    print("-" * 30)
    
    print("\nFinal 50 episodes statistics:")
    print("-" * 40)
    print(f"{'Method':<15} {'Mean Return':<12} {'Std Return':<12} {'Convergence':<12}")
    print("-" * 40)
    
    for key in ['none', 0.01, 0.05, 0.1]:
        returns = results[key]['returns']
        mean_return = np.mean(returns[-50:])
        std_return = np.std(returns[-50:])
        
        # Find convergence (first episode reaching 195)
        convergence = next((i for i, r in enumerate(returns) if r >= 195), 200)
        
        method = 'No baseline' if key == 'none' else f'EMA α={key}'
        print(f"{method:<15} {mean_return:<12.2f} {std_return:<12.2f} {convergence:<12d}")
    
    # Mathematical explanation
    print("\n3. Why Baselines Work")
    print("-" * 30)
    print("The policy gradient with baseline:")
    print("  grad = E[log pi(a|s) * (G - b(s))]")
    print("")
    print("Key property: E[log pi(a|s) * b(s)] = b(s) * E[log pi(a|s)] = 0")
    print("(The expected score function is zero)")
    print("")
    print("Therefore: grad = E[log pi(a|s) * G] - E[log pi(a|s) * b(s)]")
    print("                = E[log pi(a|s) * G] - 0")
    print("                = original gradient (unbiased!)")
    print("")
    print("But variance is reduced because:")
    print("  Var(G - b) < Var(G) when b ≈ E[G]")
    
    # EMA properties
    print("\n4. EMA Properties")
    print("-" * 30)
    print("Small α (0.01): Slow adaptation, stable but may lag")
    print("Medium α (0.05): Balanced adaptation and stability")
    print("Large α (0.1): Fast adaptation, responsive but noisy")
    print("")
    print("Update rule: b_new = (1-α)*b_old + α*return")
    print("Effective window: ~1/α episodes")
    
    print("\n" + "="*50)
    print("EMA baseline demonstration completed!")
    print("Key insight: Simple baselines provide significant benefits")
    print("with minimal computational overhead.")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
