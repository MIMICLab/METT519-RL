#!/usr/bin/env python3
"""
RL2025 - Lecture 8: Experiment 07 - Entropy Regularization

This experiment demonstrates entropy regularization to encourage
exploration and prevent premature convergence to deterministic policies.

Learning objectives:
- Implement entropy computation for categorical distributions
- Add entropy bonus to policy gradient
- Compare different entropy coefficients
- Understand exploration vs exploitation trade-off

Prerequisites: Completed exp06
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

def compute_entropy(logits):
    """Compute entropy of categorical distribution."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

def train_with_entropy(env, policy, value_net, policy_optimizer, value_optimizer,
                       n_episodes=200, gamma=0.99, entropy_coef=0.01):
    """Train REINFORCE with entropy regularization."""
    episode_returns = []
    episode_lengths = []
    entropy_values = []
    action_probs_history = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []
        entropies = []
        values = []
        
        done = False
        while not done:
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action and entropy
            logits = policy(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            # Track action probabilities
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                action_probs_history.append(probs.cpu().numpy().flatten())
            
            # Get value estimate
            value = value_net(state)
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            
            obs = next_obs
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(device)
        values = torch.cat(values).squeeze()
        
        # Compute advantages
        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss with entropy bonus
        policy_loss = 0
        entropy_bonus = 0
        for log_prob, advantage, entropy in zip(log_probs, advantages, entropies):
            policy_loss += -log_prob * advantage
            entropy_bonus += entropy
        
        policy_loss = policy_loss / len(log_probs)
        entropy_bonus = entropy_bonus / len(entropies)
        
        # Total loss (negative because we want to maximize entropy)
        total_policy_loss = policy_loss - entropy_coef * entropy_bonus
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns)
        
        # Update policy
        policy_optimizer.zero_grad()
        total_policy_loss.backward()
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
        entropy_values.append(entropy_bonus.item())
        
        if (episode + 1) % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            avg_entropy = np.mean(entropy_values[-20:])
            print(f"Episode {episode+1:3d} | Return: {avg_return:7.2f} | "
                  f"Entropy: {avg_entropy:.4f}")
    
    return episode_returns, entropy_values, action_probs_history

def compare_entropy_coefficients(env, coefficients=[0.0, 0.001, 0.01, 0.1]):
    """Compare different entropy regularization strengths."""
    results = {}
    
    for coef in coefficients:
        print(f"\nTraining with entropy coefficient: {coef}")
        print("-" * 40)
        
        setup_seed(42)
        
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        
        policy = PolicyNetwork(obs_dim, n_actions).to(device)
        value_net = ValueNetwork(obs_dim).to(device)
        
        policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
        value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
        
        returns, entropies, probs = train_with_entropy(
            env, policy, value_net, policy_optimizer, value_optimizer,
            n_episodes=200, entropy_coef=coef
        )
        
        results[coef] = {
            'returns': returns,
            'entropies': entropies,
            'probs': probs,
            'policy': policy
        }
    
    return results

def main():
    print("="*50)
    print("Experiment 07: Entropy Regularization")
    print("="*50)
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    print("\n1. Entropy Basics")
    print("-" * 30)
    print("Entropy H(π) = -Σ π(a|s) log π(a|s)")
    print("Maximum entropy: uniform distribution")
    print("Minimum entropy: deterministic policy")
    print("For 2 actions: H_max = log(2) ≈ 0.693")
    
    # Compare different entropy coefficients
    print("\n2. Comparing Entropy Coefficients")
    print("-" * 30)
    
    coefficients = [0.0, 0.001, 0.01, 0.1]
    results = compare_entropy_coefficients(env, coefficients)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Returns for different coefficients
    colors = {0.0: 'red', 0.001: 'orange', 0.01: 'green', 0.1: 'blue'}
    
    for coef in coefficients:
        returns = results[coef]['returns']
        label = f'β={coef}'
        
        axes[0, 0].plot(returns, alpha=0.3, color=colors[coef])
        if len(returns) >= 20:
            ma = np.convolve(returns, np.ones(20)/20, mode='valid')
            axes[0, 0].plot(range(19, len(returns)), ma,
                          color=colors[coef], linewidth=2, label=label)
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Entropy evolution
    for coef in coefficients:
        entropies = results[coef]['entropies']
        axes[0, 1].plot(entropies, color=colors[coef], 
                       linewidth=2, label=f'β={coef}')
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Policy Entropy')
    axes[0, 1].set_title('Entropy Evolution')
    axes[0, 1].axhline(y=np.log(2), color='black', linestyle='--', 
                       alpha=0.3, label='Max entropy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final policy action probabilities
    for i, coef in enumerate(coefficients[:2]):
        policy = results[coef]['policy']
        
        # Sample states to visualize policy
        test_states = torch.randn(100, env.observation_space.shape[0]).to(device)
        
        with torch.no_grad():
            logits = policy(test_states)
            probs = torch.softmax(logits, dim=-1)
        
        axes[0, 2].hist(probs[:, 0].cpu().numpy(), bins=20, alpha=0.5,
                       label=f'β={coef}', color=colors[coef])
    
    axes[0, 2].set_xlabel('P(action=0)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Final Policy Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Convergence speed
    target = 195
    convergence_episodes = []
    for coef in coefficients:
        returns = results[coef]['returns']
        conv = next((i for i, r in enumerate(returns) if r >= target), 200)
        convergence_episodes.append(conv)
    
    axes[1, 0].bar(range(len(coefficients)), convergence_episodes,
                  color=[colors[c] for c in coefficients])
    axes[1, 0].set_xticks(range(len(coefficients)))
    axes[1, 0].set_xticklabels([f'β={c}' for c in coefficients])
    axes[1, 0].set_ylabel('Episodes to reach 195')
    axes[1, 0].set_title('Convergence Speed')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Action probability evolution (for coef=0.01)
    probs_history = results[0.01]['probs']
    if len(probs_history) > 1000:
        # Sample every 10th step for visualization
        sampled_probs = probs_history[::10]
        sampled_probs = np.array(sampled_probs)
        
        axes[1, 1].plot(sampled_probs[:, 0], alpha=0.7, label='P(left)')
        axes[1, 1].plot(sampled_probs[:, 1], alpha=0.7, label='P(right)')
        axes[1, 1].set_xlabel('Steps (sampled)')
        axes[1, 1].set_ylabel('Action Probability')
        axes[1, 1].set_title('Action Probabilities Over Time (β=0.01)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Final performance comparison
    final_returns = []
    final_stds = []
    for coef in coefficients:
        returns = results[coef]['returns'][-50:]
        final_returns.append(np.mean(returns))
        final_stds.append(np.std(returns))
    
    axes[1, 2].bar(range(len(coefficients)), final_returns,
                  yerr=final_stds, capsize=5,
                  color=[colors[c] for c in coefficients])
    axes[1, 2].set_xticks(range(len(coefficients)))
    axes[1, 2].set_xticklabels([f'β={c}' for c in coefficients])
    axes[1, 2].set_ylabel('Mean Return')
    axes[1, 2].set_title('Final Performance (last 50 episodes)')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Entropy Regularization Analysis', fontsize=14)
    plt.tight_layout()
    save_path = FIGURES_DIR / 'entropy_regularization_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plots to {save_path}")
    
    # Statistical analysis
    print("\n3. Statistical Analysis")
    print("-" * 50)
    print(f"{'Coefficient':<12} {'Final Return':<15} {'Final Entropy':<15} {'Convergence':<12}")
    print("-" * 50)
    
    for coef in coefficients:
        returns = results[coef]['returns']
        entropies = results[coef]['entropies']
        
        final_return = np.mean(returns[-50:])
        final_entropy = np.mean(entropies[-50:])
        conv = next((i for i, r in enumerate(returns) if r >= 195), 200)
        
        print(f"β={coef:<9.3f} {final_return:<15.2f} {final_entropy:<15.4f} {conv:<12d}")
    
    # Key insights
    print("\n4. Key Insights")
    print("-" * 30)
    print("• β=0.0: No exploration bonus, can get stuck")
    print("• β=0.001: Slight exploration, good balance")
    print("• β=0.01: Moderate exploration, slower but stable")
    print("• β=0.1: Too much exploration, hinders convergence")
    print("")
    print("Trade-off: Exploration (high β) vs Exploitation (low β)")
    print("Optimal β depends on task complexity and reward sparsity")
    
    print("\n5. Mathematical Interpretation")
    print("-" * 30)
    print("Modified objective: J(θ) = E[R] + β·H(π)")
    print("")
    print("Gradient: ∇J = E[∇log π·(G-b)] + β·∇H(π)")
    print("")
    print("Entropy gradient pushes toward uniform distribution")
    print("Prevents premature convergence to suboptimal deterministic policies")
    
    print("\n" + "="*50)
    print("Entropy regularization demonstration completed!")
    print("Balance exploration and exploitation for optimal learning.")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
