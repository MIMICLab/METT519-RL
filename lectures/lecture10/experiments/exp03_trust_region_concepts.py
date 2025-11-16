#!/usr/bin/env python3
"""
RL2025 - Lecture 10: Experiment 03 - Trust Region Concepts

This experiment demonstrates the core concepts behind trust region methods,
showing why we need to constrain policy updates and how KL divergence
serves as a natural distance metric between policies.

Learning objectives:
- Understand policy collapse from large updates
- Implement KL divergence calculation between policies  
- Visualize trust region constraints
- Motivate PPO's clipped surrogate objective

Prerequisites: exp02_policy_gradient_basics.py completed successfully
"""

import time
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import FIGURES_DIR, get_device, set_seed

set_seed(42)
device = get_device()

class PolicyNetwork(nn.Module):
    """Policy network with ability to track parameter changes."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (not log probabilities)."""
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)
    
    def get_log_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Get log action probabilities."""
        logits = self.forward(obs)
        return F.log_softmax(logits, dim=-1)
    
    def get_action_and_logprob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return log probability."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def clone_params(self):
        """Return a copy of current parameters."""
        return {name: param.clone().detach() for name, param in self.named_parameters()}
    
    def load_params(self, params_dict):
        """Load parameters from dictionary."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(params_dict[name])

def compute_kl_divergence(policy_old: PolicyNetwork, policy_new: PolicyNetwork, 
                         observations: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between old and new policies."""
    with torch.no_grad():
        old_probs = policy_old.get_action_probs(observations)
        old_log_probs = torch.log(old_probs + 1e-8)
    
    new_log_probs = policy_new.get_log_probs(observations)
    new_probs = torch.exp(new_log_probs)
    
    # KL(old || new) = sum(old_probs * (old_log_probs - new_log_probs))
    kl_div = (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1).mean()
    return kl_div

def collect_batch_data(env, policy: PolicyNetwork, batch_size: int = 1000):
    """Collect a batch of transitions for analysis."""
    observations = []
    actions = []
    rewards = []
    log_probs = []
    
    while len(observations) < batch_size:
        obs, _ = env.reset()
        done = False
        
        while not done and len(observations) < batch_size:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, log_prob = policy.get_action_and_logprob(obs_tensor)
            
            observations.append(obs)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
    
    return (torch.tensor(observations[:batch_size], dtype=torch.float32, device=device),
            torch.tensor(actions[:batch_size], dtype=torch.long, device=device),
            torch.tensor(rewards[:batch_size], dtype=torch.float32, device=device),
            torch.tensor(log_probs[:batch_size], dtype=torch.float32, device=device))

def demonstrate_policy_collapse():
    """Show what happens with large policy updates."""
    print("\nDemonstrating Policy Collapse with Large Updates...")
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Create policy and collect initial data
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    observations, actions, rewards, old_log_probs = collect_batch_data(env, policy)
    
    # Evaluate initial performance
    initial_perf = evaluate_policy_performance(env, policy)
    print(f"Initial policy performance: {initial_perf:.2f}")
    
    # Store initial parameters
    initial_params = policy.clone_params()
    
    # Try different learning rates (small to large)
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    results = []
    
    for lr in learning_rates:
        # Reset to initial parameters
        policy.load_params(initial_params)
        
        # Perform one large update
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # Compute simple policy gradient loss
        logits = policy(observations)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        
        # Simple advantage (centered rewards)
        advantages = rewards - rewards.mean()
        loss = -(new_log_probs * advantages).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate performance after update
        final_perf = evaluate_policy_performance(env, policy)
        
        # Compute KL divergence
        policy_old = PolicyNetwork(obs_dim, act_dim).to(device)
        policy_old.load_params(initial_params)
        kl_div = compute_kl_divergence(policy_old, policy, observations).item()
        
        results.append((lr, initial_perf, final_perf, kl_div))
        print(f"LR: {lr:.1e}, Performance: {initial_perf:.2f} → {final_perf:.2f}, KL: {kl_div:.4f}")
    
    env.close()
    return results

def evaluate_policy_performance(env, policy: PolicyNetwork, episodes: int = 10) -> float:
    """Evaluate policy performance over multiple episodes."""
    total_rewards = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _ = policy.get_action_and_logprob(obs_tensor)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)

def visualize_trust_region_constraints():
    """Visualize the effect of trust region constraints."""
    print("\nVisualizing Trust Region Constraints...")
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    observations, actions, rewards, old_log_probs = collect_batch_data(env, policy, 500)
    
    # Define different constraint levels
    target_kls = [0.01, 0.05, 0.1, 0.2]
    colors = ['blue', 'green', 'orange', 'red']
    
    results_by_kl = {}
    
    for target_kl in target_kls:
        print(f"\nTesting target KL: {target_kl}")
        
        # Reset policy
        initial_params = policy.clone_params()
        policy.load_params(initial_params)
        
        # Perform constrained updates
        performance_history = []
        kl_history = []
        
        for step in range(20):
            # Compute gradients
            optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
            
            logits = policy(observations)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            
            advantages = rewards - rewards.mean()
            loss = -(new_log_probs * advantages).mean()
            
            optimizer.zero_grad()
            loss.backward()
            
            # Check KL before update
            policy_old = PolicyNetwork(obs_dim, act_dim).to(device)
            policy_old.load_params(initial_params)
            
            # Try the update
            with torch.no_grad():
                # Store current params
                current_params = policy.clone_params()
                
                # Take step
                optimizer.step()
                
                # Check resulting KL
                kl_div = compute_kl_divergence(policy_old, policy, observations).item()
                
                # If KL too large, scale back
                if kl_div > target_kl:
                    # Restore and take smaller step
                    policy.load_params(current_params)
                    for param in policy.parameters():
                        if param.grad is not None:
                            param.data -= 0.001 * param.grad.data  # Very small step
                    kl_div = compute_kl_divergence(policy_old, policy, observations).item()
            
            # Evaluate performance
            perf = evaluate_policy_performance(env, policy, episodes=3)
            performance_history.append(perf)
            kl_history.append(kl_div)
        
        results_by_kl[target_kl] = {
            'performance': performance_history,
            'kl': kl_history
        }
    
    env.close()
    
    # Print summary
    print("\nTrust Region Constraint Analysis:")
    for target_kl in target_kls:
        final_perf = results_by_kl[target_kl]['performance'][-1]
        avg_kl = np.mean(results_by_kl[target_kl]['kl'])
        print(f"Target KL {target_kl}: Final Performance = {final_perf:.2f}, Avg KL = {avg_kl:.4f}")
    
    return results_by_kl

def demonstrate_importance_sampling():
    """Demonstrate importance sampling ratios in policy updates."""
    print("\nDemonstrating Importance Sampling Ratios...")
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    observations, actions, rewards, old_log_probs = collect_batch_data(env, policy)
    
    # Store old policy
    old_params = policy.clone_params()
    
    # Make some updates to create new policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    for _ in range(10):
        logits = policy(observations)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        
        advantages = rewards - rewards.mean()
        loss = -(new_log_probs * advantages).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Compute importance sampling ratios
    with torch.no_grad():
        # New policy probabilities
        new_logits = policy(observations)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        new_log_probs = new_dist.log_prob(actions)
        
        # Old policy probabilities
        old_policy = PolicyNetwork(obs_dim, act_dim).to(device)
        old_policy.load_params(old_params)
        old_logits = old_policy(observations)
        old_dist = torch.distributions.Categorical(logits=old_logits)
        old_log_probs_new = old_dist.log_prob(actions)
        
        # Importance sampling ratios
        ratios = torch.exp(new_log_probs - old_log_probs_new)
        
        print(f"Importance sampling ratio statistics:")
        print(f"  Mean: {ratios.mean():.4f}")
        print(f"  Std:  {ratios.std():.4f}")
        print(f"  Min:  {ratios.min():.4f}")
        print(f"  Max:  {ratios.max():.4f}")
        print(f"  Ratios > 2.0: {(ratios > 2.0).float().mean():.2%}")
        print(f"  Ratios < 0.5: {(ratios < 0.5).float().mean():.2%}")
    
    env.close()
    return ratios.cpu().numpy()

def visualize_clipping_concept():
    """Visualize how clipping affects the surrogate objective."""
    print("\nVisualizing Clipping Concept...")
    
    # Generate synthetic ratios and advantages
    ratios = np.linspace(0.5, 2.0, 100)
    advantages = [1.0, -1.0]  # Positive and negative advantage
    clip_epsilon = 0.2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, adv in enumerate(advantages):
        ax = ax1 if i == 0 else ax2
        
        # Original surrogate
        original_surrogate = ratios * adv
        
        # Clipped surrogate
        clipped_ratios = np.clip(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
        clipped_surrogate = clipped_ratios * adv
        
        # PPO surrogate (minimum of original and clipped)
        if adv > 0:
            ppo_surrogate = np.minimum(original_surrogate, clipped_surrogate)
        else:
            ppo_surrogate = np.maximum(original_surrogate, clipped_surrogate)
        
        ax.plot(ratios, original_surrogate, 'b-', label='Original Surrogate', linewidth=2)
        ax.plot(ratios, clipped_surrogate, 'r--', label='Clipped Surrogate', linewidth=2)
        ax.plot(ratios, ppo_surrogate, 'g-', label='PPO Surrogate', linewidth=3)
        
        ax.axvline(1 - clip_epsilon, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(1 + clip_epsilon, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(1.0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Importance Ratio')
        ax.set_ylabel('Surrogate Value')
        ax.set_title(f'Advantage = {adv}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / 'trust_region_clipping.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved clipping visualization to {fig_path}")
    plt.close()

def main():
    print("="*60)
    print("Trust Region Concepts for Policy Optimization")
    print("="*60)
    
    # Demonstrate policy collapse
    collapse_results = demonstrate_policy_collapse()
    
    # Visualize trust region constraints
    trust_region_results = visualize_trust_region_constraints()
    
    # Demonstrate importance sampling
    ratios = demonstrate_importance_sampling()
    
    # Visualize clipping concept
    visualize_clipping_concept()
    
    print("\n" + "="*60)
    print("Key Insights:")
    print("1. Large policy updates can cause performance collapse")
    print("2. KL divergence serves as a natural distance metric")
    print("3. Trust region constraints stabilize learning")
    print("4. Importance sampling ratios can become extreme")
    print("5. Clipping provides a simple approximation to trust regions")
    print("="*60)
    
    print("\nNext: exp04_ppo_implementation.py")

if __name__ == "__main__":
    main()
