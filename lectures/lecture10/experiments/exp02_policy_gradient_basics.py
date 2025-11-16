#!/usr/bin/env python3
"""
RL2025 - Lecture 10: Experiment 02 - Policy Gradient Basics

This experiment introduces the basic concepts of policy gradient methods,
demonstrating the REINFORCE algorithm and likelihood ratio gradients
before introducing PPO improvements.

Learning objectives:
- Understand policy gradient theorem fundamentals
- Implement basic REINFORCE with baseline
- Observe high variance in vanilla policy gradients
- Motivate the need for trust region methods

Prerequisites: exp01_setup.py completed successfully
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

import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

class PolicyNetwork(nn.Module):
    """Simple policy network for discrete action spaces."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_action_and_logprob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability for given observation."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

class ValueNetwork(nn.Module):
    """Simple value network for baseline."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

def collect_trajectory(env, policy: PolicyNetwork, max_steps: int = 500) -> Tuple[List, List, List, float]:
    """Collect a single trajectory using the current policy."""
    observations, actions, log_probs, rewards = [], [], [], []
    
    obs, _ = env.reset()
    total_reward = 0.0
    
    for _ in range(max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = policy.get_action_and_logprob(obs_tensor)
        
        observations.append(obs)
        actions.append(action.item())
        log_probs.append(log_prob.item())
        
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return observations, actions, log_probs, rewards

def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns from rewards."""
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

def reinforce_update(policy: PolicyNetwork, value_net: ValueNetwork, 
                    trajectories: List[Tuple], policy_lr: float = 1e-3, 
                    value_lr: float = 5e-3, use_baseline: bool = True) -> Tuple[float, float]:
    """Perform REINFORCE update with optional baseline."""
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr) if use_baseline else None
    
    policy_loss = 0.0
    value_loss = 0.0
    
    for observations, actions, log_probs, rewards in trajectories:
        # Compute returns
        returns = compute_returns(rewards)
        
        # Convert to tensors
        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Compute current policy log probabilities
        logits = policy(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        current_log_probs = dist.log_prob(actions_tensor)
        
        if use_baseline:
            # Compute baseline values
            values = value_net(obs_tensor)
            advantages = returns_tensor - values
            
            # Update value network
            value_loss_batch = F.mse_loss(values, returns_tensor)
            value_optimizer.zero_grad()
            value_loss_batch.backward()
            value_optimizer.step()
            value_loss += value_loss_batch.item()
        else:
            advantages = returns_tensor
        
        # Policy gradient update
        policy_loss_batch = -(current_log_probs * advantages.detach()).mean()
        policy_optimizer.zero_grad()
        policy_loss_batch.backward()
        policy_optimizer.step()
        policy_loss += policy_loss_batch.item()
    
    return policy_loss / len(trajectories), value_loss / len(trajectories) if use_baseline else 0.0

def demonstrate_vanilla_reinforce():
    """Demonstrate vanilla REINFORCE algorithm."""
    print("\nDemonstrating Vanilla REINFORCE (without baseline)...")
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)  # Not used in vanilla REINFORCE
    
    writer = SummaryWriter(f"runs/reinforce_vanilla_{int(time.time())}")
    
    episode_rewards = []
    policy_losses = []
    
    for episode in range(100):
        # Collect trajectory
        obs_list, actions_list, log_probs_list, rewards_list = collect_trajectory(env, policy)
        episode_reward = sum(rewards_list)
        episode_rewards.append(episode_reward)
        
        # Update policy (collect multiple trajectories for stable gradients)
        trajectories = []
        for _ in range(4):  # Collect 4 trajectories per update
            traj = collect_trajectory(env, policy)
            trajectories.append(traj)
        
        policy_loss, _ = reinforce_update(policy, value_net, trajectories, use_baseline=False)
        policy_losses.append(policy_loss)
        
        # Logging
        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/policy_loss", policy_loss, episode)
        
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Policy Loss: {policy_loss:.4f}")
    
    env.close()
    writer.close()
    
    return episode_rewards, policy_losses

def demonstrate_reinforce_with_baseline():
    """Demonstrate REINFORCE with value function baseline."""
    print("\nDemonstrating REINFORCE with Baseline...")
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    
    writer = SummaryWriter(f"runs/reinforce_baseline_{int(time.time())}")
    
    episode_rewards = []
    policy_losses = []
    value_losses = []
    
    for episode in range(100):
        # Collect trajectory
        obs_list, actions_list, log_probs_list, rewards_list = collect_trajectory(env, policy)
        episode_reward = sum(rewards_list)
        episode_rewards.append(episode_reward)
        
        # Update policy with baseline
        trajectories = []
        for _ in range(4):  # Collect 4 trajectories per update
            traj = collect_trajectory(env, policy)
            trajectories.append(traj)
        
        policy_loss, value_loss = reinforce_update(policy, value_net, trajectories, use_baseline=True)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        
        # Logging
        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/policy_loss", policy_loss, episode)
        writer.add_scalar("train/value_loss", value_loss, episode)
        
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
    
    env.close()
    writer.close()
    
    return episode_rewards, policy_losses, value_losses

def analyze_gradient_variance():
    """Analyze the variance in policy gradients."""
    print("\nAnalyzing Policy Gradient Variance...")
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    
    # Collect multiple trajectories and compute gradient estimates
    gradient_norms = []
    returns_variance = []
    
    for trial in range(20):
        trajectories = []
        all_returns = []
        
        for _ in range(10):  # 10 trajectories per trial
            obs_list, actions_list, log_probs_list, rewards_list = collect_trajectory(env, policy)
            returns = compute_returns(rewards_list)
            all_returns.extend(returns)
            trajectories.append((obs_list, actions_list, log_probs_list, rewards_list))
        
        # Compute gradient norm
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        total_loss = 0.0
        
        for observations, actions, log_probs, rewards in trajectories:
            returns = compute_returns(rewards)
            
            obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            
            logits = policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            current_log_probs = dist.log_prob(actions_tensor)
            
            loss = -(current_log_probs * returns_tensor).mean()
            total_loss += loss
        
        policy_optimizer.zero_grad()
        total_loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for param in policy.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        gradient_norms.append(grad_norm)
        returns_variance.append(np.var(all_returns))
    
    env.close()
    
    print(f"Average gradient norm: {np.mean(gradient_norms):.4f} ± {np.std(gradient_norms):.4f}")
    print(f"Average returns variance: {np.mean(returns_variance):.4f} ± {np.std(returns_variance):.4f}")
    
    return gradient_norms, returns_variance

def compare_methods():
    """Compare vanilla REINFORCE vs REINFORCE with baseline."""
    print("\n" + "="*60)
    print("Comparing REINFORCE Methods")
    print("="*60)
    
    # Run vanilla REINFORCE
    vanilla_rewards, vanilla_losses = demonstrate_vanilla_reinforce()
    
    # Run REINFORCE with baseline
    baseline_rewards, baseline_policy_losses, baseline_value_losses = demonstrate_reinforce_with_baseline()
    
    # Analyze variance
    grad_norms, returns_var = analyze_gradient_variance()
    
    # Print comparison results
    print(f"\nVanilla REINFORCE - Final avg reward: {np.mean(vanilla_rewards[-10:]):.2f}")
    print(f"REINFORCE w/ Baseline - Final avg reward: {np.mean(baseline_rewards[-10:]):.2f}")
    print(f"Gradient variance analysis - Avg norm: {np.mean(grad_norms):.4f}")
    
    return {
        'vanilla_rewards': vanilla_rewards,
        'baseline_rewards': baseline_rewards,
        'gradient_norms': grad_norms,
        'returns_variance': returns_var
    }

def main():
    print("="*60)
    print("Policy Gradient Basics - REINFORCE Implementation")
    print("="*60)
    
    # Run comparison of methods
    results = compare_methods()
    
    print("\nKey Observations:")
    print("1. High variance in vanilla policy gradients")
    print("2. Baseline reduces variance but doesn't solve all issues")
    print("3. Need for more sophisticated methods (PPO!)")
    print("\nNext: exp03_trust_region_concepts.py")

if __name__ == "__main__":
    main()