#!/usr/bin/env python3
"""
RL2025 - Lecture 10: Experiment 05 - Generalized Advantage Estimation Analysis

This experiment provides a deep dive into GAE, showing how different lambda
values affect bias-variance tradeoffs and comparing GAE with other advantage
estimation methods.

Learning objectives:
- Understand GAE parameter sensitivity
- Compare different advantage estimation methods
- Visualize bias-variance tradeoffs
- Analyze the effect of lambda on learning stability

Prerequisites: exp04_ppo_implementation.py completed successfully
"""

# PyTorch 2.x Standard Practice Header
from pathlib import Path

import numpy as np
import torch

from helpers import FIGURES_DIR, get_device, set_seed

device = get_device()
set_seed(42)

import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, List
import time
from dataclasses import dataclass

# Import from previous experiment
from exp04_ppo_implementation import ActorCritic, PPOConfig, PPOTrainer, make_env

class AdvantageAnalyzer:
    """Class for analyzing different advantage estimation methods."""
    
    def __init__(self, env_id: str = "CartPole-v1", device: torch.device = device):
        self.env_id = env_id
        self.device = device
        
        # Create single environment for analysis
        self.env = gym.make(env_id)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        
        # Get environment dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        
        # Create actor-critic for analysis
        self.agent = ActorCritic(self.obs_dim, self.act_dim).to(device)
    
    def collect_trajectories(self, num_trajectories: int = 10) -> List[Dict]:
        """Collect multiple trajectories for analysis."""
        trajectories = []
        
        for _ in range(num_trajectories):
            obs_list = []
            action_list = []
            reward_list = []
            value_list = []
            done_list = []
            
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    action, logprob, value = self.agent.get_action_and_value(obs_tensor)
                    value = value.item()
                
                obs_list.append(obs)
                action_list.append(action.item())
                value_list.append(value)
                
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                reward_list.append(reward)
                done_list.append(terminated or truncated)
                done = terminated or truncated
            
            trajectories.append({
                'observations': np.array(obs_list),
                'actions': np.array(action_list),
                'rewards': np.array(reward_list),
                'values': np.array(value_list),
                'dones': np.array(done_list)
            })
        
        return trajectories
    
    def compute_monte_carlo_advantages(self, trajectory: Dict, gamma: float = 0.99) -> np.ndarray:
        """Compute Monte Carlo (lambda=1) advantage estimates."""
        rewards = trajectory['rewards']
        values = trajectory['values']
        
        # Compute Monte Carlo returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        advantages = returns - values
        
        return advantages, returns
    
    def compute_td_advantages(self, trajectory: Dict, gamma: float = 0.99) -> np.ndarray:
        """Compute TD (lambda=0) advantage estimates."""
        rewards = trajectory['rewards']
        values = trajectory['values']
        dones = trajectory['dones']
        
        advantages = []
        returns = []
        
        for t in range(len(rewards)):
            if t == len(rewards) - 1 or dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            td_target = rewards[t] + gamma * next_value
            advantage = td_target - values[t]
            
            advantages.append(advantage)
            returns.append(td_target)
        
        return np.array(advantages), np.array(returns)
    
    def compute_gae_advantages(self, trajectory: Dict, gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantage estimates."""
        rewards = trajectory['rewards']
        values = trajectory['values']
        dones = trajectory['dones']
        
        advantages = []
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1 or dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae_lam = delta + gamma * lam * (1 - dones[t]) * last_gae_lam
            advantages.insert(0, last_gae_lam)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def compare_advantage_methods(self, num_trajectories: int = 20) -> Dict:
        """Compare different advantage estimation methods."""
        print("Comparing advantage estimation methods...")
        
        trajectories = self.collect_trajectories(num_trajectories)
        
        results = {
            'monte_carlo': {'advantages': [], 'returns': [], 'variance': []},
            'td': {'advantages': [], 'returns': [], 'variance': []},
            'gae_095': {'advantages': [], 'returns': [], 'variance': []},
            'gae_090': {'advantages': [], 'returns': [], 'variance': []},
            'gae_099': {'advantages': [], 'returns': [], 'variance': []}
        }
        
        for traj in trajectories:
            # Monte Carlo
            mc_adv, mc_ret = self.compute_monte_carlo_advantages(traj)
            results['monte_carlo']['advantages'].extend(mc_adv)
            results['monte_carlo']['returns'].extend(mc_ret)
            results['monte_carlo']['variance'].append(np.var(mc_adv))
            
            # TD
            td_adv, td_ret = self.compute_td_advantages(traj)
            results['td']['advantages'].extend(td_adv)
            results['td']['returns'].extend(td_ret)
            results['td']['variance'].append(np.var(td_adv))
            
            # GAE with different lambdas
            for lam, key in [(0.95, 'gae_095'), (0.90, 'gae_090'), (0.99, 'gae_099')]:
                gae_adv, gae_ret = self.compute_gae_advantages(traj, lam=lam)
                results[key]['advantages'].extend(gae_adv)
                results[key]['returns'].extend(gae_ret)
                results[key]['variance'].append(np.var(gae_adv))
        
        # Compute statistics
        for method in results:
            results[method]['mean_advantage'] = np.mean(results[method]['advantages'])
            results[method]['std_advantage'] = np.std(results[method]['advantages'])
            results[method]['mean_variance'] = np.mean(results[method]['variance'])
        
        return results
    
    def visualize_advantage_comparison(self, results: Dict):
        """Visualize advantage estimation comparison."""
        methods = list(results.keys())
        mean_advantages = [results[m]['mean_advantage'] for m in methods]
        std_advantages = [results[m]['std_advantage'] for m in methods]
        mean_variances = [results[m]['mean_variance'] for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot mean and std of advantages
        x = np.arange(len(methods))
        ax1.bar(x, mean_advantages, yerr=std_advantages, alpha=0.7, capsize=5)
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Average Advantage')
        ax1.set_title('Advantage Mean and Standard Deviation')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot variance across trajectories
        ax2.bar(x, mean_variances, alpha=0.7)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Mean Variance per Trajectory')
        ax2.set_title('Advantage Variance per Trajectory')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = FIGURES_DIR / 'advantage_comparison.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved advantage comparison to '{out_path}'")
        plt.close()

def run_gae_lambda_sweep():
    """Run PPO with different GAE lambda values."""
    print("Running GAE lambda parameter sweep...")
    
    lambda_values = [0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    results = {}
    
    base_config = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=50_000,  # Shorter for sweep
        num_envs=4,
        num_steps=128,
        seed=42
    )
    
    for lam in lambda_values:
        print(f"\nTesting GAE lambda: {lam}")
        
        config = base_config
        config.gae_lambda = lam
        config.seed = 42  # Keep seed consistent
        
        # Modify run name for logging
        original_writer_creation = PPOTrainer.__init__
        
        def custom_init(self, config):
            original_writer_creation(self, config)
            self.writer = SummaryWriter(f"runs/ppo_gae_lambda_{lam}_{int(time.time())}")
        
        PPOTrainer.__init__ = custom_init
        
        trainer = PPOTrainer(config)
        trainer.train()
        
        results[lam] = trainer
        
        # Restore original method
        PPOTrainer.__init__ = original_writer_creation
    
    return results

def analyze_gae_bias_variance():
    """Analyze bias-variance tradeoff in GAE."""
    print("Analyzing GAE bias-variance tradeoff...")
    
    analyzer = AdvantageAnalyzer()
    
    # Compare different methods
    comparison_results = analyzer.compare_advantage_methods(num_trajectories=50)
    
    # Print results
    print("\nAdvantage Estimation Comparison:")
    print("-" * 50)
    for method in comparison_results:
        mean_adv = comparison_results[method]['mean_advantage']
        std_adv = comparison_results[method]['std_advantage']
        mean_var = comparison_results[method]['mean_variance']
        
        print(f"{method:12}: Mean={mean_adv:6.3f}, Std={std_adv:6.3f}, Var={mean_var:6.3f}")
    
    # Visualize results
    analyzer.visualize_advantage_comparison(comparison_results)
    
    return comparison_results

def demonstrate_gae_computation():
    """Step-by-step demonstration of GAE computation."""
    print("Demonstrating GAE computation step-by-step...")
    
    # Create synthetic trajectory data
    rewards = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    values = np.array([0.5, 0.3, 0.7, 0.2, 0.1, 0.8])
    gamma = 0.9
    lam = 0.95
    
    print(f"Synthetic trajectory:")
    print(f"Rewards: {rewards}")
    print(f"Values:  {values}")
    print(f"Gamma: {gamma}, Lambda: {lam}")
    
    # Compute GAE step by step
    print("\nGAE Computation:")
    print("-" * 40)
    
    advantages = []
    last_gae_lam = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # Terminal state
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae_lam = delta + gamma * lam * last_gae_lam
        
        print(f"Step {t}: delta={delta:.3f}, GAE={last_gae_lam:.3f}")
        advantages.insert(0, last_gae_lam)
    
    advantages = np.array(advantages)
    returns = advantages + values
    
    print(f"\nFinal Results:")
    print(f"Advantages: {advantages}")
    print(f"Returns:    {returns}")
    
    # Compare with Monte Carlo
    mc_returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        mc_returns.insert(0, G)
    
    mc_advantages = np.array(mc_returns) - values
    
    print(f"\nComparison with Monte Carlo:")
    print(f"MC Returns:     {mc_returns}")
    print(f"MC Advantages:  {mc_advantages}")
    print(f"Difference:     {advantages - mc_advantages}")

def plot_lambda_sensitivity():
    """Plot how advantages change with different lambda values."""
    print("Plotting lambda sensitivity...")
    
    # Create synthetic trajectory
    rewards = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    values = np.array([0.5, 0.3, 0.7, 0.2, 0.1, 0.8])
    gamma = 0.9
    
    lambda_values = np.linspace(0.0, 1.0, 21)
    advantage_profiles = []
    
    for lam in lambda_values:
        advantages = []
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae_lam = delta + gamma * lam * last_gae_lam
            advantages.insert(0, last_gae_lam)
        
        advantage_profiles.append(advantages)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    advantage_profiles = np.array(advantage_profiles)
    for t in range(len(rewards)):
        ax.plot(lambda_values, advantage_profiles[:, t], 'o-', label=f'Step {t}', alpha=0.7)
    
    ax.set_xlabel('GAE Lambda')
    ax.set_ylabel('Advantage')
    ax.set_title('GAE Advantage Sensitivity to Lambda')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / 'gae_lambda_sensitivity.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved lambda sensitivity plot to '{fig_path}'")
    plt.close()

def main():
    print("="*60)
    print("Generalized Advantage Estimation Analysis")
    print("="*60)
    
    # Demonstrate GAE computation
    demonstrate_gae_computation()
    
    # Analyze bias-variance tradeoff
    comparison_results = analyze_gae_bias_variance()
    
    # Plot lambda sensitivity
    plot_lambda_sensitivity()
    
    # Run lambda sweep (commented out for speed - uncomment if desired)
    # lambda_results = run_gae_lambda_sweep()
    
    print("\nKey Insights:")
    print("1. GAE trades off bias and variance with lambda parameter")
    print("2. Lower lambda = lower variance, higher bias")
    print("3. Higher lambda = higher variance, lower bias") 
    print("4. Lambda ~0.95 often provides good balance")
    print("5. Problem-specific tuning may be needed")
    
    print("\nNext: exp06_hyperparameter_sensitivity.py")

if __name__ == "__main__":
    main()
