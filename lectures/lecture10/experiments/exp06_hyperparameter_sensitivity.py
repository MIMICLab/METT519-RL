#!/usr/bin/env python3
"""
RL2025 - Lecture 10: Experiment 06 - PPO Hyperparameter Sensitivity Analysis

This experiment systematically analyzes the sensitivity of PPO to its key
hyperparameters, providing insights into tuning and robustness.

Learning objectives:
- Understand impact of key PPO hyperparameters
- Learn systematic hyperparameter tuning approaches
- Identify robust vs sensitive hyperparameter ranges
- Develop intuition for PPO debugging

Prerequisites: exp05_gae_analysis.py completed successfully
"""

# PyTorch 2.x Standard Practice Header
import random, numpy as np, torch
from pathlib import Path

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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import time
from dataclasses import dataclass, replace
import json

from helpers import FIGURES_DIR

import torch.nn as nn

# Import from previous experiments
from exp04_ppo_implementation import PPOConfig, PPOTrainer

class HyperparameterSweep:
    """Class for systematic hyperparameter analysis."""
    
    def __init__(self, base_config: PPOConfig, results_dir: Path | str | None = None):
        self.base_config = base_config
        self.results_dir = Path(results_dir) if results_dir is not None else FIGURES_DIR / "hyperparameter_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.sweep_results = {}
    
    def run_single_experiment(self, config: PPOConfig, name: str) -> Dict:
        """Run a single experiment and return key metrics."""
        print(f"Running experiment: {name}")
        
        # Track metrics during training
        metrics = {
            'final_performance': 0.0,
            'training_stability': 0.0,
            'sample_efficiency': 0.0,
            'convergence_speed': 0.0
        }
        
        try:
            trainer = PPOTrainer(config)
            
            # Modify trainer to collect metrics
            original_update = trainer.update_policy
            episode_rewards = []
            policy_losses = []
            value_losses = []
            kl_divergences = []
            
            def modified_update():
                update_info = original_update()
                policy_losses.append(update_info['policy_loss'])
                value_losses.append(update_info['value_loss'])
                kl_divergences.append(update_info['approx_kl'])
                return update_info
            
            trainer.update_policy = modified_update
            
            # Train the agent
            trainer.train()
            
            # Evaluate final performance
            final_performance = evaluate_trained_agent(trainer.agent, config.env_id)
            
            # Compute derived metrics
            metrics['final_performance'] = final_performance
            metrics['training_stability'] = 1.0 / (np.std(policy_losses[-10:]) + 1e-6)
            metrics['sample_efficiency'] = final_performance / config.total_timesteps * 100000
            
            # Convergence speed (steps to reach 90% of final performance)
            if len(policy_losses) > 10:
                target_performance = final_performance * 0.9
                metrics['convergence_speed'] = len(policy_losses) - 10  # Placeholder
            
            print(f"  Final performance: {final_performance:.2f}")
            
            # Save detailed results
            detailed_results = {
                'config': config.__dict__,
                'metrics': metrics,
                'policy_losses': policy_losses,
                'value_losses': value_losses,
                'kl_divergences': kl_divergences
            }
            
            with open(self.results_dir / f"{name}.json", 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
        except Exception as e:
            print(f"  Failed: {e}")
            metrics['final_performance'] = 0.0
        
        return metrics
    
    def sweep_learning_rate(self) -> Dict:
        """Sweep learning rate values."""
        print("\n=== Learning Rate Sweep ===")
        
        lr_values = [1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3]
        results = {}
        
        for lr in lr_values:
            config = replace(self.base_config, learning_rate=lr, seed=42)
            metrics = self.run_single_experiment(config, f"lr_{lr:.0e}")
            results[lr] = metrics
        
        self.sweep_results['learning_rate'] = results
        return results
    
    def sweep_clip_coefficient(self) -> Dict:
        """Sweep clipping coefficient values."""
        print("\n=== Clip Coefficient Sweep ===")
        
        clip_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
        results = {}
        
        for clip_coef in clip_values:
            config = replace(self.base_config, clip_coef=clip_coef, seed=42)
            metrics = self.run_single_experiment(config, f"clip_{clip_coef:.2f}")
            results[clip_coef] = metrics
        
        self.sweep_results['clip_coefficient'] = results
        return results
    
    def sweep_batch_size(self) -> Dict:
        """Sweep effective batch size (num_envs * num_steps)."""
        print("\n=== Batch Size Sweep ===")
        
        # Different combinations of num_envs and num_steps for different batch sizes
        batch_configs = [
            (2, 64),    # 128
            (4, 64),    # 256  
            (4, 128),   # 512
            (8, 128),   # 1024
            (8, 256),   # 2048
            (16, 128),  # 2048 (different split)
        ]
        
        results = {}
        
        for num_envs, num_steps in batch_configs:
            batch_size = num_envs * num_steps
            config = replace(self.base_config, 
                           num_envs=num_envs, 
                           num_steps=num_steps,
                           seed=42)
            metrics = self.run_single_experiment(config, f"batch_{batch_size}")
            results[batch_size] = metrics
        
        self.sweep_results['batch_size'] = results
        return results
    
    def sweep_entropy_coefficient(self) -> Dict:
        """Sweep entropy coefficient values."""
        print("\n=== Entropy Coefficient Sweep ===")
        
        ent_values = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
        results = {}
        
        for ent_coef in ent_values:
            config = replace(self.base_config, ent_coef=ent_coef, seed=42)
            metrics = self.run_single_experiment(config, f"ent_{ent_coef:.3f}")
            results[ent_coef] = metrics
        
        self.sweep_results['entropy_coefficient'] = results
        return results
    
    def sweep_value_coefficient(self) -> Dict:
        """Sweep value function coefficient values."""
        print("\n=== Value Function Coefficient Sweep ===")
        
        vf_values = [0.1, 0.25, 0.5, 1.0, 2.0]
        results = {}
        
        for vf_coef in vf_values:
            config = replace(self.base_config, vf_coef=vf_coef, seed=42)
            metrics = self.run_single_experiment(config, f"vf_{vf_coef:.2f}")
            results[vf_coef] = metrics
        
        self.sweep_results['value_coefficient'] = results
        return results
    
    def sweep_update_epochs(self) -> Dict:
        """Sweep number of update epochs."""
        print("\n=== Update Epochs Sweep ===")
        
        epoch_values = [1, 2, 4, 8, 10]
        results = {}
        
        for epochs in epoch_values:
            config = replace(self.base_config, update_epochs=epochs, seed=42)
            metrics = self.run_single_experiment(config, f"epochs_{epochs}")
            results[epochs] = metrics
        
        self.sweep_results['update_epochs'] = results
        return results
    
    def run_comprehensive_sweep(self) -> Dict:
        """Run comprehensive hyperparameter sweep."""
        print("="*60)
        print("PPO Hyperparameter Sensitivity Analysis")
        print("="*60)
        
        # Run all sweeps
        self.sweep_learning_rate()
        self.sweep_clip_coefficient()
        self.sweep_batch_size()
        self.sweep_entropy_coefficient()
        self.sweep_value_coefficient()
        self.sweep_update_epochs()
        
        # Save complete results
        with open(self.results_dir / 'complete_sweep.json', 'w') as f:
            json.dump(self.sweep_results, f, indent=2)
        
        return self.sweep_results
    
    def visualize_results(self):
        """Create visualizations of sweep results."""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        sweep_names = list(self.sweep_results.keys())
        
        for i, sweep_name in enumerate(sweep_names):
            if i >= 6:
                break
                
            ax = axes[i]
            sweep_data = self.sweep_results[sweep_name]
            
            x_values = list(sweep_data.keys())
            y_values = [sweep_data[x]['final_performance'] for x in x_values]
            
            # Convert string keys to float if possible
            try:
                x_values = [float(x) for x in x_values]
                sort_idx = np.argsort(x_values)
                x_values = [x_values[i] for i in sort_idx]
                y_values = [y_values[i] for i in sort_idx]
            except:
                pass
            
            ax.plot(x_values, y_values, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel(sweep_name.replace('_', ' ').title())
            ax.set_ylabel('Final Performance')
            ax.set_title(f'{sweep_name.replace("_", " ").title()} Sensitivity')
            ax.grid(True, alpha=0.3)
            
            # Highlight best value
            best_idx = np.argmax(y_values)
            ax.plot(x_values[best_idx], y_values[best_idx], 'ro', markersize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'hyperparameter_sensitivity.png', dpi=150, bbox_inches='tight')
        print(f"Saved sensitivity plots to '{self.results_dir / 'hyperparameter_sensitivity.png'}'")
        plt.close()
    
    def generate_recommendations(self) -> Dict:
        """Generate hyperparameter recommendations based on results."""
        print("\nGenerating hyperparameter recommendations...")
        
        recommendations = {}
        
        for sweep_name, sweep_data in self.sweep_results.items():
            # Find best performing hyperparameter value
            best_value = max(sweep_data.keys(), key=lambda x: sweep_data[x]['final_performance'])
            best_performance = sweep_data[best_value]['final_performance']
            
            # Find values within 95% of best performance (robust range)
            threshold = best_performance * 0.95
            robust_values = [v for v in sweep_data.keys() 
                           if sweep_data[v]['final_performance'] >= threshold]
            
            recommendations[sweep_name] = {
                'best_value': best_value,
                'best_performance': best_performance,
                'robust_range': robust_values,
                'sensitivity': 'High' if len(robust_values) <= 2 else 'Low'
            }
        
        # Print recommendations
        print("\nHyperparameter Recommendations:")
        print("-" * 50)
        
        for param, rec in recommendations.items():
            print(f"{param.replace('_', ' ').title()}:")
            print(f"  Best value: {rec['best_value']}")
            print(f"  Performance: {rec['best_performance']:.2f}")
            print(f"  Robust range: {rec['robust_range']}")
            print(f"  Sensitivity: {rec['sensitivity']}")
            print()
        
        # Save recommendations
        with open(self.results_dir / 'recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        return recommendations

def evaluate_trained_agent(agent: nn.Module, env_id: str, episodes: int = 10) -> float:
    """Evaluate a trained agent's performance."""
    env = gym.make(env_id)
    total_rewards = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
    
    env.close()
    return float(np.mean(total_rewards))

def analyze_hyperparameter_interactions():
    """Analyze interactions between key hyperparameters."""
    print("\n=== Hyperparameter Interaction Analysis ===")
    
    base_config = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=30_000,  # Shorter for interaction study
        num_envs=4,
        num_steps=64,
        seed=42
    )
    
    # Test combinations of learning rate and clip coefficient
    lr_values = [1e-4, 2.5e-4, 5e-4]
    clip_values = [0.1, 0.2, 0.3]
    
    interaction_results = {}
    
    for lr in lr_values:
        for clip_coef in clip_values:
            config = replace(base_config, learning_rate=lr, clip_coef=clip_coef)
            
            try:
                trainer = PPOTrainer(config)
                trainer.train()
                performance = evaluate_trained_agent(trainer.agent, config.env_id)
            except:
                performance = 0.0
            
            interaction_results[(lr, clip_coef)] = performance
            print(f"LR={lr:.0e}, Clip={clip_coef:.1f}: Performance={performance:.2f}")
    
    # Visualize interaction
    fig, ax = plt.subplots(figsize=(8, 6))
    
    lr_grid, clip_grid = np.meshgrid(lr_values, clip_values)
    performance_grid = np.zeros_like(lr_grid)
    
    for i, lr in enumerate(lr_values):
        for j, clip_coef in enumerate(clip_values):
            performance_grid[j, i] = interaction_results[(lr, clip_coef)]
    
    im = ax.imshow(performance_grid, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(lr_values)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lr_values])
    ax.set_yticks(range(len(clip_values)))
    ax.set_yticklabels([f"{clip:.1f}" for clip in clip_values])
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Clip Coefficient')
    ax.set_title('Learning Rate vs Clip Coefficient Interaction')
    
    # Add text annotations
    for i in range(len(lr_values)):
        for j in range(len(clip_values)):
            ax.text(i, j, f"{performance_grid[j, i]:.1f}", 
                   ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Performance')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hyperparameter_interactions.png', dpi=150, bbox_inches='tight')
    print(f"Saved interaction plot to '{FIGURES_DIR / 'hyperparameter_interactions.png'}'")
    plt.close()
    
    return interaction_results

def main():
    print("="*60)
    print("PPO Hyperparameter Sensitivity Analysis")
    print("="*60)
    
    # Base configuration for sweeps
    base_config = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=40_000,  # Shorter for sweep efficiency
        num_envs=4,
        num_steps=128,
        seed=42
    )
    
    # Run comprehensive sweep
    sweep_analyzer = HyperparameterSweep(base_config)
    results = sweep_analyzer.run_comprehensive_sweep()
    
    # Visualize results
    sweep_analyzer.visualize_results()
    
    # Generate recommendations
    recommendations = sweep_analyzer.generate_recommendations()
    
    # Analyze interactions
    interaction_results = analyze_hyperparameter_interactions()
    
    print("\n" + "="*60)
    print("Key Insights:")
    print("1. Learning rate is typically the most sensitive hyperparameter")
    print("2. Clipping coefficient has a sweet spot around 0.2")
    print("3. Batch size affects sample efficiency and stability")
    print("4. Entropy coefficient prevents premature convergence")
    print("5. Multiple epochs can improve sample efficiency")
    print("6. Value coefficient balances critic vs actor learning")
    print("="*60)
    
    print("\nNext: exp07_debugging_techniques.py")

if __name__ == "__main__":
    main()
