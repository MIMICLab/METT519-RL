#!/usr/bin/env python3
"""
RL2025 - Lecture 10: Experiment 09 - PPO Final Integration and Benchmarking

This final experiment integrates all PPO components into a comprehensive
implementation, runs benchmarks, and provides a complete evaluation framework.

Learning objectives:
- Integrate all PPO components into production-ready code
- Benchmark performance across multiple environments
- Implement proper evaluation and testing protocols
- Create reproducible experimental setups

Prerequisites: exp08_continuous_control.py completed successfully
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
import numpy as np
import json
import time
import hashlib
from typing import Dict, List, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# Import all previous components
from exp04_ppo_implementation import PPOConfig, ActorCritic, RolloutBuffer, PPOTrainer
from exp08_continuous_control import ContinuousActorCritic, ContinuousPPOConfig, ContinuousPPOTrainer

@dataclass 
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    environments: List[str] = None
    seeds: List[int] = None
    total_timesteps: int = 200_000
    num_runs: int = 3
    eval_episodes: int = 10
    save_results: bool = True
    results_dir: str = "benchmark_results"
    
    def __post_init__(self):
        if self.environments is None:
            self.environments = [
                "CartPole-v1",      # Discrete, simple
                "LunarLander-v2",   # Discrete, harder
                "Pendulum-v1",      # Continuous, simple
            ]
        if self.seeds is None:
            self.seeds = [42, 123, 456]

class ComprehensivePPO:
    """Comprehensive PPO implementation supporting both discrete and continuous control."""
    
    def __init__(self, config: Union[PPOConfig, ContinuousPPOConfig]):
        self.config = config
        self.device = device
        
        # Create environment to get specs
        test_env = gym.make(config.env_id)
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        test_env.close()
        
        self.obs_dim = obs_space.shape[0]
        self.is_continuous = isinstance(act_space, gym.spaces.Box)
        
        if self.is_continuous:
            self.act_dim = act_space.shape[0]
            self.action_low = torch.tensor(act_space.low, device=device)
            self.action_high = torch.tensor(act_space.high, device=device)
        else:
            self.act_dim = act_space.n
            self.action_low = None
            self.action_high = None
        
        # Initialize appropriate trainer
        if self.is_continuous:
            self.trainer = ContinuousPPOTrainer(config)
        else:
            self.trainer = PPOTrainer(config)
        
        print(f"Initialized {'Continuous' if self.is_continuous else 'Discrete'} PPO")
        print(f"Environment: {config.env_id}")
        print(f"Obs dim: {self.obs_dim}, Act dim: {self.act_dim}")
    
    def train(self) -> Dict[str, Any]:
        """Train the agent and return training metrics."""
        start_time = time.time()
        
        self.trainer.train()
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'total_timesteps': self.config.total_timesteps,
            'final_performance': self.evaluate()
        }
    
    def evaluate(self, episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """Evaluate trained agent."""
        env = gym.make(self.config.env_id, render_mode='human' if render else None)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            obs, _ = env.reset(seed=self.config.seed + episode)
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    if self.is_continuous:
                        action, _, _, _ = self.trainer.agent.get_action_and_value(obs_tensor)
                        # Process action for environment
                        if hasattr(self.trainer, 'process_action'):
                            env_action = self.trainer.process_action(action).cpu().numpy().flatten()
                        else:
                            env_action = action.cpu().numpy().flatten()
                    else:
                        action, _, _ = self.trainer.agent.get_action_and_value(obs_tensor)
                        env_action = action.item()
                
                obs, reward, terminated, truncated, _ = env.step(env_action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        env.close()
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'std_length': float(np.std(episode_lengths))
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        save_dict = {
            'model_state_dict': self.trainer.agent.state_dict(),
            'config': asdict(self.config),
            'is_continuous': self.is_continuous,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim
        }
        
        torch.save(save_dict, path)
        
        # Compute hash for reproducibility
        with open(path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        print(f"Model saved to {path}")
        print(f"File hash: {file_hash}")
        
        return file_hash

class BenchmarkSuite:
    """Comprehensive benchmarking suite for PPO."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        
        # Create results directory
        Path(config.results_dir).mkdir(exist_ok=True)
    
    def get_optimal_config(self, env_id: str) -> Union[PPOConfig, ContinuousPPOConfig]:
        """Get optimized configuration for each environment."""
        
        # Base configurations optimized for each environment
        configs = {
            "CartPole-v1": PPOConfig(
                env_id=env_id,
                total_timesteps=self.config.total_timesteps,
                num_envs=8,
                num_steps=128,
                learning_rate=2.5e-4,
                clip_coef=0.2,
                gae_lambda=0.95,
                vf_coef=0.5,
                ent_coef=0.01
            ),
            "LunarLander-v2": PPOConfig(
                env_id=env_id,
                total_timesteps=self.config.total_timesteps,
                num_envs=16,
                num_steps=512,
                learning_rate=2.5e-4,
                clip_coef=0.2,
                gae_lambda=0.98,
                vf_coef=0.5,
                ent_coef=0.01
            ),
            "Pendulum-v1": ContinuousPPOConfig(
                env_id=env_id,
                total_timesteps=self.config.total_timesteps,
                num_envs=8,
                num_steps=256,
                learning_rate=3e-4,
                clip_coef=0.2,
                gae_lambda=0.95,
                vf_coef=0.5,
                ent_coef=0.01
            )
        }
        
        return configs.get(env_id, PPOConfig(env_id=env_id, total_timesteps=self.config.total_timesteps))
    
    def run_single_benchmark(self, env_id: str, seed: int) -> Dict[str, Any]:
        """Run a single benchmark experiment."""
        print(f"\nRunning benchmark: {env_id} with seed {seed}")
        print("-" * 50)
        
        # Get optimal configuration
        config = self.get_optimal_config(env_id)
        config.seed = seed
        
        # Create PPO agent
        ppo = ComprehensivePPO(config)
        
        # Train
        training_metrics = ppo.train()
        
        # Evaluate
        eval_metrics = ppo.evaluate(episodes=self.config.eval_episodes)
        
        # Save model
        model_path = f"{self.config.results_dir}/{env_id}_seed{seed}_model.pt"
        model_hash = ppo.save_model(model_path)
        
        # Compile results
        results = {
            'environment': env_id,
            'seed': seed,
            'config': asdict(config),
            'training_metrics': training_metrics,
            'evaluation_metrics': eval_metrics,
            'model_path': model_path,
            'model_hash': model_hash,
            'timestamp': time.time()
        }
        
        print(f"Final performance: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all environments and seeds."""
        print("="*60)
        print("PPO Comprehensive Benchmark Suite")
        print("="*60)
        
        all_results = []
        
        for env_id in self.config.environments:
            env_results = []
            
            for seed in self.config.seeds:
                try:
                    result = self.run_single_benchmark(env_id, seed)
                    env_results.append(result)
                    all_results.append(result)
                except Exception as e:
                    print(f"Failed benchmark {env_id} seed {seed}: {e}")
                    continue
            
            # Compute environment summary statistics
            if env_results:
                performances = [r['evaluation_metrics']['mean_reward'] for r in env_results]
                env_summary = {
                    'environment': env_id,
                    'num_runs': len(env_results),
                    'mean_performance': float(np.mean(performances)),
                    'std_performance': float(np.std(performances)),
                    'min_performance': float(np.min(performances)),
                    'max_performance': float(np.max(performances))
                }
                self.results[env_id] = env_summary
                
                print(f"\n{env_id} Summary:")
                print(f"  Runs: {env_summary['num_runs']}")
                print(f"  Mean: {env_summary['mean_performance']:.2f}")
                print(f"  Std:  {env_summary['std_performance']:.2f}")
                print(f"  Range: [{env_summary['min_performance']:.2f}, {env_summary['max_performance']:.2f}]")
        
        # Save detailed results
        if self.config.save_results:
            results_file = f"{self.config.results_dir}/benchmark_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'summary': self.results,
                    'detailed_results': all_results,
                    'benchmark_config': asdict(self.config)
                }, f, indent=2)
            print(f"\nResults saved to {results_file}")
        
        return self.results
    
    def create_performance_plots(self):
        """Create performance comparison plots."""
        if not self.results:
            print("No results to plot")
            return
        
        # Performance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        env_names = list(self.results.keys())
        mean_perfs = [self.results[env]['mean_performance'] for env in env_names]
        std_perfs = [self.results[env]['std_performance'] for env in env_names]
        
        # Bar plot with error bars
        x = np.arange(len(env_names))
        ax1.bar(x, mean_perfs, yerr=std_perfs, capsize=5, alpha=0.7)
        ax1.set_xlabel('Environment')
        ax1.set_ylabel('Mean Episode Return')
        ax1.set_title('PPO Performance Across Environments')
        ax1.set_xticks(x)
        ax1.set_xticklabels(env_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Coefficient of variation (stability measure)
        cv_values = [std / abs(mean) if mean != 0 else 0 for mean, std in zip(mean_perfs, std_perfs)]
        ax2.bar(x, cv_values, alpha=0.7)
        ax2.set_xlabel('Environment')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_title('Training Stability (Lower is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(env_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.results_dir}/performance_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Performance plots saved to {self.config.results_dir}/performance_comparison.png")
        plt.close()

def run_ablation_studies():
    """Run ablation studies on key PPO components."""
    print("\n=== PPO Ablation Studies ===")
    
    base_config = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=50_000,
        num_envs=4,
        seed=42
    )
    
    ablations = {
        'baseline': base_config,
        'no_clipping': PPOConfig(**{**asdict(base_config), 'clip_coef': float('inf')}),
        'no_entropy': PPOConfig(**{**asdict(base_config), 'ent_coef': 0.0}),
        'no_value_clipping': PPOConfig(**{**asdict(base_config), 'clip_vloss': False}),
        'single_epoch': PPOConfig(**{**asdict(base_config), 'update_epochs': 1}),
        'high_gae_lambda': PPOConfig(**{**asdict(base_config), 'gae_lambda': 1.0}),
        'low_gae_lambda': PPOConfig(**{**asdict(base_config), 'gae_lambda': 0.8}),
    }
    
    ablation_results = {}
    
    for name, config in ablations.items():
        print(f"\nTesting ablation: {name}")
        try:
            ppo = ComprehensivePPO(config)
            ppo.train()
            performance = ppo.evaluate(episodes=5)
            ablation_results[name] = performance['mean_reward']
            print(f"  Performance: {performance['mean_reward']:.2f}")
        except Exception as e:
            print(f"  Failed: {e}")
            ablation_results[name] = 0.0
    
    print("\nAblation Study Results:")
    print("-" * 30)
    baseline_perf = ablation_results.get('baseline', 0.0)
    
    for name, perf in sorted(ablation_results.items(), key=lambda x: x[1], reverse=True):
        diff = perf - baseline_perf
        print(f"{name:20}: {perf:6.2f} ({diff:+5.2f})")
    
    return ablation_results

def demonstrate_reproducibility():
    """Demonstrate reproducible training."""
    print("\n=== Reproducibility Test ===")
    
    config = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=20_000,
        num_envs=4,
        seed=42  # Fixed seed
    )
    
    results = []
    
    for run in range(3):
        print(f"Run {run + 1}/3...")
        
        ppo = ComprehensivePPO(config)
        ppo.train()
        performance = ppo.evaluate(episodes=5)
        results.append(performance['mean_reward'])
        
        print(f"  Performance: {performance['mean_reward']:.4f}")
    
    print(f"\nReproducibility Results:")
    print(f"Mean: {np.mean(results):.4f}")
    print(f"Std:  {np.std(results):.4f}")
    print(f"Range: [{np.min(results):.4f}, {np.max(results):.4f}]")
    
    if np.std(results) < 1e-3:
        print("✓ High reproducibility achieved")
    else:
        print("⚠ Reproducibility may need improvement")
    
    return results

def create_learning_curves():
    """Create learning curves for different environments."""
    print("\n=== Creating Learning Curves ===")
    
    environments = ["CartPole-v1", "Pendulum-v1"]
    
    for env_id in environments:
        print(f"Generating learning curve for {env_id}...")
        
        if env_id == "CartPole-v1":
            config = PPOConfig(env_id=env_id, total_timesteps=100_000, num_envs=4, seed=42)
        else:
            config = ContinuousPPOConfig(env_id=env_id, total_timesteps=100_000, num_envs=4, seed=42)
        
        ppo = ComprehensivePPO(config)
        ppo.train()
        
        print(f"  Training completed for {env_id}")
        print(f"  Check TensorBoard logs: runs/")

def main():
    print("="*60)
    print("PPO Final Integration and Benchmarking")
    print("="*60)
    
    # 1. Run comprehensive benchmark
    benchmark_config = BenchmarkConfig(
        environments=["CartPole-v1", "Pendulum-v1"],  # Reduced for demo
        seeds=[42, 123],  # Reduced for demo
        total_timesteps=50_000,  # Reduced for demo
        num_runs=2
    )
    
    benchmark = BenchmarkSuite(benchmark_config)
    results = benchmark.run_comprehensive_benchmark()
    benchmark.create_performance_plots()
    
    # 2. Run ablation studies
    ablation_results = run_ablation_studies()
    
    # 3. Test reproducibility
    repro_results = demonstrate_reproducibility()
    
    # 4. Create learning curves
    create_learning_curves()
    
    print("\n" + "="*60)
    print("Final Integration Summary:")
    print("1. ✓ Comprehensive PPO implementation")
    print("2. ✓ Both discrete and continuous control")
    print("3. ✓ Benchmarking suite with multiple environments")
    print("4. ✓ Ablation studies on key components")
    print("5. ✓ Reproducibility testing")
    print("6. ✓ Performance visualization")
    print("7. ✓ Model saving and evaluation")
    print("="*60)
    
    print("\nPPO Implementation Complete!")
    print("Ready for real-world applications.")

if __name__ == "__main__":
    main()