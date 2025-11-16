#!/usr/bin/env python3
"""
RL2025 - Lecture 10: Experiment 07 - PPO Debugging and Diagnostic Techniques

This experiment demonstrates common PPO failure modes and debugging techniques,
helping students identify and fix issues in their implementations.

Learning objectives:
- Recognize common PPO failure patterns
- Use diagnostic metrics effectively
- Implement debugging visualizations
- Develop systematic debugging approaches

Prerequisites: exp06_hyperparameter_sensitivity.py completed successfully
"""

# PyTorch 2.x Standard Practice Header
import random, numpy as np, torch

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
from typing import Dict, List, Tuple, Any
import time
from dataclasses import dataclass, replace
import warnings

from helpers import FIGURES_DIR

# Import from previous experiments
from exp04_ppo_implementation import PPOConfig, PPOTrainer, ActorCritic

class PPODiagnosticTrainer(PPOTrainer):
    """Enhanced PPO trainer with extensive diagnostics."""
    
    def __init__(self, config: PPOConfig):
        super().__init__(config)
        
        # Diagnostic tracking
        self.diagnostics = {
            'grad_norms': [],
            'weight_norms': [],
            'clip_fractions': [],
            'kl_divergences': [],
            'value_errors': [],
            'policy_entropies': [],
            'advantage_stats': [],
            'ratio_stats': [],
            'loss_components': []
        }
        
        # Add hooks for gradient monitoring
        self.add_gradient_hooks()
    
    def add_gradient_hooks(self):
        """Add hooks to monitor gradients."""
        def grad_hook(name):
            def hook(grad):
                if grad is not None:
                    self.diagnostics.setdefault(f'grad_norm_{name}', []).append(grad.norm().item())
                return grad
            return hook
        
        for name, param in self.agent.named_parameters():
            if param.requires_grad:
                param.register_hook(grad_hook(name.replace('.', '_')))
    
    def update_policy(self):
        """Enhanced policy update with diagnostics."""
        # Get flattened batch
        batch = self.buffer.get_batch(self.config.minibatch_size)
        
        # Normalize advantages
        advantages = batch['advantages']
        adv_mean, adv_std = advantages.mean(), advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        # Store advantage statistics
        self.diagnostics['advantage_stats'].append({
            'mean': adv_mean.item(),
            'std': adv_std.item(),
            'min': advantages.min().item(),
            'max': advantages.max().item()
        })
        
        # Training loop with enhanced diagnostics
        clip_fracs = []
        kl_divs = []
        value_errors = []
        entropies = []
        ratio_stats = []
        
        for epoch in range(self.config.update_epochs):
            batch_size = batch['obs'].shape[0]
            minibatch_size = self.config.minibatch_size
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Get minibatch
                mb_obs = batch['obs'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_logprobs = batch['old_logprobs'][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = batch['returns'][mb_indices]
                mb_old_values = batch['old_values'][mb_indices]
                
                # Forward pass
                logits, values = self.agent(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                values = values.flatten()
                
                # Compute ratios and diagnostics
                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()
                
                # Diagnostic metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()
                    value_error = (values - mb_returns).abs().mean()
                    
                    clip_fracs.append(clip_frac.item())
                    kl_divs.append(approx_kl.item())
                    value_errors.append(value_error.item())
                    entropies.append(entropy.item())
                    
                    ratio_stats.append({
                        'mean': ratio.mean().item(),
                        'std': ratio.std().item(),
                        'min': ratio.min().item(),
                        'max': ratio.max().item()
                    })
                
                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                if self.config.clip_vloss:
                    v_loss_unclipped = (values - mb_returns) ** 2
                    v_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values, -self.config.clip_coef, self.config.clip_coef
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                
                # Total loss
                total_loss = pg_loss + self.config.vf_coef * v_loss - self.config.ent_coef * entropy
                
                # Store loss components
                self.diagnostics['loss_components'].append({
                    'policy_loss': pg_loss.item(),
                    'value_loss': v_loss.item(),
                    'entropy_loss': -self.config.ent_coef * entropy.item(),
                    'total_loss': total_loss.item()
                })
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient norm before clipping
                grad_norm = 0.0
                for param in self.agent.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                self.diagnostics['grad_norms'].append(grad_norm)
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Weight norms
                weight_norm = 0.0
                for param in self.agent.parameters():
                    weight_norm += param.norm().item() ** 2
                weight_norm = weight_norm ** 0.5
                self.diagnostics['weight_norms'].append(weight_norm)
            
            # Early stopping based on KL divergence
            if self.config.target_kl is not None and np.mean(kl_divs) > self.config.target_kl:
                print(f"Early stopping at epoch {epoch} due to KL divergence {np.mean(kl_divs):.4f}")
                break
        
        # Store epoch-level diagnostics
        self.diagnostics['clip_fractions'].append(np.mean(clip_fracs))
        self.diagnostics['kl_divergences'].append(np.mean(kl_divs))
        self.diagnostics['value_errors'].append(np.mean(value_errors))
        self.diagnostics['policy_entropies'].append(np.mean(entropies))
        self.diagnostics['ratio_stats'].append({
            'mean': np.mean([r['mean'] for r in ratio_stats]),
            'std': np.mean([r['std'] for r in ratio_stats]),
            'min': np.min([r['min'] for r in ratio_stats]),
            'max': np.max([r['max'] for r in ratio_stats])
        })
        
        # Enhanced logging
        self.writer.add_scalar("diagnostics/grad_norm", np.mean(self.diagnostics['grad_norms'][-10:]), self.global_step)
        self.writer.add_scalar("diagnostics/weight_norm", np.mean(self.diagnostics['weight_norms'][-10:]), self.global_step)
        self.writer.add_scalar("diagnostics/advantage_std", adv_std.item(), self.global_step)
        self.writer.add_scalar("diagnostics/value_error", np.mean(value_errors), self.global_step)
        self.writer.add_scalar("diagnostics/ratio_mean", np.mean([r['mean'] for r in ratio_stats]), self.global_step)
        self.writer.add_scalar("diagnostics/ratio_std", np.mean([r['std'] for r in ratio_stats]), self.global_step)
        
        return {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': np.mean(kl_divs),
            'clipfrac': np.mean(clip_fracs)
        }

def create_broken_configs() -> Dict[str, PPOConfig]:
    """Create configurations that demonstrate common failure modes."""
    
    base_config = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=30_000,
        num_envs=4,
        num_steps=128,
        seed=42
    )
    
    broken_configs = {
        'learning_rate_too_high': replace(base_config, learning_rate=0.1),
        'learning_rate_too_low': replace(base_config, learning_rate=1e-6),
        'clip_too_small': replace(base_config, clip_coef=0.01),
        'clip_too_large': replace(base_config, clip_coef=1.0),
        'no_entropy': replace(base_config, ent_coef=0.0),
        'too_much_entropy': replace(base_config, ent_coef=1.0),
        'value_coef_too_high': replace(base_config, vf_coef=10.0),
        'too_many_epochs': replace(base_config, update_epochs=20),
        'batch_too_small': replace(base_config, num_envs=1, num_steps=32),
        'no_grad_clipping': replace(base_config, max_grad_norm=float('inf'))
    }
    
    return broken_configs

def diagnose_failure_mode(trainer: PPODiagnosticTrainer, config_name: str):
    """Analyze diagnostics to identify failure mode."""
    diagnostics = trainer.diagnostics
    
    print(f"\nDiagnosing failure mode: {config_name}")
    print("-" * 50)
    
    # Check various diagnostic signals
    issues = []
    
    # Gradient issues
    if len(diagnostics['grad_norms']) > 10:
        avg_grad_norm = np.mean(diagnostics['grad_norms'][-10:])
        if avg_grad_norm > 10.0:
            issues.append("Gradient explosion detected")
        elif avg_grad_norm < 1e-6:
            issues.append("Vanishing gradients detected")
    
    # KL divergence issues
    if len(diagnostics['kl_divergences']) > 0:
        kl_values = diagnostics['kl_divergences']
        if np.mean(kl_values) > 0.1:
            issues.append("KL divergence too high - policy changing too fast")
        elif np.mean(kl_values) < 1e-5:
            issues.append("KL divergence too low - policy not learning")
    
    # Clip fraction issues
    if len(diagnostics['clip_fractions']) > 0:
        clip_values = diagnostics['clip_fractions']
        if np.mean(clip_values) > 0.8:
            issues.append("Excessive clipping - consider larger clip range")
        elif np.mean(clip_values) < 0.1:
            issues.append("Minimal clipping - policy updates may be too conservative")
    
    # Entropy issues
    if len(diagnostics['policy_entropies']) > 0:
        entropy_trend = diagnostics['policy_entropies']
        if len(entropy_trend) > 5:
            recent_entropy = np.mean(entropy_trend[-5:])
            if recent_entropy < 0.1:
                issues.append("Very low entropy - premature convergence risk")
            elif recent_entropy > 2.0:
                issues.append("Very high entropy - policy too random")
    
    # Value function issues
    if len(diagnostics['value_errors']) > 0:
        value_errors = diagnostics['value_errors']
        if np.mean(value_errors) > 100:
            issues.append("High value function error - poor baseline")
    
    # Ratio statistics
    if len(diagnostics['ratio_stats']) > 0:
        ratio_stats = diagnostics['ratio_stats'][-5:]  # Recent stats
        max_ratio = max([r['max'] for r in ratio_stats])
        min_ratio = min([r['min'] for r in ratio_stats])
        
        if max_ratio > 5.0 or min_ratio < 0.2:
            issues.append("Extreme importance sampling ratios detected")
    
    # Print diagnosis
    if issues:
        print("Issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No obvious issues detected in diagnostics")
    
    return issues

def create_diagnostic_plots(trainer: PPODiagnosticTrainer, config_name: str):
    """Create comprehensive diagnostic plots."""
    diagnostics = trainer.diagnostics
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    plot_configs = [
        ('grad_norms', 'Gradient Norm', 'Update Step', 'Norm'),
        ('weight_norms', 'Weight Norm', 'Update Step', 'Norm'),
        ('clip_fractions', 'Clip Fraction', 'Update', 'Fraction'),
        ('kl_divergences', 'KL Divergence', 'Update', 'KL'),
        ('value_errors', 'Value Error', 'Update', 'MAE'),
        ('policy_entropies', 'Policy Entropy', 'Update', 'Entropy'),
    ]
    
    for i, (key, title, xlabel, ylabel) in enumerate(plot_configs):
        if i >= len(axes):
            break
            
        ax = axes[i]
        data = diagnostics.get(key, [])
        
        if data:
            ax.plot(data, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            # Add trend line if enough data
            if len(data) > 10:
                z = np.polyfit(range(len(data)), data, 1)
                p = np.poly1d(z)
                ax.plot(range(len(data)), p(range(len(data))), "r--", alpha=0.8, linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    # Advantage statistics plot
    if 'advantage_stats' in diagnostics and diagnostics['advantage_stats']:
        ax = axes[6]
        adv_stats = diagnostics['advantage_stats']
        means = [stat['mean'] for stat in adv_stats]
        stds = [stat['std'] for stat in adv_stats]
        
        ax.plot(means, label='Mean', linewidth=2)
        ax.plot(stds, label='Std', linewidth=2)
        ax.set_title('Advantage Statistics')
        ax.set_xlabel('Update')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Ratio statistics plot
    if 'ratio_stats' in diagnostics and diagnostics['ratio_stats']:
        ax = axes[7]
        ratio_stats = diagnostics['ratio_stats']
        means = [stat['mean'] for stat in ratio_stats]
        maxs = [stat['max'] for stat in ratio_stats]
        mins = [stat['min'] for stat in ratio_stats]
        
        ax.plot(means, label='Mean', linewidth=2)
        ax.plot(maxs, label='Max', linewidth=1, alpha=0.7)
        ax.plot(mins, label='Min', linewidth=1, alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Importance Sampling Ratios')
        ax.set_xlabel('Update')
        ax.set_ylabel('Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Loss components plot
    if 'loss_components' in diagnostics and diagnostics['loss_components']:
        ax = axes[8]
        loss_data = diagnostics['loss_components']
        
        policy_losses = [l['policy_loss'] for l in loss_data]
        value_losses = [l['value_loss'] for l in loss_data]
        entropy_losses = [l['entropy_loss'] for l in loss_data]
        
        # Smooth the data
        if len(policy_losses) > 50:
            window = 20
            policy_losses = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
            value_losses = np.convolve(value_losses, np.ones(window)/window, mode='valid')
            entropy_losses = np.convolve(entropy_losses, np.ones(window)/window, mode='valid')
        
        ax.plot(policy_losses, label='Policy Loss', linewidth=2)
        ax.plot(value_losses, label='Value Loss', linewidth=2)
        ax.plot(entropy_losses, label='Entropy Loss', linewidth=2)
        ax.set_title('Loss Components')
        ax.set_xlabel('Minibatch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'PPO Diagnostics: {config_name}', fontsize=16)
    plt.tight_layout()
    out_path = FIGURES_DIR / f'diagnostics_{config_name}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved diagnostic plots to '{out_path}'")
    plt.close()

def run_diagnostic_experiments():
    """Run experiments with known failure modes."""
    print("="*60)
    print("PPO Debugging and Diagnostic Experiments")
    print("="*60)
    
    broken_configs = create_broken_configs()
    
    # Test each broken configuration
    for config_name, config in broken_configs.items():
        print(f"\nTesting: {config_name}")
        print("-" * 30)
        
        try:
            trainer = PPODiagnosticTrainer(config)
            trainer.train()
            
            # Diagnose issues
            issues = diagnose_failure_mode(trainer, config_name)
            
            # Create diagnostic plots
            create_diagnostic_plots(trainer, config_name)
            
        except Exception as e:
            print(f"Training failed: {e}")
            continue

def demonstrate_debugging_workflow():
    """Demonstrate systematic debugging workflow."""
    print("\n=== Systematic Debugging Workflow ===")
    
    print("1. Start with a known good configuration")
    good_config = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=20_000,
        num_envs=4,
        num_steps=128,
        learning_rate=2.5e-4,
        clip_coef=0.2,
        seed=42
    )
    
    print("2. Train and collect diagnostics")
    trainer = PPODiagnosticTrainer(good_config)
    trainer.train()
    
    print("3. Analyze diagnostic signals")
    issues = diagnose_failure_mode(trainer, "baseline")
    
    print("4. Create diagnostic plots")
    create_diagnostic_plots(trainer, "baseline")
    
    print("\nDebugging Checklist:")
    checklist = [
        "Check gradient norms (should be 0.1-10)",
        "Monitor KL divergence (should be < 0.05)",
        "Watch clip fraction (should be 0.1-0.3)",
        "Track policy entropy (should decrease gradually)",
        "Monitor value function error",
        "Check importance sampling ratios",
        "Verify advantage normalization",
        "Look for loss component balance"
    ]
    
    for i, item in enumerate(checklist, 1):
        print(f"  {i}. {item}")

def main():
    print("="*60)
    print("PPO Debugging and Diagnostic Techniques")
    print("="*60)
    
    # Run diagnostic experiments
    run_diagnostic_experiments()
    
    # Demonstrate debugging workflow
    demonstrate_debugging_workflow()
    
    print("\n" + "="*60)
    print("Debugging Summary:")
    print("1. Use comprehensive diagnostics to identify issues")
    print("2. Start with known good configurations")
    print("3. Change one parameter at a time")
    print("4. Monitor key metrics: KL, clip fraction, entropy")
    print("5. Check gradients and importance sampling ratios")
    print("6. Visualize learning curves and loss components")
    print("="*60)
    
    print("\nNext: exp08_continuous_control.py")

if __name__ == "__main__":
    main()
