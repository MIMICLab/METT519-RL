#!/usr/bin/env python3
"""
RL2025 - Lecture 10: Experiment 08 - PPO for Continuous Control

This experiment extends PPO to continuous action spaces using Gaussian policies,
demonstrating the key differences and implementation details for continuous control.

Learning objectives:
- Implement Gaussian policy heads for continuous actions
- Handle action bounds and squashing functions
- Understand log probability calculations for continuous actions
- Compare discrete vs continuous control performance

Prerequisites: exp07_debugging_techniques.py completed successfully
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
from typing import Tuple, Dict, Any
import time
from dataclasses import dataclass, replace

# Import base classes from previous experiments
from exp04_ppo_implementation import PPOConfig, make_env

class ContinuousActorCritic(nn.Module):
    """Actor-Critic network for continuous action spaces."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...] = (64, 64)):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh()
            ])
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Actor head - outputs mean of Gaussian distribution
        self.actor_mean = nn.Linear(in_dim, act_dim)
        
        # Log standard deviation (learned parameter or network output)
        # Option 1: State-independent log std (simpler)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        
        # Option 2: State-dependent log std (uncomment to use)
        # self.actor_logstd = nn.Linear(in_dim, act_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(in_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with orthogonal initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean, log std, and value."""
        features = self.feature_extractor(x)
        
        action_mean = self.actor_mean(features)
        
        # Option 1: State-independent log std
        action_logstd = self.actor_logstd.expand_as(action_mean)
        
        # Option 2: State-dependent log std (uncomment to use)
        # action_logstd = self.actor_logstd(features)
        
        value = self.critic(features)
        
        return action_mean, action_logstd, value
    
    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value."""
        action_mean, action_logstd, value = self.forward(x)
        action_std = torch.exp(action_logstd)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value."""
        features = self.feature_extractor(x)
        return self.critic(features).squeeze(-1)

class TanhSquashedGaussianActor(nn.Module):
    """Gaussian actor with tanh squashing for bounded actions."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256),
                 action_scale: float = 1.0, action_bias: float = 0.0):
        super().__init__()
        
        self.action_scale = action_scale
        self.action_bias = action_bias
        
        # Feature extractor
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Mean and log std heads
        self.mean_head = nn.Linear(in_dim, act_dim)
        self.logstd_head = nn.Linear(in_dim, act_dim)
        
        # Constrain log std to reasonable range
        self.log_std_min = -20
        self.log_std_max = 2
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        mean = self.mean_head(features)
        log_std = self.logstd_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions with tanh squashing."""
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(x_t)
        
        # Compute log probability with squashing correction
        log_prob = normal.log_prob(x_t)
        # Jacobian correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Scale actions to desired range
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob

@dataclass
class ContinuousPPOConfig(PPOConfig):
    """PPO configuration for continuous control."""
    env_id: str = "Pendulum-v1"
    action_bound_method: str = "clip"  # "clip", "tanh", or "none"
    action_scale: float = 2.0  # For tanh squashing
    action_bias: float = 0.0   # For tanh squashing

class ContinuousPPOTrainer:
    """PPO trainer for continuous action spaces."""
    
    def __init__(self, config: ContinuousPPOConfig):
        self.config = config
        self.device = device
        
        # Create vectorized environments
        self.envs = gym.vector.SyncVectorEnv([
            make_env(config.env_id, config.seed, i) 
            for i in range(config.num_envs)
        ])
        
        # Get environment dimensions
        obs_space = self.envs.single_observation_space
        act_space = self.envs.single_action_space
        
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        
        print(f"Environment: {config.env_id}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.act_dim}")
        print(f"Action space: {act_space}")
        
        # Create actor-critic network
        if config.action_bound_method == "tanh":
            # Use tanh squashed Gaussian for bounded actions
            self.agent = ContinuousActorCritic(self.obs_dim, self.act_dim).to(self.device)
            # Set action bounds
            if hasattr(act_space, 'low') and hasattr(act_space, 'high'):
                self.action_scale = torch.tensor((act_space.high - act_space.low) / 2.0, device=self.device)
                self.action_bias = torch.tensor((act_space.high + act_space.low) / 2.0, device=self.device)
            else:
                self.action_scale = config.action_scale
                self.action_bias = config.action_bias
        else:
            self.agent = ContinuousActorCritic(self.obs_dim, self.act_dim).to(self.device)
            # Action bounds for clipping
            if hasattr(act_space, 'low') and hasattr(act_space, 'high'):
                self.action_low = torch.tensor(act_space.low, device=self.device)
                self.action_high = torch.tensor(act_space.high, device=self.device)
            else:
                self.action_low = None
                self.action_high = None
        
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
        
        # Storage for rollouts
        self.obs_buffer = []
        self.action_buffer = []
        self.logprob_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.value_buffer = []
        
        # Tracking
        self.global_step = 0
        self.writer = SummaryWriter(f"runs/continuous_ppo_{config.env_id}_{int(time.time())}")
    
    def process_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Process raw network output to valid environment actions."""
        if self.config.action_bound_method == "clip" and self.action_low is not None:
            # Clip actions to valid range
            return torch.clamp(raw_action, self.action_low, self.action_high)
        elif self.config.action_bound_method == "tanh":
            # Apply tanh squashing and scaling
            return torch.tanh(raw_action) * self.action_scale + self.action_bias
        else:
            # No processing
            return raw_action
    
    def collect_rollouts(self):
        """Collect rollouts using current policy."""
        # Clear buffers
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.logprob_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.value_buffer.clear()
        
        next_obs, _ = self.envs.reset(seed=self.config.seed)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.config.num_envs, device=self.device)
        
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            
            # Store observation
            self.obs_buffer.append(next_obs.clone())
            
            # Get action from current policy
            with torch.no_grad():
                action, logprob, entropy, value = self.agent.get_action_and_value(next_obs)
                
                # Process actions for environment
                env_action = self.process_action(action)
            
            # Store policy outputs
            self.action_buffer.append(action.clone())
            self.logprob_buffer.append(logprob.clone())
            self.value_buffer.append(value.clone())
            
            # Take environment step
            next_obs, reward, terminated, truncated, infos = self.envs.step(env_action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            # Store step results
            self.reward_buffer.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            self.done_buffer.append(torch.tensor(done, dtype=torch.float32, device=self.device))
            
            # Update observations
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            next_done = torch.tensor(done, dtype=torch.float32, device=self.device)
            
            # Log episode statistics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        print(f"global_step={self.global_step}, episode_reward={info['episode']['r']:.2f}")
                        self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                        self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
        
        return next_obs, next_done
    
    def compute_gae(self, next_obs: torch.Tensor, next_done: torch.Tensor):
        """Compute GAE advantages and returns."""
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs)
            
            advantages = torch.zeros_like(self.reward_buffer[0])
            last_gae_lam = 0
            
            for t in reversed(range(len(self.reward_buffer))):
                if t == len(self.reward_buffer) - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.done_buffer[t + 1]
                    nextvalues = self.value_buffer[t + 1]
                
                delta = self.reward_buffer[t] + self.config.gamma * nextvalues * nextnonterminal - self.value_buffer[t]
                advantages = last_gae_lam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * last_gae_lam
                
                # Store advantages (we'll reverse later)
                if t == len(self.reward_buffer) - 1:
                    self.advantage_buffer = [advantages]
                else:
                    self.advantage_buffer.insert(0, advantages)
            
            # Compute returns
            self.return_buffer = [adv + val for adv, val in zip(self.advantage_buffer, self.value_buffer)]
    
    def update_policy(self):
        """Update policy using PPO."""
        # Flatten buffers
        b_obs = torch.stack(self.obs_buffer).reshape(-1, self.obs_dim)
        b_actions = torch.stack(self.action_buffer).reshape(-1, self.act_dim)
        b_logprobs = torch.stack(self.logprob_buffer).reshape(-1)
        b_advantages = torch.stack(self.advantage_buffer).reshape(-1)
        b_returns = torch.stack(self.return_buffer).reshape(-1)
        b_values = torch.stack(self.value_buffer).reshape(-1)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Training loop
        clipfracs = []
        
        for epoch in range(self.config.update_epochs):
            # Create random minibatches
            batch_size = len(b_obs)
            minibatch_size = self.config.minibatch_size
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Get minibatch
                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_old_logprobs = b_logprobs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_old_values = b_values[mb_indices]
                
                # Forward pass
                _, new_logprobs, entropy, new_values = self.agent.get_action_and_value(mb_obs, mb_actions)
                
                # Compute ratios
                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()
                
                # Diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()
                    clipfracs.append(clipfrac)
                
                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                if self.config.clip_vloss:
                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    v_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values, -self.config.clip_coef, self.config.clip_coef
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                
                # Entropy loss (encourage exploration)
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss + self.config.vf_coef * v_loss - self.config.ent_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            # Early stopping based on KL divergence
            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                print(f"Early stopping at epoch {epoch} due to KL divergence {approx_kl:.4f}")
                break
        
        # Logging
        clipfrac_value = torch.stack(clipfracs).mean().item() if clipfracs else 0.0
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", clipfrac_value, self.global_step)
        
        # Log action statistics
        with torch.no_grad():
            action_mean, action_logstd, _ = self.agent(b_obs[:100])  # Sample for stats
            action_std = torch.exp(action_logstd)
            self.writer.add_scalar("policy/action_mean", action_mean.mean().item(), self.global_step)
            self.writer.add_scalar("policy/action_std", action_std.mean().item(), self.global_step)
        
        return {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy': entropy_loss.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': clipfrac_value
        }
    
    def train(self):
        """Main training loop."""
        print("Starting continuous PPO training...")
        
        num_updates = self.config.total_timesteps // (self.config.num_steps * self.config.num_envs)
        start_time = time.time()
        
        for update in range(1, num_updates + 1):
            # Collect rollouts
            next_obs, next_done = self.collect_rollouts()
            
            # Compute advantages
            self.compute_gae(next_obs, next_done)
            
            # Update policy
            update_info = self.update_policy()
            
            # Print progress
            if update % 10 == 0:
                elapsed_time = time.time() - start_time
                sps = self.global_step / elapsed_time
                print(f"Update {update}/{num_updates}")
                print(f"  Global step: {self.global_step}")
                print(f"  Steps per second: {sps:.0f}")
                print(f"  Policy loss: {update_info['policy_loss']:.4f}")
                print(f"  Value loss: {update_info['value_loss']:.4f}")
                print(f"  Entropy: {update_info['entropy']:.4f}")
                print()
        
        self.envs.close()
        self.writer.close()
        print("Training completed!")

def compare_discrete_vs_continuous():
    """Compare discrete and continuous control environments."""
    print("\n=== Comparing Discrete vs Continuous Control ===")
    
    # Discrete control (CartPole)
    discrete_config = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=50_000,
        num_envs=4,
        seed=42
    )
    
    print("Training discrete control (CartPole)...")
    # Would train discrete PPO here
    
    # Continuous control (Pendulum)
    continuous_config = ContinuousPPOConfig(
        env_id="Pendulum-v1",
        total_timesteps=50_000,
        num_envs=4,
        seed=42
    )
    
    print("Training continuous control (Pendulum)...")
    trainer = ContinuousPPOTrainer(continuous_config)
    trainer.train()
    
    print("\nKey differences:")
    print("1. Action spaces: Discrete (Categorical) vs Continuous (Gaussian)")
    print("2. Policy outputs: Logits vs Mean/Std")
    print("3. Action sampling: Categorical vs Normal distribution")
    print("4. Action bounds: N/A vs Clipping/Squashing")
    print("5. Exploration: Entropy vs Std deviation")

def test_action_bound_methods():
    """Test different action bounding methods."""
    print("\n=== Testing Action Bound Methods ===")
    
    methods = ["clip", "tanh", "none"]
    
    for method in methods:
        print(f"\nTesting {method} method...")
        
        config = ContinuousPPOConfig(
            env_id="Pendulum-v1",
            total_timesteps=20_000,
            num_envs=2,
            action_bound_method=method,
            seed=42
        )
        
        trainer = ContinuousPPOTrainer(config)
        # Train for shorter time for comparison
        trainer.config.total_timesteps = 10_000
        trainer.train()

def main():
    print("="*60)
    print("PPO for Continuous Control")
    print("="*60)
    
    # Main continuous control experiment
    config = ContinuousPPOConfig(
        env_id="Pendulum-v1",
        total_timesteps=100_000,
        num_envs=4,
        learning_rate=3e-4,
        seed=42
    )
    
    trainer = ContinuousPPOTrainer(config)
    trainer.train()
    
    # Compare different approaches
    # compare_discrete_vs_continuous()  # Commented for brevity
    # test_action_bound_methods()       # Commented for brevity
    
    print("\nKey Takeaways:")
    print("1. Continuous control uses Gaussian policies")
    print("2. Action bounds require special handling")
    print("3. Entropy regularization prevents collapse")
    print("4. Log probability calculation includes Jacobian terms")
    print("5. Exploration balance is crucial")
    
    print("\nNext: exp09_final_integration.py")

if __name__ == "__main__":
    main()
