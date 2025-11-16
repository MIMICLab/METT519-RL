#!/usr/bin/env python3
"""
RL2025 - Lecture 10: Experiment 04 - PPO Core Implementation

This experiment implements the core PPO algorithm with clipped surrogate
objective, GAE advantage estimation, and all essential components for
stable policy optimization.

Learning objectives:
- Implement PPO clipped surrogate objective
- Add Generalized Advantage Estimation (GAE)
- Create actor-critic architecture
- Understand rollout buffer and minibatch training

Prerequisites: exp03_trust_region_concepts.py completed successfully
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
import numpy as np
from typing import Tuple, Dict, Any
import time
from dataclasses import dataclass

@dataclass
class PPOConfig:
    """PPO configuration parameters."""
    env_id: str = "CartPole-v1"
    total_timesteps: int = 100_000
    num_envs: int = 4
    num_steps: int = 128      # rollout length per env
    gamma: float = 0.99       # discount factor
    gae_lambda: float = 0.95  # GAE lambda
    learning_rate: float = 2.5e-4
    update_epochs: int = 4    # epochs per PPO update
    minibatch_size: int = 512 # minibatch size for SGD
    clip_coef: float = 0.2    # PPO clipping parameter
    clip_vloss: bool = True   # whether to clip value loss
    vf_coef: float = 0.5      # value function coefficient
    ent_coef: float = 0.01    # entropy coefficient
    max_grad_norm: float = 0.5 # gradient clipping
    target_kl: float = 0.03   # target KL for early stopping
    seed: int = 42
    device: str = "auto"

class ActorCritic(nn.Module):
    """Actor-Critic network with shared features."""
    
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
        
        # Actor head (policy)
        self.actor = nn.Linear(in_dim, act_dim)
        
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and value."""
        features = self.feature_extractor(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value."""
        features = self.feature_extractor(x)
        return self.critic(features)
    
    def get_action_and_value(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value."""
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        
        # Initialize storage
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        
        # Will be computed later
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
    
    def add(self, step: int, obs: torch.Tensor, action: torch.Tensor, 
            logprob: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, value: torch.Tensor):
        """Add a step of data to the buffer."""
        self.obs[step] = obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value
    
    def compute_gae(self, next_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - self.dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * nextnonterminal * last_gae_lam
        
        self.advantages = advantages
        self.returns = advantages + self.values
    
    def get_batch(self, batch_size: int):
        """Get a flattened batch of data."""
        # Flatten time and env dimensions: [T, N, ...] -> [T*N, ...]
        batch_size = self.num_steps * self.num_envs
        indices = torch.randperm(batch_size, device=self.obs.device)
        
        flat_obs = self.obs.reshape(-1, self.obs.shape[-1])
        flat_actions = self.actions.reshape(-1)
        flat_logprobs = self.logprobs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)
        
        return {
            'obs': flat_obs[indices],
            'actions': flat_actions[indices],
            'old_logprobs': flat_logprobs[indices],
            'advantages': flat_advantages[indices],
            'returns': flat_returns[indices],
            'old_values': flat_values[indices]
        }

def make_env(env_id: str, seed: int, idx: int):
    """Create a single environment."""
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        return env
    return thunk

class PPOTrainer:
    """PPO training class."""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # Set up device
        if config.device == "auto":
            self.device = device
        else:
            self.device = torch.device(config.device)
        
        # Create vectorized environments
        self.envs = gym.vector.SyncVectorEnv([
            make_env(config.env_id, config.seed, i) 
            for i in range(config.num_envs)
        ])
        
        # Get environment dimensions
        obs_space = self.envs.single_observation_space
        act_space = self.envs.single_action_space
        
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.n
        
        # Create actor-critic network
        self.agent = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
        
        # Create rollout buffer
        self.buffer = RolloutBuffer(config.num_steps, config.num_envs, self.obs_dim, self.device)
        
        # Tracking variables
        self.global_step = 0
        self.writer = SummaryWriter(f"runs/ppo_{config.env_id}_{int(time.time())}")
        
    def collect_rollouts(self):
        """Collect rollouts using current policy."""
        next_obs, _ = self.envs.reset(seed=self.config.seed)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.config.num_envs, device=self.device)
        
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            
            # Get action from current policy
            with torch.no_grad():
                action, logprob, value = self.agent.get_action_and_value(next_obs)
                value = value.flatten()
            
            # Take environment step
            obs = next_obs
            next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            next_done = torch.tensor(done, dtype=torch.float32, device=self.device)
            
            # Store in buffer
            self.buffer.add(step, obs, action, logprob, reward, next_done, value)
            
            # Log episode statistics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        print(f"global_step={self.global_step}, episode_reward={info['episode']['r']}")
                        self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                        self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
        
        # Compute advantages using GAE
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).flatten()
            self.buffer.compute_gae(next_value, self.config.gamma, self.config.gae_lambda)
        
        return next_obs, next_done
    
    def update_policy(self):
        """Update policy using PPO."""
        # Get flattened batch
        batch = self.buffer.get_batch(self.config.minibatch_size)
        
        # Normalize advantages
        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        clip_fracs = []
        
        for epoch in range(self.config.update_epochs):
            # Create minibatches
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
                
                # Compute ratios and surrogate losses
                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()
                
                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()
                    clip_fracs.append(clip_frac)
                
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
                loss = pg_loss + self.config.vf_coef * v_loss - self.config.ent_coef * entropy
                
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
        clipfrac_value = torch.stack(clip_fracs).mean().item() if clip_fracs else 0.0
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", clipfrac_value, self.global_step)
        
        return {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': clipfrac_value
        }
    
    def train(self):
        """Main training loop."""
        print("Starting PPO training...")
        print(f"Device: {self.device}")
        print(f"Environment: {self.config.env_id}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        
        num_updates = self.config.total_timesteps // (self.config.num_steps * self.config.num_envs)
        
        start_time = time.time()
        
        for update in range(1, num_updates + 1):
            # Collect rollouts
            self.collect_rollouts()
            
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
                print(f"  Approx KL: {update_info['approx_kl']:.4f}")
                print(f"  Clip fraction: {update_info['clipfrac']:.4f}")
                print()
        
        self.envs.close()
        self.writer.close()
        
        print("Training completed!")

def run_ppo_experiment():
    """Run PPO experiment with default configuration."""
    config = PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=100_000,
        num_envs=4,
        num_steps=128,
        learning_rate=2.5e-4,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01
    )
    
    trainer = PPOTrainer(config)
    trainer.train()
    
    return trainer

def compare_clipping_values():
    """Compare different clipping values."""
    print("Comparing different clipping values...")
    
    clip_values = [0.1, 0.2, 0.3]
    results = {}
    
    for clip_coef in clip_values:
        print(f"\nTesting clip coefficient: {clip_coef}")
        
        config = PPOConfig(
            env_id="CartPole-v1",
            total_timesteps=50_000,  # Shorter for comparison
            num_envs=4,
            num_steps=128,
            clip_coef=clip_coef,
            seed=42
        )
        
        trainer = PPOTrainer(config)
        trainer.train()
        
        results[clip_coef] = trainer
    
    return results

def main():
    print("="*60)
    print("PPO Core Implementation")
    print("="*60)
    
    # Run main PPO experiment
    trainer = run_ppo_experiment()
    
    print("\nExperiment completed!")
    print("Check TensorBoard logs for training curves.")
    print("Next: exp05_gae_analysis.py")

if __name__ == "__main__":
    main()
