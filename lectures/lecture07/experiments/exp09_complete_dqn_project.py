#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 09 - Complete DQN Project Integration

This experiment integrates all components from previous experiments into a
production-ready DQN implementation with logging, checkpointing, and evaluation.

Learning objectives:
- Build a complete, production-ready DQN implementation
- Integrate all best practices and improvements
- Implement proper logging and checkpointing
- Create a reusable DQN framework

Prerequisites: Completed exp01-exp08
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Literal
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

BOOL_DTYPE = np.bool8 if hasattr(np, "bool8") else np.bool_

# Standard setup for reproducibility
def setup_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# Device selection
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)


def get_cuda_rng_state_all_safe():
    """Return CUDA RNG state if available, guarding against CPU-only builds."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_rng_state_all()
    except RuntimeError:
        return None

@dataclass
class DQNConfig:
    """Complete configuration for DQN training"""
    # Environment
    env_name: str = "CartPole-v1"
    seed: int = 42
    
    # Network
    hidden_dim: int = 128
    dueling: bool = True
    double: bool = True
    
    # Training
    total_steps: int = 100000
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    grad_clip: float = 10.0
    
    # Replay buffer
    buffer_size: int = 50000
    warmup_steps: int = 1000
    
    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 50000
    eps_schedule: Literal["linear", "exponential"] = "linear"
    
    # Target network
    target_update_every: int = 1000
    soft_update: bool = False
    tau: float = 0.005
    
    # Logging
    log_every: int = 1000
    eval_every: int = 5000
    save_every: int = 10000
    eval_episodes: int = 10
    
    # Paths
    log_dir: str = "runs/dqn_project"
    checkpoint_dir: str = "checkpoints"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

class DQNNetwork(nn.Module):
    """Flexible DQN network supporting vanilla and dueling architectures"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        
        if dueling:
            # Shared feature extraction
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            # Value stream
            self.value = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # Advantage stream
            self.advantage = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        else:
            # Standard DQN
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            features = self.feature(x)
            value = self.value(features)
            advantage = self.advantage(features)
            # Combine with advantage mean subtraction for stability
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.net(x)
        return q_values

class ReplayBuffer:
    """Efficient replay buffer implementation"""
    def __init__(self, state_dim: int, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.full = False
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=BOOL_DTYPE)
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0
    
    def sample(self, batch_size: int):
        max_idx = self.capacity if self.full else self.pos
        idx = np.random.randint(0, max_idx, size=batch_size)
        
        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.dones[idx].astype(np.float32)).to(self.device)
        )
    
    def __len__(self):
        return self.capacity if self.full else self.pos

class DQNAgent:
    """Complete DQN agent with all features"""
    def __init__(self, config: DQNConfig):
        self.config = config
        setup_seed(config.seed)
        
        # Environment
        self.env = gym.make(config.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Networks
        self.policy_net = DQNNetwork(
            self.state_dim, self.action_dim, 
            config.hidden_dim, config.dueling
        ).to(device)
        
        self.target_net = DQNNetwork(
            self.state_dim, self.action_dim,
            config.hidden_dim, config.dueling
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(self.state_dim, config.buffer_size, device)
        
        # Logging
        self.setup_logging()
        
        # Metrics
        self.global_step = 0
        self.episode = 0
        self.best_eval_reward = -float('inf')
        
    def setup_logging(self):
        """Setup TensorBoard logging and checkpoint directory"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config_hash = self.config.get_hash()
        
        self.log_dir = Path(self.config.log_dir) / f"{timestamp}_{config_hash}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Save configuration
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        print(f"Logging to: {self.log_dir}")
    
    def get_epsilon(self) -> float:
        """Calculate current epsilon value"""
        if self.config.eps_schedule == "linear":
            progress = min(1.0, self.global_step / self.config.eps_decay_steps)
            return self.config.eps_start + progress * (self.config.eps_end - self.config.eps_start)
        else:  # exponential
            decay_rate = -np.log(self.config.eps_end / self.config.eps_start) / self.config.eps_decay_steps
            return self.config.eps_start * np.exp(-decay_rate * self.global_step)
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self) -> Optional[float]:
        """Perform one gradient update"""
        if len(self.memory) < self.config.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            if self.config.double:
                # Double DQN: use policy network to select actions
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_net(next_states).max(1)[0]
            
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        
        # Update target network
        if self.config.soft_update:
            # Soft update
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    self.config.tau * policy_param.data + (1 - self.config.tau) * target_param.data
                )
        elif self.global_step % self.config.target_update_every == 0:
            # Hard update
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def evaluate(self, episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent"""
        eval_rewards = []
        eval_lengths = []
        
        for ep in range(episodes):
            state, _ = self.env.reset(seed=self.config.seed + 1000 + ep)
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.select_action(state, epsilon=0.0)  # Greedy
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            'reward_mean': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
            'length_mean': np.mean(eval_lengths),
            'length_std': np.std(eval_lengths)
        }
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save model checkpoint"""
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        
        checkpoint = {
            'global_step': self.global_step,
            'episode': self.episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_eval_reward': self.best_eval_reward,
            'rng_states': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': get_cuda_rng_state_all_safe(),
            },
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        self.global_step = checkpoint['global_step']
        self.episode = checkpoint['episode']
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_eval_reward = checkpoint.get('best_eval_reward', -float('inf'))
        # Restore RNG states if present
        rng = checkpoint.get('rng_states')
        if rng is not None:
            try:
                random.setstate(rng['python'])
                np.random.set_state(rng['numpy'])
                if rng.get('torch') is not None:
                    torch.set_rng_state(rng['torch'])
                if torch.cuda.is_available() and rng.get('cuda') is not None:
                    try:
                        torch.cuda.set_rng_state_all(rng['cuda'])
                    except RuntimeError as err:
                        print(f"Warning: Unable to restore CUDA RNG state: {err}")
            except Exception as e:
                print(f"Warning: Failed to restore RNG states: {e}")
        
        print(f"Checkpoint loaded from: {path}")
        print(f"Resuming from step {self.global_step}, episode {self.episode}")
    
    def train(self):
        """Main training loop"""
        print("="*50)
        print("Starting DQN Training")
        print("="*50)
        print(f"Environment: {self.config.env_name}")
        print(f"Device: {device}")
        print(f"Total steps: {self.config.total_steps}")
        print(f"Architecture: {'Dueling' if self.config.dueling else 'Standard'} "
              f"{'Double' if self.config.double else ''} DQN")
        print("="*50)
        
        state, _ = self.env.reset(seed=self.config.seed)
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        start_time = time.time()
        
        while self.global_step < self.config.total_steps:
            # Select and perform action
            epsilon = self.get_epsilon()
            action = self.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.memory.add(state, action, reward, next_state, done)
            
            # Update
            loss = self.update()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.global_step += 1
            
            # Episode end
            if done:
                # Log episode metrics
                self.writer.add_scalar('episode/reward', episode_reward, self.global_step)
                self.writer.add_scalar('episode/length', episode_length, self.global_step)
                if episode_losses:
                    self.writer.add_scalar('episode/loss', np.mean(episode_losses), self.global_step)
                self.writer.add_scalar('train/epsilon', epsilon, self.global_step)
                
                # Reset episode
                state, _ = self.env.reset(seed=self.config.seed + self.episode)
                self.episode += 1
                episode_reward = 0
                episode_length = 0
                episode_losses = []
            
            # Periodic logging
            if self.global_step % self.config.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed
                print(f"Step {self.global_step}/{self.config.total_steps} | "
                      f"Episode {self.episode} | "
                      f"Epsilon {epsilon:.3f} | "
                      f"Buffer {len(self.memory)} | "
                      f"Speed {steps_per_sec:.1f} steps/s")
            
            # Periodic evaluation
            if self.global_step % self.config.eval_every == 0:
                eval_metrics = self.evaluate(self.config.eval_episodes)
                
                self.writer.add_scalar('eval/reward_mean', eval_metrics['reward_mean'], self.global_step)
                self.writer.add_scalar('eval/reward_std', eval_metrics['reward_std'], self.global_step)
                self.writer.add_scalar('eval/length_mean', eval_metrics['length_mean'], self.global_step)
                
                print(f"Evaluation at step {self.global_step}: "
                      f"Reward {eval_metrics['reward_mean']:.1f} ± {eval_metrics['reward_std']:.1f}")
                
                # Save best model
                if eval_metrics['reward_mean'] > self.best_eval_reward:
                    self.best_eval_reward = eval_metrics['reward_mean']
                    self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
                    print(f"New best model! Reward: {self.best_eval_reward:.1f}")
            
            # Periodic checkpointing
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()
        
        # Final evaluation
        print("\n" + "="*50)
        print("Training Complete - Final Evaluation")
        print("="*50)
        
        final_metrics = self.evaluate(20)
        print(f"Final performance: {final_metrics['reward_mean']:.1f} ± {final_metrics['reward_std']:.1f}")
        
        # Save final model
        self.save_checkpoint(self.checkpoint_dir / "final_model.pt")
        
        # Close resources
        self.writer.close()
        self.env.close()
        
        print(f"\nTraining summary:")
        print(f"  Total steps: {self.global_step}")
        print(f"  Total episodes: {self.episode}")
        print(f"  Best eval reward: {self.best_eval_reward:.1f}")
        print(f"  Training time: {(time.time() - start_time) / 60:.1f} minutes")
        print(f"  Logs saved to: {self.log_dir}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Complete DQN Project')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = DQNConfig(**config_dict)
    else:
        config = DQNConfig()
    
    # Create agent
    agent = DQNAgent(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        agent.load_checkpoint(Path(args.checkpoint))
    
    # Train
    agent.train()
    
    print("\n" + "="*50)
    print("Complete DQN project execution finished!")
    print("="*50)

if __name__ == "__main__":
    main()
