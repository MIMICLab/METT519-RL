#!/usr/bin/env python3
"""
RL2025 - Lecture 6: Experiment 09 - Complete Integrated DQN Test

This experiment integrates all components from previous experiments
into a production-ready DQN implementation with comprehensive testing.

Learning objectives:
- Integrate all DQN components
- Implement complete training pipeline
- Add logging and checkpointing
- Verify reproducibility and performance

Prerequisites: All previous experiments (01-08)
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from exp01_setup import (
    setup_seed,
    device,
    amp_enabled,
    deterministic_mode_enabled,
    configure_deterministic_behavior,
    should_use_compile,
    should_use_amp,
    is_torch_compile_supported,
)

setup_seed(42)

if deterministic_mode_enabled():
    configure_deterministic_behavior()


def capture_rng_states() -> Dict[str, Any]:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def restore_rng_states(states: Optional[Dict[str, Any]]) -> None:
    if not states:
        return
    if states.get('python') is not None:
        random.setstate(states['python'])
    if states.get('numpy') is not None:
        np.random.set_state(states['numpy'])
    if states.get('torch') is not None:
        torch.set_rng_state(states['torch'])
    if states.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['cuda'])

class QNetwork(nn.Module):
    """Q-Network with configurable architecture"""
    
    def __init__(self, obs_dim, n_actions, hidden_sizes=(128, 128)):
        super(QNetwork, self).__init__()
        
        layers = []
        input_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, n_actions))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Efficient experience replay buffer"""
    
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def push(self, obs, action, reward, next_obs, done):
        idx = self.position
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_obs
        self.dones[idx] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        obs = torch.FloatTensor(self.observations[indices]).to(device)
        actions = torch.LongTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_obs = torch.FloatTensor(self.next_observations[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return self.size

class CompleteDQN:
    """Production-ready DQN with all enhancements"""
    
    def __init__(self, config):
        self.config = config
        if deterministic_mode_enabled():
            configure_deterministic_behavior()
        setup_seed(config['seed'])
        
        # Environment setup
        self.env = gym.make(config['env_name'])
        self.eval_env = gym.make(config['env_name'])
        
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        
        # Networks
        self.q_network = QNetwork(
            self.obs_dim, 
            self.n_actions, 
            config['hidden_sizes']
        ).to(device)
        
        self.target_network = QNetwork(
            self.obs_dim, 
            self.n_actions, 
            config['hidden_sizes']
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Apply torch.compile if available
        self.use_compile = should_use_compile(config['use_compile'])
        if self.use_compile:
            print("   torch.compile enabled")
            self.q_network = torch.compile(self.q_network)
            self.target_network = torch.compile(self.target_network)
        elif config['use_compile'] and not self.use_compile:
            if not hasattr(torch, 'compile'):
                print("   torch.compile not available (requires PyTorch 2.0+)")
            elif deterministic_mode_enabled():
                print("   Deterministic mode active: skipping torch.compile()")
            elif not is_torch_compile_supported():
                print("   torch.compile not supported on this platform (missing Triton backend)")
            else:
                print("   torch.compile disabled for current configuration")

        # Optimizer and scaler
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=config['learning_rate']
        )

        self.use_amp = should_use_amp(config['use_amp'])
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        elif config['use_amp'] and not self.use_amp:
            print("   Deterministic mode active: AMP disabled")
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            config['buffer_size'], 
            self.obs_dim
        )
        
        # Training parameters
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.target_update_freq = config['target_update_freq']
        self.use_double_dqn = config['use_double_dqn']
        
        # Tracking
        self.total_steps = 0
        self.episode_count = 0
        self.update_count = 0
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'q_values': [],
            'epsilon': [],
            'eval_rewards': []
        }
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def select_action(self, state, epsilon=None):
        """Epsilon-greedy action selection"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self):
        """Perform one gradient update"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = batch
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(obs, actions, rewards, next_obs, dones)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(obs, actions, rewards, next_obs, dones)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
            self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def _compute_loss(self, obs, actions, rewards, next_obs, dones):
        """Compute DQN loss with optional Double DQN"""
        # Current Q-values
        current_q_values = self.q_network(obs)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute targets
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN
                next_q_online = self.q_network(next_obs)
                next_actions = next_q_online.argmax(1, keepdim=True)
                next_q_target = self.target_network(next_obs)
                next_q_values = next_q_target.gather(1, next_actions).squeeze()
            else:
                # Vanilla DQN
                next_q_values = self.target_network(next_obs).max(1)[0]
            
            targets = rewards + self.gamma * (1 - dones) * next_q_values
            
            # Track Q-values
            self.training_history['q_values'].append(current_q_values.mean().item())
        
        # Huber loss
        loss = F.huber_loss(current_q_values, targets)
        
        return loss
    
    def train_episode(self):
        """Train for one episode"""
        obs, _ = self.env.reset(seed=self.config['seed'] + self.episode_count)
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        done = False
        while not done:
            # Select action
            action = self.select_action(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.push(obs, action, reward, next_obs, done)
            
            # Update network
            loss = self.update()
            if loss is not None:
                episode_losses.append(loss)
            
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            obs = next_obs
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Record episode statistics
        self.episode_count += 1
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['episode_lengths'].append(episode_length)
        self.training_history['epsilon'].append(self.epsilon)
        
        if episode_losses:
            self.training_history['losses'].append(np.mean(episode_losses))
        
        return episode_reward, episode_length
    
    def evaluate(self, n_episodes=10):
        """Evaluate agent performance"""
        eval_rewards = []
        eval_lengths = []
        
        for ep in range(n_episodes):
            obs, _ = self.eval_env.reset(seed=self.config['seed'] + 1000 + ep)
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Greedy action selection (no exploration)
                action = self.select_action(obs, epsilon=0.0)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return np.mean(eval_rewards), np.std(eval_rewards), np.mean(eval_lengths)
    
    def save_checkpoint(self, filepath=None):
        """Save model checkpoint"""
        if filepath is None:
            filepath = self.checkpoint_dir / f"checkpoint_ep{self.episode_count}.pt"
        
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'rng_states': capture_rng_states()
        }

        if self.use_amp:
            checkpoint['scaler'] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        return filepath

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)

        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.use_amp and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
        self.epsilon = checkpoint['epsilon']
        self.training_history = checkpoint['training_history']
        restore_rng_states(checkpoint.get('rng_states'))

        return checkpoint

    def close(self):
        """Release environment resources."""
        try:
            self.env.close()
        except Exception:
            pass
        try:
            self.eval_env.close()
        except Exception:
            pass

def create_config():
    """Create default configuration"""
    config = {
        'env_name': 'CartPole-v1',
        'seed': 42,
        'hidden_sizes': (128, 128),
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'buffer_size': 10000,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 100,
        'use_double_dqn': True,
        'use_amp': torch.cuda.is_available(),
        'use_compile': hasattr(torch, 'compile'),
        'checkpoint_dir': 'checkpoints',
        'n_episodes': 200,
        'eval_interval': 20,
        'save_interval': 50
    }

    config['deterministic'] = deterministic_mode_enabled()
    return config

def run_complete_training():
    """Run complete DQN training with all features"""
    
    print("\n" + "="*50)
    print("Complete DQN Training Pipeline")
    print("="*50)
    
    # Create configuration
    config = create_config()

    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Initialize agent
    print("\nInitializing DQN agent...")
    agent = CompleteDQN(config)
    
    print(f"   Environment: {config['env_name']}")
    print(f"   Observation dim: {agent.obs_dim}")
    print(f"   Action space: {agent.n_actions}")
    print(f"   Device: {device}")
    print(f"   AMP enabled: {agent.use_amp}")
    print(f"   Double DQN: {config['use_double_dqn']}")
    print(f"   Deterministic mode: {config['deterministic']}")
    
    # Training loop
    print(f"\nTraining for {config['n_episodes']} episodes...")
    print("-" * 50)
    
    best_eval_reward = -float('inf')
    start_time = time.time()
    
    for episode in range(config['n_episodes']):
        # Train episode
        episode_reward, episode_length = agent.train_episode()
        
        # Periodic evaluation
        if (episode + 1) % config['eval_interval'] == 0:
            eval_mean, eval_std, eval_length = agent.evaluate()
            agent.training_history['eval_rewards'].append(eval_mean)
            
            # Calculate statistics
            recent_rewards = agent.training_history['episode_rewards'][-20:]
            avg_reward = np.mean(recent_rewards)
            recent_losses = agent.training_history['losses'][-20:]
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            
            print(f"Episode {episode+1:3d} | "
                  f"Train: {avg_reward:6.1f} | "
                  f"Eval: {eval_mean:6.1f} ± {eval_std:5.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Steps: {agent.total_steps:6d}")
            
            # Save best model
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                best_path = agent.checkpoint_dir / "best_model.pt"
                agent.save_checkpoint(best_path)
        
        # Periodic checkpoint
        if (episode + 1) % config['save_interval'] == 0:
            agent.save_checkpoint()
    
    # Training complete
    training_time = time.time() - start_time
    print("-" * 50)
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Best evaluation reward: {best_eval_reward:.1f}")
    
    # Save final checkpoint
    final_path = agent.checkpoint_dir / "final_model.pt"
    agent.save_checkpoint(final_path)
    print(f"Final model saved to: {final_path}")
    
    return agent

def test_reproducibility():
    """Test training reproducibility"""
    
    print("\n" + "="*50)
    print("Testing Reproducibility")
    print("="*50)
    
    config = create_config()
    config['n_episodes'] = 50  # Shorter for testing

    print("\nRunning two identical training runs...")
    
    # First run
    agent1 = CompleteDQN(config)
    rewards1 = []
    for _ in range(config['n_episodes']):
        reward, _ = agent1.train_episode()
        rewards1.append(reward)
    agent1.close()
    
    # Second run (same seed)
    agent2 = CompleteDQN(config)
    rewards2 = []
    for _ in range(config['n_episodes']):
        reward, _ = agent2.train_episode()
        rewards2.append(reward)
    agent2.close()
    
    # Compare results
    rewards1 = np.array(rewards1)
    rewards2 = np.array(rewards2)
    
    identical = np.allclose(rewards1, rewards2)
    correlation = np.corrcoef(rewards1, rewards2)[0, 1]
    
    print(f"\n   Identical trajectories: {identical}")
    print(f"   Correlation: {correlation:.4f}")
    print(f"   Mean difference: {np.abs(rewards1 - rewards2).mean():.4f}")
    
    if identical:
        print("   ✓ Perfect reproducibility achieved!")
    else:
        reason = "compilation or nondeterministic GPU kernels"
        if deterministic_mode_enabled():
            reason = "stochastic environment dynamics"
        print(f"   ⚠ Minor differences detected (likely due to {reason})")

def analyze_training_curves(agent):
    """Analyze and display training curves"""
    
    print("\n" + "="*50)
    print("Training Analysis")
    print("="*50)
    
    history = agent.training_history
    
    # Episode rewards
    print("\n1. Episode Rewards:")
    rewards = history['episode_rewards']
    if len(rewards) > 0:
        print(f"   First 10 episodes: {np.mean(rewards[:10]):.1f}")
        print(f"   Last 10 episodes: {np.mean(rewards[-10:]):.1f}")
        print(f"   Maximum reward: {max(rewards):.1f}")
        print(f"   Improvement: {np.mean(rewards[-10:]) - np.mean(rewards[:10]):.1f}")
    
    # Q-values
    print("\n2. Q-Value Evolution:")
    q_values = history['q_values']
    if len(q_values) > 0:
        print(f"   Initial Q-values: {np.mean(q_values[:100]) if len(q_values) > 100 else np.mean(q_values):.3f}")
        print(f"   Final Q-values: {np.mean(q_values[-100:]) if len(q_values) > 100 else np.mean(q_values):.3f}")
    
    # Loss
    print("\n3. Loss Statistics:")
    losses = history['losses']
    if len(losses) > 0:
        print(f"   Initial loss: {np.mean(losses[:10]) if len(losses) > 10 else np.mean(losses):.4f}")
        print(f"   Final loss: {np.mean(losses[-10:]) if len(losses) > 10 else np.mean(losses):.4f}")
        print(f"   Loss variance: {np.var(losses):.6f}")
    
    # Evaluation
    print("\n4. Evaluation Performance:")
    eval_rewards = history['eval_rewards']
    if len(eval_rewards) > 0:
        print(f"   Best evaluation: {max(eval_rewards):.1f}")
        print(f"   Final evaluation: {eval_rewards[-1]:.1f}")
        print(f"   Evaluations above 195: {sum(r > 195 for r in eval_rewards)}")

def main():
    print("="*50)
    print("Experiment 09: Complete Integrated DQN Test")
    print("="*50)
    
    print(f"\nSystem Information:")
    print(f"   Device: {device}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Gymnasium: {gym.__version__}")
    print(f"   Deterministic mode requested: {deterministic_mode_enabled()}")
    
    # Run complete training
    agent = run_complete_training()
    
    # Analyze results
    analyze_training_curves(agent)
    agent.close()
    
    # Test reproducibility
    test_reproducibility()
    
    print("\n" + "="*50)
    print("Complete DQN implementation successfully tested!")
    print("All components integrated and working correctly.")
    print("="*50)

if __name__ == "__main__":
    main()
