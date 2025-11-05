#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 05 - DQN Training Loop

This experiment implements a complete DQN training loop with proper
epsilon decay, target updates, and loss computation.

Learning objectives:
- Implement the main DQN training algorithm
- Handle epsilon-greedy exploration schedule
- Perform gradient updates with Huber loss
- Monitor training progress and convergence

Prerequisites: Completed exp01-exp04
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# Standard setup
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR = Path(os.environ.get("LECTURE07_FIGURES_DIR", DEFAULT_FIGURES_DIR))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state), dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.long, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_state), dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device)
        )
    
    def __len__(self):
        return len(self.buffer)

class RunningMeanStd:
    """Track running mean/variance for observation normalization."""
    def __init__(self, shape, epsilon=1e-4, device=None):
        self.device = device or torch.device('cpu')
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def update(self, x):
        tensor = self._to_tensor(x)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        batch_mean = tensor.mean(dim=0)
        batch_var = tensor.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(tensor.shape[0]), dtype=torch.float32, device=self.device)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = torch.clamp(m2 / total_count, min=1e-6)
        self.count = total_count

    def normalize(self, x):
        tensor = self._to_tensor(x)
        return (tensor - self.mean) / (self.var.sqrt() + 1e-8)


def select_action(normalized_state, policy_net, epsilon, action_dim):
    """Epsilon-greedy action selection operating on normalized states"""
    if random.random() < epsilon:
        return random.randrange(action_dim)
    
    with torch.no_grad():
        q_values = policy_net(normalized_state.unsqueeze(0))
        return q_values.argmax(dim=1).item()

def compute_loss(batch, policy_net, target_net, gamma):
    """Compute DQN loss using Huber loss"""
    states, actions, rewards, next_states, dones = batch
    
    # Current Q values
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    # Huber loss
    loss = F.smooth_l1_loss(current_q_values, target_q_values)
    return loss

def main():
    print("="*50)
    print("Experiment 05: DQN Training Loop")
    print("="*50)
    
    setup_seed(42)
    
    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.995
    TARGET_UPDATE = 10
    MEMORY_SIZE = 10000
    LR = 1e-3
    NUM_EPISODES = 200
    
    print("\nHyperparameters:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Epsilon: {EPS_START} -> {EPS_END}")
    print(f"  Learning rate: {LR}")
    print(f"  Memory size: {MEMORY_SIZE}")
    print(f"  Target update: every {TARGET_UPDATE} episodes")
    
    # Environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    obs_rms = RunningMeanStd(state_dim, device=device)
    
    # Networks
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Optimizer and memory
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilon_values = []
    
    epsilon = EPS_START
    
    print("\nStarting training...")
    print("\nEpisode | Reward | Length | Epsilon | Loss    | Buffer")
    print("-"*60)
    
    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=42 + episode)
        obs_rms.update(state)
        normalized_state = obs_rms.normalize(state)
        episode_reward = 0
        episode_length = 0
        episode_loss = []
        
        done = False
        while not done:
            # Select action
            action = select_action(normalized_state, policy_net, epsilon, action_dim)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            obs_rms.update(next_state)
            normalized_next_state = obs_rms.normalize(next_state)
            memory.push(
                normalized_state.detach().cpu().numpy(),
                action,
                reward,
                normalized_next_state.detach().cpu().numpy(),
                done,
            )
            
            # Update state
            state = next_state
            normalized_state = normalized_next_state
            episode_reward += reward
            episode_length += 1
            
            # Train if enough samples
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                loss = compute_loss(batch, policy_net, target_net, GAMMA)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                
                optimizer.step()
                episode_loss.append(loss.item())
        
        # Update metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilon_values.append(epsilon)
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        
        # Print progress
        if episode % 10 == 0:
            print(f"{episode:7d} | {episode_reward:6.0f} | {episode_length:6d} | "
                  f"{epsilon:7.4f} | {avg_loss:7.4f} | {len(memory):6d}")
    
    print("\nTraining complete!")
    
    # Evaluation
    print("\n" + "="*50)
    print("Final Evaluation (10 episodes, epsilon=0):")
    print("="*50)
    
    eval_rewards = []
    policy_net.eval()
    
    for eval_episode in range(10):
        state, _ = env.reset(seed=1000 + eval_episode)
        episode_reward = 0
        done = False
        
        while not done:
            normalized_state = obs_rms.normalize(state)
            action = select_action(normalized_state, policy_net, epsilon=0.0, action_dim=action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
        print(f"  Episode {eval_episode + 1}: {episode_reward:.0f}")
    
    print(f"\nMean reward: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    
    # Analysis
    print("\n" + "="*50)
    print("Training Analysis:")
    print("="*50)
    
    # Compute moving averages
    window = 10
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    
    print(f"\n1. Performance Improvement:")
    print(f"   First {window} episodes avg: {np.mean(episode_rewards[:window]):.1f}")
    print(f"   Last {window} episodes avg: {np.mean(episode_rewards[-window:]):.1f}")
    print(f"   Best episode: {max(episode_rewards):.0f} (Episode {np.argmax(episode_rewards)})")
    
    print(f"\n2. Exploration vs Exploitation:")
    print(f"   Initial epsilon: {EPS_START}")
    print(f"   Final epsilon: {epsilon_values[-1]:.4f}")
    print(f"   Episodes to epsilon < 0.1: {np.argmax(np.array(epsilon_values) < 0.1)}")
    
    print(f"\n3. Learning Stability:")
    print(f"   Mean loss: {np.mean(losses[50:]):.4f}")  # Skip initial high losses
    print(f"   Loss std: {np.std(losses[50:]):.4f}")
    
    # Check for convergence
    last_20_rewards = episode_rewards[-20:]
    if min(last_20_rewards) >= 195:
        print(f"\n4. Convergence: SOLVED! (20 consecutive episodes with reward >= 195)")
    else:
        print(f"\n4. Convergence: Not fully solved (Best streak: {max(last_20_rewards):.0f})")
    
    # Visualize results
    if True:  # Set to True to generate plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(episode_rewards, alpha=0.3, label='Raw')
        if len(moving_avg) > 0:
            axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                          label=f'{window}-episode average', linewidth=2)
        axes[0, 0].axhline(y=195, color='r', linestyle='--', label='Solved threshold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(losses)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epsilon decay
        axes[1, 0].plot(epsilon_values)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Episode length
        axes[1, 1].plot(episode_lengths)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Episode Length')
        axes[1, 1].set_title('Episode Duration')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = FIGURES_DIR / 'dqn_training_results.png'
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\n5. Visualization saved to: {save_path}")
        plt.close()
    
    print("\n" + "="*50)
    print("DQN training loop experiment complete!")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
