#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 07 - Advanced DQN Variants

This experiment implements and compares advanced DQN variants including
Double DQN, Dueling DQN, and combinations thereof.

Learning objectives:
- Implement Double DQN to reduce overestimation bias
- Implement Dueling DQN architecture
- Compare performance of different variants
- Understand the benefits of each improvement

Prerequisites: Completed exp01-exp06
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


class VanillaDQN(nn.Module):
    """Standard DQN network"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDQN(nn.Module):
    """Dueling DQN with separate value and advantage streams"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Shared feature layers
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine streams (subtracting mean for stability)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                action,
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(done),
            )
        )
    
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

class DQNAgent:
    """Base DQN agent with configurable variants"""
    def __init__(self, state_dim, action_dim, config):
        self.action_dim = action_dim
        self.config = config
        
        # Select network architecture
        if config['dueling']:
            self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
            self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        else:
            self.policy_net = VanillaDQN(state_dim, action_dim).to(device)
            self.target_net = VanillaDQN(state_dim, action_dim).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['lr'])
        self.memory = ReplayBuffer(config['buffer_size'])
        
        self.epsilon = config['eps_start']
        self.steps = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self):
        if len(self.memory) < self.config['batch_size']:
            return None
        
        batch = self.memory.sample(self.config['batch_size'])
        states, actions, rewards, next_states, dones = batch
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            if self.config['double']:
                # Double DQN: use policy network to select actions
                next_actions = self.policy_net(next_states).argmax(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_net(next_states).max(1)[0]
            
            target_q_values = rewards + self.config['gamma'] * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        self.steps += 1
        
        # Update target network
        if self.steps % self.config['target_update'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.config['eps_end'], self.epsilon * self.config['eps_decay'])
        
        return loss.item()

def train_agent(agent_config, env_seed=42, num_episodes=200, verbose=False):
    """Train a DQN agent with given configuration"""
    setup_seed(env_seed)
    
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, agent_config)
    
    episode_rewards = []
    losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=env_seed + episode)
        episode_reward = 0
        episode_losses = []
        
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        if verbose and episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            print(f"  Episode {episode}: Avg Reward = {avg_reward:.1f}")
    
    env.close()
    return episode_rewards, losses

def main():
    print("="*50)
    print("Experiment 07: Advanced DQN Variants")
    print("="*50)
    
    # Base configuration
    base_config = {
        'lr': 1e-4,
        'batch_size': 64,
        'gamma': 0.99,
        'buffer_size': 50000,
        'eps_start': 1.0,
        'eps_end': 0.02,
        'eps_decay': 0.995,
        'target_update': 10,
        'double': False,
        'dueling': False
    }
    
    # Configurations for different variants
    variants = {
        'Vanilla DQN': {'double': False, 'dueling': False},
        'Double DQN': {'double': True, 'dueling': False},
        'Dueling DQN': {'double': False, 'dueling': True},
        'Double Dueling DQN': {'double': True, 'dueling': True}
    }
    
    training_episodes = 400

    print("\nTraining different DQN variants...")
    print(f"Each variant will be trained for {training_episodes} episodes")
    
    results = {}
    
    for name, variant_config in variants.items():
        print(f"\n" + "="*50)
        print(f"Training: {name}")
        print("="*50)
        
        config = base_config.copy()
        config.update(variant_config)
        
        print(f"Configuration: Double={config['double']}, Dueling={config['dueling']}")
        
        # Train with 3 different seeds for robustness
        all_rewards = []
        all_losses = []
        
        for seed in [42, 123, 456]:
            rewards, losses = train_agent(config, env_seed=seed, num_episodes=training_episodes)
            all_rewards.append(rewards)
            all_losses.append(losses)
            
            final_avg = np.mean(rewards[-50:])
            print(f"  Seed {seed}: Final avg reward = {final_avg:.1f}")
        
        results[name] = {
            'rewards': all_rewards,
            'losses': all_losses,
            'config': config
        }
    
    # Analysis
    print("\n" + "="*50)
    print("Performance Comparison")
    print("="*50)
    
    print("\n   Variant            | Mean Final | Std Final | Best | Convergence")
    print("   " + "-"*65)
    
    for name, result in results.items():
        # Calculate statistics across seeds
        final_rewards = [np.mean(rewards[-50:]) for rewards in result['rewards']]
        mean_final = np.mean(final_rewards)
        std_final = np.std(final_rewards)
        
        best_rewards = [max(rewards) for rewards in result['rewards']]
        best = max(best_rewards)
        
        # Check convergence (episodes reaching 195+)
        convergence_rates = []
        for rewards in result['rewards']:
            converged = sum(1 for r in rewards if r >= 195) / len(rewards)
            convergence_rates.append(converged)
        mean_convergence = np.mean(convergence_rates) * 100
        
        print(f"   {name:18s} | {mean_final:10.1f} | {std_final:9.1f} | {best:4.0f} | {mean_convergence:10.1f}%")
    
    # Overestimation bias analysis
    print("\n" + "="*50)
    print("Overestimation Bias Analysis")
    print("="*50)
    
    print("\nComparing Q-value estimates between Vanilla and Double DQN:")
    
    # Train small models to analyze Q-values
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    vanilla_agent = DQNAgent(state_dim, action_dim, {**base_config, 'double': False, 'dueling': False})
    double_agent = DQNAgent(state_dim, action_dim, {**base_config, 'double': True, 'dueling': False})
    
    # Collect some experiences
    for _ in range(100):
        state, _ = env.reset()
        for _ in range(20):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            vanilla_agent.memory.push(state, action, reward, next_state, done)
            double_agent.memory.push(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
    
    # Train briefly
    for _ in range(50):
        vanilla_agent.update()
        double_agent.update()
    
    # Compare Q-value estimates
    test_states = torch.randn(100, state_dim, device=device)
    
    with torch.no_grad():
        vanilla_q = vanilla_agent.policy_net(test_states)
        double_q = double_agent.policy_net(test_states)
        
        vanilla_max = vanilla_q.max(1)[0].mean().item()
        double_max = double_q.max(1)[0].mean().item()
        
        print(f"\nAverage max Q-value:")
        print(f"  Vanilla DQN: {vanilla_max:.4f}")
        print(f"  Double DQN:  {double_max:.4f}")
        print(f"  Reduction:   {(vanilla_max - double_max):.4f} ({(vanilla_max - double_max)/vanilla_max*100:.1f}%)")
    
    # Visualize learning curves
    print("\n" + "="*50)
    print("Learning Curves Visualization")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for idx, (name, result) in enumerate(results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Plot all seeds
        for seed_idx, rewards in enumerate(result['rewards']):
            # Smooth with moving average
            window = 10
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, alpha=0.3, label=f'Seed {[42, 123, 456][seed_idx]}')
        
        # Plot mean
        mean_rewards = np.mean(result['rewards'], axis=0)
        smoothed_mean = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed_mean, linewidth=2, label='Mean', color='black')
        
        ax.axhline(y=195, color='r', linestyle='--', alpha=0.5, label='Solved')
        ax.set_title(name)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / 'advanced_dqn_comparison.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()
    
    # Key insights
    print("\n" + "="*50)
    print("Key Insights")
    print("="*50)
    
    print("\n1. Double DQN:")
    print("   - Reduces overestimation bias in Q-values")
    print("   - More stable learning, especially in later episodes")
    print("   - May learn slightly slower initially")
    
    print("\n2. Dueling DQN:")
    print("   - Separates value and advantage estimation")
    print("   - Better generalization across actions")
    print("   - Particularly effective when many actions have similar values")
    
    print("\n3. Double Dueling DQN:")
    print("   - Combines benefits of both approaches")
    print("   - Most robust performance across seeds")
    print("   - Best for complex environments")
    
    print("\n4. Performance Summary:")
    best_variant = max(results.items(), 
                       key=lambda x: np.mean([np.mean(r[-50:]) for r in x[1]['rewards']]))
    print(f"   Best overall: {best_variant[0]}")
    
    print("\n" + "="*50)
    print("Advanced DQN variants experiment complete!")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
