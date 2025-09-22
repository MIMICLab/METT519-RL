#!/usr/bin/env python3
"""
RL2025 - Lecture 6: Experiment 05 - Basic DQN Agent

This experiment implements a basic DQN agent combining Q-network
and experience replay, but without target networks yet.

Learning objectives:
- Combine Q-network with experience replay
- Implement DQN loss function
- Understand training loop structure
- Observe instability without target network

Prerequisites: Experiments 03 and 04 (Q-network and replay buffer)
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

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

class QNetwork(nn.Module):
    """Q-Network for value function approximation"""
    
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
    """Experience replay buffer"""
    
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

class BasicDQNAgent:
    """Basic DQN agent without target network"""
    
    def __init__(self, env, learning_rate=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32):
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        # Initialize Q-network
        self.q_network = QNetwork(self.obs_dim, self.n_actions).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, self.obs_dim)
        
        # Tracking
        self.losses = []
        self.episode_rewards = []
        self.q_values = []
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            self.q_values.append(q_values.max().item())  # Track max Q-value
            return q_values.argmax().item()
    
    def compute_loss(self, batch):
        """Compute DQN loss (without target network)"""
        obs, actions, rewards, next_obs, dones = batch
        
        # Current Q-values for taken actions
        current_q_values = self.q_network(obs)  # [batch_size, n_actions]
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()  # [batch_size]
        
        # Next Q-values (using same network - this causes instability!)
        with torch.no_grad():
            next_q_values = self.q_network(next_obs)  # [batch_size, n_actions]
            next_q_values = next_q_values.max(1)[0]  # [batch_size]
            
            # Compute targets
            targets = rewards + self.gamma * (1 - dones) * next_q_values
        
        # MSE loss
        loss = F.mse_loss(current_q_values, targets)
        
        return loss
    
    def update(self):
        """Perform one gradient update"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        
        self.optimizer.step()
        
        # Track loss
        self.losses.append(loss.item())
    
    def train_episode(self):
        """Train for one episode"""
        obs, _ = self.env.reset()
        episode_reward = 0
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
            self.update()
            
            episode_reward += reward
            obs = next_obs
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.episode_rewards.append(episode_reward)
        return episode_reward

def train_basic_dqn():
    """Train basic DQN and observe instability"""
    
    print("\n1. Training Basic DQN (No Target Network):")
    
    env = gym.make("CartPole-v1")
    agent = BasicDQNAgent(
        env,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32
    )
    
    n_episodes = 100
    print(f"   Training for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        reward = agent.train_episode()
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(agent.episode_rewards[-20:])
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
            avg_q = np.mean(agent.q_values[-100:]) if agent.q_values else 0
            
            print(f"   Episode {episode+1:3d}: "
                  f"Avg Reward = {avg_reward:6.1f}, "
                  f"Loss = {avg_loss:6.4f}, "
                  f"Avg Q = {avg_q:6.2f}, "
                  f"Epsilon = {agent.epsilon:.3f}")
    
    env.close()
    return agent

def analyze_instability(agent):
    """Analyze sources of instability in basic DQN"""
    
    print("\n2. Analyzing Training Instability:")
    
    # Plot learning curves
    print("\n   a) Episode Rewards (Moving Average):")
    window = 10
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards, 
                                 np.ones(window)/window, mode='valid')
        
        # ASCII plot
        max_reward = max(moving_avg)
        min_reward = min(moving_avg)
        range_reward = max_reward - min_reward if max_reward != min_reward else 1
        
        for i in range(0, len(moving_avg), 5):
            normalized = int(((moving_avg[i] - min_reward) / range_reward) * 20)
            bar = '#' * normalized
            print(f"      Episode {i:3d}: {bar} ({moving_avg[i]:.1f})")
    
    # Analyze Q-value evolution
    print("\n   b) Q-Value Statistics:")
    if agent.q_values:
        q_array = np.array(agent.q_values)
        print(f"      Initial Q-values (first 10): {q_array[:10].round(2)}")
        print(f"      Final Q-values (last 10): {q_array[-10:].round(2)}")
        print(f"      Max Q-value: {q_array.max():.2f}")
        print(f"      Mean Q-value: {q_array.mean():.2f}")
        print(f"      Std Q-value: {q_array.std():.2f}")
    
    # Loss analysis
    print("\n   c) Loss Evolution:")
    if agent.losses:
        losses = np.array(agent.losses)
        print(f"      Initial loss (first 10 avg): {losses[:10].mean():.4f}")
        print(f"      Final loss (last 100 avg): {losses[-100:].mean():.4f}")
        print(f"      Loss variance: {losses.var():.4f}")

def demonstrate_moving_target_problem():
    """Demonstrate the moving target problem"""
    
    print("\n3. Moving Target Problem Demonstration:")
    
    # Create a simple scenario
    q_net = QNetwork(4, 2, hidden_sizes=(32,)).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=0.01)
    
    # Fixed state and action
    state = torch.FloatTensor([[0.5, 0.5, 0.5, 0.5]]).to(device)
    action = torch.LongTensor([0]).to(device)
    reward = 1.0
    next_state = torch.FloatTensor([[0.6, 0.6, 0.6, 0.6]]).to(device)
    gamma = 0.99
    
    print("\n   Updating same network for target and prediction:")
    
    q_values_history = []
    target_history = []
    
    for step in range(20):
        # Current Q-value
        q_values = q_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze()
        
        # Target (using SAME network - problematic!)
        with torch.no_grad():
            next_q_values = q_net(next_state)
            next_q_max = next_q_values.max()
            target = reward + gamma * next_q_max
        
        q_values_history.append(q_value.item())
        target_history.append(target.item())
        
        # Loss and update
        loss = F.mse_loss(q_value, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"      Step {step:2d}: Q(s,a)={q_value.item():.3f}, "
                  f"Target={target.item():.3f}, Loss={loss.item():.4f}")
    
    print("\n   Problem: Target keeps changing as we update!")
    print(f"   Target drift: {target_history[0]:.3f} -> {target_history[-1]:.3f}")
    print("   This causes oscillations and instability in learning.")

def compare_loss_functions():
    """Compare MSE vs Huber loss for DQN"""
    
    print("\n4. Loss Function Comparison (MSE vs Huber):")
    
    # Create sample predictions and targets with outliers
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    targets = torch.tensor([1.1, 2.2, 3.1, 10.0, 5.2], device=device)  # 4th is outlier
    
    # MSE Loss
    mse_loss = F.mse_loss(predictions, targets, reduction='none')
    mse_total = mse_loss.mean()
    
    # Huber Loss
    huber_loss = F.huber_loss(predictions, targets, reduction='none', delta=1.0)
    huber_total = huber_loss.mean()
    
    print("\n   Individual losses:")
    print("   Sample | Target | Pred | MSE Loss | Huber Loss")
    print("   -------|--------|------|----------|------------")
    
    for i in range(len(predictions)):
        is_outlier = "*" if abs(targets[i] - predictions[i]) > 2 else " "
        print(f"     {i+1}{is_outlier}   |  {targets[i]:.1f}  | {predictions[i]:.1f} |  {mse_loss[i]:.3f}   |   {huber_loss[i]:.3f}")
    
    print(f"\n   Total MSE Loss: {mse_total:.3f}")
    print(f"   Total Huber Loss: {huber_total:.3f}")
    print("\n   -> Huber loss is more robust to outliers!")

def main():
    print("="*50)
    print("Experiment 05: Basic DQN Agent")
    print("="*50)
    
    # Train basic DQN
    agent = train_basic_dqn()
    
    # Analyze instability
    analyze_instability(agent)
    
    # Demonstrate moving target problem
    demonstrate_moving_target_problem()
    
    # Compare loss functions
    compare_loss_functions()
    
    print("\n5. Key Observations:")
    print("   - Training is unstable without target network")
    print("   - Q-values can diverge or oscillate")
    print("   - Targets change with every update (moving target)")
    print("   - Huber loss provides more stability than MSE")
    
    print("\n" + "="*50)
    print("Basic DQN implementation completed!")
    print("Next: Adding target network for stability")
    print("="*50)

if __name__ == "__main__":
    main()