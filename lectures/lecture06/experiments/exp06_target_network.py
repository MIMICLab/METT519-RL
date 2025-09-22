#!/usr/bin/env python3
"""
RL2025 - Lecture 6: Experiment 06 - Target Network Implementation

This experiment adds a target network to DQN, demonstrating how it
stabilizes training by fixing targets during updates.

Learning objectives:
- Implement target network with hard/soft updates
- Compare stability with and without target network
- Understand update frequencies and their effects
- Visualize target network benefits

Prerequisites: Basic DQN implementation from Experiment 05
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import copy

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

class DQNWithTargetNetwork:
    """DQN agent with target network for stability"""
    
    def __init__(self, env, learning_rate=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32,
                 target_update_freq=100, use_soft_update=False, tau=0.005):
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_soft_update = use_soft_update
        self.tau = tau
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        # Initialize Q-network and target network
        self.q_network = QNetwork(self.obs_dim, self.n_actions).to(device)
        self.target_network = QNetwork(self.obs_dim, self.n_actions).to(device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, self.obs_dim)
        
        # Tracking
        self.losses = []
        self.episode_rewards = []
        self.q_values = []
        self.target_q_values = []
        self.update_counter = 0
        self.target_updates = []
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            self.q_values.append(q_values.max().item())
            return q_values.argmax().item()
    
    def hard_update_target(self):
        """Copy weights from online to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_updates.append(self.update_counter)
    
    def soft_update_target(self):
        """Polyak averaging update of target network"""
        with torch.no_grad():
            for target_param, param in zip(self.target_network.parameters(), 
                                          self.q_network.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
    
    def compute_loss(self, batch):
        """Compute DQN loss with target network"""
        obs, actions, rewards, next_obs, dones = batch
        
        # Current Q-values for taken actions
        current_q_values = self.q_network(obs)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Next Q-values from TARGET network (key difference!)
        with torch.no_grad():
            next_q_values = self.target_network(next_obs)  # Using target network
            next_q_values = next_q_values.max(1)[0]
            self.target_q_values.append(next_q_values.mean().item())
            
            # Compute targets
            targets = rewards + self.gamma * (1 - dones) * next_q_values
        
        # Huber loss for stability
        loss = F.huber_loss(current_q_values, targets)
        
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        
        if self.use_soft_update:
            self.soft_update_target()
        elif self.update_counter % self.target_update_freq == 0:
            self.hard_update_target()
        
        # Track loss
        self.losses.append(loss.item())
    
    def train_episode(self):
        """Train for one episode"""
        obs, _ = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            self.replay_buffer.push(obs, action, reward, next_obs, done)
            self.update()
            
            episode_reward += reward
            obs = next_obs
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_rewards.append(episode_reward)
        return episode_reward

def compare_with_without_target():
    """Compare DQN with and without target network"""
    
    print("\n1. Comparing DQN Variants:")
    
    # Train without target network (from previous experiment)
    print("\n   a) Training WITHOUT Target Network:")
    env1 = gym.make("CartPole-v1")
    
    # Simplified version without target
    class SimpleDQN:
        def __init__(self, env):
            self.env = env
            self.q_network = QNetwork(4, 2, (64,)).to(device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
            self.buffer = ReplayBuffer(5000, 4)
            self.epsilon = 1.0
            self.rewards = []
            self.losses = []
        
        def train_step(self, obs, action, reward, next_obs, done):
            self.buffer.push(obs, action, reward, next_obs, done)
            
            if len(self.buffer) >= 32:
                batch = self.buffer.sample(32)
                obs_b, act_b, rew_b, next_b, done_b = batch
                
                # Without target network
                q_current = self.q_network(obs_b).gather(1, act_b.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next = self.q_network(next_b).max(1)[0]  # Same network!
                    targets = rew_b + 0.99 * (1 - done_b) * q_next
                
                loss = F.mse_loss(q_current, targets)
                self.losses.append(loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    agent1 = SimpleDQN(env1)
    
    # Quick training
    for ep in range(50):
        obs, _ = env1.reset()
        ep_reward = 0
        done = False
        
        while not done:
            if random.random() < agent1.epsilon:
                action = env1.action_space.sample()
            else:
                with torch.no_grad():
                    action = agent1.q_network(torch.FloatTensor(obs).unsqueeze(0).to(device)).argmax().item()
            
            next_obs, reward, terminated, truncated, _ = env1.step(action)
            done = terminated or truncated
            agent1.train_step(obs, action, reward, next_obs, done)
            ep_reward += reward
            obs = next_obs
        
        agent1.rewards.append(ep_reward)
        agent1.epsilon *= 0.95
    
    print(f"      Final avg reward: {np.mean(agent1.rewards[-10:]):.1f}")
    print(f"      Loss variance: {np.var(agent1.losses[-100:]) if len(agent1.losses) > 100 else 0:.4f}")
    
    env1.close()
    
    # Train with target network
    print("\n   b) Training WITH Target Network:")
    env2 = gym.make("CartPole-v1")
    agent2 = DQNWithTargetNetwork(
        env2,
        target_update_freq=100,
        use_soft_update=False
    )
    
    for ep in range(50):
        reward = agent2.train_episode()
    
    print(f"      Final avg reward: {np.mean(agent2.episode_rewards[-10:]):.1f}")
    print(f"      Loss variance: {np.var(agent2.losses[-100:]) if len(agent2.losses) > 100 else 0:.4f}")
    print(f"      Target updates: {len(agent2.target_updates)}")
    
    env2.close()
    
    return agent2

def analyze_update_frequencies(agent):
    """Analyze effect of different target update frequencies"""
    
    print("\n2. Target Update Frequency Analysis:")
    
    update_freqs = [10, 50, 100, 500, 1000]
    
    print("\n   Testing different update frequencies:")
    for freq in update_freqs:
        env = gym.make("CartPole-v1")
        test_agent = DQNWithTargetNetwork(
            env,
            target_update_freq=freq,
            use_soft_update=False
        )
        
        # Quick training
        for _ in range(30):
            test_agent.train_episode()
        
        avg_reward = np.mean(test_agent.episode_rewards[-10:])
        n_updates = len(test_agent.target_updates)
        
        print(f"      Freq={freq:4d}: Avg Reward={avg_reward:6.1f}, "
              f"Target Updates={n_updates:3d}")
        
        env.close()
    
    print("\n   -> Too frequent: instability")
    print("   -> Too infrequent: slow learning")
    print("   -> Sweet spot: 100-500 steps for CartPole")

def compare_hard_soft_updates():
    """Compare hard vs soft target updates"""
    
    print("\n3. Hard vs Soft Target Updates:")
    
    # Hard update
    print("\n   a) Hard Update (periodic copy):")
    env1 = gym.make("CartPole-v1")
    hard_agent = DQNWithTargetNetwork(
        env1,
        target_update_freq=100,
        use_soft_update=False
    )
    
    for _ in range(50):
        hard_agent.train_episode()
    
    print(f"      Final avg reward: {np.mean(hard_agent.episode_rewards[-10:]):.1f}")
    print(f"      Number of hard updates: {len(hard_agent.target_updates)}")
    
    env1.close()
    
    # Soft update
    print("\n   b) Soft Update (Polyak averaging):")
    env2 = gym.make("CartPole-v1")
    soft_agent = DQNWithTargetNetwork(
        env2,
        use_soft_update=True,
        tau=0.005
    )
    
    for _ in range(50):
        soft_agent.train_episode()
    
    print(f"      Final avg reward: {np.mean(soft_agent.episode_rewards[-10:]):.1f}")
    print(f"      Continuous soft updates with tau={soft_agent.tau}")
    
    env2.close()
    
    return hard_agent, soft_agent

def visualize_q_value_evolution(hard_agent, soft_agent):
    """Visualize Q-value evolution with different update strategies"""
    
    print("\n4. Q-Value Evolution Analysis:")
    
    # Hard update Q-values
    if hard_agent.q_values:
        print("\n   a) Hard Update Q-Values:")
        q_hard = np.array(hard_agent.q_values)
        
        # Show evolution at update points
        update_points = hard_agent.target_updates[:5] if len(hard_agent.target_updates) >= 5 else hard_agent.target_updates
        
        for i, update_point in enumerate(update_points):
            if update_point < len(q_hard):
                window_start = max(0, update_point - 10)
                window_end = min(len(q_hard), update_point + 10)
                window_q = q_hard[window_start:window_end]
                
                print(f"      Update {i+1} at step {update_point}:")
                print(f"        Before: {window_q[:10].mean() if len(window_q) > 0 else 0:.2f}")
                print(f"        After:  {window_q[10:].mean() if len(window_q) > 10 else 0:.2f}")
    
    # Soft update Q-values
    if soft_agent.q_values:
        print("\n   b) Soft Update Q-Values:")
        q_soft = np.array(soft_agent.q_values)
        
        # Show smooth evolution
        checkpoints = [0, len(q_soft)//4, len(q_soft)//2, 3*len(q_soft)//4, len(q_soft)-1]
        
        for checkpoint in checkpoints:
            if checkpoint < len(q_soft):
                window = q_soft[max(0, checkpoint-5):checkpoint+5]
                print(f"      Step {checkpoint:4d}: Q={window.mean():.2f} (std={window.std():.3f})")

def demonstrate_stability_benefits():
    """Demonstrate stability benefits of target network"""
    
    print("\n5. Stability Benefits of Target Network:")
    
    print("\n   Without Target Network:")
    print("   - Targets change with every update")
    print("   - Creates feedback loop: Q -> Target -> Q")
    print("   - Can lead to divergence or oscillation")
    print("   - High variance in loss")
    
    print("\n   With Target Network:")
    print("   - Targets remain fixed between updates")
    print("   - Breaks harmful feedback loops")
    print("   - More stable convergence")
    print("   - Lower variance in loss")
    
    # Simple demonstration
    print("\n   Mathematical Intuition:")
    print("   Without target: Q(s,a) <- r + γ max Q(s',·)")
    print("                   ↑__________________|")
    print("   (Circular dependency - Q affects its own target)")
    
    print("\n   With target:    Q(s,a) <- r + γ max Q⁻(s',·)")
    print("                   (Q⁻ is fixed during updates)")

def main():
    print("="*50)
    print("Experiment 06: Target Network Implementation")
    print("="*50)
    
    # Compare with and without target network
    agent = compare_with_without_target()
    
    # Analyze update frequencies
    analyze_update_frequencies(agent)
    
    # Compare hard vs soft updates
    hard_agent, soft_agent = compare_hard_soft_updates()
    
    # Visualize Q-value evolution
    visualize_q_value_evolution(hard_agent, soft_agent)
    
    # Explain stability benefits
    demonstrate_stability_benefits()
    
    print("\n" + "="*50)
    print("Target network implementation completed!")
    print("Next: Double DQN for reducing overestimation")
    print("="*50)

if __name__ == "__main__":
    main()