#!/usr/bin/env python3
"""
RL2025 - Lecture 6: Experiment 07 - Double DQN Enhancement

This experiment implements Double DQN to reduce overestimation bias
by decoupling action selection from evaluation.

Learning objectives:
- Understand overestimation bias in Q-learning
- Implement Double DQN algorithm
- Compare Double DQN with vanilla DQN
- Analyze reduction in overestimation

Prerequisites: DQN with target network from Experiment 06
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

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

class DoubleDQNAgent:
    """Double DQN agent with reduced overestimation"""
    
    def __init__(self, env, learning_rate=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32,
                 target_update_freq=100, use_double_dqn=True):
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        # Initialize networks
        self.q_network = QNetwork(self.obs_dim, self.n_actions).to(device)
        self.target_network = QNetwork(self.obs_dim, self.n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, self.obs_dim)
        
        # Tracking
        self.losses = []
        self.episode_rewards = []
        self.q_values_online = []
        self.q_values_target = []
        self.overestimation = []
        self.update_counter = 0
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def compute_loss_vanilla_dqn(self, batch):
        """Vanilla DQN loss computation"""
        obs, actions, rewards, next_obs, dones = batch
        
        # Current Q-values
        current_q_values = self.q_network(obs)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Vanilla DQN: use target network for both selection and evaluation
        with torch.no_grad():
            next_q_values_target = self.target_network(next_obs)
            next_q_max = next_q_values_target.max(1)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q_max
        
        loss = F.huber_loss(current_q_values, targets)
        
        # Track for analysis
        self.q_values_target.append(next_q_max.mean().item())
        
        return loss
    
    def compute_loss_double_dqn(self, batch):
        """Double DQN loss computation"""
        obs, actions, rewards, next_obs, dones = batch
        
        # Current Q-values
        current_q_values = self.q_network(obs)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: use online network for selection, target for evaluation
        with torch.no_grad():
            # Action selection using online network
            next_q_values_online = self.q_network(next_obs)
            next_actions = next_q_values_online.argmax(1, keepdim=True)
            
            # Action evaluation using target network
            next_q_values_target = self.target_network(next_obs)
            next_q_selected = next_q_values_target.gather(1, next_actions).squeeze()
            
            targets = rewards + self.gamma * (1 - dones) * next_q_selected
            
            # Track overestimation
            vanilla_max = next_q_values_target.max(1)[0]
            overest = (vanilla_max - next_q_selected).mean().item()
            self.overestimation.append(overest)
        
        loss = F.huber_loss(current_q_values, targets)
        
        # Track for analysis
        self.q_values_online.append(next_q_values_online.max(1)[0].mean().item())
        self.q_values_target.append(next_q_selected.mean().item())
        
        return loss
    
    def update(self):
        """Perform one gradient update"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Compute loss
        if self.use_double_dqn:
            loss = self.compute_loss_double_dqn(batch)
        else:
            loss = self.compute_loss_vanilla_dqn(batch)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
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

def demonstrate_overestimation_bias():
    """Demonstrate overestimation bias in Q-learning"""
    
    print("\n1. Understanding Overestimation Bias:")
    
    print("\n   The max operator in Q-learning can cause overestimation:")
    print("   - With noise: max E[Q] ≤ E[max Q]")
    print("   - Max operator is biased upward")
    print("   - Accumulates over many updates")
    
    # Simple demonstration
    print("\n   Example with random Q-values:")
    
    # Simulate noisy Q-value estimates
    true_values = torch.tensor([1.0, 0.5, 0.3, 0.8])
    noise_std = 0.3
    n_samples = 1000
    
    max_estimates = []
    for _ in range(n_samples):
        noise = torch.randn_like(true_values) * noise_std
        noisy_q = true_values + noise
        max_estimates.append(noisy_q.max().item())
    
    true_max = true_values.max().item()
    estimated_max = np.mean(max_estimates)
    overestimation = estimated_max - true_max
    
    print(f"   True max Q-value: {true_max:.3f}")
    print(f"   Average estimated max: {estimated_max:.3f}")
    print(f"   Overestimation bias: {overestimation:.3f}")
    print("   -> Max operator consistently overestimates!")

def compare_vanilla_vs_double():
    """Compare vanilla DQN with Double DQN"""
    
    print("\n2. Comparing Vanilla DQN vs Double DQN:")
    
    # Train vanilla DQN
    print("\n   a) Training Vanilla DQN:")
    env1 = gym.make("CartPole-v1")
    vanilla_agent = DoubleDQNAgent(
        env1,
        use_double_dqn=False,  # Vanilla DQN
        buffer_size=5000,
        target_update_freq=100
    )
    
    for ep in range(50):
        reward = vanilla_agent.train_episode()
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(vanilla_agent.episode_rewards[-10:])
            avg_q = np.mean(vanilla_agent.q_values_target[-100:]) if vanilla_agent.q_values_target else 0
            print(f"      Episode {ep+1}: Avg Reward={avg_reward:.1f}, Avg Q={avg_q:.2f}")
    
    vanilla_final_q = np.mean(vanilla_agent.q_values_target[-100:]) if vanilla_agent.q_values_target else 0
    env1.close()
    
    # Train Double DQN
    print("\n   b) Training Double DQN:")
    env2 = gym.make("CartPole-v1")
    double_agent = DoubleDQNAgent(
        env2,
        use_double_dqn=True,  # Double DQN
        buffer_size=5000,
        target_update_freq=100
    )
    
    for ep in range(50):
        reward = double_agent.train_episode()
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(double_agent.episode_rewards[-10:])
            avg_q = np.mean(double_agent.q_values_target[-100:]) if double_agent.q_values_target else 0
            avg_overest = np.mean(double_agent.overestimation[-100:]) if double_agent.overestimation else 0
            print(f"      Episode {ep+1}: Avg Reward={avg_reward:.1f}, "
                  f"Avg Q={avg_q:.2f}, Overest={avg_overest:.3f}")
    
    double_final_q = np.mean(double_agent.q_values_target[-100:]) if double_agent.q_values_target else 0
    env2.close()
    
    print("\n   Results:")
    print(f"   Vanilla DQN final Q: {vanilla_final_q:.2f}")
    print(f"   Double DQN final Q: {double_final_q:.2f}")
    print(f"   Q-value reduction: {vanilla_final_q - double_final_q:.2f}")
    
    return vanilla_agent, double_agent

def analyze_action_selection():
    """Analyze how Double DQN decouples selection and evaluation"""
    
    print("\n3. Double DQN Action Selection Analysis:")
    
    print("\n   Vanilla DQN:")
    print("   1. Next action: a* = argmax_a Q_target(s', a)")
    print("   2. Next value: Q_target(s', a*)")
    print("   -> Same network for both selection and evaluation")
    print("   -> If one action is overestimated, it's both selected AND evaluated high")
    
    print("\n   Double DQN:")
    print("   1. Next action: a* = argmax_a Q_online(s', a)")
    print("   2. Next value: Q_target(s', a*)")
    print("   -> Different networks for selection and evaluation")
    print("   -> Reduces correlation between selection and evaluation errors")
    
    # Demonstrate with example
    print("\n   Example scenario:")
    
    # Create sample Q-values
    q_online = torch.tensor([[2.5, 3.8, 1.2, 2.0]])  # Online network
    q_target = torch.tensor([[2.2, 3.0, 1.5, 2.3]])  # Target network
    
    print(f"   Q_online values: {q_online.numpy()[0]}")
    print(f"   Q_target values: {q_target.numpy()[0]}")
    
    # Vanilla DQN
    vanilla_action = q_target.argmax(1).item()
    vanilla_value = q_target[0, vanilla_action].item()
    print(f"\n   Vanilla DQN:")
    print(f"   - Selected action: {vanilla_action}")
    print(f"   - Estimated value: {vanilla_value:.2f}")
    
    # Double DQN
    double_action = q_online.argmax(1).item()
    double_value = q_target[0, double_action].item()
    print(f"\n   Double DQN:")
    print(f"   - Selected action: {double_action}")
    print(f"   - Estimated value: {double_value:.2f}")
    
    print(f"\n   Difference: {vanilla_value - double_value:.2f}")

def visualize_q_value_evolution(vanilla_agent, double_agent):
    """Visualize Q-value evolution for both algorithms"""
    
    print("\n4. Q-Value Evolution Comparison:")
    
    # Vanilla DQN Q-values
    if vanilla_agent.q_values_target:
        vanilla_q = np.array(vanilla_agent.q_values_target)
        
        print("\n   Vanilla DQN Q-values:")
        checkpoints = [0, len(vanilla_q)//4, len(vanilla_q)//2, 3*len(vanilla_q)//4, len(vanilla_q)-1]
        for checkpoint in checkpoints:
            if checkpoint < len(vanilla_q):
                window = vanilla_q[max(0, checkpoint-10):checkpoint+10]
                if len(window) > 0:
                    print(f"      Step {checkpoint:4d}: Q={window.mean():.2f}")
    
    # Double DQN Q-values
    if double_agent.q_values_target:
        double_q = np.array(double_agent.q_values_target)
        
        print("\n   Double DQN Q-values:")
        checkpoints = [0, len(double_q)//4, len(double_q)//2, 3*len(double_q)//4, len(double_q)-1]
        for checkpoint in checkpoints:
            if checkpoint < len(double_q):
                window = double_q[max(0, checkpoint-10):checkpoint+10]
                if len(window) > 0:
                    print(f"      Step {checkpoint:4d}: Q={window.mean():.2f}")
    
    # Overestimation tracking
    if double_agent.overestimation:
        overest = np.array(double_agent.overestimation)
        print("\n   Overestimation reduction over time:")
        print(f"      Initial: {np.mean(overest[:100]) if len(overest) > 100 else np.mean(overest):.3f}")
        print(f"      Final:   {np.mean(overest[-100:]) if len(overest) > 100 else np.mean(overest):.3f}")

def practical_implications():
    """Discuss practical implications of Double DQN"""
    
    print("\n5. Practical Implications:")
    
    print("\n   Benefits of Double DQN:")
    print("   + Reduces overestimation bias")
    print("   + More accurate Q-value estimates")
    print("   + Often improves final performance")
    print("   + Minimal computational overhead")
    
    print("\n   When to use Double DQN:")
    print("   - Default choice for most problems")
    print("   - Especially important for:")
    print("     * Environments with many actions")
    print("     * Stochastic environments")
    print("     * Long training runs")
    
    print("\n   Implementation tips:")
    print("   - Easy to implement (few lines of code change)")
    print("   - Compatible with other improvements (PER, Dueling, etc.)")
    print("   - No additional hyperparameters needed")

def main():
    print("="*50)
    print("Experiment 07: Double DQN Enhancement")
    print("="*50)
    
    # Demonstrate overestimation bias
    demonstrate_overestimation_bias()
    
    # Compare vanilla vs Double DQN
    vanilla_agent, double_agent = compare_vanilla_vs_double()
    
    # Analyze action selection mechanism
    analyze_action_selection()
    
    # Visualize Q-value evolution
    visualize_q_value_evolution(vanilla_agent, double_agent)
    
    # Discuss practical implications
    practical_implications()
    
    print("\n" + "="*50)
    print("Double DQN implementation completed!")
    print("Next: Training optimizations (AMP, compile)")
    print("="*50)

if __name__ == "__main__":
    main()