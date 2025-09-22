#!/usr/bin/env python3
"""
RL2025 - Lecture 6: Experiment 04 - Experience Replay Buffer

This experiment implements and demonstrates the experience replay buffer,
a key component for stabilizing DQN training by breaking correlations.

Learning objectives:
- Implement an efficient replay buffer
- Understand why replay breaks correlation
- Compare with and without replay
- Analyze memory efficiency

Prerequisites: Understanding of Q-learning and neural networks
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import time

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

class ReplayBuffer:
    """
    Efficient circular buffer for experience replay.
    Stores transitions as numpy arrays for memory efficiency.
    """
    
    def __init__(self, capacity, obs_dim):
        """
        Args:
            capacity: Maximum number of transitions to store
            obs_dim: Dimension of observation space
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        
        # Pre-allocate arrays for efficiency
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def push(self, obs, action, reward, next_obs, done):
        """Store a transition in the buffer"""
        idx = self.position
        
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_obs
        self.dones[idx] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        Returns:
            Tuple of tensors: (obs, actions, rewards, next_obs, dones)
        """
        if self.size < batch_size:
            return None

        # Random sampling without replacement
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Convert to tensors and move to device
        obs = torch.FloatTensor(self.observations[indices]).to(device)
        actions = torch.LongTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_obs = torch.FloatTensor(self.next_observations[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return self.size
    
    def get_memory_usage(self):
        """Calculate memory usage in MB"""
        total_bytes = (
            self.observations.nbytes +
            self.actions.nbytes +
            self.rewards.nbytes +
            self.next_observations.nbytes +
            self.dones.nbytes
        )
        return total_bytes / (1024 * 1024)

class SimpleDequeBuffer:
    """Simple deque-based buffer for comparison"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def demonstrate_replay_buffer():
    """Demonstrate replay buffer operations"""
    
    print("\n1. Creating Replay Buffer:")
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    buffer_size = 10000
    
    buffer = ReplayBuffer(buffer_size, obs_dim)
    
    print(f"   Buffer capacity: {buffer_size:,}")
    print(f"   Observation dimension: {obs_dim}")
    print(f"   Memory allocated: {buffer.get_memory_usage():.2f} MB")
    
    # 2. Fill buffer with experiences
    print("\n2. Collecting Experiences:")
    
    obs, _ = env.reset(seed=42)
    transitions_collected = 0
    episodes = 0
    
    while transitions_collected < 1000:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buffer.push(obs, action, reward, next_obs, done)
        transitions_collected += 1
        
        if done:
            obs, _ = env.reset()
            episodes += 1
        else:
            obs = next_obs
    
    print(f"   Transitions collected: {len(buffer)}")
    print(f"   Episodes completed: {episodes}")
    print(f"   Buffer fill ratio: {len(buffer) / buffer_size * 100:.1f}%")
    
    # 3. Sample from buffer
    print("\n3. Sampling from Buffer:")
    
    batch_size = 32
    sample = buffer.sample(batch_size)
    if sample is None:
        print("   Not enough samples collected to draw a batch yet.")
    else:
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = sample

        print(f"   Batch size: {batch_size}")
        print(f"   Observations shape: {obs_batch.shape}")
        print(f"   Actions shape: {actions_batch.shape}")
        print(f"   Rewards shape: {rewards_batch.shape}")
        print(f"   Next observations shape: {next_obs_batch.shape}")
        print(f"   Dones shape: {dones_batch.shape}")

        # Show sample statistics
        print(f"\n   Sample statistics:")
        print(f"   - Mean reward: {rewards_batch.mean().item():.3f}")
        print(f"   - Std reward: {rewards_batch.std().item():.3f}")
        print(f"   - Episode endings: {dones_batch.sum().item():.0f}/{batch_size}")
        print(f"   - Action distribution: {[(actions_batch == i).sum().item() for i in range(2)]}")
    
    env.close()
    return buffer

def analyze_correlation():
    """Analyze correlation in sequential vs random sampling"""
    
    print("\n4. Correlation Analysis:")
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    
    # Collect a trajectory
    obs, _ = env.reset(seed=42)
    trajectory = []
    
    for _ in range(50):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        trajectory.append({
            'obs': obs.copy(),
            'action': action,
            'reward': reward,
            'next_obs': next_obs.copy(),
            'done': done
        })
        
        if done:
            break
        obs = next_obs
    
    print(f"   Trajectory length: {len(trajectory)}")
    
    # Sequential sampling (correlated)
    print("\n   a) Sequential Sampling (Highly Correlated):")
    sequential_rewards = [t['reward'] for t in trajectory[:10]]
    sequential_obs = np.array([t['obs'][0] for t in trajectory[:10]])  # Cart position
    
    print(f"      Rewards: {sequential_rewards}")
    print(f"      Positions: {sequential_obs.round(3)}")
    
    # Calculate autocorrelation
    if len(sequential_obs) > 1:
        corr = np.corrcoef(sequential_obs[:-1], sequential_obs[1:])[0, 1]
        print(f"      Position autocorrelation: {corr:.3f}")
    
    # Random sampling (decorrelated)
    print("\n   b) Random Sampling from Replay (Decorrelated):")
    
    # Fill buffer with multiple episodes
    buffer = ReplayBuffer(1000, obs_dim)
    obs, _ = env.reset()
    
    for _ in range(500):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buffer.push(obs, action, reward, next_obs, done)
        
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    # Sample randomly
    sample = buffer.sample(10)
    if sample is None:
        print("      Not enough samples collected for random analysis yet.")
    else:
        obs_batch, _, rewards_batch, _, _ = sample
        random_rewards = rewards_batch.cpu().numpy()
        random_positions = obs_batch[:, 0].cpu().numpy()  # Cart position

        print(f"      Rewards: {random_rewards.tolist()}")
        print(f"      Positions: {random_positions.round(3)}")

        # Check correlation
        if len(random_positions) > 1:
            # Sort by position to check if there's any pattern
            sorted_indices = np.argsort(random_positions)
            sorted_pos = random_positions[sorted_indices]

            # Calculate differences between consecutive sorted positions
            pos_diffs = np.diff(sorted_pos)
            print(f"      Position differences (sorted): {pos_diffs.round(3)}")
            print(f"      -> Random sampling breaks temporal correlation!")

    env.close()

def benchmark_performance():
    """Compare performance of different buffer implementations"""
    
    print("\n5. Performance Benchmark:")
    
    obs_dim = 4
    capacity = 100000
    batch_size = 32
    n_operations = 1000
    
    # Numpy-based buffer
    print("\n   a) Numpy-based Replay Buffer:")
    buffer1 = ReplayBuffer(capacity, obs_dim)
    
    # Fill
    start = time.time()
    for _ in range(n_operations):
        obs = np.random.randn(obs_dim)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_obs = np.random.randn(obs_dim)
        done = np.random.random() < 0.1
        buffer1.push(obs, action, reward, next_obs, done)
    fill_time1 = time.time() - start
    
    # Sample
    start = time.time()
    for _ in range(n_operations // 10):
        _ = buffer1.sample(batch_size)
    sample_time1 = time.time() - start
    
    print(f"      Fill time ({n_operations} ops): {fill_time1:.3f}s")
    print(f"      Sample time ({n_operations//10} ops): {sample_time1:.3f}s")
    print(f"      Memory usage: {buffer1.get_memory_usage():.2f} MB")
    
    # Deque-based buffer
    print("\n   b) Deque-based Buffer:")
    buffer2 = SimpleDequeBuffer(capacity)
    
    # Fill
    start = time.time()
    for _ in range(n_operations):
        transition = (
            np.random.randn(obs_dim),
            np.random.randint(2),
            np.random.randn(),
            np.random.randn(obs_dim),
            np.random.random() < 0.1
        )
        buffer2.push(transition)
    fill_time2 = time.time() - start
    
    # Sample
    start = time.time()
    for _ in range(min(n_operations // 10, len(buffer2) // batch_size)):
        _ = buffer2.sample(batch_size)
    sample_time2 = time.time() - start
    
    print(f"      Fill time ({n_operations} ops): {fill_time2:.3f}s")
    print(f"      Sample time: {sample_time2:.3f}s")
    
    print("\n   Performance comparison:")
    print(f"      Fill speedup: {fill_time2/fill_time1:.1f}x")
    print(f"      Sample speedup: {sample_time2/sample_time1:.1f}x")
    print("      -> Numpy-based buffer is more efficient!")

def demonstrate_replay_importance():
    """Show why replay is important for stability"""
    
    print("\n6. Why Experience Replay Matters:")
    
    print("\n   Without Replay (Online Learning):")
    print("   - Samples are highly correlated (sequential)")
    print("   - Recent experiences dominate gradients")
    print("   - Forgetting of earlier experiences")
    print("   - Oscillations and instability")
    
    print("\n   With Replay:")
    print("   - Samples are decorrelated (random)")
    print("   - Balanced gradient updates")
    print("   - Reuse of experiences (sample efficiency)")
    print("   - Smoother, more stable learning")
    
    print("\n   Key Benefits:")
    print("   1. Break temporal correlations")
    print("   2. Improve sample efficiency")
    print("   3. Stabilize training")
    print("   4. Enable off-policy learning")

def main():
    print("="*50)
    print("Experiment 04: Experience Replay Buffer")
    print("="*50)
    
    # Demonstrate basic replay buffer operations
    buffer = demonstrate_replay_buffer()
    
    # Analyze correlation with and without replay
    analyze_correlation()
    
    # Benchmark different implementations
    benchmark_performance()
    
    # Explain importance of replay
    demonstrate_replay_importance()
    
    print("\n" + "="*50)
    print("Experience replay implementation completed!")
    print("Next: Building a basic DQN agent")
    print("="*50)

if __name__ == "__main__":
    main()
