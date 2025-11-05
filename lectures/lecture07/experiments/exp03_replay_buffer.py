#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 03 - Experience Replay Buffer

This experiment implements and tests an experience replay buffer,
demonstrating how it decorrelates samples and improves learning stability.

Learning objectives:
- Implement a circular replay buffer
- Understand the benefits of experience replay
- Compare sequential vs random sampling
- Analyze buffer capacity effects

Prerequisites: Completed exp01-exp02
"""

import os
import random
import numpy as np
import torch
import gymnasium as gym
from collections import deque

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

BOOL_DTYPE = np.bool8 if hasattr(np, "bool8") else np.bool_


class ReplayBuffer:
    """
    Efficient circular buffer for storing transitions.
    Uses numpy arrays for memory efficiency.
    """
    def __init__(self, state_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device
        self.pos = 0
        self.full = False
        
        # Pre-allocate arrays
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=BOOL_DTYPE)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0
    
    def sample(self, batch_size):
        """Sample a random batch of transitions (returns tensors on target device)"""
        max_idx = self.capacity if self.full else self.pos
        idx = np.random.randint(0, max_idx, size=batch_size)
        
        # Convert to tensors and move to device
        states = torch.from_numpy(self.states[idx]).to(self.device)
        actions = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idx]).to(self.device)
        dones = torch.from_numpy(self.dones[idx].astype(np.float32)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.capacity if self.full else self.pos
    
    def get_statistics(self):
        """Get buffer statistics"""
        current_size = len(self)
        if current_size == 0:
            return {}
        
        valid_rewards = self.rewards[:current_size]
        valid_dones = self.dones[:current_size]
        
        return {
            'size': current_size,
            'capacity': self.capacity,
            'full': self.full,
            'mean_reward': np.mean(valid_rewards),
            'std_reward': np.std(valid_rewards),
            'done_ratio': np.mean(valid_dones),
        }

def main():
    print("="*50)
    print("Experiment 03: Experience Replay Buffer")
    print("="*50)
    
    setup_seed(42)
    
    # 1. Create and test replay buffer
    print("\n1. Replay Buffer Implementation:")
    
    state_dim = 4  # CartPole observation dimension
    buffer_capacity = 1000
    buffer = ReplayBuffer(state_dim, buffer_capacity, device)
    
    print(f"   Buffer capacity: {buffer_capacity}")
    print(f"   State dimension: {state_dim}")
    print(f"   Device: {device}")
    
    # 2. Fill buffer with sample data
    print("\n2. Filling Buffer with Sample Transitions:")
    
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=42)
    
    n_steps = 150
    for step in range(n_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buffer.add(obs, action, reward, next_obs, done)
        
        obs = next_obs
        if done:
            obs, _ = env.reset()
    
    stats = buffer.get_statistics()
    print(f"   Added {n_steps} transitions")
    print(f"   Buffer size: {stats['size']}")
    print(f"   Buffer full: {stats['full']}")
    print(f"   Mean reward: {stats['mean_reward']:.3f}")
    print(f"   Done ratio: {stats['done_ratio']:.3f}")
    
    # 3. Test sampling
    print("\n3. Random Sampling Test:")
    
    batch_size = 32
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print(f"   Batch size: {batch_size}")
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Rewards shape: {rewards.shape}")
    print(f"   Sample rewards: {rewards[:5].cpu().numpy()}")
    print(f"   Sample actions: {actions[:5].cpu().numpy()}")
    
    # 4. Compare sequential vs random sampling
    print("\n4. Sequential vs Random Sampling:")
    
    # Sequential sampling (bad for learning)
    sequential_states = buffer.states[:batch_size]
    
    # Random sampling (good for learning)
    random_sample = buffer.sample(batch_size)[0].cpu().numpy()
    
    # Compute correlation between consecutive samples
    def compute_correlation(samples):
        """Compute average correlation between consecutive samples"""
        correlations = []
        for i in range(len(samples) - 1):
            corr = np.corrcoef(samples[i], samples[i+1])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        return np.mean(correlations) if correlations else 0.0
    
    seq_corr = compute_correlation(sequential_states)
    rand_corr = compute_correlation(random_sample)
    
    print(f"   Sequential correlation: {seq_corr:.4f}")
    print(f"   Random correlation: {rand_corr:.4f}")
    if abs(seq_corr) > 1e-12:
        improvement = (seq_corr - rand_corr) / seq_corr * 100.0
        print(f"   Decorrelation improvement: {improvement:.1f}%")
    else:
        print("   Decorrelation improvement: N/A (zero sequential correlation)")
    
    # 5. Buffer capacity analysis
    print("\n5. Buffer Capacity Effects:")
    
    capacities = [100, 500, 1000, 5000]
    buffer_stats = []
    
    for capacity in capacities:
        test_buffer = ReplayBuffer(state_dim, capacity, device)
        
        # Fill with 2000 transitions
        obs, _ = env.reset(seed=42)
        for _ in range(2000):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            test_buffer.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
            if done:
                obs, _ = env.reset()
        
        stats = test_buffer.get_statistics()
        buffer_stats.append(stats)
        
        print(f"   Capacity: {capacity:5d} | Size: {stats['size']:5d} | "
              f"Mean reward: {stats['mean_reward']:.3f}")
    
    # 6. Circular buffer behavior
    print("\n6. Circular Buffer Overwriting:")
    
    small_buffer = ReplayBuffer(state_dim, 10, device)
    
    # Add 15 transitions to a buffer of size 10
    for i in range(15):
        state = np.array([i, i, i, i], dtype=np.float32)
        small_buffer.add(state, 0, float(i), state, False)
    
    print(f"   Buffer capacity: 10")
    print(f"   Added transitions: 15")
    print(f"   Current size: {len(small_buffer)}")
    print(f"   Position pointer: {small_buffer.pos}")
    print(f"   First 3 stored states (oldest if full):")
    for i in range(3):
        idx = (small_buffer.pos + i) % small_buffer.capacity if small_buffer.full else i
        print(f"     {small_buffer.states[idx]}")
    
    # 7. Memory efficiency
    print("\n7. Memory Efficiency:")
    
    large_capacity = 100000
    large_buffer = ReplayBuffer(state_dim, large_capacity, device)
    
    # Calculate memory usage
    memory_bytes = (
        large_buffer.states.nbytes + 
        large_buffer.next_states.nbytes +
        large_buffer.actions.nbytes +
        large_buffer.rewards.nbytes +
        large_buffer.dones.nbytes
    )
    memory_mb = memory_bytes / (1024 * 1024)
    
    print(f"   Large buffer capacity: {large_capacity}")
    print(f"   Memory usage: {memory_mb:.2f} MB")
    print(f"   Per-transition memory: {memory_bytes/large_capacity:.1f} bytes")
    
    # 8. Batch diversity analysis
    print("\n8. Batch Diversity Analysis:")
    
    # Fill buffer with episodes
    obs, _ = env.reset(seed=42)
    episode_buffer = ReplayBuffer(state_dim, 1000, device)
    episode_markers = []
    episode_id = 0
    
    for _ in range(500):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        episode_buffer.add(obs, action, reward, next_obs, done)
        episode_markers.append(episode_id)
        
        obs = next_obs
        if done:
            obs, _ = env.reset()
            episode_id += 1
    
    # Sample batch and check episode diversity
    batch_indices = np.random.randint(0, len(episode_buffer), size=32)
    batch_episodes = [episode_markers[i] for i in batch_indices]
    unique_episodes = len(set(batch_episodes))
    
    print(f"   Batch size: 32")
    print(f"   Unique episodes in batch: {unique_episodes}")
    print(f"   Episode diversity: {unique_episodes/32*100:.1f}%")
    
    print("\n" + "="*50)
    print("Replay buffer implementation complete!")
    print("Key benefits demonstrated:")
    print("- Decorrelates sequential samples")
    print("- Enables efficient memory reuse")
    print("- Improves sample diversity")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
