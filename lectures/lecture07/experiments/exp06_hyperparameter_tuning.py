#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 06 - Hyperparameter Tuning

This experiment explores the impact of different hyperparameters on DQN
performance, including learning rate, batch size, and target update frequency.

Learning objectives:
- Understand sensitivity to hyperparameter choices
- Implement systematic hyperparameter search
- Analyze the impact of each hyperparameter
- Learn to identify optimal configurations

Prerequisites: Completed exp01-exp05
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
import json
from itertools import product

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

def train_dqn(config, verbose=False):
    """Train DQN with given configuration"""
    setup_seed(config['seed'])
    
    # Environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Networks
    policy_net = DQN(state_dim, action_dim, config['hidden_dim']).to(device)
    target_net = DQN(state_dim, action_dim, config['hidden_dim']).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # Optimizer and memory
    optimizer = optim.Adam(policy_net.parameters(), lr=config['lr'])
    memory = ReplayBuffer(config['buffer_size'])
    
    # Training
    epsilon = config['eps_start']
    episode_rewards = []
    
    for episode in range(config['num_episodes']):
        state, _ = env.reset(seed=config['seed'] + episode)
        episode_reward = 0
        
        done = False
        while not done:
            # Select action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()
            
            # Step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            # Train
            if len(memory) >= config['batch_size']:
                batch = memory.sample(config['batch_size'])
                states, actions, rewards, next_states, dones = batch
                
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + config['gamma'] * next_q * (1 - dones)
                
                loss = F.smooth_l1_loss(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), config['grad_clip'])
                optimizer.step()
        
        episode_rewards.append(episode_reward)
        
        # Update target network
        if episode % config['target_update'] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(config['eps_end'], epsilon * config['eps_decay'])
        
        if verbose and episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            print(f"  Episode {episode}: Avg Reward = {avg_reward:.1f}, Epsilon = {epsilon:.3f}")
    
    env.close()
    
    # Calculate metrics
    last_50 = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
    return {
        'final_avg_reward': np.mean(last_50),
        'best_reward': max(episode_rewards),
        'convergence_episode': next((i for i, r in enumerate(episode_rewards) if r >= 195), -1),
        'all_rewards': episode_rewards
    }

def main():
    print("="*50)
    print("Experiment 06: Hyperparameter Tuning")
    print("="*50)
    
    # Base configuration
    base_config = {
        'seed': 42,
        'num_episodes': 150,
        'hidden_dim': 128,
        'lr': 1e-3,
        'batch_size': 128,
        'gamma': 0.99,
        'buffer_size': 10000,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'eps_decay': 0.995,
        'target_update': 10,
        'grad_clip': 10.0
    }
    
    print("\nBase configuration:")
    for key, value in base_config.items():
        print(f"  {key}: {value}")
    
    # 1. Learning rate sensitivity
    print("\n" + "="*50)
    print("1. Learning Rate Sensitivity Analysis")
    print("="*50)
    
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    lr_results = []
    
    print("\n   LR      | Final Avg | Best  | Converged")
    print("   " + "-"*45)
    
    for lr in learning_rates:
        config = base_config.copy()
        config['lr'] = lr
        config['num_episodes'] = 100  # Faster for testing
        
        result = train_dqn(config)
        lr_results.append((lr, result))
        
        converged = "Yes" if result['convergence_episode'] >= 0 else "No"
        print(f"   {lr:7.5f} | {result['final_avg_reward']:9.1f} | {result['best_reward']:5.0f} | {converged}")
    
    best_lr = max(lr_results, key=lambda x: x[1]['final_avg_reward'])[0]
    print(f"\nBest learning rate: {best_lr}")
    
    # 2. Batch size impact
    print("\n" + "="*50)
    print("2. Batch Size Impact")
    print("="*50)
    
    batch_sizes = [32, 64, 128, 256]
    batch_results = []
    
    print("\n   Batch | Final Avg | Best  | Converged")
    print("   " + "-"*45)
    
    for batch_size in batch_sizes:
        config = base_config.copy()
        config['batch_size'] = batch_size
        config['num_episodes'] = 100
        
        result = train_dqn(config)
        batch_results.append((batch_size, result))
        
        converged = "Yes" if result['convergence_episode'] >= 0 else "No"
        print(f"   {batch_size:5d} | {result['final_avg_reward']:9.1f} | {result['best_reward']:5.0f} | {converged}")
    
    # 3. Target update frequency
    print("\n" + "="*50)
    print("3. Target Update Frequency")
    print("="*50)
    
    target_updates = [1, 5, 10, 20, 50]
    target_results = []
    
    print("\n   Update | Final Avg | Best  | Converged")
    print("   " + "-"*45)
    
    for target_update in target_updates:
        config = base_config.copy()
        config['target_update'] = target_update
        config['num_episodes'] = 100
        
        result = train_dqn(config)
        target_results.append((target_update, result))
        
        converged = "Yes" if result['convergence_episode'] >= 0 else "No"
        print(f"   {target_update:6d} | {result['final_avg_reward']:9.1f} | {result['best_reward']:5.0f} | {converged}")
    
    # 4. Epsilon decay strategies
    print("\n" + "="*50)
    print("4. Epsilon Decay Strategies")
    print("="*50)
    
    eps_configs = [
        {'eps_decay': 0.99, 'name': 'Slow (0.99)'},
        {'eps_decay': 0.995, 'name': 'Medium (0.995)'},
        {'eps_decay': 0.999, 'name': 'Very Slow (0.999)'},
        {'eps_decay': 0.98, 'name': 'Fast (0.98)'}
    ]
    
    print("\n   Strategy       | Final Avg | Best  | Converged")
    print("   " + "-"*50)
    
    for eps_config in eps_configs:
        config = base_config.copy()
        config['eps_decay'] = eps_config['eps_decay']
        config['num_episodes'] = 100
        
        result = train_dqn(config)
        
        converged = "Yes" if result['convergence_episode'] >= 0 else "No"
        print(f"   {eps_config['name']:14s} | {result['final_avg_reward']:9.1f} | {result['best_reward']:5.0f} | {converged}")
    
    # 5. Buffer size effect
    print("\n" + "="*50)
    print("5. Replay Buffer Size Effect")
    print("="*50)
    
    buffer_sizes = [1000, 5000, 10000, 50000]
    buffer_results = []
    
    print("\n   Buffer  | Final Avg | Best  | Converged")
    print("   " + "-"*45)
    
    for buffer_size in buffer_sizes:
        config = base_config.copy()
        config['buffer_size'] = buffer_size
        config['num_episodes'] = 100
        
        result = train_dqn(config)
        buffer_results.append((buffer_size, result))
        
        converged = "Yes" if result['convergence_episode'] >= 0 else "No"
        print(f"   {buffer_size:7d} | {result['final_avg_reward']:9.1f} | {result['best_reward']:5.0f} | {converged}")
    
    # 6. Network architecture
    print("\n" + "="*50)
    print("6. Network Architecture (Hidden Dimension)")
    print("="*50)
    
    hidden_dims = [32, 64, 128, 256]
    arch_results = []
    
    print("\n   Hidden | Final Avg | Best  | Parameters")
    print("   " + "-"*45)
    
    for hidden_dim in hidden_dims:
        config = base_config.copy()
        config['hidden_dim'] = hidden_dim
        config['num_episodes'] = 100
        
        # Calculate parameter count
        param_count = (4 * hidden_dim + hidden_dim) + (hidden_dim * hidden_dim + hidden_dim) + (hidden_dim * 2 + 2)
        
        result = train_dqn(config)
        arch_results.append((hidden_dim, result))
        
        print(f"   {hidden_dim:6d} | {result['final_avg_reward']:9.1f} | {result['best_reward']:5.0f} | {param_count:10d}")
    
    # 7. Grid search for best combination
    print("\n" + "="*50)
    print("7. Grid Search (Best 3 Hyperparameters)")
    print("="*50)
    
    # Reduced grid for efficiency
    grid = {
        'lr': [5e-4, 1e-3],
        'batch_size': [64, 128],
        'target_update': [5, 10]
    }
    
    print("\nSearching over", np.prod([len(v) for v in grid.values()]), "configurations...")
    
    best_score = -np.inf
    best_params = None
    
    for lr, batch_size, target_update in product(grid['lr'], grid['batch_size'], grid['target_update']):
        config = base_config.copy()
        config.update({
            'lr': lr,
            'batch_size': batch_size,
            'target_update': target_update,
            'num_episodes': 50  # Quick evaluation
        })
        
        result = train_dqn(config)
        
        if result['final_avg_reward'] > best_score:
            best_score = result['final_avg_reward']
            best_params = {'lr': lr, 'batch_size': batch_size, 'target_update': target_update}
    
    print(f"\nBest configuration found:")
    print(f"  Learning rate: {best_params['lr']}")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  Target update: {best_params['target_update']}")
    print(f"  Score: {best_score:.1f}")
    
    # 8. Final training with best parameters
    print("\n" + "="*50)
    print("8. Training with Optimized Hyperparameters")
    print("="*50)
    
    optimal_config = base_config.copy()
    optimal_config.update(best_params)
    optimal_config['num_episodes'] = 200
    
    print("\nTraining with optimal configuration...")
    final_result = train_dqn(optimal_config, verbose=True)
    
    print(f"\nFinal Results:")
    print(f"  Average reward (last 50): {final_result['final_avg_reward']:.1f}")
    print(f"  Best episode reward: {final_result['best_reward']:.0f}")
    if final_result['convergence_episode'] >= 0:
        print(f"  Converged at episode: {final_result['convergence_episode']}")
    else:
        print(f"  Did not converge to 195+ reward")
    
    # Summary
    print("\n" + "="*50)
    print("Key Insights from Hyperparameter Tuning:")
    print("="*50)
    print("\n1. Learning rate: Too high causes instability, too low slows learning")
    print("2. Batch size: Larger batches provide more stable gradients")
    print("3. Target updates: Balance between stability and staleness")
    print("4. Buffer size: Larger buffers improve sample diversity")
    print("5. Epsilon decay: Controls exploration-exploitation tradeoff")
    print("6. Network size: Bigger isn't always better for simple tasks")
    
    print("\n" + "="*50)
    print("Hyperparameter tuning experiment complete!")
    print("="*50)

if __name__ == "__main__":
    main()