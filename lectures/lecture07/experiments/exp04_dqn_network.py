#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 04 - DQN Network Architecture

This experiment explores different neural network architectures for DQN,
including the impact of network depth, width, and activation functions.

Learning objectives:
- Design and compare different Q-network architectures
- Understand the role of hidden layers and activation functions
- Implement target network mechanism
- Analyze network capacity and parameter count

Prerequisites: Completed exp01-exp03
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

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

class BasicDQN(nn.Module):
    """Basic DQN with 2 hidden layers"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        # Shape annotations for clarity
        x = F.relu(self.fc1(x))  # [B, state_dim] -> [B, hidden_dim]
        x = F.relu(self.fc2(x))  # [B, hidden_dim] -> [B, hidden_dim]
        x = self.fc3(x)           # [B, hidden_dim] -> [B, action_dim]
        return x

class DeepDQN(nn.Module):
    """Deeper DQN with 4 hidden layers"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DuelingDQN(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Shared layers
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
    
    def forward(self, x):
        features = self.feature(x)                    # [B, hidden_dim]
        value = self.value(features)                  # [B, 1]
        advantage = self.advantage(features)          # [B, action_dim]
        
        # Combine value and advantage (mean subtraction for stability)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class NoisyDQN(nn.Module):
    """DQN with noisy linear layers for exploration"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Add learnable noise parameters
        self.noise_std = nn.Parameter(torch.ones(hidden_dim) * 0.1)
    
    def forward(self, x, add_noise=False):
        x = F.relu(self.fc1(x))
        
        if add_noise and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("="*50)
    print("Experiment 04: DQN Network Architectures")
    print("="*50)
    
    setup_seed(42)
    
    # Environment setup
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"\nEnvironment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # 1. Compare different architectures
    print("\n1. Network Architecture Comparison:")
    
    architectures = {
        'Basic DQN': BasicDQN(state_dim, action_dim),
        'Deep DQN': DeepDQN(state_dim, action_dim),
        'Dueling DQN': DuelingDQN(state_dim, action_dim),
        'Noisy DQN': NoisyDQN(state_dim, action_dim)
    }
    
    # Test batch of states
    batch_size = 32
    test_states = torch.randn(batch_size, state_dim, device=device)
    
    print("\n   Architecture     | Parameters | Output Shape | Mean Q")
    print("   " + "-"*55)
    
    for name, model in architectures.items():
        model = model.to(device)
        param_count = count_parameters(model)
        
        with torch.no_grad():
            q_values = model(test_states)
            mean_q = q_values.mean().item()
        
        print(f"   {name:15s} | {param_count:10d} | {str(q_values.shape):12s} | {mean_q:7.4f}")
    
    # 2. Hidden layer size analysis
    print("\n2. Hidden Layer Size Impact:")
    
    hidden_sizes = [32, 64, 128, 256, 512]
    
    print("\n   Hidden Size | Parameters | Forward Time (ms)")
    print("   " + "-"*45)
    
    for hidden_size in hidden_sizes:
        model = BasicDQN(state_dim, action_dim, hidden_dim=hidden_size).to(device)
        param_count = count_parameters(model)
        
        # Time forward pass
        import time
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                _ = model(test_states)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.time() - start) * 10  # ms per forward pass
        
        print(f"   {hidden_size:11d} | {param_count:10d} | {elapsed:15.3f}")
    
    # 3. Target network mechanism
    print("\n3. Target Network Mechanism:")
    
    policy_net = BasicDQN(state_dim, action_dim).to(device)
    target_net = BasicDQN(state_dim, action_dim).to(device)
    
    # Initialize target with policy weights
    target_net.load_state_dict(policy_net.state_dict())
    
    print("   Initial state: Target and policy networks identical")
    
    # Check initial similarity
    with torch.no_grad():
        policy_q = policy_net(test_states)
        target_q = target_net(test_states)
        initial_diff = (policy_q - target_q).abs().mean().item()
    
    print(f"   Initial Q-value difference: {initial_diff:.6f}")
    
    # Train policy network for a few steps
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    
    for step in range(10):
        fake_targets = torch.randn_like(policy_q)
        loss = F.mse_loss(policy_net(test_states), fake_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check divergence
    with torch.no_grad():
        policy_q_new = policy_net(test_states)
        target_q_same = target_net(test_states)
        divergence = (policy_q_new - target_q_same).abs().mean().item()
    
    print(f"   After 10 updates: Q-value divergence: {divergence:.4f}")
    
    # Hard update
    target_net.load_state_dict(policy_net.state_dict())
    
    with torch.no_grad():
        target_q_updated = target_net(test_states)
        sync_diff = (policy_q_new - target_q_updated).abs().mean().item()
    
    print(f"   After hard update: Q-value difference: {sync_diff:.6f}")
    
    # 4. Soft update mechanism
    print("\n4. Soft Update Mechanism:")
    
    tau = 0.005  # Soft update rate
    
    # Perform soft update
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    with torch.no_grad():
        target_q_soft = target_net(test_states)
        soft_diff = (policy_q_new - target_q_soft).abs().mean().item()
    
    print(f"   Tau: {tau}")
    print(f"   After soft update: Q-value difference: {soft_diff:.4f}")
    print(f"   Soft update reduces divergence by {(1-tau)*100:.1f}%")
    
    # 5. Gradient flow analysis
    print("\n5. Gradient Flow Analysis:")
    
    model = BasicDQN(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Forward and backward pass
    states = torch.randn(32, state_dim, device=device, requires_grad=True)
    q_values = model(states)
    target_q = torch.randn_like(q_values)
    loss = F.mse_loss(q_values, target_q)
    
    optimizer.zero_grad()
    loss.backward()
    
    print("\n   Layer         | Grad Norm | Grad Mean | Grad Std")
    print("   " + "-"*55)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            print(f"   {name:13s} | {grad_norm:9.4f} | {grad_mean:9.4f} | {grad_std:8.4f}")
    
    # 6. Activation function comparison
    print("\n6. Activation Function Impact:")
    
    class CustomDQN(nn.Module):
        def __init__(self, state_dim, action_dim, activation='relu'):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, action_dim)
            
            if activation == 'relu':
                self.activation = F.relu
            elif activation == 'tanh':
                self.activation = torch.tanh
            elif activation == 'elu':
                self.activation = F.elu
            elif activation == 'leaky_relu':
                self.activation = F.leaky_relu
        
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            return self.fc3(x)
    
    activations = ['relu', 'tanh', 'elu', 'leaky_relu']
    
    print("\n   Activation | Mean Output | Std Output | Dead Neurons")
    print("   " + "-"*55)
    
    for activation in activations:
        model = CustomDQN(state_dim, action_dim, activation).to(device)
        
        with torch.no_grad():
            # Check for dead neurons (neurons that never activate)
            test_batch = torch.randn(1000, state_dim, device=device)
            hidden1 = model.activation(model.fc1(test_batch))
            dead_neurons = (hidden1 == 0).all(dim=0).sum().item()
            
            outputs = model(test_states)
            mean_out = outputs.mean().item()
            std_out = outputs.std().item()
        
        print(f"   {activation:10s} | {mean_out:11.4f} | {std_out:10.4f} | {dead_neurons:12d}")
    
    # 7. Initialization strategies
    print("\n7. Weight Initialization Strategies:")
    
    def init_weights(model, strategy='xavier'):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if strategy == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif strategy == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif strategy == 'normal':
                    nn.init.normal_(module.weight, mean=0, std=0.02)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    strategies = ['xavier', 'kaiming', 'normal']
    
    print("\n   Strategy | Initial Q Mean | Initial Q Std")
    print("   " + "-"*45)
    
    for strategy in strategies:
        model = BasicDQN(state_dim, action_dim).to(device)
        init_weights(model, strategy)
        
        with torch.no_grad():
            q_values = model(test_states)
            mean_q = q_values.mean().item()
            std_q = q_values.std().item()
        
        print(f"   {strategy:8s} | {mean_q:14.4f} | {std_q:13.4f}")
    
    print("\n" + "="*50)
    print("DQN network architecture analysis complete!")
    print("Key insights:")
    print("- Network depth vs width tradeoffs")
    print("- Target network stabilizes learning")
    print("- Activation and initialization matter")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
