#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 02 - Q-Learning Basics

This experiment demonstrates fundamental Q-learning concepts including
Q-value computation, Bellman updates, and the role of discount factor.

Learning objectives:
- Understand Q-value representation for state-action pairs
- Implement basic Bellman update equation
- Visualize how Q-values evolve during learning
- Compare greedy vs epsilon-greedy action selection

Prerequisites: Completed exp01
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
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

def main():
    print("="*50)
    print("Experiment 02: Q-Learning Basics")
    print("="*50)
    
    setup_seed(42)
    
    # 1. Simple Q-table demonstration
    print("\n1. Q-Table Representation:")
    print("   For discrete state/action spaces, Q-values can be stored in a table")
    
    # Simplified discrete version of CartPole for demonstration
    n_states = 10  # Discretized states
    n_actions = 2  # Left, Right
    q_table = np.zeros((n_states, n_actions))
    print(f"   Q-table shape: {q_table.shape}")
    print(f"   Initial Q-values (all zeros):\n{q_table[:3]}")
    
    # 2. Bellman update demonstration
    print("\n2. Bellman Update Equation:")
    print("   Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]")
    
    # Example update
    state = 2
    action = 1
    reward = 1.0
    next_state = 3
    done = False
    
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    
    # Current Q-value
    q_current = q_table[state, action]
    
    # Target Q-value
    if done:
        q_target = reward
    else:
        q_target = reward + gamma * np.max(q_table[next_state])
    
    # Update
    q_table[state, action] = q_current + alpha * (q_target - q_current)
    
    print(f"   State: {state}, Action: {action}")
    print(f"   Reward: {reward}, Next State: {next_state}")
    print(f"   Q-value before: {q_current:.4f}")
    print(f"   Q-value after: {q_table[state, action]:.4f}")
    print(f"   TD error: {(q_target - q_current):.4f}")
    
    # 3. Neural network as Q-function approximator
    print("\n3. Neural Network Q-Function:")
    
    class SimpleQNetwork(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # CartPole dimensions
    state_dim = 4  # Cart position, velocity, pole angle, angular velocity
    action_dim = 2  # Left, Right
    
    q_network = SimpleQNetwork(state_dim, action_dim).to(device)
    print(f"   Network architecture:")
    print(f"   Input: {state_dim} -> Hidden: 128 -> Hidden: 128 -> Output: {action_dim}")
    
    # Test forward pass
    test_state = torch.randn(1, state_dim, device=device)  # Batch size 1
    q_values = q_network(test_state)
    print(f"   Sample input shape: {test_state.shape}")
    print(f"   Q-values output shape: {q_values.shape}")
    print(f"   Q-values: {q_values.detach().cpu().numpy()[0]}")
    
    # 4. Epsilon-greedy action selection
    print("\n4. Action Selection Strategies:")
    
    def select_action_greedy(q_values):
        """Select action with highest Q-value"""
        return torch.argmax(q_values, dim=-1).item()
    
    def select_action_epsilon_greedy(q_values, epsilon=0.1):
        """Select random action with probability epsilon"""
        if random.random() < epsilon:
            return random.randint(0, q_values.shape[-1] - 1)
        return torch.argmax(q_values, dim=-1).item()
    
    # Compare selections
    epsilon_values = [0.0, 0.1, 0.5, 1.0]
    n_trials = 1000
    
    print("   Epsilon | Random actions | Greedy actions")
    print("   " + "-"*40)
    
    for epsilon in epsilon_values:
        random_count = 0
        setup_seed(42)  # Reset seed for consistency
        
        for _ in range(n_trials):
            test_q = torch.tensor([[0.5, 0.3]], device=device)
            action = select_action_epsilon_greedy(test_q, epsilon)
            if action != 0:  # Action 0 has higher Q-value
                random_count += 1
        
        greedy_count = n_trials - random_count
        print(f"   {epsilon:4.1f}    | {random_count:14d} | {greedy_count:14d}")
    
    # 5. Discount factor impact
    print("\n5. Discount Factor Impact:")
    print("   gamma=0.0  -> Only immediate rewards matter")
    print("   gamma=0.99 -> Future rewards are important")
    print("   gamma=1.0  -> All future rewards equally important")
    
    # Simulate episode returns with different gammas
    rewards = [1.0] * 10  # 10 steps with reward 1
    gammas = [0.0, 0.5, 0.9, 0.99, 1.0]
    
    print("\n   Gamma | Total Return")
    print("   " + "-"*20)
    
    for gamma in gammas:
        total_return = sum(r * (gamma ** t) for t, r in enumerate(rewards))
        print(f"   {gamma:4.2f}  | {total_return:8.4f}")
    
    # 6. Loss function for DQN
    print("\n6. DQN Loss Function (Huber Loss):")
    
    # Sample batch
    batch_size = 4
    states = torch.randn(batch_size, state_dim, device=device)
    actions = torch.randint(0, action_dim, (batch_size,), device=device)
    rewards = torch.rand(batch_size, device=device)
    next_states = torch.randn(batch_size, state_dim, device=device)
    dones = torch.zeros(batch_size, device=device)
    
    # Compute Q-values
    q_values = q_network(states)  # [batch_size, action_dim]
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]
    
    # Compute targets (simplified, no target network)
    with torch.no_grad():
        next_q_values = q_network(next_states)  # [batch_size, action_dim]
        next_q_max = next_q_values.max(dim=1).values  # [batch_size]
        targets = rewards + 0.99 * (1 - dones) * next_q_max  # [batch_size]
    
    # Huber loss
    loss_fn = nn.HuberLoss()
    loss = loss_fn(q_selected, targets)
    
    print(f"   Batch size: {batch_size}")
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Selected Q-values: {q_selected.detach().cpu().numpy()}")
    print(f"   Target values: {targets.cpu().numpy()}")
    print(f"   Huber loss: {loss.item():.4f}")
    
    print("\n" + "="*50)
    print("Q-Learning basics demonstrated successfully!")
    print("="*50)

if __name__ == "__main__":
    main()