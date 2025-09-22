#!/usr/bin/env python3
"""
RL2025 - Lecture 6: Experiment 03 - Neural Q-Network Basics

This experiment introduces neural networks for Q-function approximation,
showing how they replace Q-tables in high-dimensional spaces.

Learning objectives:
- Build a Q-network architecture in PyTorch
- Understand input/output dimensions for Q-learning
- Compare network predictions with tabular Q-values
- Demonstrate gradient-based updates

Prerequisites: PyTorch basics from Lecture 2
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
    """
    Neural network for Q-function approximation.
    Input: state observation (continuous)
    Output: Q-values for all actions
    """
    
    def __init__(self, obs_dim, n_actions, hidden_sizes=(128, 128)):
        super(QNetwork, self).__init__()
        
        # Build network layers
        layers = []
        input_size = obs_dim
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer (no activation - Q-values can be any real number)
        layers.append(nn.Linear(input_size, n_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through network.
        Args:
            state: Tensor of shape [batch_size, obs_dim]
        Returns:
            q_values: Tensor of shape [batch_size, n_actions]
        """
        return self.network(state)
    
    def get_action(self, state, epsilon=0.0):
        """
        Epsilon-greedy action selection.
        Args:
            state: Numpy array of shape [obs_dim]
            epsilon: Exploration probability
        Returns:
            action: Integer action index
        """
        if random.random() < epsilon:
            return random.randrange(self.network[-1].out_features)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, obs_dim]
            q_values = self.forward(state_tensor)  # [1, n_actions]
            return int(q_values.argmax().item())

def demonstrate_q_network():
    """Demonstrate Q-network forward pass and properties"""
    
    print("\n1. Q-Network Architecture:")
    
    # Create environment to get dimensions
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    print(f"   Input dimension (observation): {obs_dim}")
    print(f"   Output dimension (actions): {n_actions}")
    
    # Create Q-network
    q_net = QNetwork(obs_dim, n_actions, hidden_sizes=(128, 128)).to(device)
    
    print(f"\n   Network architecture:")
    print(f"   {q_net}")
    
    # Count parameters
    total_params = sum(p.numel() for p in q_net.parameters())
    trainable_params = sum(p.numel() for p in q_net.parameters() if p.requires_grad)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 2. Forward Pass Example
    print("\n2. Forward Pass Example:")
    
    # Get initial observation
    obs, _ = env.reset(seed=42)
    print(f"   Observation: {obs}")
    print(f"   Observation shape: {obs.shape}")
    
    # Convert to tensor and add batch dimension
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # [1, 4]
    print(f"   Tensor shape: {obs_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        q_values = q_net(obs_tensor)
    
    print(f"   Q-values: {q_values.squeeze().cpu().numpy()}")
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Best action: {q_values.argmax().item()}")
    
    # 3. Batch Processing
    print("\n3. Batch Processing:")
    
    # Create batch of observations
    batch_size = 32
    batch_obs = torch.randn(batch_size, obs_dim).to(device)
    print(f"   Batch input shape: {batch_obs.shape}")
    
    with torch.no_grad():
        batch_q_values = q_net(batch_obs)
    
    print(f"   Batch output shape: {batch_q_values.shape}")
    print(f"   First 3 Q-value pairs:")
    for i in range(3):
        q_vals = batch_q_values[i].cpu().numpy()
        print(f"     Sample {i}: Q(s,0)={q_vals[0]:.3f}, Q(s,1)={q_vals[1]:.3f}")
    
    env.close()
    return q_net, obs_dim, n_actions

def demonstrate_gradient_update(q_net, obs_dim, n_actions):
    """Show how gradients flow through Q-network"""
    
    print("\n4. Gradient-Based Learning:")
    
    # Create optimizer
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    
    # Simulate a mini-batch of experiences
    batch_size = 16
    states = torch.randn(batch_size, obs_dim).to(device)
    actions = torch.randint(0, n_actions, (batch_size,)).to(device)
    rewards = torch.randn(batch_size).to(device)
    next_states = torch.randn(batch_size, obs_dim).to(device)
    dones = torch.zeros(batch_size).to(device)  # No episodes done
    
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: 1e-3")
    
    # Forward pass - compute Q(s,a) for taken actions
    q_values = q_net(states)  # [batch_size, n_actions]
    q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze()  # [batch_size]
    
    # Compute targets (simplified - no target network yet)
    with torch.no_grad():
        next_q_values = q_net(next_states)  # [batch_size, n_actions]
        next_q_max = next_q_values.max(1)[0]  # [batch_size]
        targets = rewards + 0.99 * (1 - dones) * next_q_max
    
    # Compute loss
    loss = F.mse_loss(q_values_taken, targets)
    
    print(f"   Initial loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in q_net.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    print(f"   Gradient norms (min/mean/max): {min(grad_norms):.4f} / {np.mean(grad_norms):.4f} / {max(grad_norms):.4f}")
    
    # Update weights
    optimizer.step()
    
    # Recompute loss to show improvement
    with torch.no_grad():
        q_values_new = q_net(states)
        q_values_taken_new = q_values_new.gather(1, actions.unsqueeze(1)).squeeze()
        loss_new = F.mse_loss(q_values_taken_new, targets)
    
    print(f"   Loss after update: {loss_new.item():.4f}")
    print(f"   Loss reduction: {(loss.item() - loss_new.item()):.4f}")

def compare_with_tabular():
    """Compare neural network with tabular representation"""
    
    print("\n5. Comparison: Neural vs Tabular Q-Function:")
    
    print("\n   Tabular Q-Learning:")
    print("   - Exact value for each state-action pair")
    print("   - No generalization between states")
    print("   - Memory: O(|S| × |A|)")
    print("   - Updates: Direct assignment")
    
    print("\n   Neural Q-Network:")
    print("   - Approximate values via function")
    print("   - Generalizes to similar states")
    print("   - Memory: O(network parameters)")
    print("   - Updates: Gradient descent")
    
    # Demonstrate generalization
    print("\n6. Generalization Example:")
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    q_net = QNetwork(obs_dim, n_actions).to(device)
    
    # Two very similar states
    state1 = torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0).to(device)
    state2 = torch.FloatTensor([0.01, 0.01, 0.01, 0.01]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q1 = q_net(state1).cpu().numpy()[0]
        q2 = q_net(state2).cpu().numpy()[0]
    
    print(f"   State 1: [0.00, 0.00, 0.00, 0.00]")
    print(f"   Q-values: [{q1[0]:.3f}, {q1[1]:.3f}]")
    
    print(f"\n   State 2: [0.01, 0.01, 0.01, 0.01]")
    print(f"   Q-values: [{q2[0]:.3f}, {q2[1]:.3f}]")
    
    diff = np.abs(q1 - q2)
    print(f"\n   Q-value difference: [{diff[0]:.4f}, {diff[1]:.4f}]")
    print("   -> Neural network naturally generalizes to similar states!")
    
    env.close()

def main():
    print("="*50)
    print("Experiment 03: Neural Q-Network Basics")
    print("="*50)
    
    # Demonstrate Q-network architecture and forward pass
    q_net, obs_dim, n_actions = demonstrate_q_network()
    
    # Show gradient-based updates
    demonstrate_gradient_update(q_net, obs_dim, n_actions)
    
    # Compare with tabular approach
    compare_with_tabular()
    
    print("\n" + "="*50)
    print("Neural Q-Network basics completed!")
    print("Next: Experience replay buffer implementation")
    print("="*50)

if __name__ == "__main__":
    main()