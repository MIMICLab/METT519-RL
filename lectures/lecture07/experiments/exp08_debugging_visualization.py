#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 08 - Debugging and Visualization

This experiment provides tools and techniques for debugging DQN implementations
and visualizing the learning process for better understanding.

Learning objectives:
- Implement debugging tools for DQN
- Visualize Q-value evolution
- Monitor gradient flow and weight changes
- Create diagnostic plots for troubleshooting

Prerequisites: Completed exp01-exp07
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
from collections import deque, defaultdict
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:  # pragma: no cover - optional dependency
    sns = None
    HAS_SEABORN = False

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


class DebugDQN(nn.Module):
    """DQN with debugging hooks"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Debug storage
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for debugging"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                # grad_output is a tuple; take first tensor
                if grad_output and grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for each layer
        self.hooks.append(self.fc1.register_forward_hook(get_activation('fc1')))
        self.hooks.append(self.fc2.register_forward_hook(get_activation('fc2')))
        self.hooks.append(self.fc3.register_forward_hook(get_activation('fc3')))
        
        # Use full backward hooks to avoid deprecation warnings
        self.hooks.append(self.fc1.register_full_backward_hook(get_gradient('fc1')))
        self.hooks.append(self.fc2.register_full_backward_hook(get_gradient('fc2')))
        self.hooks.append(self.fc3.register_full_backward_hook(get_gradient('fc3')))
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_debug_info(self):
        """Get debugging information"""
        info = {
            'activations': {k: v.cpu().numpy() for k, v in self.activations.items()},
            'gradients': {k: v.cpu().numpy() for k, v in self.gradients.items() if k in self.gradients}
        }
        return info

class DiagnosticBuffer:
    """Enhanced replay buffer with diagnostics"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.stats = defaultdict(list)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
        # Collect statistics
        self.stats['rewards'].append(reward)
        self.stats['dones'].append(done)
        self.stats['state_mean'].append(np.mean(state))
        self.stats['state_std'].append(np.std(state))
    
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
    
    def get_diagnostics(self):
        """Get buffer diagnostics"""
        if not self.stats['rewards']:
            return {}
        
        return {
            'mean_reward': np.mean(self.stats['rewards'][-1000:]),
            'std_reward': np.std(self.stats['rewards'][-1000:]),
            'done_ratio': np.mean(self.stats['dones'][-1000:]),
            'state_mean': np.mean(self.stats['state_mean'][-1000:]),
            'state_std': np.mean(self.stats['state_std'][-1000:])
        }
    
    def __len__(self):
        return len(self.buffer)

class DQNDebugger:
    """Comprehensive debugging tools for DQN"""
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize network
        self.policy_net = DebugDQN(state_dim, action_dim).to(device)
        self.target_net = DebugDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = DiagnosticBuffer(10000)
        
        # Tracking
        self.training_history = defaultdict(list)
        self.weight_history = []
        self.q_value_history = []
    
    def check_gradient_flow(self):
        """Check if gradients are flowing properly"""
        print("\nGradient Flow Check:")
        print("=" * 40)
        
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_max = param.grad.max().item()
                grad_min = param.grad.min().item()
                
                print(f"{name:15s} | norm: {grad_norm:8.4f} | mean: {grad_mean:8.4f} | "
                      f"max: {grad_max:8.4f} | min: {grad_min:8.4f}")
                
                # Check for issues
                if grad_norm == 0:
                    print(f"  WARNING: Zero gradients in {name}")
                elif grad_norm > 100:
                    print(f"  WARNING: Large gradients in {name}")
                elif np.isnan(grad_norm):
                    print(f"  ERROR: NaN gradients in {name}")
    
    def check_dead_neurons(self):
        """Check for dead ReLU neurons"""
        print("\nDead Neuron Check:")
        print("=" * 40)
        
        # Generate test batch
        test_batch = torch.randn(1000, self.state_dim, device=device)
        
        with torch.no_grad():
            _ = self.policy_net(test_batch)
            activations = self.policy_net.activations
            
            for layer_name, activation in activations.items():
                if layer_name != 'fc3':  # Skip output layer
                    dead_count = (activation == 0).all(dim=0).sum().item()
                    total_neurons = activation.shape[1]
                    dead_ratio = dead_count / total_neurons * 100
                    
                    print(f"{layer_name}: {dead_count}/{total_neurons} dead neurons ({dead_ratio:.1f}%)")
                    
                    if dead_ratio > 50:
                        print(f"  WARNING: High ratio of dead neurons in {layer_name}")
    
    def visualize_q_values(self, states):
        """Visualize Q-value distribution"""
        with torch.no_grad():
            q_values = self.policy_net(states).cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Q-value distribution
        axes[0].hist(q_values.flatten(), bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Q-value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Q-value Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Q-value heatmap
        if HAS_SEABORN:
            sns.heatmap(
                q_values[:20],
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                ax=axes[1],
                cbar_kws={'label': 'Q-value'},
            )
        else:
            im = axes[1].imshow(q_values[:20], aspect='auto', cmap='coolwarm')
            axes[1].figure.colorbar(im, ax=axes[1], label='Q-value')
            for i in range(min(20, q_values.shape[0])):
                for j in range(q_values.shape[1]):
                    axes[1].text(j, i, f"{q_values[i, j]:.2f}", ha='center', va='center', fontsize=6)
        axes[1].set_xlabel('Action')
        axes[1].set_ylabel('State Sample')
        axes[1].set_title('Q-values for First 20 States')
        
        plt.tight_layout()
        return fig
    
    def monitor_training_step(self, batch):
        """Monitor a single training step"""
        states, actions, rewards, next_states, dones = batch
        
        # Forward pass
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Store diagnostics
        diagnostics = {
            'loss': loss.item(),
            'current_q_mean': current_q.mean().item(),
            'current_q_std': current_q.std().item(),
            'target_q_mean': target_q.mean().item(),
            'target_q_std': target_q.std().item(),
            'td_error_mean': (target_q - current_q).mean().item(),
            'td_error_std': (target_q - current_q).std().item(),
            'reward_mean': rewards.mean().item(),
            'done_ratio': dones.mean().item()
        }
        
        # Check gradient flow
        self.check_gradient_flow()
        
        # Update weights
        self.optimizer.step()
        
        return diagnostics

def main():
    print("="*50)
    print("Experiment 08: Debugging and Visualization")
    print("="*50)
    
    setup_seed(42)
    
    # Setup environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    debugger = DQNDebugger(state_dim, action_dim)
    
    print(f"\nEnvironment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")
    
    # 1. Initial network analysis
    print("\n" + "="*50)
    print("1. Initial Network Analysis")
    print("="*50)
    
    # Check initial weights
    print("\nInitial Weight Statistics:")
    for name, param in debugger.policy_net.named_parameters():
        weight_mean = param.data.mean().item()
        weight_std = param.data.std().item()
        print(f"{name:15s} | mean: {weight_mean:8.4f} | std: {weight_std:8.4f}")
    
    # Check dead neurons
    debugger.check_dead_neurons()
    
    # 2. Collect some experience
    print("\n" + "="*50)
    print("2. Collecting Experience")
    print("="*50)
    
    for episode in range(10):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            debugger.memory.push(state, action, reward, next_state, done)
            state = next_state
            steps += 1
        
        print(f"Episode {episode + 1}: {steps} steps")
    
    print(f"\nBuffer size: {len(debugger.memory)}")
    buffer_diagnostics = debugger.memory.get_diagnostics()
    for key, value in buffer_diagnostics.items():
        print(f"  {key}: {value:.4f}")
    
    # 3. Training diagnostics
    print("\n" + "="*50)
    print("3. Training Step Diagnostics")
    print("="*50)
    
    if len(debugger.memory) >= 32:
        batch = debugger.memory.sample(32)
        diagnostics = debugger.monitor_training_step(batch)
        
        print("\nTraining Metrics:")
        for key, value in diagnostics.items():
            print(f"  {key}: {value:.4f}")
    
    # 4. Q-value analysis
    print("\n" + "="*50)
    print("4. Q-Value Analysis")
    print("="*50)
    
    # Sample states for analysis
    test_states = []
    for _ in range(50):
        state, _ = env.reset()
        test_states.append(state)
    test_states = torch.tensor(np.array(test_states), dtype=torch.float32, device=device)
    
    fig = debugger.visualize_q_values(test_states)
    save_path = FIGURES_DIR / 'q_value_analysis.png'
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Q-value visualization saved to: {save_path}")
    plt.close()
    
    # 5. Common issues detection
    print("\n" + "="*50)
    print("5. Common Issues Detection")
    print("="*50)
    
    issues_found = []
    
    # Check for exploding Q-values
    with torch.no_grad():
        q_values = debugger.policy_net(test_states)
        max_q = q_values.max().item()
        min_q = q_values.min().item()
        
        if abs(max_q) > 100:
            issues_found.append("Exploding Q-values detected")
        if np.isnan(max_q) or np.isnan(min_q):
            issues_found.append("NaN Q-values detected")
    
    # Check weight magnitudes
    for name, param in debugger.policy_net.named_parameters():
        weight_norm = param.data.norm().item()
        if weight_norm > 100:
            issues_found.append(f"Large weights in {name}")
        elif weight_norm < 0.001:
            issues_found.append(f"Very small weights in {name}")
    
    if issues_found:
        print("\nIssues detected:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("\nNo major issues detected")
    
    # 6. Training progression monitoring
    print("\n" + "="*50)
    print("6. Training Progression Monitoring")
    print("="*50)
    
    print("\nTraining for 100 steps with monitoring...")
    
    training_metrics = defaultdict(list)
    
    for step in range(100):
        if len(debugger.memory) >= 32:
            batch = debugger.memory.sample(32)
            
            # Get diagnostics
            diagnostics = debugger.monitor_training_step(batch)
            
            # Store metrics
            for key, value in diagnostics.items():
                training_metrics[key].append(value)
            
            # Collect new experience periodically
            if step % 10 == 0:
                state, _ = env.reset()
                for _ in range(50):
                    action = env.action_space.sample()
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    debugger.memory.push(state, action, reward, next_state, done)
                    if done:
                        break
                    state = next_state
    
    # Plot training metrics
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    metrics_to_plot = ['loss', 'current_q_mean', 'target_q_mean', 
                      'td_error_mean', 'reward_mean', 'done_ratio']
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        if metric in training_metrics:
            ax.plot(training_metrics[metric])
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / 'training_diagnostics.png'
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"\nTraining diagnostics saved to: {save_path}")
    plt.close()
    
    # 7. Debugging recommendations
    print("\n" + "="*50)
    print("7. Debugging Recommendations")
    print("="*50)
    
    print("\nCommon debugging strategies:")
    print("1. Gradient Issues:")
    print("   - Vanishing gradients: Reduce network depth or use skip connections")
    print("   - Exploding gradients: Use gradient clipping or reduce learning rate")
    
    print("\n2. Q-value Issues:")
    print("   - Overestimation: Use Double DQN")
    print("   - Instability: Increase target update frequency or use soft updates")
    
    print("\n3. Learning Issues:")
    print("   - Slow learning: Increase learning rate or reduce epsilon")
    print("   - Unstable learning: Decrease learning rate or increase batch size")
    
    print("\n4. Dead Neurons:")
    print("   - Use LeakyReLU or ELU instead of ReLU")
    print("   - Better weight initialization (He or Xavier)")
    
    print("\n5. Memory Issues:")
    print("   - Poor sample diversity: Increase buffer size")
    print("   - Overfitting to recent: Use prioritized replay")
    
    # Final summary
    print("\n" + "="*50)
    print("Debugging Summary")
    print("="*50)
    
    print("\nKey metrics from analysis:")
    if training_metrics:
        print(f"  Final loss: {training_metrics['loss'][-1]:.4f}")
        print(f"  Final Q-value mean: {training_metrics['current_q_mean'][-1]:.4f}")
        print(f"  Final TD error: {training_metrics['td_error_mean'][-1]:.4f}")
    
    print("\nDebugging tools demonstrated:")
    print("  - Gradient flow monitoring")
    print("  - Dead neuron detection")
    print("  - Q-value visualization")
    print("  - Training metrics tracking")
    print("  - Common issue detection")
    
    print("\n" + "="*50)
    print("Debugging and visualization experiment complete!")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    main()
