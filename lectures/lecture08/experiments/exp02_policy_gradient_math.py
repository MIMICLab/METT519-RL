#!/usr/bin/env python3
"""
RL2025 - Lecture 8: Experiment 02 - Policy Gradient Mathematics

This experiment demonstrates the mathematical foundations of policy gradients,
including the score function estimator and log-derivative trick.

Learning objectives:
- Understand the policy gradient theorem
- Implement score function calculation
- Visualize gradient directions
- Compare analytical and numerical gradients

Prerequisites: Completed exp01_setup.py
"""

# PyTorch 2.x Standard Practice Header
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
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

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR = Path(os.environ.get("LECTURE08_FIGURES_DIR", DEFAULT_FIGURES_DIR))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("="*50)
    print("Experiment 02: Policy Gradient Mathematics")
    print("="*50)
    
    # 1. Simple 2-action policy for visualization
    print("\n1. Policy Parameterization:")
    
    class SimplePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            # Single parameter theta for 2-action policy
            self.theta = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
        def forward(self, x=None):
            # Policy: pi(a=0) = sigmoid(theta), pi(a=1) = 1 - sigmoid(theta)
            # Logits for categorical: [theta, 0]
            logits = torch.stack([self.theta, torch.zeros_like(self.theta)], dim=-1)
            return logits
    
    policy = SimplePolicy().to(device)
    print(f"   Initial parameter theta: {policy.theta.item():.4f}")
    
    # 2. Compute probabilities
    print("\n2. Action Probabilities:")
    with torch.no_grad():
        logits = policy()
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        print(f"   Logits: {logits.cpu().numpy()}")
        print(f"   P(action=0): {probs[0].item():.4f}")
        print(f"   P(action=1): {probs[1].item():.4f}")
    
    # 3. Score function (gradient of log probability)
    print("\n3. Score Function Calculation:")
    
    # Sample multiple trajectories
    n_samples = 1000
    returns = []
    log_probs = []
    actions = []
    
    for _ in range(n_samples):
        logits = policy()
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Simulate returns: R(a=0) = 1, R(a=1) = -1
        ret = 1.0 if action.item() == 0 else -1.0
        
        returns.append(ret)
        log_probs.append(log_prob)
        actions.append(action.item())
    
    # Monte Carlo policy gradient estimate
    returns_tensor = torch.tensor(returns, device=device, dtype=torch.float32)
    log_probs_tensor = torch.stack(log_probs)
    
    # Compute gradient estimate: E[grad log pi * R]
    policy_gradient = -(log_probs_tensor * returns_tensor).mean()
    
    policy.zero_grad()
    policy_gradient.backward()
    
    print(f"   Sampled action distribution:")
    print(f"   - Action 0: {actions.count(0)/n_samples:.3f}")
    print(f"   - Action 1: {actions.count(1)/n_samples:.3f}")
    print(f"   Average return: {np.mean(returns):.4f}")
    print(f"   Policy gradient estimate: {policy.theta.grad.item():.4f}")
    
    # 4. Analytical gradient computation
    print("\n4. Analytical vs Numerical Gradient:")
    
    with torch.no_grad():
        # Analytical gradient: sum_a pi(a) * grad_log_pi(a) * R(a)
        probs = torch.softmax(policy(), dim=-1).squeeze(0)
        p0, p1 = probs.tolist()
        
        # For our parameterization: grad_log_pi(0) = 1 - pi(0)
        #                          grad_log_pi(1) = -pi(0)
        analytical_grad = p0 * (1 - p0) * 1.0 + p1 * (-p0) * (-1.0)
        print(f"   Analytical gradient: {analytical_grad:.4f}")
        print(f"   Monte Carlo estimate: {policy.theta.grad.item():.4f}")
        print(f"   Difference: {abs(analytical_grad - policy.theta.grad.item()):.4f}")
    
    # 5. Gradient direction visualization
    print("\n5. Gradient Direction Analysis:")
    
    theta_values = np.linspace(-3, 3, 100)
    expected_returns = []
    gradients = []
    
    for theta_val in theta_values:
        # Set parameter value
        with torch.no_grad():
            policy.theta.data = torch.tensor([theta_val], device=device, dtype=torch.float32)
        
        # Compute expected return
        logits = policy()
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        expected_return = probs[0] * 1.0 + probs[1] * (-1.0)
        expected_returns.append(expected_return.item())
        
        # Compute gradient
        policy.zero_grad()
        loss = -expected_return
        loss.backward()
        gradients.append(policy.theta.grad.item())
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(theta_values, expected_returns, 'b-', linewidth=2)
    ax1.set_xlabel('Parameter theta')
    ax1.set_ylabel('Expected Return J(theta)')
    ax1.set_title('Objective Function')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    ax2.plot(theta_values, gradients, 'r-', linewidth=2)
    ax2.set_xlabel('Parameter theta')
    ax2.set_ylabel('Gradient dJ/dtheta')
    ax2.set_title('Policy Gradient')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / 'policy_gradient_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to {save_path}")
    
    # 6. Variance of gradient estimates
    print("\n6. Gradient Estimator Variance:")
    
    sample_sizes = [10, 50, 100, 500, 1000]
    variances = []
    
    with torch.no_grad():
        policy.theta.data = torch.tensor([0.0], device=device, dtype=torch.float32)
    
    for n in sample_sizes:
        gradient_estimates = []
        
        for _ in range(100):  # 100 independent estimates
            grad_sum = 0.0
            for _ in range(n):
                logits = policy()
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                ret = 1.0 if action.item() == 0 else -1.0
                
                policy.zero_grad()
                loss = -(log_prob * ret)
                loss.backward()
                grad_sum += policy.theta.grad.item()
            
            gradient_estimates.append(grad_sum / n)
        
        variance = np.var(gradient_estimates)
        variances.append(variance)
        print(f"   N={n:4d}: Variance = {variance:.6f}, Std = {np.sqrt(variance):.6f}")
    
    # 7. Log-derivative trick demonstration
    print("\n7. Log-Derivative Trick:")
    print("   d/dtheta p(a|theta) = p(a|theta) * d/dtheta log p(a|theta)")
    print("   This allows us to express gradients as expectations!")
    print("   E[grad log pi(a) * R(a)] = grad E[R(a)]")
    
    print("\n" + "="*50)
    print("Policy gradient mathematics demonstrated successfully!")
    print("Key insight: We can optimize expected returns without")
    print("knowing the environment dynamics!")
    print("="*50)

if __name__ == "__main__":
    main()
