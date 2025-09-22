#!/usr/bin/env python3
"""
RL2025 - Lecture 4: Experiment 01 - Environment Setup and MDP Verification

This experiment verifies the PyTorch environment and introduces basic
MDP concepts through a simple tabular representation.

Learning objectives:
- Verify PyTorch installation and device availability
- Understand MDP state/action spaces
- Validate tensor operations for tabular RL

Prerequisites: PyTorch 2.x installed
"""

import os
import sys
import random
import numpy as np
import torch
import platform
from datetime import datetime

def setup_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Proper device selection (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def print_system_info():
    """Print comprehensive system information"""
    print("="*50)
    print("System Information")
    print("="*50)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == 'mps':
        print("Using Apple Metal Performance Shaders")
    
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def test_basic_mdp_tensors():
    """Test basic tensor operations for tabular MDPs"""
    print("\n" + "="*50)
    print("Testing MDP Tensor Operations")
    print("="*50)
    
    device = get_device()
    
    # Simple 3-state, 2-action MDP
    n_states = 3
    n_actions = 2
    
    print(f"\nMDP Configuration:")
    print(f"  States (S): {n_states}")
    print(f"  Actions (A): {n_actions}")
    
    # Transition probability tensor P[s, a, s']
    # Shape: [S, A, S]
    P = torch.zeros((n_states, n_actions, n_states), dtype=torch.float32, device=device)
    
    # Example transitions (deterministic for simplicity)
    P[0, 0, 1] = 1.0  # State 0, action 0 -> state 1
    P[0, 1, 2] = 1.0  # State 0, action 1 -> state 2
    P[1, 0, 0] = 1.0  # State 1, action 0 -> state 0
    P[1, 1, 2] = 1.0  # State 1, action 1 -> state 2
    P[2, 0, 2] = 1.0  # State 2, action 0 -> state 2 (terminal)
    P[2, 1, 2] = 1.0  # State 2, action 1 -> state 2 (terminal)
    
    print(f"\nTransition tensor P shape: {P.shape}")
    print(f"P dtype: {P.dtype}")
    print(f"P device: {P.device}")
    
    # Verify probability constraints
    row_sums = P.sum(dim=2)  # Sum over next states
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "Transition probabilities must sum to 1"
    print("✓ Transition probabilities sum to 1 for each (s,a) pair")
    
    # Reward tensor R[s, a]
    # Shape: [S, A]
    R = torch.tensor([
        [0.0, 0.0],   # Rewards from state 0
        [-1.0, 0.0],  # Rewards from state 1
        [10.0, 10.0]  # Rewards from state 2 (terminal)
    ], dtype=torch.float32, device=device)
    
    print(f"\nReward tensor R shape: {R.shape}")
    print(f"R dtype: {R.dtype}")
    print(f"R device: {R.device}")
    
    # Test Bellman backup operation
    gamma = 0.9
    V = torch.zeros(n_states, dtype=torch.float32, device=device)
    
    # Q(s,a) = R(s,a) + gamma * sum_s' P(s,a,s') * V(s')
    # Using einsum for clarity
    Q = R + gamma * torch.einsum('sas,s->sa', P, V)
    
    print(f"\nAction-value tensor Q shape: {Q.shape}")
    print(f"Q values:\n{Q}")
    
    # Greedy policy extraction
    pi = torch.argmax(Q, dim=1)
    print(f"\nGreedy policy shape: {pi.shape}")
    print(f"Policy actions: {pi}")
    
    print("\n✓ All MDP tensor operations completed successfully!")
    
    return P, R, V, Q, pi

def test_bellman_operator():
    """Test Bellman expectation operator properties"""
    print("\n" + "="*50)
    print("Testing Bellman Operator Properties")
    print("="*50)
    
    device = get_device()
    n_states = 4
    n_actions = 2
    gamma = 0.95
    
    # Random MDP for testing
    P = torch.rand((n_states, n_actions, n_states), device=device)
    P = P / P.sum(dim=2, keepdim=True)  # Normalize
    R = torch.randn((n_states, n_actions), device=device)
    
    # Random policy (stochastic)
    pi = torch.softmax(torch.randn((n_states, n_actions), device=device), dim=1)
    
    # Compute P_pi and R_pi for policy evaluation
    # P_pi[s,s'] = sum_a pi(a|s) * P(s,a,s')
    P_pi = torch.einsum('sa,sat->st', pi, P)  # [S, S]
    # R_pi[s] = sum_a pi(a|s) * R(s,a)
    R_pi = torch.einsum('sa,sa->s', pi, R)  # [S]
    
    print(f"P_pi shape: {P_pi.shape}")
    print(f"R_pi shape: {R_pi.shape}")
    
    # Bellman expectation operator: T_pi(V) = R_pi + gamma * P_pi @ V
    V = torch.randn(n_states, device=device)
    T_pi_V = R_pi + gamma * (P_pi @ V)
    
    print(f"\nInitial V shape: {V.shape}")
    print(f"T_pi(V) shape: {T_pi_V.shape}")
    
    # Test contraction property
    V1 = torch.randn(n_states, device=device)
    V2 = torch.randn(n_states, device=device)
    
    T_V1 = R_pi + gamma * (P_pi @ V1)
    T_V2 = R_pi + gamma * (P_pi @ V2)
    
    dist_V = torch.max(torch.abs(V1 - V2)).item()
    dist_T = torch.max(torch.abs(T_V1 - T_V2)).item()
    
    print(f"\n||V1 - V2||_inf = {dist_V:.6f}")
    print(f"||T(V1) - T(V2)||_inf = {dist_T:.6f}")
    print(f"Contraction factor: {dist_T / dist_V:.6f} (should be <= {gamma})")
    
    assert dist_T <= gamma * dist_V + 1e-6, "Bellman operator should be a contraction"
    print("✓ Bellman operator contraction property verified!")

def main():
    print("="*50)
    print("Experiment 01: Environment Setup and MDP Verification")
    print("="*50)
    
    # Set seed for reproducibility
    setup_seed(42)
    
    # Print system information
    print_system_info()
    
    # Test basic MDP tensor operations
    P, R, V, Q, pi = test_basic_mdp_tensors()
    
    # Test Bellman operator properties
    test_bellman_operator()
    
    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()