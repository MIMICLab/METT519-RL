#!/usr/bin/env python3
"""
RL2025 - Lecture 4: Experiment 03 - Policy Evaluation

This experiment implements policy evaluation using the Bellman expectation
equation, demonstrating fixed-point iteration and contraction properties.

Learning objectives:
- Implement iterative policy evaluation
- Observe contraction and convergence
- Compare different policies (random vs. fixed)
- Understand the role of gamma in convergence

Prerequisites: exp02_gridworld.py completed
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import GridWorld from previous experiment
from exp02_gridworld import GridWorldMDP, GridWorldSpec, create_classic_gridworld, setup_seed, get_device

def policy_evaluation(
    P: torch.Tensor,           # [S, A, S]
    R: torch.Tensor,           # [S, A]
    pi: torch.Tensor,          # [S] for deterministic or [S, A] for stochastic
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
    method: str = 'auto',      # 'iterative', 'direct', or 'auto'
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Evaluate a policy using iterative policy evaluation.
    
    Args:
        P: Transition probabilities [S, A, S]
        R: Reward function [S, A]
        pi: Policy - either [S] for deterministic or [S, A] for stochastic
        gamma: Discount factor
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        method: Evaluation strategy ('iterative', 'direct', or 'auto')
        verbose: Print progress
    
    Returns:
        Dictionary with V (values), history, and convergence info
    """
    device = P.device
    n_states = P.shape[0]
    n_actions = P.shape[1]

    method = method.lower()
    if method not in {'iterative', 'direct', 'auto'}:
        raise ValueError(f"Invalid policy evaluation method: {method}")

    # Handle both deterministic and stochastic policies
    if pi.dim() == 1:
        pi_stoch = torch.zeros((n_states, n_actions), device=device)
        pi_stoch[torch.arange(n_states), pi] = 1.0
    else:
        pi_stoch = pi

    # Compute P_pi: [S, S] transition matrix under policy
    P_pi = torch.einsum('sa,sat->st', pi_stoch, P)

    # Compute R_pi: [S] expected reward under policy
    R_pi = torch.einsum('sa,sa->s', pi_stoch, R)

    # Initialize value function
    V = torch.zeros(n_states, dtype=torch.float32, device=device)

    history = {
        'values': [],
        'deltas': [],
        'bellman_residuals': []
    }

    iterations = 0
    delta = float('inf')
    bellman_residual = float('inf')
    converged = False

    if method != 'direct':
        if verbose:
            print(f"\nPolicy Evaluation:")
            print(f"  States: {n_states}, Actions: {n_actions}")
            print(f"  Gamma: {gamma}, Tolerance: {tolerance}")

        for iteration in range(max_iterations):
            iterations = iteration + 1
            V_old = V.clone()

            # Bellman expectation update: V = R_pi + gamma * P_pi @ V
            V = R_pi + gamma * (P_pi @ V_old)

            # Compute convergence metrics
            delta = torch.max(torch.abs(V - V_old)).item()
            bellman_residual = torch.max(torch.abs(V - (R_pi + gamma * P_pi @ V))).item()

            history['values'].append(V.clone().cpu())
            history['deltas'].append(delta)
            history['bellman_residuals'].append(bellman_residual)

            if verbose and ((iteration < 100 and iteration % 10 == 0) or (iteration % 100 == 0)):
                print(f"  Iteration {iteration:4d}: delta = {delta:.2e}, residual = {bellman_residual:.2e}")

            if delta < tolerance:
                converged = True
                break

        if converged or method == 'iterative':
            if verbose:
                if converged:
                    print(f"\n✓ Converged after {iterations} iterations")
                    print(f"  Final delta: {delta:.2e}")
                    print(f"  Final residual: {bellman_residual:.2e}")
                else:
                    print(f"\n⚠ Maximum iterations ({max_iterations}) reached without meeting tolerance")

            return {
                'V': V,
                'iterations': iterations,
                'converged': converged,
                'final_delta': delta,
                'history': history,
                'method': 'iterative'
            }

        if verbose and method == 'auto':
            print(f"\n⚠ Iterative evaluation did not reach tolerance; switching to direct solve")
    else:
        # Direct-only mode keeps history focused on final solution
        history = {
            'values': [],
            'deltas': [],
            'bellman_residuals': []
        }

    # Direct solution using linear system: (I - gamma * P_pi) V = R_pi
    identity = torch.eye(n_states, device=device, dtype=P_pi.dtype)
    system_matrix = identity - gamma * P_pi
    # Use double precision for numerical stability before casting back
    solution = torch.linalg.solve(system_matrix.double(), R_pi.double().unsqueeze(-1)).squeeze(-1)
    V = solution.to(P_pi.dtype)

    bellman_residual = torch.max(torch.abs(V - (R_pi + gamma * P_pi @ V))).item()
    history['values'].append(V.clone().cpu())
    history['deltas'].append(0.0)
    history['bellman_residuals'].append(bellman_residual)

    return {
        'V': V,
        'iterations': iterations if method != 'direct' else 0,
        'converged': True,
        'final_delta': 0.0,
        'history': history,
        'method': 'direct',
        'bellman_residual': bellman_residual
    }

def test_contraction_property():
    """Demonstrate that Bellman operator is a contraction"""
    print("\n" + "="*50)
    print("Testing Contraction Property")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    # Random policy
    pi = torch.ones((mdp.n_states, mdp.n_actions), device=device) / mdp.n_actions
    
    # Compute operators
    P_pi = torch.einsum('sa,sat->st', pi, mdp.P)
    R_pi = torch.einsum('sa,sa->s', pi, mdp.R)
    
    # Test with two random value functions
    V1 = torch.randn(mdp.n_states, device=device)
    V2 = torch.randn(mdp.n_states, device=device)
    
    # Apply Bellman operator
    T_V1 = R_pi + mdp.spec.gamma * (P_pi @ V1)
    T_V2 = R_pi + mdp.spec.gamma * (P_pi @ V2)
    
    # Compute distances
    dist_V = torch.max(torch.abs(V1 - V2)).item()
    dist_TV = torch.max(torch.abs(T_V1 - T_V2)).item()
    
    contraction_factor = dist_TV / dist_V if dist_V > 0 else 0
    
    print(f"\nContraction test:")
    print(f"  ||V1 - V2||∞ = {dist_V:.6f}")
    print(f"  ||T(V1) - T(V2)||∞ = {dist_TV:.6f}")
    print(f"  Contraction factor: {contraction_factor:.6f}")
    print(f"  Gamma: {mdp.spec.gamma}")
    print(f"  Valid contraction: {contraction_factor <= mdp.spec.gamma}")

def compare_policies():
    """Compare value functions for different policies"""
    print("\n" + "="*50)
    print("Comparing Different Policies")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    policies = {
        'Random': torch.ones((mdp.n_states, mdp.n_actions), device=device) / mdp.n_actions,
        'Always UP': torch.zeros(mdp.n_states, dtype=torch.long, device=device),
        'Always RIGHT': torch.ones(mdp.n_states, dtype=torch.long, device=device),
        'Cautious': torch.zeros(mdp.n_states, dtype=torch.long, device=device)  # Modified below
    }
    
    # Cautious policy: avoid moving toward pit
    for s in range(mdp.n_states):
        r, c = mdp.state_to_pos[s]
        # If we're in the leftmost column, go right; otherwise go up
        if c == 0:
            policies['Cautious'][s] = 1  # RIGHT
        else:
            policies['Cautious'][s] = 0  # UP
    
    results = {}
    
    for name, policy in policies.items():
        print(f"\nEvaluating {name} policy...")
        result = policy_evaluation(
            mdp.P, mdp.R, policy, 
            gamma=mdp.spec.gamma,
            tolerance=1e-8,
            verbose=False
        )
        results[name] = result
        
        print(f"  Converged in {result['iterations']} iterations")
        print(f"  Mean value: {result['V'].mean().item():.3f}")
        
        # Print grid with values
        print(f"\n  Value function for {name}:")
        mdp.render_grid(result['V'])
    
    return results, mdp

def analyze_convergence_rates():
    """Analyze how gamma affects convergence rate"""
    print("\n" + "="*50)
    print("Analyzing Convergence Rates")
    print("="*50)
    
    device = get_device()
    
    gammas = [0.5, 0.9, 0.95, 0.99, 0.999]
    convergence_data = {}
    
    for gamma in gammas:
        # Create GridWorld with specific gamma
        spec = GridWorldSpec(
            grid=["S..G", ".#.P", "...."],
            terminal_rewards={(0, 3): +1.0, (1, 3): -1.0},
            step_cost=-0.04,
            slip_prob=0.2,
            gamma=gamma
        )
        mdp = GridWorldMDP(spec, device)
        
        # Random policy
        pi = torch.ones((mdp.n_states, mdp.n_actions), device=device) / mdp.n_actions
        
        # Evaluate
        result = policy_evaluation(
            mdp.P, mdp.R, pi,
            gamma=gamma,
            tolerance=1e-8,
            max_iterations=10000,
            verbose=False
        )
        
        convergence_data[gamma] = {
            'iterations': result['iterations'],
            'deltas': result['history']['deltas']
        }
        
        print(f"  γ = {gamma:.3f}: converged in {result['iterations']:4d} iterations")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    
    for gamma in gammas:
        deltas = convergence_data[gamma]['deltas'][:100]  # First 100 iterations
        plt.semilogy(deltas, label=f'γ = {gamma}', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Max |V(s) - V_old(s)|')
    plt.title('Policy Evaluation Convergence for Different Discount Factors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/policy_eval_convergence.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Convergence plot saved to ./figures/policy_eval_convergence.png")
    plt.close()

def main():
    print("="*50)
    print("Experiment 03: Policy Evaluation")
    print("="*50)
    
    setup_seed(42)
    
    # Test contraction property
    test_contraction_property()
    
    # Compare different policies
    results, mdp = compare_policies()
    
    # Analyze convergence rates
    analyze_convergence_rates()
    
    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
