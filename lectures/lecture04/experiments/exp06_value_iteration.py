#!/usr/bin/env python3
"""
RL2025 - Lecture 4: Experiment 06 - Value Iteration

This experiment implements value iteration algorithm using the Bellman
optimality operator, demonstrating direct convergence to optimal values.

Learning objectives:
- Implement value iteration algorithm
- Compare with policy iteration efficiency
- Understand Bellman optimality operator
- Analyze convergence properties

Prerequisites: exp05_policy_iteration.py completed
"""

import os
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

from exp02_gridworld import GridWorldMDP, GridWorldSpec, create_classic_gridworld, setup_seed, get_device, ACTION_NAMES
from exp04_policy_improvement import compute_q_values

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR = Path(os.environ.get("LECTURE04_FIGURES_DIR", DEFAULT_FIGURES_DIR))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from exp05_policy_iteration import policy_iteration

def value_iteration(
    P: torch.Tensor,           # [S, A, S]
    R: torch.Tensor,           # [S, A]
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    initial_values: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> Dict:
    """
    Value Iteration algorithm using Bellman optimality operator.
    
    Args:
        P: Transition probabilities [S, A, S]
        R: Reward function [S, A]
        gamma: Discount factor
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        initial_values: Initial value function (zeros if None)
        verbose: Print progress
    
    Returns:
        Dictionary with optimal policy, values, and history
    """
    device = P.device
    n_states = P.shape[0]
    n_actions = P.shape[1]
    
    # Initialize values
    if initial_values is None:
        V = torch.zeros(n_states, dtype=torch.float32, device=device)
        if verbose:
            print("Starting with zero value function")
    else:
        V = initial_values.clone()
        if verbose:
            print("Starting with provided initial values")
    
    history = {
        'values': [],
        'deltas': [],
        'bellman_residuals': [],
        'q_values': [],
        'greedy_policies': [],
        'timestamps': []
    }
    
    start_time = time.time()
    
    if verbose:
        print(f"\nValue Iteration:")
        print(f"  States: {n_states}, Actions: {n_actions}")
        print(f"  Gamma: {gamma}, Tolerance: {tolerance}")
        print(f"  {'Iteration':<10} {'Delta':<12} {'Bellman Res':<12} {'Mean V':<10}")
        print(f"  {'-'*46}")
    
    for iteration in range(max_iterations):
        V_old = V.clone()
        
        # Compute Q-values: Q(s,a) = R(s,a) + gamma * sum_s' P(s,a,s') * V(s')
        Q = compute_q_values(P, R, V_old, gamma)
        
        # Bellman optimality update: V(s) = max_a Q(s,a)
        V, _ = torch.max(Q, dim=1)
        
        # Compute convergence metrics
        delta = torch.max(torch.abs(V - V_old)).item()
        
        # Bellman residual: ||V - T*V||
        Q_current = compute_q_values(P, R, V, gamma)
        V_bellman, _ = torch.max(Q_current, dim=1)
        bellman_residual = torch.max(torch.abs(V - V_bellman)).item()
        
        # Extract greedy policy
        greedy_policy = torch.argmax(Q, dim=1)
        
        # Store history
        history['values'].append(V.clone().cpu())
        history['deltas'].append(delta)
        history['bellman_residuals'].append(bellman_residual)
        history['q_values'].append(Q.clone().cpu())
        history['greedy_policies'].append(greedy_policy.clone().cpu())
        history['timestamps'].append(time.time() - start_time)
        
        # Print progress
        if verbose and (iteration < 20 or iteration % 10 == 0):
            print(f"  {iteration:<10} {delta:<12.6e} {bellman_residual:<12.6e} {V.mean().item():<10.3f}")
        
        # Check convergence
        if delta < tolerance:
            if verbose:
                print(f"\n✓ Converged after {iteration + 1} iterations")
                print(f"  Final delta: {delta:.2e}")
                print(f"  Final residual: {bellman_residual:.2e}")
                print(f"  Total time: {time.time() - start_time:.3f}s")
            break
    else:
        if verbose:
            print(f"\n⚠ Maximum iterations ({max_iterations}) reached")
    
    # Final Q-values and policy
    Q_star = compute_q_values(P, R, V, gamma)
    pi_star = torch.argmax(Q_star, dim=1)
    
    return {
        'V_star': V,
        'Q_star': Q_star,
        'pi_star': pi_star,
        'iterations': iteration + 1,
        'converged': delta < tolerance,
        'final_delta': delta,
        'history': history,
        'total_time': time.time() - start_time
    }

def compare_vi_pi():
    """Compare Value Iteration with Policy Iteration"""
    print("\n" + "="*50)
    print("Comparing Value Iteration vs Policy Iteration")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    # Run Value Iteration
    print("\n1. Value Iteration:")
    vi_start = time.time()
    vi_result = value_iteration(
        mdp.P, mdp.R,
        gamma=mdp.spec.gamma,
        tolerance=1e-8,
        verbose=False
    )
    vi_time = time.time() - vi_start
    
    print(f"   Iterations: {vi_result['iterations']}")
    print(f"   Time: {vi_time:.4f}s")
    print(f"   Final mean value: {vi_result['V_star'].mean().item():.6f}")
    
    # Run Policy Iteration
    print("\n2. Policy Iteration:")
    pi_start = time.time()
    pi_result = policy_iteration(
        mdp.P, mdp.R,
        gamma=mdp.spec.gamma,
        eval_tolerance=1e-8,
        verbose=False
    )
    pi_time = time.time() - pi_start
    
    print(f"   Iterations: {pi_result['iterations']}")
    print(f"   Time: {pi_time:.4f}s")
    print(f"   Final mean value: {pi_result['V_star'].mean().item():.6f}")
    
    # Compare optimal values
    value_diff = torch.max(torch.abs(vi_result['V_star'] - pi_result['V_star'])).item()
    policy_same = torch.equal(vi_result['pi_star'], pi_result['pi_star'])
    
    print(f"\n3. Comparison:")
    print(f"   Max value difference: {value_diff:.2e}")
    print(f"   Same optimal policy: {policy_same}")
    print(f"   VI/PI time ratio: {vi_time/pi_time:.2f}x")
    
    # Plot convergence comparison
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Delta convergence
    plt.subplot(1, 2, 1)
    plt.semilogy(vi_result['history']['deltas'], label='Value Iteration', linewidth=2)
    
    # For PI, we need to aggregate deltas from evaluations
    pi_deltas = []
    for i in range(len(pi_result['history']['mean_values'])):
        pi_deltas.append(abs(pi_result['history']['mean_values'][i] - 
                           (pi_result['history']['mean_values'][i-1] 
                            if i > 0 else 0)))
    plt.semilogy(pi_deltas[1:], label='Policy Iteration', linewidth=2, linestyle='--')
    
    plt.xlabel('Iteration')
    plt.ylabel('Delta (log scale)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Value evolution
    plt.subplot(1, 2, 2)
    vi_means = [v.mean().item() for v in vi_result['history']['values']]
    plt.plot(vi_means, label='Value Iteration', linewidth=2)
    plt.plot(pi_result['history']['mean_values'], 
            label='Policy Iteration', linewidth=2, linestyle='--', marker='o')
    
    plt.xlabel('Iteration')
    plt.ylabel('Mean State Value')
    plt.title('Mean Value Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = FIGURES_DIR / 'vi_pi_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to {comparison_path}")
    plt.close()
    
    return vi_result, pi_result

def analyze_bellman_operator():
    """Analyze properties of Bellman optimality operator"""
    print("\n" + "="*50)
    print("Analyzing Bellman Optimality Operator")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    # Test contraction property
    V1 = torch.randn(mdp.n_states, device=device)
    V2 = torch.randn(mdp.n_states, device=device)
    
    # Apply Bellman optimality operator
    Q1 = compute_q_values(mdp.P, mdp.R, V1, mdp.spec.gamma)
    T_V1, _ = torch.max(Q1, dim=1)
    
    Q2 = compute_q_values(mdp.P, mdp.R, V2, mdp.spec.gamma)
    T_V2, _ = torch.max(Q2, dim=1)
    
    # Compute distances
    dist_V = torch.max(torch.abs(V1 - V2)).item()
    dist_TV = torch.max(torch.abs(T_V1 - T_V2)).item()
    
    print(f"\nContraction test (Bellman optimality):")
    print(f"  ||V1 - V2||∞ = {dist_V:.6f}")
    print(f"  ||T*V1 - T*V2||∞ = {dist_TV:.6f}")
    print(f"  Contraction factor: {dist_TV / dist_V if dist_V > 0 else 0:.6f}")
    print(f"  Gamma: {mdp.spec.gamma}")
    print(f"  Valid contraction: {dist_TV <= mdp.spec.gamma * dist_V + 1e-6}")
    
    # Test monotonicity: if V1 <= V2, then T*V1 <= T*V2
    V1_mono = torch.zeros(mdp.n_states, device=device)
    V2_mono = torch.ones(mdp.n_states, device=device)
    
    Q1_mono = compute_q_values(mdp.P, mdp.R, V1_mono, mdp.spec.gamma)
    T_V1_mono, _ = torch.max(Q1_mono, dim=1)
    
    Q2_mono = compute_q_values(mdp.P, mdp.R, V2_mono, mdp.spec.gamma)
    T_V2_mono, _ = torch.max(Q2_mono, dim=1)
    
    monotonic = torch.all(T_V1_mono <= T_V2_mono + 1e-6)
    print(f"\nMonotonicity test:")
    print(f"  V1 ≤ V2: True (by construction)")
    print(f"  T*V1 ≤ T*V2: {monotonic}")

def test_different_initializations():
    """Test value iteration with different initial values"""
    print("\n" + "="*50)
    print("Testing Different Initial Values")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    initializations = {
        'Zero': torch.zeros(mdp.n_states, device=device),
        'Optimistic': torch.ones(mdp.n_states, device=device) * 10.0,
        'Pessimistic': torch.ones(mdp.n_states, device=device) * -10.0,
        'Random': torch.randn(mdp.n_states, device=device),
        'Terminal': torch.zeros(mdp.n_states, device=device)
    }
    
    # Set terminal values to their rewards
    for s in range(mdp.n_states):
        if mdp.terminal_states[s]:
            initializations['Terminal'][s] = mdp.terminal_rewards[s]
    
    results = {}
    
    plt.figure(figsize=(12, 8))
    
    for idx, (name, init_V) in enumerate(initializations.items()):
        print(f"\nInitialization: {name}")
        print(f"  Initial mean: {init_V.mean().item():.3f}")
        print(f"  Initial range: [{init_V.min().item():.3f}, {init_V.max().item():.3f}]")
        
        result = value_iteration(
            mdp.P, mdp.R,
            gamma=mdp.spec.gamma,
            tolerance=1e-8,
            initial_values=init_V,
            verbose=False
        )
        
        results[name] = result
        print(f"  Converged in {result['iterations']} iterations")
        print(f"  Final mean: {result['V_star'].mean().item():.6f}")
        
        # Plot convergence
        plt.subplot(2, 3, idx + 1)
        mean_values = [v.mean().item() for v in result['history']['values']]
        plt.plot(mean_values, linewidth=2)
        plt.axhline(y=result['V_star'].mean().item(), color='r', 
                   linestyle='--', alpha=0.5, label='Optimal')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Value')
        plt.title(f'{name} Initialization')
        plt.grid(True, alpha=0.3)
        if idx == 0:
            plt.legend()
    
    plt.tight_layout()
    init_path = FIGURES_DIR / 'vi_initializations.png'
    plt.savefig(init_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Initialization comparison saved to {init_path}")
    plt.close()
    
    # Verify all converge to same solution
    V_star_ref = results['Zero']['V_star']
    print("\nVerifying convergence to unique solution:")
    for name, result in results.items():
        diff = torch.max(torch.abs(result['V_star'] - V_star_ref)).item()
        print(f"  {name:12s}: max difference = {diff:.2e}")

def main():
    print("="*50)
    print("Experiment 06: Value Iteration")
    print("="*50)
    
    setup_seed(42)
    
    # Basic value iteration test
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    print("\nRunning Value Iteration on Classic GridWorld...")
    result = value_iteration(
        mdp.P, mdp.R,
        gamma=mdp.spec.gamma,
        verbose=True
    )
    
    print(f"\nOptimal Policy:")
    mdp.render_grid(result['V_star'])
    
    # Compare with policy iteration
    vi_result, pi_result = compare_vi_pi()
    
    # Analyze Bellman operator
    analyze_bellman_operator()
    
    # Test different initializations
    test_different_initializations()
    
    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
