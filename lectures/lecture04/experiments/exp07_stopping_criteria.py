#!/usr/bin/env python3
"""
RL2025 - Lecture 4: Experiment 07 - Stopping Criteria and Error Bounds

This experiment explores different stopping criteria for DP algorithms
and demonstrates theoretical error bounds for approximate solutions.

Learning objectives:
- Understand different stopping criteria
- Compute and verify error bounds
- Analyze trade-offs between accuracy and computation
- Implement efficient early stopping

Prerequisites: exp06_value_iteration.py completed
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

from exp02_gridworld import GridWorldMDP, GridWorldSpec, create_classic_gridworld, setup_seed, get_device
from exp04_policy_improvement import compute_q_values
from exp06_value_iteration import value_iteration

FIGURES_DIR = Path(__file__).parent / "figures"


def value_iteration_with_bounds(
    P: torch.Tensor,
    R: torch.Tensor,
    gamma: float = 0.99,
    epsilon: float = 0.01,
    max_iterations: int = 10000,
    verbose: bool = True
) -> Dict:
    """
    Value iteration with theoretical error bounds.
    
    Stops when guaranteed to be epsilon-optimal:
    ||V_k - V*||_inf <= epsilon
    
    Uses stopping criterion:
    delta_k < (1-gamma) * epsilon / (2*gamma)
    
    Args:
        P: Transition probabilities [S, A, S]
        R: Reward function [S, A]
        gamma: Discount factor
        epsilon: Desired accuracy for ||V - V*||_inf
        max_iterations: Maximum iterations
        verbose: Print progress
    
    Returns:
        Dictionary with values, bounds, and history
    """
    device = P.device
    n_states = P.shape[0]
    
    if gamma < 0 or gamma >= 1:
        raise ValueError("This implementation assumes a discount factor gamma in [0, 1).")

    # Compute stopping threshold from desired accuracy
    if gamma == 0:
        threshold = epsilon
    else:
        threshold = (1 - gamma) * epsilon / (2 * gamma)
    
    V = torch.zeros(n_states, dtype=torch.float32, device=device)
    
    history = {
        'values': [],
        'deltas': [],
        'upper_bounds': [],
        'lower_bounds': [],
        'true_errors': []  # Will be computed if we have V_star
    }
    
    if verbose:
        print(f"\nValue Iteration with Error Bounds:")
        print(f"  Target accuracy (ε): {epsilon}")
        print(f"  Stopping threshold: {threshold:.6e}")
        print(f"  Theoretical bound: ||V - V*||∞ ≤ {epsilon}")
    
    start_time = time.time()
    
    for k in range(max_iterations):
        V_old = V.clone()
        
        # Bellman update
        Q = compute_q_values(P, R, V_old, gamma)
        V, _ = torch.max(Q, dim=1)
        
        # Compute delta
        delta = torch.max(torch.abs(V - V_old)).item()
        
        # Compute error bounds
        # Upper bound: V* <= V_k + gamma/(1-gamma) * delta_k * 1
        # Lower bound: V* >= V_k - gamma/(1-gamma) * delta_k * 1
        bound_term = (gamma / (1 - gamma)) * delta if gamma != 1 else float('inf')
        upper_bound = V + bound_term
        lower_bound = V - bound_term
        
        history['values'].append(V.clone().cpu())
        history['deltas'].append(delta)
        history['upper_bounds'].append(upper_bound.clone().cpu())
        history['lower_bounds'].append(lower_bound.clone().cpu())
        
        if verbose and (k < 10 or k % 10 == 0):
            max_bound_width = 2 * bound_term
            print(f"  Iter {k:4d}: δ={delta:.6e}, bound width={max_bound_width:.6e}")
        
        # Check stopping criterion
        if delta < threshold:
            if verbose:
                print(f"\n✓ Converged to ε-optimal solution")
                print(f"  Iterations: {k + 1}")
                print(f"  Final delta: {delta:.6e}")
                print(f"  Guaranteed accuracy: ||V - V*||∞ ≤ {epsilon}")
                print(f"  Time: {time.time() - start_time:.3f}s")
            break
    
    # Extract policy
    Q_final = compute_q_values(P, R, V, gamma)
    pi = torch.argmax(Q_final, dim=1)
    
    return {
        'V': V,
        'pi': pi,
        'iterations': k + 1,
        'epsilon': epsilon,
        'threshold': threshold,
        'final_delta': delta,
        'history': history,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound
    }

def compare_stopping_criteria():
    """Compare different stopping criteria and their effects"""
    print("\n" + "="*50)
    print("Comparing Different Stopping Criteria")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    # First, get true optimal values for comparison
    true_result = value_iteration(
        mdp.P, mdp.R,
        gamma=mdp.spec.gamma,
        tolerance=1e-12,  # Very tight tolerance
        max_iterations=10000,
        verbose=False
    )
    V_star = true_result['V_star']
    
    # Different epsilon values
    epsilons = [1.0, 0.1, 0.01, 0.001, 0.0001]
    results = {}
    
    for eps in epsilons:
        print(f"\nTesting ε = {eps}:")
        
        result = value_iteration_with_bounds(
            mdp.P, mdp.R,
            gamma=mdp.spec.gamma,
            epsilon=eps,
            verbose=False
        )
        
        # Compute true error
        true_error = torch.max(torch.abs(result['V'] - V_star)).item()
        
        results[eps] = {
            'result': result,
            'true_error': true_error,
            'iterations': result['iterations'],
            'threshold': result['threshold']
        }
        
        print(f"  Iterations: {result['iterations']}")
        print(f"  Threshold used: {result['threshold']:.6e}")
        print(f"  True error: {true_error:.6e}")
        print(f"  Guaranteed bound: {eps}")
        print(f"  Bound satisfied: {true_error <= eps}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Iterations vs epsilon
    ax = axes[0, 0]
    iterations = [results[eps]['iterations'] for eps in epsilons]
    ax.semilogx(epsilons, iterations, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Target Accuracy (ε)')
    ax.set_ylabel('Iterations Required')
    ax.set_title('Computational Cost vs Accuracy')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Plot 2: True error vs guaranteed bound
    ax = axes[0, 1]
    true_errors = [results[eps]['true_error'] for eps in epsilons]
    ax.loglog(epsilons, epsilons, 'k--', label='Guaranteed bound', alpha=0.5)
    ax.loglog(epsilons, true_errors, 'o-', label='True error', linewidth=2, markersize=8)
    ax.set_xlabel('Target Accuracy (ε)')
    ax.set_ylabel('Error')
    ax.set_title('True Error vs Guaranteed Bound')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Plot 3: Convergence for different epsilons
    ax = axes[1, 0]
    for eps in [0.1, 0.01, 0.001]:
        deltas = results[eps]['result']['history']['deltas']
        ax.semilogy(deltas[:100], label=f'ε={eps}', linewidth=2)
        threshold = results[eps]['threshold']
        ax.axhline(y=threshold, linestyle='--', alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Delta')
    ax.set_title('Convergence with Different Stopping Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Bound width evolution
    ax = axes[1, 1]
    eps = 0.01
    result = results[eps]['result']
    upper = torch.stack(result['history']['upper_bounds'])
    lower = torch.stack(result['history']['lower_bounds'])
    width = (upper - lower).max(dim=1)[0]
    
    ax.semilogy(width.numpy(), linewidth=2)
    ax.axhline(y=2*eps, color='r', linestyle='--', label=f'Target: 2ε={2*eps}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Bound Width')
    ax.set_title(f'Error Bound Width Evolution (ε={eps})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stopping_path = FIGURES_DIR / "stopping_criteria.png"
    plt.savefig(stopping_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Stopping criteria analysis saved to {stopping_path}")
    plt.close(fig)

def test_span_seminorm():
    """Test span seminorm as alternative stopping criterion"""
    print("\n" + "="*50)
    print("Testing Span Seminorm Criterion")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    def span(V):
        """Compute span seminorm: max(V) - min(V)"""
        return (V.max() - V.min()).item()
    
    # Custom value iteration with span criterion
    V = torch.zeros(mdp.n_states, dtype=torch.float32, device=device)
    gamma = mdp.spec.gamma
    epsilon = 0.01
    
    # Span-based threshold
    span_threshold = (1 - gamma) * epsilon / gamma if gamma != 0 else epsilon
    
    print(f"\nSpan Seminorm Stopping:")
    print(f"  Target accuracy: {epsilon}")
    print(f"  Span threshold: {span_threshold:.6e}")
    
    history_span = []
    history_delta = []
    
    for k in range(1000):
        V_old = V.clone()
        Q = compute_q_values(mdp.P, mdp.R, V_old, gamma)
        V, _ = torch.max(Q, dim=1)
        
        delta = torch.max(torch.abs(V - V_old)).item()
        span_V = span(V - V_old)
        
        history_delta.append(delta)
        history_span.append(span_V)
        
        if k < 10 or k % 10 == 0:
            print(f"  Iter {k:3d}: δ={delta:.6e}, span={span_V:.6e}")
        
        if span_V < span_threshold:
            print(f"\n✓ Converged using span criterion after {k+1} iterations")
            break
    
    # Compare with standard criterion
    standard_result = value_iteration_with_bounds(
        mdp.P, mdp.R,
        gamma=gamma,
        epsilon=epsilon,
        verbose=False
    )
    
    print(f"\nComparison:")
    print(f"  Span criterion iterations: {k+1}")
    print(f"  Standard criterion iterations: {standard_result['iterations']}")
    
    # Plot comparison
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(history_delta, label='Delta (||V_k - V_{k-1}||∞)', linewidth=2)
    plt.semilogy(history_span, label='Span (max-min)', linewidth=2)
    plt.axhline(y=standard_result['threshold'], color='r', linestyle='--', 
               alpha=0.5, label='Standard threshold')
    plt.axhline(y=span_threshold, color='g', linestyle='--', 
               alpha=0.5, label='Span threshold')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Stopping Metrics Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    ratio = np.array(history_span) / (np.array(history_delta) + 1e-10)
    plt.plot(ratio, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Span / Delta Ratio')
    plt.title('Relative Tightness of Bounds')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    span_path = FIGURES_DIR / "span_seminorm.png"
    plt.savefig(span_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Span seminorm analysis saved to {span_path}")
    plt.close()

def analyze_policy_switching():
    """Analyze when the greedy policy stabilizes during VI"""
    print("\n" + "="*50)
    print("Analyzing Policy Switching During Value Iteration")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    # Run value iteration and track policy changes
    V = torch.zeros(mdp.n_states, dtype=torch.float32, device=device)
    gamma = mdp.spec.gamma
    
    policy_history = []
    policy_changes = []
    value_not_converged = []
    
    prev_policy = None
    policy_stable_iteration = None
    
    for k in range(200):
        V_old = V.clone()
        Q = compute_q_values(mdp.P, mdp.R, V_old, gamma)
        V, _ = torch.max(Q, dim=1)
        
        policy = torch.argmax(Q, dim=1)
        policy_history.append(policy.clone().cpu())
        
        delta = torch.max(torch.abs(V - V_old)).item()
        value_not_converged.append(delta)
        
        if prev_policy is not None:
            changes = (policy != prev_policy).sum().item()
            policy_changes.append(changes)
            
            if changes == 0 and policy_stable_iteration is None:
                policy_stable_iteration = k
                print(f"\nPolicy stabilized at iteration {k}")
                print(f"Value delta at stabilization: {delta:.6e}")
        else:
            policy_changes.append(mdp.n_states)  # All "changed" on first iteration
        
        prev_policy = policy.clone()
        
        if delta < 1e-10:
            print(f"Values converged at iteration {k}")
            break
    
    # Plot analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Policy changes over time
    ax = axes[0, 0]
    ax.plot(policy_changes, linewidth=2)
    if policy_stable_iteration:
        ax.axvline(x=policy_stable_iteration, color='r', linestyle='--', 
                  label=f'Policy stable (iter {policy_stable_iteration})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of State Policy Changes')
    ax.set_title('Policy Changes During Value Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Value convergence
    ax = axes[0, 1]
    ax.semilogy(value_not_converged, linewidth=2)
    if policy_stable_iteration:
        ax.axvline(x=policy_stable_iteration, color='r', linestyle='--',
                  label='Policy stable')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value Delta (log scale)')
    ax.set_title('Value Function Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Policy evolution for specific states
    ax = axes[1, 0]
    n_states_to_plot = min(5, mdp.n_states)
    for s in range(n_states_to_plot):
        policies_s = [p[s].item() for p in policy_history]
        ax.plot(policies_s, label=f'State {s}', linewidth=2, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Action')
    ax.set_yticks(range(4))
    ax.set_yticklabels(['UP', 'RIGHT', 'DOWN', 'LEFT'])
    ax.set_title('Policy Evolution for Individual States')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.text(0.1, 0.8, f"Total iterations: {k+1}", fontsize=12)
    ax.text(0.1, 0.6, f"Policy stable at: iteration {policy_stable_iteration}", fontsize=12)
    if policy_stable_iteration:
        ax.text(0.1, 0.4, f"Additional iterations for value convergence: {k - policy_stable_iteration}", fontsize=12)
        ax.text(0.1, 0.2, f"Ratio (value iters / policy stable): {(k+1) / policy_stable_iteration:.2f}x", fontsize=12)
    ax.set_title('Summary')
    ax.axis('off')
    
    plt.tight_layout()
    policy_switch_path = FIGURES_DIR / "policy_switching.png"
    plt.savefig(policy_switch_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Policy switching analysis saved to {policy_switch_path}")
    plt.close()

def main():
    print("="*50)
    print("Experiment 07: Stopping Criteria and Error Bounds")
    print("="*50)
    
    setup_seed(42)
    
    # Test error bounds
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    print("\nTesting Error Bounds with ε = 0.01:")
    result = value_iteration_with_bounds(
        mdp.P, mdp.R,
        gamma=mdp.spec.gamma,
        epsilon=0.01,
        verbose=True
    )
    
    # Compare different stopping criteria
    compare_stopping_criteria()
    
    # Test span seminorm
    test_span_seminorm()
    
    # Analyze policy switching
    analyze_policy_switching()
    
    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
