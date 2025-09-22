#!/usr/bin/env python3
"""
RL2025 - Lecture 4: Experiment 05 - Policy Iteration

This experiment implements the complete policy iteration algorithm,
alternating between policy evaluation and policy improvement until convergence.

Learning objectives:
- Implement full policy iteration algorithm
- Observe convergence to optimal policy
- Track policy changes through iterations
- Compare convergence with different initial policies

Prerequisites: exp04_policy_improvement.py completed
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

from exp02_gridworld import GridWorldMDP, GridWorldSpec, create_classic_gridworld, setup_seed, get_device, ACTION_NAMES
from exp03_policy_evaluation import policy_evaluation
from exp04_policy_improvement import policy_improvement, compute_q_values

def policy_iteration(
    P: torch.Tensor,           # [S, A, S]
    R: torch.Tensor,           # [S, A]
    gamma: float = 0.99,
    eval_tolerance: float = 1e-8,
    max_eval_iterations: int = 1000,
    max_policy_iterations: int = 100,
    initial_policy: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> Dict:
    """
    Policy Iteration algorithm.
    
    Args:
        P: Transition probabilities [S, A, S]
        R: Reward function [S, A]
        gamma: Discount factor
        eval_tolerance: Tolerance for policy evaluation
        max_eval_iterations: Max iterations for each evaluation
        max_policy_iterations: Max policy improvement iterations
        initial_policy: Initial policy (random if None)
        verbose: Print progress
    
    Returns:
        Dictionary with optimal policy, values, and history
    """
    device = P.device
    n_states = P.shape[0]
    n_actions = P.shape[1]
    
    # Initialize policy
    if initial_policy is None:
        # Start with uniform random policy
        pi = torch.ones((n_states, n_actions), device=device) / n_actions
        if verbose:
            print("Starting with uniform random policy")
    else:
        pi = initial_policy
        if verbose:
            print("Starting with provided initial policy")
    
    history = {
        'policies': [],
        'values': [],
        'mean_values': [],
        'policy_changes': [],
        'eval_iterations': [],
        'timestamps': []
    }
    
    start_time = time.time()
    
    for iteration in range(max_policy_iterations):
        if verbose:
            print(f"\n{'='*40}")
            print(f"Policy Iteration {iteration + 1}")
            print(f"{'='*40}")
        
        # Step 1: Policy Evaluation
        eval_start = time.time()
        eval_result = policy_evaluation(
            P, R, pi, gamma,
            tolerance=eval_tolerance,
            max_iterations=max_eval_iterations,
            verbose=False
        )
        V = eval_result['V']
        eval_time = time.time() - eval_start
        
        if verbose:
            print(f"Policy Evaluation:")
            print(f"  Converged in {eval_result['iterations']} iterations")
            print(f"  Time: {eval_time:.3f}s")
            print(f"  Mean value: {V.mean().item():.3f}")
        
        # Step 2: Policy Improvement
        improvement_result = policy_improvement(
            P, R, V, gamma,
            temperature=0.0,  # Deterministic greedy
            verbose=False
        )
        
        # Extract deterministic policy
        pi_new_det = improvement_result['pi']
        
        # Convert to stochastic for comparison
        pi_new = torch.zeros((n_states, n_actions), device=device)
        pi_new[torch.arange(n_states), pi_new_det] = 1.0
        
        # Check if policy changed
        if pi.dim() == 1:
            # Previous policy was deterministic
            pi_old_det = pi
        else:
            # Extract deterministic from stochastic
            pi_old_det = torch.argmax(pi, dim=1)
        
        policy_changed = not torch.equal(pi_new_det, pi_old_det)
        n_changes = (pi_new_det != pi_old_det).sum().item()
        
        if verbose:
            print(f"Policy Improvement:")
            print(f"  States with policy change: {n_changes}/{n_states}")
        
        # Store history
        history['policies'].append(pi_new_det.clone().cpu())
        history['values'].append(V.clone().cpu())
        history['mean_values'].append(V.mean().item())
        history['policy_changes'].append(n_changes)
        history['eval_iterations'].append(eval_result['iterations'])
        history['timestamps'].append(time.time() - start_time)
        
        # Check convergence
        if not policy_changed:
            if verbose:
                print(f"\n✓ Policy converged after {iteration + 1} iterations!")
                print(f"Total time: {time.time() - start_time:.3f}s")
            break
        
        # Update policy for next iteration
        pi = pi_new
    else:
        if verbose:
            print(f"\n⚠ Maximum iterations ({max_policy_iterations}) reached")
    
    # Final evaluation for optimal values
    final_eval = policy_evaluation(
        P, R, pi_new_det, gamma,
        tolerance=eval_tolerance,
        max_iterations=max_eval_iterations,
        verbose=False
    )
    
    return {
        'pi_star': pi_new_det,
        'V_star': final_eval['V'],
        'Q_star': compute_q_values(P, R, final_eval['V'], gamma),
        'iterations': iteration + 1,
        'converged': not policy_changed,
        'history': history,
        'total_time': time.time() - start_time
    }

def visualize_policy_evolution(mdp: GridWorldMDP, history: Dict):
    """Visualize how the policy evolves during iteration"""
    print("\n" + "="*50)
    print("Visualizing Policy Evolution")
    print("="*50)
    
    n_iterations = len(history['policies'])
    
    # Create figure with subplots for each iteration
    n_cols = min(4, n_iterations)
    n_rows = (n_iterations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_iterations == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    for i in range(n_iterations):
        ax = axes[i]
        ax.set_xlim(-0.5, mdp.width - 0.5)
        ax.set_ylim(-0.5, mdp.height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Iteration {i+1}\n(Mean V: {history["mean_values"][i]:.3f})')
        
        policy = history['policies'][i]
        
        for s in range(mdp.n_states):
            r, c = mdp.state_to_pos[s]
            if not mdp.terminal_states[s]:
                action = policy[s].item()
                ax.text(c, r, arrow_map[action], ha='center', va='center', fontsize=16)
            else:
                if mdp.terminal_rewards[s] > 0:
                    ax.text(c, r, 'G', ha='center', va='center', fontsize=14, color='green')
                else:
                    ax.text(c, r, 'P', ha='center', va='center', fontsize=14, color='red')
        
        # Add walls
        for r in range(mdp.height):
            for c in range(mdp.width):
                if not mdp.passable[r][c]:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=True, color='black'))
        
        ax.set_xticks(range(mdp.width))
        ax.set_yticks(range(mdp.height))
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_iterations, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/policy_evolution.png', dpi=150, bbox_inches='tight')
    print("✓ Policy evolution saved to ./figures/policy_evolution.png")
    plt.close()

def compare_initial_policies():
    """Compare convergence from different initial policies"""
    print("\n" + "="*50)
    print("Comparing Different Initial Policies")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    # Different initial policies
    initial_policies = {
        'Random': torch.ones((mdp.n_states, mdp.n_actions), device=device) / mdp.n_actions,
        'All UP': None,  # Will be created below
        'All RIGHT': None,
        'Diagonal': None
    }
    
    # Create deterministic initial policies
    all_up = torch.zeros((mdp.n_states, mdp.n_actions), device=device)
    all_up[:, 0] = 1.0  # All states choose UP
    initial_policies['All UP'] = all_up
    
    all_right = torch.zeros((mdp.n_states, mdp.n_actions), device=device)
    all_right[:, 1] = 1.0  # All states choose RIGHT
    initial_policies['All RIGHT'] = all_right
    
    # Diagonal policy: alternate UP and RIGHT
    diagonal = torch.zeros((mdp.n_states, mdp.n_actions), device=device)
    for s in range(mdp.n_states):
        r, c = mdp.state_to_pos[s]
        if (r + c) % 2 == 0:
            diagonal[s, 0] = 1.0  # UP
        else:
            diagonal[s, 1] = 1.0  # RIGHT
    initial_policies['Diagonal'] = diagonal
    
    results = {}
    
    for name, init_pi in initial_policies.items():
        print(f"\nRunning Policy Iteration with {name} initial policy...")
        
        result = policy_iteration(
            mdp.P, mdp.R,
            gamma=mdp.spec.gamma,
            initial_policy=init_pi,
            verbose=False
        )
        
        results[name] = result
        
        print(f"  Converged in {result['iterations']} iterations")
        print(f"  Final mean value: {result['V_star'].mean().item():.3f}")
        print(f"  Total time: {result['total_time']:.3f}s")
        print(f"  Total policy evaluations: {sum(result['history']['eval_iterations'])}")
    
    # Plot convergence comparison
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Mean value over iterations
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(range(1, len(result['history']['mean_values']) + 1),
                result['history']['mean_values'],
                marker='o', label=name, linewidth=2)
    
    plt.xlabel('Policy Iteration')
    plt.ylabel('Mean State Value')
    plt.title('Convergence from Different Initial Policies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Policy changes over iterations
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(range(1, len(result['history']['policy_changes']) + 1),
                result['history']['policy_changes'],
                marker='s', label=name, linewidth=2)
    
    plt.xlabel('Policy Iteration')
    plt.ylabel('Number of State Policy Changes')
    plt.title('Policy Changes per Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/policy_iteration_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison saved to ./figures/policy_iteration_comparison.png")
    plt.close()
    
    return results

def test_larger_gridworld():
    """Test policy iteration on a larger grid"""
    print("\n" + "="*50)
    print("Testing on Larger GridWorld")
    print("="*50)
    
    device = get_device()
    
    # Create a larger 6x6 grid
    grid = [
        "S.....",
        ".####.",
        ".....G",
        "P.....",
        ".####.",
        "......"
    ]
    
    spec = GridWorldSpec(
        grid=grid,
        terminal_rewards={(2, 5): +1.0, (3, 0): -1.0},
        step_cost=-0.01,
        slip_prob=0.1,
        gamma=0.95
    )
    
    mdp = GridWorldMDP(spec, device)
    
    print(f"\nLarge GridWorld: {mdp.height}x{mdp.width}")
    print(f"States: {mdp.n_states}, Actions: {mdp.n_actions}")
    
    # Run policy iteration
    result = policy_iteration(
        mdp.P, mdp.R,
        gamma=spec.gamma,
        eval_tolerance=1e-7,
        verbose=True
    )
    
    print(f"\nFinal Statistics:")
    print(f"  Optimal value (mean): {result['V_star'].mean().item():.3f}")
    print(f"  Optimal value (max): {result['V_star'].max().item():.3f}")
    print(f"  Optimal value (min): {result['V_star'].min().item():.3f}")
    
    # Visualize final policy
    visualize_policy_evolution(mdp, {'policies': [result['pi_star']], 
                                     'mean_values': [result['V_star'].mean().item()]})

def main():
    print("="*50)
    print("Experiment 05: Policy Iteration")
    print("="*50)
    
    setup_seed(42)
    
    # Basic policy iteration test
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    print("\nRunning Policy Iteration on Classic GridWorld...")
    result = policy_iteration(
        mdp.P, mdp.R,
        gamma=mdp.spec.gamma,
        verbose=True
    )
    
    # Visualize policy evolution
    visualize_policy_evolution(mdp, result['history'])
    
    # Compare different initial policies
    comparison_results = compare_initial_policies()
    
    # Test on larger grid
    test_larger_gridworld()
    
    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()