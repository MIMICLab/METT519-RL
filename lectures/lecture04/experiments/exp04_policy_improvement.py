#!/usr/bin/env python3
"""
RL2025 - Lecture 4: Experiment 04 - Policy Improvement

This experiment implements policy improvement using the greedy operator,
demonstrating how to extract better policies from value functions.

Learning objectives:
- Implement greedy policy extraction
- Verify policy improvement theorem
- Compare Q-values for action selection
- Understand deterministic vs stochastic improvements

Prerequisites: exp03_policy_evaluation.py completed
"""

import os
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR = Path(os.environ.get("LECTURE04_FIGURES_DIR", DEFAULT_FIGURES_DIR))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from exp02_gridworld import GridWorldMDP, GridWorldSpec, create_classic_gridworld, setup_seed, get_device, ACTION_NAMES
from exp03_policy_evaluation import policy_evaluation

def compute_q_values(
    P: torch.Tensor,    # [S, A, S]
    R: torch.Tensor,    # [S, A]
    V: torch.Tensor,    # [S]
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute Q-values from value function.
    
    Q(s,a) = R(s,a) + gamma * sum_s' P(s,a,s') * V(s')
    
    Args:
        P: Transition probabilities [S, A, S]
        R: Reward function [S, A]
        V: State values [S]
        gamma: Discount factor
    
    Returns:
        Q-values [S, A]
    """
    # Efficient computation using einsum
    # 'sat,t->sa' sums over next-state dimension
    Q = R + gamma * torch.einsum('sat,t->sa', P, V)
    return Q

def policy_improvement(
    P: torch.Tensor,
    R: torch.Tensor,
    V: torch.Tensor,
    gamma: float = 0.99,
    temperature: float = 0.0,  # 0 for deterministic, >0 for soft
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Improve policy by acting greedily with respect to value function.
    
    Args:
        P: Transition probabilities [S, A, S]
        R: Reward function [S, A]
        V: Current value function [S]
        gamma: Discount factor
        temperature: Softmax temperature (0 = greedy, >0 = soft)
        verbose: Print improvement statistics
    
    Returns:
        Dictionary with improved policy and Q-values
    """
    device = P.device
    n_states = P.shape[0]
    n_actions = P.shape[1]
    
    # Compute Q-values
    Q = compute_q_values(P, R, V, gamma)
    
    if temperature == 0:
        # Deterministic greedy policy
        pi_new = torch.argmax(Q, dim=1)  # [S]
        pi_probs = torch.zeros((n_states, n_actions), device=device)
        pi_probs[torch.arange(n_states), pi_new] = 1.0
    else:
        # Stochastic softmax policy
        pi_probs = torch.softmax(Q / temperature, dim=1)  # [S, A]
        pi_new = torch.multinomial(pi_probs, 1).squeeze()  # Sample actions
    
    if verbose:
        print(f"\nPolicy Improvement:")
        print(f"  Temperature: {temperature} {'(deterministic)' if temperature == 0 else '(stochastic)'}")
        
        # Analyze Q-value gaps
        Q_max, _ = Q.max(dim=1)
        Q_min, _ = Q.min(dim=1)
        Q_gap = Q_max - Q_min
        
        print(f"  Mean Q-value gap: {Q_gap.mean().item():.3f}")
        print(f"  Max Q-value gap: {Q_gap.max().item():.3f}")
        print(f"  States with unique best action: {(Q_gap > 1e-6).sum().item()}/{n_states}")
    
    return {
        'pi': pi_new,           # Deterministic policy [S]
        'pi_probs': pi_probs,   # Stochastic policy [S, A]
        'Q': Q,                 # Q-values [S, A]
        'advantages': Q - Q.mean(dim=1, keepdim=True)  # Advantages [S, A]
    }

def verify_policy_improvement_theorem():
    """Verify that greedy policy improvement never makes things worse"""
    print("\n" + "="*50)
    print("Verifying Policy Improvement Theorem")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    # Start with a random policy
    pi_old = torch.ones((mdp.n_states, mdp.n_actions), device=device) / mdp.n_actions
    
    print("\nInitial random policy:")
    eval_old = policy_evaluation(mdp.P, mdp.R, pi_old, mdp.spec.gamma, verbose=False)
    V_old = eval_old['V']
    print(f"  Mean value: {V_old.mean().item():.3f}")
    
    # Improve the policy
    improvement = policy_improvement(mdp.P, mdp.R, V_old, mdp.spec.gamma, temperature=0)
    pi_new = improvement['pi']
    Q = improvement['Q']
    
    # Evaluate the new policy
    print("\nImproved (greedy) policy:")
    eval_new = policy_evaluation(mdp.P, mdp.R, pi_new, mdp.spec.gamma, verbose=False)
    V_new = eval_new['V']
    print(f"  Mean value: {V_new.mean().item():.3f}")
    
    # Verify improvement
    improvement_per_state = V_new - V_old
    print(f"\nImprovement statistics:")
    print(f"  States improved: {(improvement_per_state > 1e-6).sum().item()}")
    print(f"  States unchanged: {torch.abs(improvement_per_state) <= 1e-6}.sum().item()")
    print(f"  States worsened: {(improvement_per_state < -1e-6).sum().item()}")
    print(f"  Mean improvement: {improvement_per_state.mean().item():.6f}")
    print(f"  Max improvement: {improvement_per_state.max().item():.6f}")
    
    # Verify the mathematical relationship: V_new >= V_old
    assert torch.all(V_new >= V_old - 1e-6), "Policy improvement theorem violated!"
    print("\n✓ Policy Improvement Theorem verified: V_π' >= V_π for all states")
    
    return mdp, V_old, V_new, Q

def visualize_policy_improvement(mdp, V_old, V_new, Q):
    """Visualize the improvement process"""
    print("\n" + "="*50)
    print("Visualizing Policy Improvement")
    print("="*50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert values to grids for visualization
    def values_to_grid(V):
        grid = np.full((mdp.height, mdp.width), np.nan)
        for s in range(mdp.n_states):
            r, c = mdp.state_to_pos[s]
            grid[r, c] = V[s].item()
        return grid
    
    # Plot old values
    im1 = axes[0, 0].imshow(values_to_grid(V_old), cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 0].set_title('V_π (Random Policy)')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot new values
    im2 = axes[0, 1].imshow(values_to_grid(V_new), cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 1].set_title('V_π\' (Improved Policy)')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot improvement
    improvement = V_new - V_old
    im3 = axes[0, 2].imshow(values_to_grid(improvement), cmap='Greens', vmin=0)
    axes[0, 2].set_title('Improvement (V_π\' - V_π)')
    axes[0, 2].set_xlabel('Column')
    axes[0, 2].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot Q-values for a sample state
    sample_state = 4  # Middle state
    r, c = mdp.state_to_pos[sample_state]
    
    q_vals = Q[sample_state].cpu().numpy()
    axes[1, 0].bar(ACTION_NAMES, q_vals)
    axes[1, 0].set_title(f'Q-values at State {sample_state} ({r},{c})')
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('Q-value')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot advantage function
    advantages = Q - Q.mean(dim=1, keepdim=True)
    adv_vals = advantages[sample_state].cpu().numpy()
    axes[1, 1].bar(ACTION_NAMES, adv_vals)
    axes[1, 1].set_title(f'Advantages at State {sample_state}')
    axes[1, 1].set_xlabel('Action')
    axes[1, 1].set_ylabel('Advantage')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot policy arrows
    pi_greedy = torch.argmax(Q, dim=1)
    arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    axes[1, 2].set_xlim(-0.5, mdp.width - 0.5)
    axes[1, 2].set_ylim(-0.5, mdp.height - 0.5)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title('Greedy Policy')
    axes[1, 2].set_xlabel('Column')
    axes[1, 2].set_ylabel('Row')
    
    for s in range(mdp.n_states):
        r, c = mdp.state_to_pos[s]
        if not mdp.terminal_states[s]:
            action = pi_greedy[s].item()
            axes[1, 2].text(c, r, arrow_map[action], ha='center', va='center', fontsize=20)
        else:
            # Terminal states
            if mdp.terminal_rewards[s] > 0:
                axes[1, 2].text(c, r, 'G', ha='center', va='center', fontsize=16, color='green')
            else:
                axes[1, 2].text(c, r, 'P', ha='center', va='center', fontsize=16, color='red')
    
    # Add walls
    for r in range(mdp.height):
        for c in range(mdp.width):
            if not mdp.passable[r][c]:
                axes[1, 2].add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=True, color='black'))
    
    axes[1, 2].set_xticks(range(mdp.width))
    axes[1, 2].set_yticks(range(mdp.height))
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / 'policy_improvement.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {out_path}")
    plt.close()

def test_soft_improvement():
    """Test soft (stochastic) policy improvement"""
    print("\n" + "="*50)
    print("Testing Soft Policy Improvement")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    # Start with uniform random policy
    pi_uniform = torch.ones((mdp.n_states, mdp.n_actions), device=device) / mdp.n_actions
    eval_uniform = policy_evaluation(mdp.P, mdp.R, pi_uniform, mdp.spec.gamma, verbose=False)
    V_uniform = eval_uniform['V']
    
    temperatures = [0.0, 0.1, 0.5, 1.0, 5.0]
    results = {}
    
    for temp in temperatures:
        # Soft improvement
        improvement = policy_improvement(mdp.P, mdp.R, V_uniform, mdp.spec.gamma, 
                                       temperature=temp, verbose=False)
        
        # Evaluate the soft policy
        eval_soft = policy_evaluation(mdp.P, mdp.R, improvement['pi_probs'], 
                                     mdp.spec.gamma, verbose=False)
        
        results[temp] = {
            'V': eval_soft['V'],
            'pi_probs': improvement['pi_probs'],
            'mean_value': eval_soft['V'].mean().item()
        }
        
        # Compute entropy of policy
        entropy = -(improvement['pi_probs'] * torch.log(improvement['pi_probs'] + 1e-8)).sum(dim=1).mean()
        
        print(f"\nTemperature τ = {temp}:")
        print(f"  Mean value: {results[temp]['mean_value']:.3f}")
        print(f"  Policy entropy: {entropy:.3f}")
        
        if temp == 0:
            print(f"  Deterministic actions: {mdp.n_states}")
        else:
            # Count near-deterministic states
            max_probs = improvement['pi_probs'].max(dim=1)[0]
            near_det = (max_probs > 0.9).sum().item()
            print(f"  Near-deterministic states (p>0.9): {near_det}/{mdp.n_states}")

def main():
    print("="*50)
    print("Experiment 04: Policy Improvement")
    print("="*50)
    
    setup_seed(42)
    
    # Verify policy improvement theorem
    mdp, V_old, V_new, Q = verify_policy_improvement_theorem()
    
    # Visualize the improvement
    visualize_policy_improvement(mdp, V_old, V_new, Q)
    
    # Test soft improvement
    test_soft_improvement()
    
    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
