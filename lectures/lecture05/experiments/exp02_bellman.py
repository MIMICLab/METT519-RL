#!/usr/bin/env python3
"""
RL2025 - Lecture 5: Experiment 02 - Bellman Equations and Value Functions

This experiment demonstrates the Bellman equations and value iteration.

Learning objectives:
- Understand state-value and action-value functions
- Implement Bellman backup operations
- Visualize value function convergence

Prerequisites: exp01_setup.py completed successfully
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import gymnasium as gym
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

def get_frozenlake_dynamics():
    """Extract transition dynamics from FrozenLake for analysis."""
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Build transition and reward matrices
    # P[s,a,s'] = probability of s->s' under action a
    # R[s,a] = expected reward for (s,a)
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    
    # FrozenLake uses a specific internal structure
    # We'll use the environment's P attribute if available
    if hasattr(env, 'P'):
        for s in range(n_states):
            for a in range(n_actions):
                for prob, next_s, reward, done in env.P[s][a]:
                    P[s, a, next_s] += prob
                    R[s, a] += prob * reward
    
    env.close()
    return P, R, n_states, n_actions

def bellman_expectation_backup(V, P, R, gamma, policy):
    """
    Compute one Bellman expectation backup for policy evaluation.
    V: current value function [S]
    P: transition probabilities [S, A, S']
    R: rewards [S, A]
    gamma: discount factor
    policy: action probabilities [S, A]
    """
    n_states = V.shape[0]
    V_new = np.zeros_like(V)
    
    for s in range(n_states):
        value = 0.0
        for a in range(policy.shape[1]):
            # Expected value under action a
            action_value = R[s, a]
            for s_next in range(n_states):
                action_value += gamma * P[s, a, s_next] * V[s_next]
            # Weight by policy probability
            value += policy[s, a] * action_value
        V_new[s] = value
    
    return V_new

def bellman_optimality_backup(V, P, R, gamma):
    """
    Compute one Bellman optimality backup for value iteration.
    Returns new value function and greedy policy.
    """
    n_states, n_actions = R.shape
    V_new = np.zeros_like(V)
    policy = np.zeros((n_states, n_actions))
    
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            # Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) * V(s')
            q_values[a] = R[s, a]
            for s_next in range(n_states):
                q_values[a] += gamma * P[s, a, s_next] * V[s_next]
        
        # Take maximum over actions
        V_new[s] = np.max(q_values)
        # Greedy policy
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    
    return V_new, policy

def value_iteration(P, R, gamma=0.99, theta=1e-6, max_iters=1000):
    """
    Run value iteration to convergence.
    Returns optimal value function and policy.
    """
    n_states = R.shape[0]
    V = np.zeros(n_states)
    
    history = []
    
    for i in range(max_iters):
        V_old = V.copy()
        V, policy = bellman_optimality_backup(V, P, R, gamma)
        
        # Check convergence
        delta = np.max(np.abs(V - V_old))
        history.append({
            'iteration': i,
            'V': V.copy(),
            'delta': delta
        })
        
        if delta < theta:
            print(f"Value iteration converged in {i+1} iterations")
            break
    
    return V, policy, history

def q_from_v(V, P, R, gamma):
    """Compute Q-values from state values."""
    n_states, n_actions = R.shape
    Q = np.zeros((n_states, n_actions))
    
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = R[s, a]
            for s_next in range(n_states):
                Q[s, a] += gamma * P[s, a, s_next] * V[s_next]
    
    return Q

def visualize_values(V, title="State Values", size=4):
    """Visualize value function as a heatmap."""
    V_grid = V.reshape((size, size))
    
    plt.figure(figsize=(6, 5))
    plt.imshow(V_grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(title)
    
    # Add value text
    for i in range(size):
        for j in range(size):
            plt.text(j, i, f'{V_grid[i,j]:.2f}', 
                    ha='center', va='center', color='white', fontsize=10)
    
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    plt.show()

def demonstrate_bellman_operator():
    """Show Bellman operator properties."""
    print("="*50)
    print("Bellman Operator Properties")
    print("="*50)
    
    # Get environment dynamics
    P, R, n_states, n_actions = get_frozenlake_dynamics()
    gamma = 0.99
    
    # Initialize two different value functions
    V1 = np.random.rand(n_states)
    V2 = np.random.rand(n_states)
    
    print(f"Initial ||V1 - V2|| = {np.max(np.abs(V1 - V2)):.4f}")
    
    # Apply Bellman optimality operator
    V1_new, _ = bellman_optimality_backup(V1, P, R, gamma)
    V2_new, _ = bellman_optimality_backup(V2, P, R, gamma)
    
    print(f"After backup ||T*V1 - T*V2|| = {np.max(np.abs(V1_new - V2_new)):.4f}")
    
    # Verify contraction
    ratio = np.max(np.abs(V1_new - V2_new)) / np.max(np.abs(V1 - V2))
    print(f"Contraction ratio: {ratio:.4f} (should be <= {gamma})")
    
    # Show convergence over iterations
    print("\nConvergence to fixed point:")
    V = np.zeros(n_states)
    for i in range(10):
        V_old = V.copy()
        V, _ = bellman_optimality_backup(V, P, R, gamma)
        delta = np.max(np.abs(V - V_old))
        print(f"  Iteration {i+1}: delta = {delta:.6f}")

def compare_policies():
    """Compare random, greedy, and optimal policies."""
    print("="*50)
    print("Policy Comparison")
    print("="*50)
    
    P, R, n_states, n_actions = get_frozenlake_dynamics()
    gamma = 0.99
    
    # Random policy
    random_policy = np.ones((n_states, n_actions)) / n_actions
    
    # Evaluate random policy
    V_random = np.zeros(n_states)
    for _ in range(100):
        V_random = bellman_expectation_backup(V_random, P, R, gamma, random_policy)
    
    print(f"Random policy value at start: {V_random[0]:.4f}")
    
    # Optimal policy via value iteration
    V_optimal, optimal_policy, _ = value_iteration(P, R, gamma)
    print(f"Optimal policy value at start: {V_optimal[0]:.4f}")
    
    # Improvement ratio
    if V_random[0] > 0:
        improvement = (V_optimal[0] - V_random[0]) / V_random[0]
        print(f"Improvement: {improvement:.1%}")

def analyze_q_values():
    """Analyze Q-value structure."""
    print("="*50)
    print("Q-Value Analysis")
    print("="*50)
    
    P, R, n_states, n_actions = get_frozenlake_dynamics()
    gamma = 0.99
    
    # Get optimal V and Q
    V_optimal, _, _ = value_iteration(P, R, gamma)
    Q_optimal = q_from_v(V_optimal, P, R, gamma)
    
    # Show Q-values for a few states
    print("Q-values for selected states:")
    print("(Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)")
    
    for s in [0, 5, 10, 15]:  # Sample states
        print(f"\nState {s}:")
        for a in range(n_actions):
            print(f"  Q({s},{a}) = {Q_optimal[s,a]:.4f}")
        best_a = np.argmax(Q_optimal[s])
        print(f"  Best action: {best_a}")
    
    # Verify Bellman optimality equation
    print("\nBellman optimality verification (should be ~0):")
    for s in range(min(5, n_states)):
        v_from_q = np.max(Q_optimal[s])
        error = abs(V_optimal[s] - v_from_q)
        print(f"  State {s}: |V(s) - max_a Q(s,a)| = {error:.6f}")

def main():
    print("="*50)
    print("Experiment 02: Bellman Equations & Value Functions")
    print("="*50)
    
    # 1. Demonstrate Bellman operator properties
    demonstrate_bellman_operator()
    print()
    
    # 2. Run value iteration
    print("="*50)
    print("Value Iteration")
    print("="*50)
    P, R, n_states, n_actions = get_frozenlake_dynamics()
    V_optimal, optimal_policy, history = value_iteration(P, R, gamma=0.99)
    
    print(f"Optimal value at start state: {V_optimal[0]:.4f}")
    print(f"Number of iterations: {len(history)}")
    
    # 3. Compare policies
    compare_policies()
    print()
    
    # 4. Analyze Q-values
    analyze_q_values()
    
    # 5. Show convergence
    print("\n" + "="*50)
    print("Convergence Analysis")
    print("="*50)
    deltas = [h['delta'] for h in history[:20]]
    print("First 20 iteration deltas:")
    for i, delta in enumerate(deltas):
        print(f"  Iter {i+1}: {delta:.6f}")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()