#!/usr/bin/env python3
"""
RL2025 - Lecture 4: Experiment 08 - Algorithmic Optimizations

This experiment explores various optimizations for dynamic programming
algorithms including asynchronous updates, prioritized sweeping, and
GPU acceleration techniques.

Learning objectives:
- Implement asynchronous value iteration
- Understand prioritized sweeping
- Compare in-place vs. two-array updates
- Explore GPU acceleration benefits

Prerequisites: exp07_stopping_criteria.py completed
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
import time
import heapq
from pathlib import Path

from exp02_gridworld import GridWorldMDP, GridWorldSpec, setup_seed, get_device
from exp04_policy_improvement import compute_q_values

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR = Path(os.environ.get("LECTURE04_FIGURES_DIR", DEFAULT_FIGURES_DIR))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def synchronous_value_iteration(
    P: torch.Tensor,
    R: torch.Tensor,
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 1000
) -> Dict:
    """Standard synchronous value iteration (baseline)"""
    device = P.device
    n_states = P.shape[0]
    
    V = torch.zeros(n_states, dtype=torch.float32, device=device)
    
    start_time = time.time()
    iterations = 0
    
    for iterations in range(max_iterations):
        V_old = V.clone()  # Create copy for synchronous update
        Q = compute_q_values(P, R, V_old, gamma)
        V, _ = torch.max(Q, dim=1)
        
        delta = torch.max(torch.abs(V - V_old)).item()
        if delta < tolerance:
            break
    
    return {
        'V': V,
        'iterations': iterations + 1,
        'time': time.time() - start_time,
        'type': 'synchronous'
    }

def asynchronous_value_iteration(
    P: torch.Tensor,
    R: torch.Tensor,
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    sweep_order: str = 'sequential'  # 'sequential', 'random', 'reverse'
) -> Dict:
    """Asynchronous (Gauss-Seidel) value iteration"""
    device = P.device
    n_states = P.shape[0]
    n_actions = P.shape[1]
    
    V = torch.zeros(n_states, dtype=torch.float32, device=device)
    
    # Determine sweep order
    if sweep_order == 'sequential':
        state_order = list(range(n_states))
    elif sweep_order == 'reverse':
        state_order = list(range(n_states - 1, -1, -1))
    elif sweep_order == 'random':
        state_order = list(range(n_states))
        random.shuffle(state_order)
    
    start_time = time.time()
    iterations = 0
    
    for iterations in range(max_iterations):
        delta = 0.0
        
        # Update each state using latest values
        for s in state_order:
            v_old = V[s].item()
            
            # Compute Q(s,a) using current V (not V_old)
            q_values = torch.zeros(n_actions, device=device)
            for a in range(n_actions):
                q_values[a] = R[s, a] + gamma * torch.sum(P[s, a, :] * V)
            
            V[s] = torch.max(q_values)
            delta = max(delta, abs(V[s].item() - v_old))
        
        if delta < tolerance:
            break
    
    return {
        'V': V,
        'iterations': iterations + 1,
        'time': time.time() - start_time,
        'type': f'asynchronous_{sweep_order}'
    }

def prioritized_sweeping_vi(
    P: torch.Tensor,
    R: torch.Tensor,
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    priority_threshold: float = 1e-10
) -> Dict:
    """Value iteration with prioritized sweeping"""
    device = P.device
    n_states = P.shape[0]
    n_actions = P.shape[1]
    
    V = torch.zeros(n_states, dtype=torch.float32, device=device)
    
    # Priority queue: (-priority, state)
    # Using negative priority for max-heap behavior
    priority_queue = []
    
    # Initialize with all states
    for s in range(n_states):
        heapq.heappush(priority_queue, (0.0, s))
    
    # Track which states are in queue
    in_queue = set(range(n_states))
    
    start_time = time.time()
    iterations = 0
    updates = 0
    
    while priority_queue and iterations < max_iterations:
        iterations += 1
        
        # Pop state with highest priority
        neg_priority, s = heapq.heappop(priority_queue)
        priority = -neg_priority
        in_queue.discard(s)
        
        if priority < priority_threshold:
            continue
        
        # Compute new value for state s
        v_old = V[s].item()
        q_values = torch.zeros(n_actions, device=device)
        for a in range(n_actions):
            q_values[a] = R[s, a] + gamma * torch.sum(P[s, a, :] * V)
        V[s] = torch.max(q_values)
        
        delta_s = abs(V[s].item() - v_old)
        updates += 1
        
        # Update priorities of predecessors
        if delta_s > priority_threshold:
            # Find predecessors: states that can transition to s
            for s_pred in range(n_states):
                for a_pred in range(n_actions):
                    if P[s_pred, a_pred, s] > 0:
                        # Compute Bellman error for predecessor
                        v_pred_old = V[s_pred].item()
                        q_pred = R[s_pred, a_pred] + gamma * torch.sum(P[s_pred, a_pred, :] * V)
                        
                        # Only consider if this action could be optimal
                        q_values_pred = torch.zeros(n_actions, device=device)
                        for a in range(n_actions):
                            q_values_pred[a] = R[s_pred, a] + gamma * torch.sum(P[s_pred, a, :] * V)
                        v_pred_new = torch.max(q_values_pred).item()
                        
                        priority_pred = abs(v_pred_new - v_pred_old)
                        
                        if priority_pred > priority_threshold and s_pred not in in_queue:
                            heapq.heappush(priority_queue, (-priority_pred, s_pred))
                            in_queue.add(s_pred)
        
        # Check convergence
        if not priority_queue or (priority_queue and -priority_queue[0][0] < tolerance):
            break
    
    return {
        'V': V,
        'iterations': iterations,
        'updates': updates,
        'time': time.time() - start_time,
        'type': 'prioritized_sweeping'
    }

def gpu_accelerated_vi(
    P: torch.Tensor,
    R: torch.Tensor,
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    batch_size: Optional[int] = None
) -> Dict:
    """GPU-optimized value iteration with batched operations"""
    device = P.device
    n_states = P.shape[0]
    
    V = torch.zeros(n_states, dtype=torch.float32, device=device)
    
    # Use batch processing for large state spaces
    if batch_size is None:
        batch_size = n_states
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        V_old = V.clone()
        
        if batch_size >= n_states:
            # Full batch update (most efficient for small spaces)
            # Vectorized Q-value computation
            Q = R + gamma * torch.einsum('sas,s->sa', P, V_old)
            V, _ = torch.max(Q, dim=1)
        else:
            # Mini-batch updates for large spaces
            for batch_start in range(0, n_states, batch_size):
                batch_end = min(batch_start + batch_size, n_states)
                batch_slice = slice(batch_start, batch_end)
                
                # Compute Q for batch
                P_batch = P[batch_slice]
                R_batch = R[batch_slice]
                Q_batch = R_batch + gamma * torch.einsum('sas,s->sa', P_batch, V_old)
                V[batch_slice], _ = torch.max(Q_batch, dim=1)
        
        delta = torch.max(torch.abs(V - V_old)).item()
        if delta < tolerance:
            break
    
    return {
        'V': V,
        'iterations': iteration + 1,
        'time': time.time() - start_time,
        'type': f'gpu_batch_{batch_size}'
    }

def compare_algorithms():
    """Compare different algorithmic optimizations"""
    print("\n" + "="*50)
    print("Comparing Algorithm Optimizations")
    print("="*50)
    
    device = get_device()
    
    # Test on different grid sizes
    grid_sizes = [(3, 4), (5, 5), (8, 8)]
    
    results = {}
    
    for height, width in grid_sizes:
        print(f"\nGrid size: {height}x{width}")
        
        # Create grid
        grid = []
        for r in range(height):
            row = ""
            for c in range(width):
                if r == 0 and c == 0:
                    row += "S"
                elif r == height-1 and c == width-1:
                    row += "G"
                elif (r + c) % 7 == 3:  # Some walls
                    row += "#"
                else:
                    row += "."
            grid.append(row)
        
        spec = GridWorldSpec(
            grid=grid,
            terminal_rewards={(height-1, width-1): 1.0},
            step_cost=-0.01,
            slip_prob=0.1,
            gamma=0.95
        )
        
        mdp = GridWorldMDP(spec, device)
        grid_key = f"{height}x{width}"
        results[grid_key] = {}
        
        # Test algorithms
        algorithms = [
            ('Synchronous', lambda: synchronous_value_iteration(
                mdp.P, mdp.R, spec.gamma)),
            ('Async-Sequential', lambda: asynchronous_value_iteration(
                mdp.P, mdp.R, spec.gamma, sweep_order='sequential')),
            ('Async-Random', lambda: asynchronous_value_iteration(
                mdp.P, mdp.R, spec.gamma, sweep_order='random')),
            ('Prioritized', lambda: prioritized_sweeping_vi(
                mdp.P, mdp.R, spec.gamma)),
            ('GPU-Full', lambda: gpu_accelerated_vi(
                mdp.P, mdp.R, spec.gamma)),
        ]
        
        for name, algo_fn in algorithms:
            result = algo_fn()
            results[grid_key][name] = result
            
            print(f"  {name:20s}: {result['iterations']:4d} iters, {result['time']:6.4f}s")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot iterations
    ax = axes[0, 0]
    x_pos = np.arange(len(grid_sizes))
    width = 0.15
    
    for i, algo_name in enumerate(['Synchronous', 'Async-Sequential', 'Async-Random', 'Prioritized']):
        iterations = [results[f"{h}x{w}"][algo_name]['iterations'] 
                     for h, w in grid_sizes]
        ax.bar(x_pos + i * width, iterations, width, label=algo_name)
    
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Iterations')
    ax.set_title('Iterations to Convergence')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels([f"{h}x{w}" for h, w in grid_sizes])
    ax.legend()
    
    # Plot time
    ax = axes[0, 1]
    for i, algo_name in enumerate(['Synchronous', 'Async-Sequential', 'Async-Random', 'Prioritized']):
        times = [results[f"{h}x{w}"][algo_name]['time'] 
                for h, w in grid_sizes]
        ax.bar(x_pos + i * width, times, width, label=algo_name)
    
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels([f"{h}x{w}" for h, w in grid_sizes])
    ax.legend()
    
    # Plot speedup relative to synchronous
    ax = axes[1, 0]
    for algo_name in ['Async-Sequential', 'Async-Random', 'Prioritized', 'GPU-Full']:
        speedups = []
        for h, w in grid_sizes:
            grid_key = f"{h}x{w}"
            sync_time = results[grid_key]['Synchronous']['time']
            algo_time = results[grid_key][algo_name]['time']
            speedups.append(sync_time / algo_time)
        ax.plot([f"{h}x{w}" for h, w in grid_sizes], speedups, 
               'o-', label=algo_name, linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Speedup vs Synchronous')
    ax.set_title('Relative Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary for largest grid
    grid_key = f"{grid_sizes[-1][0]}x{grid_sizes[-1][1]}"
    table_data = []
    for algo in ['Synchronous', 'Async-Sequential', 'Async-Random', 'Prioritized', 'GPU-Full']:
        if algo in results[grid_key]:
            r = results[grid_key][algo]
            table_data.append([
                algo,
                f"{r['iterations']}",
                f"{r['time']:.4f}s",
                f"{results[grid_key]['Synchronous']['time'] / r['time']:.2f}x"
            ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Algorithm', 'Iterations', 'Time', 'Speedup'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title(f'Performance Summary ({grid_key} grid)', y=0.8)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / 'algorithm_optimizations.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Algorithm comparison saved to {out_path}")
    plt.close()

def test_memory_efficiency():
    """Compare memory usage of different approaches"""
    print("\n" + "="*50)
    print("Testing Memory Efficiency")
    print("="*50)
    
    device = get_device()
    
    # Create a larger grid for memory testing
    size = 10
    grid = []
    for r in range(size):
        row = ""
        for c in range(size):
            if r == 0 and c == 0:
                row += "S"
            elif r == size-1 and c == size-1:
                row += "G"
            else:
                row += "."
        grid.append(row)
    
    spec = GridWorldSpec(
        grid=grid,
        terminal_rewards={(size-1, size-1): 1.0},
        step_cost=-0.01,
        slip_prob=0.1,
        gamma=0.95
    )
    
    mdp = GridWorldMDP(spec, device)
    
    print(f"\nGrid size: {size}x{size} = {mdp.n_states} states")
    
    # Measure memory for transition matrix
    P_memory = mdp.P.element_size() * mdp.P.nelement() / 1024  # KB
    R_memory = mdp.R.element_size() * mdp.R.nelement() / 1024  # KB
    
    print(f"\nMemory usage:")
    print(f"  Transition matrix P: {P_memory:.2f} KB")
    print(f"  Reward matrix R: {R_memory:.2f} KB")
    print(f"  Total MDP: {P_memory + R_memory:.2f} KB")
    
    # Memory for value iteration
    V_memory = mdp.n_states * 4 / 1024  # float32
    Q_memory = mdp.n_states * mdp.n_actions * 4 / 1024  # float32
    
    print(f"\nValue Iteration memory:")
    print(f"  V vector: {V_memory:.2f} KB")
    print(f"  Q matrix: {Q_memory:.2f} KB")
    print(f"  Two-array update: {2 * V_memory:.2f} KB")
    print(f"  In-place update: {V_memory:.2f} KB")
    
    # Test sparse representation potential
    sparsity = (mdp.P == 0).float().mean().item()
    print(f"\nSparsity analysis:")
    print(f"  P matrix sparsity: {sparsity:.2%} zeros")
    print(f"  Potential sparse storage savings: {sparsity:.0%}")

def main():
    print("="*50)
    print("Experiment 08: Algorithmic Optimizations")
    print("="*50)
    
    setup_seed(42)
    
    # Compare different algorithms
    compare_algorithms()
    
    # Test memory efficiency
    test_memory_efficiency()
    
    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
