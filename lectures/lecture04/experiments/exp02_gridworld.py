#!/usr/bin/env python3
"""
RL2025 - Lecture 4: Experiment 02 - GridWorld MDP Implementation

This experiment implements a complete GridWorld MDP with walls, terminals,
and stochastic transitions, demonstrating tabular representation of P and R.

Learning objectives:
- Build transition probability matrix P[s,a,s']
- Build reward matrix R[s,a]
- Handle terminal states and walls
- Implement stochastic transitions (slip probability)

Prerequisites: exp01_setup.py completed
"""

import os
import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Standard setup
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Action definitions
ACTIONS = {
    0: (-1, 0),  # UP
    1: (0, 1),   # RIGHT
    2: (1, 0),   # DOWN
    3: (0, -1)   # LEFT
}
ACTION_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT']

@dataclass
class GridWorldSpec:
    """Specification for a GridWorld MDP"""
    grid: List[str]  # List of strings, each char is a cell
    terminal_rewards: Dict[Tuple[int, int], float]  # (row, col) -> reward
    step_cost: float = -0.04  # Cost for each step
    slip_prob: float = 0.1   # Probability of slipping to perpendicular direction
    gamma: float = 0.99      # Discount factor

class GridWorldMDP:
    """Complete GridWorld MDP with tabular P and R"""
    
    def __init__(self, spec: GridWorldSpec, device: torch.device):
        self.spec = spec
        self.device = device
        self.height = len(spec.grid)
        self.width = len(spec.grid[0]) if self.height > 0 else 0
        
        # Parse grid to identify passable cells
        self.passable = [[c != '#' for c in row] for row in spec.grid]
        
        # Create state indexing
        self.state_to_pos = {}  # state_idx -> (row, col)
        self.pos_to_state = {}  # (row, col) -> state_idx
        
        state_idx = 0
        for r in range(self.height):
            for c in range(self.width):
                if self.passable[r][c]:
                    self.state_to_pos[state_idx] = (r, c)
                    self.pos_to_state[(r, c)] = state_idx
                    state_idx += 1
        
        self.n_states = len(self.state_to_pos)
        self.n_actions = 4  # UP, RIGHT, DOWN, LEFT
        
        print(f"GridWorld created: {self.height}x{self.width} grid")
        print(f"  Total cells: {self.height * self.width}")
        print(f"  Passable states: {self.n_states}")
        print(f"  Actions: {self.n_actions}")
        
        # Identify terminal states
        self.terminal_states = torch.zeros(self.n_states, dtype=torch.bool, device=device)
        self.terminal_rewards = torch.zeros(self.n_states, dtype=torch.float32, device=device)
        
        for (r, c), reward in spec.terminal_rewards.items():
            if (r, c) in self.pos_to_state:
                s = self.pos_to_state[(r, c)]
                self.terminal_states[s] = True
                self.terminal_rewards[s] = reward
        
        # Build transition and reward matrices
        self.P, self.R = self._build_dynamics()
    
    def _is_valid_pos(self, r: int, c: int) -> bool:
        """Check if position is valid and passable"""
        return (0 <= r < self.height and 
                0 <= c < self.width and 
                self.passable[r][c])
    
    def _move(self, r: int, c: int, action: int) -> Tuple[int, int]:
        """Execute movement, handling walls"""
        dr, dc = ACTIONS[action]
        new_r, new_c = r + dr, c + dc
        
        if self._is_valid_pos(new_r, new_c):
            return (new_r, new_c)
        else:
            return (r, c)  # Hit wall, stay in place
    
    def _build_dynamics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build P[s,a,s'] and R[s,a] tensors"""
        P = torch.zeros((self.n_states, self.n_actions, self.n_states), 
                       dtype=torch.float32, device=self.device)
        R = torch.full((self.n_states, self.n_actions), 
                      self.spec.step_cost, 
                      dtype=torch.float32, device=self.device)
        
        for s in range(self.n_states):
            r, c = self.state_to_pos[s]
            
            # Terminal states are absorbing
            if self.terminal_states[s]:
                P[s, :, s] = 1.0  # Stay in terminal state
                R[s, :] = 0.0     # No further rewards
                continue
            
            for a in range(self.n_actions):
                # Stochastic transitions with slip
                # Intended direction: 1 - 2*slip_prob
                # Perpendicular directions: slip_prob each
                
                intended = a
                left_slip = (a - 1) % 4
                right_slip = (a + 1) % 4
                
                transitions = [
                    (intended, 1.0 - 2 * self.spec.slip_prob),
                    (left_slip, self.spec.slip_prob),
                    (right_slip, self.spec.slip_prob)
                ]
                
                expected_reward = 0.0
                
                for actual_action, prob in transitions:
                    new_r, new_c = self._move(r, c, actual_action)
                    new_s = self.pos_to_state[(new_r, new_c)]
                    P[s, a, new_s] += prob
                    
                    # Immediate reward
                    if self.terminal_states[new_s]:
                        immediate_reward = self.terminal_rewards[new_s].item()
                    else:
                        immediate_reward = self.spec.step_cost
                    
                    expected_reward += prob * immediate_reward
                
                R[s, a] = expected_reward
        
        # Verify probability constraints
        row_sums = P.sum(dim=2)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
            "Transition probabilities must sum to 1"
        
        return P, R
    
    def render_grid(self, values: Optional[torch.Tensor] = None):
        """Render the grid world with optional value overlay"""
        print("\nGridWorld Layout:")
        print("  S: Start, G: Goal, P: Pit, #: Wall")
        print("  .: Empty passable cell")
        print()
        
        for r in range(self.height):
            row_str = "  "
            for c in range(self.width):
                if not self.passable[r][c]:
                    row_str += "###  "
                elif (r, c) in self.spec.terminal_rewards:
                    reward = self.spec.terminal_rewards[(r, c)]
                    if reward > 0:
                        row_str += " G   "  # Goal
                    else:
                        row_str += " P   "  # Pit
                elif self.spec.grid[r][c] == 'S':
                    row_str += " S   "  # Start
                else:
                    if values is not None and (r, c) in self.pos_to_state:
                        s = self.pos_to_state[(r, c)]
                        val = values[s].item()
                        row_str += f"{val:5.2f}"
                    else:
                        row_str += "  .  "
            print(row_str)

def create_classic_gridworld(device: torch.device) -> GridWorldMDP:
    """Create the classic 4x3 gridworld from Russell & Norvig"""
    grid = [
        "S..G",
        ".#.P",
        "....",
    ]
    
    terminal_rewards = {
        (0, 3): +1.0,  # Goal at top-right
        (1, 3): -1.0,  # Pit at middle-right
    }
    
    spec = GridWorldSpec(
        grid=grid,
        terminal_rewards=terminal_rewards,
        step_cost=-0.04,
        slip_prob=0.2,  # 20% slip probability
        gamma=0.99
    )
    
    return GridWorldMDP(spec, device)

def test_dynamics():
    """Test MDP dynamics with sample trajectories"""
    print("\n" + "="*50)
    print("Testing MDP Dynamics")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    # Test from start state
    start_pos = (0, 0)  # Top-left
    start_state = mdp.pos_to_state[start_pos]
    
    print(f"\nStart state: {start_state} at position {start_pos}")
    
    # Show transition probabilities for each action from start
    for a in range(mdp.n_actions):
        print(f"\nAction {a} ({ACTION_NAMES[a]}):")
        probs = mdp.P[start_state, a, :]
        reward = mdp.R[start_state, a].item()
        
        print(f"  Expected reward: {reward:.3f}")
        print("  Transition probabilities:")
        
        for next_s in range(mdp.n_states):
            if probs[next_s] > 0:
                next_pos = mdp.state_to_pos[next_s]
                print(f"    -> State {next_s} at {next_pos}: {probs[next_s]:.3f}")
    
    # Verify conservation of probability
    print("\nVerifying probability conservation...")
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            total_prob = mdp.P[s, a, :].sum().item()
            assert abs(total_prob - 1.0) < 1e-6, f"State {s}, Action {a}: probabilities sum to {total_prob}"
    print("✓ All transition probabilities sum to 1.0")

def analyze_mdp_structure():
    """Analyze the structure of the MDP"""
    print("\n" + "="*50)
    print("MDP Structure Analysis")
    print("="*50)
    
    device = get_device()
    mdp = create_classic_gridworld(device)
    
    print(f"\nMDP Statistics:")
    print(f"  State space size: {mdp.n_states}")
    print(f"  Action space size: {mdp.n_actions}")
    print(f"  Terminal states: {mdp.terminal_states.sum().item()}")
    print(f"  Discount factor: {mdp.spec.gamma}")
    print(f"  Step cost: {mdp.spec.step_cost}")
    print(f"  Slip probability: {mdp.spec.slip_prob}")
    
    # Analyze reward structure
    print(f"\nReward structure:")
    print(f"  Min reward: {mdp.R.min().item():.3f}")
    print(f"  Max reward: {mdp.R.max().item():.3f}")
    print(f"  Mean reward: {mdp.R.mean().item():.3f}")
    
    # Analyze connectivity
    print(f"\nTransition matrix properties:")
    print(f"  P shape: {mdp.P.shape}")
    print(f"  R shape: {mdp.R.shape}")
    print(f"  Sparsity: {(mdp.P == 0).float().mean().item():.2%} zeros")
    
    # Show grid
    mdp.render_grid()

def main():
    print("="*50)
    print("Experiment 02: GridWorld MDP Implementation")
    print("="*50)
    
    setup_seed(42)
    
    # Analyze MDP structure
    analyze_mdp_structure()
    
    # Test dynamics
    test_dynamics()
    
    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()