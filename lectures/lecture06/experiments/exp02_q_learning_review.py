#!/usr/bin/env python3
"""
RL2025 - Lecture 6: Experiment 02 - Q-Learning Review

This experiment reviews tabular Q-learning and demonstrates why
it fails with high-dimensional state spaces, motivating DQN.

Learning objectives:
- Review tabular Q-learning algorithm
- Understand Q-value updates and convergence
- See limitations with continuous state spaces
- Motivate function approximation

Prerequisites: Basic understanding of Q-learning from Lecture 5
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import gymnasium as gym
from collections import defaultdict
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

class TabularQLearning:
    """Simple tabular Q-learning for discrete environments"""
    
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            q_values = self.Q[state]
            # Break ties randomly
            max_q = np.max(q_values)
            actions = np.where(q_values == max_q)[0]
            return np.random.choice(actions)
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule"""
        # Current Q-value
        q_current = self.Q[state][action]
        
        # Bootstrap target
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD error
        td_error = q_target - q_current
        
        # Update Q-value
        self.Q[state][action] += self.alpha * td_error
        
        return td_error

def discretize_observation(obs, bins_per_dim=10):
    """Discretize continuous observation for tabular Q-learning"""
    # CartPole observation: [position, velocity, angle, angular_velocity]
    # Define reasonable bounds for discretization
    bounds = [
        (-2.4, 2.4),     # Cart position
        (-3.0, 3.0),     # Cart velocity  
        (-0.21, 0.21),   # Pole angle (radians)
        (-3.0, 3.0)      # Pole angular velocity
    ]
    
    discrete_obs = []
    for i, (low, high) in enumerate(bounds):
        # Clip to bounds
        val = np.clip(obs[i], low, high)
        # Discretize
        bin_idx = int((val - low) / (high - low) * (bins_per_dim - 1))
        discrete_obs.append(bin_idx)
    
    # Convert to single state index
    state = 0
    for i, bin_idx in enumerate(discrete_obs):
        state = state * bins_per_dim + bin_idx
    
    return state

def main():
    print("="*50)
    print("Experiment 02: Q-Learning Review")
    print("="*50)
    
    # 1. Demonstrate Tabular Q-Learning on Discretized CartPole
    print("\n1. Tabular Q-Learning with Discretized States:")
    print("   (This shows the limitation of tabular methods)")
    
    env = gym.make("CartPole-v1")
    bins_per_dim = 8  # 8^4 = 4096 possible states
    n_states = bins_per_dim ** 4
    n_actions = env.action_space.n
    
    print(f"   State space discretization: {bins_per_dim} bins per dimension")
    print(f"   Total discrete states: {n_states:,}")
    print(f"   Action space: {n_actions} actions")
    
    # Initialize Q-learning agent
    agent = TabularQLearning(
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    # Training loop
    n_episodes = 100
    episode_returns = []
    td_errors = []
    
    print("\n2. Training Tabular Q-Learning:")
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        state = discretize_observation(obs, bins_per_dim)
        
        episode_return = 0
        episode_td_errors = []
        
        done = False
        while not done:
            # Select action
            action = agent.get_action(state)
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize_observation(next_obs, bins_per_dim)
            
            # Update Q-values
            td_error = agent.update(state, action, reward, next_state, done)
            episode_td_errors.append(abs(td_error))
            
            episode_return += reward
            state = next_state
        
        episode_returns.append(episode_return)
        td_errors.append(np.mean(episode_td_errors))
        
        if (episode + 1) % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            print(f"   Episode {episode+1:3d}: Average Return = {avg_return:.1f}")
    
    # 3. Analyze Q-table Coverage
    print("\n3. Q-Table Analysis:")
    states_visited = len(agent.Q)
    coverage = (states_visited / n_states) * 100
    print(f"   States visited: {states_visited:,} / {n_states:,} ({coverage:.2f}%)")
    print(f"   Q-table sparsity: {100 - coverage:.2f}%")
    
    # Count non-zero Q-values
    non_zero_count = sum(1 for state_q in agent.Q.values() 
                        for q_val in state_q if q_val != 0)
    total_entries = states_visited * n_actions
    print(f"   Non-zero Q-values: {non_zero_count} / {total_entries}")
    
    # 4. Problems with Continuous Spaces
    print("\n4. Problems with High-Dimensional/Continuous Spaces:")
    
    print("\n   a) State Space Explosion:")
    for bins in [4, 8, 16, 32]:
        states = bins ** 4
        memory_mb = (states * n_actions * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"      {bins} bins/dim -> {states:,} states ({memory_mb:.1f} MB)")
    
    print("\n   b) Discretization Error:")
    # Show how similar observations map to different states
    obs1 = np.array([0.0, 0.0, 0.0, 0.0])
    obs2 = np.array([0.01, 0.01, 0.01, 0.01])  # Very similar
    
    state1 = discretize_observation(obs1, bins_per_dim)
    state2 = discretize_observation(obs2, bins_per_dim)
    
    print(f"      Observation 1: {obs1} -> State {state1}")
    print(f"      Observation 2: {obs2} -> State {state2}")
    print(f"      Same state? {state1 == state2}")
    
    # 5. Why We Need Function Approximation
    print("\n5. Motivation for Deep Q-Networks (DQN):")
    
    print("\n   Tabular Q-Learning Limitations:")
    print("   - Cannot handle continuous state spaces")
    print("   - Memory grows exponentially with state dimensions")
    print("   - No generalization between similar states")
    print("   - Requires visiting every state-action pair")
    
    print("\n   DQN Solutions:")
    print("   - Neural network approximates Q-function")
    print("   - Handles high-dimensional inputs (images, etc.)")
    print("   - Generalizes across similar states")
    print("   - Efficient memory usage")
    
    # 6. Visualize Learning Progress
    print("\n6. Learning Curves:")
    
    # Simple ASCII plot of returns
    print("\n   Episode Returns (last 50 episodes):")
    last_50 = episode_returns[-50:] if len(episode_returns) >= 50 else episode_returns
    
    # Normalize to 0-10 scale for ASCII plot
    if len(last_50) > 0:
        max_return = max(last_50)
        min_return = min(last_50)
        range_return = max_return - min_return if max_return != min_return else 1
        
        for i, ret in enumerate(last_50):
            normalized = int(((ret - min_return) / range_return) * 20)
            bar = '#' * normalized
            print(f"   {i+1:2d}: {bar} ({ret:.0f})")
    
    # 7. Q-Learning Update Equation Review
    print("\n7. Q-Learning Mathematics Review:")
    print("\n   Update equation:")
    print("   Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]")
    print("\n   Where:")
    print("   - α = learning rate (step size)")
    print("   - γ = discount factor")
    print("   - r = immediate reward")
    print("   - s' = next state")
    print("   - TD target = r + γ max_a' Q(s',a')")
    print("   - TD error = TD target - Q(s,a)")
    
    env.close()
    
    print("\n" + "="*50)
    print("Q-Learning review completed!")
    print("Next: Neural networks for Q-function approximation")
    print("="*50)

if __name__ == "__main__":
    main()