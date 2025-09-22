#!/usr/bin/env python3
"""
RL2025 - Lecture 5: Experiment 03 - Tabular Q-Learning Basics

This experiment implements basic tabular Q-learning with manual updates.

Learning objectives:
- Implement Q-table initialization and updates
- Understand TD error and Q-learning update rule
- Visualize Q-value evolution during learning

Prerequisites: exp02_bellman.py completed successfully
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

class TabularQLearning:
    """Basic tabular Q-learning implementation."""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99):
        """
        Initialize Q-learning agent.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: Step size alpha
            gamma: Discount factor
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = gamma
        
        # Initialize Q-table to zeros
        # Shape: [n_states, n_actions]
        self.Q = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        
        # Track statistics
        self.td_errors = []
        self.q_updates = []
    
    def get_action(self, state, epsilon=0.1):
        """
        Epsilon-greedy action selection.
        
        Args:
            state: Current state
            epsilon: Exploration probability
        
        Returns:
            Selected action
        """
        if random.random() < epsilon:
            # Explore: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: greedy action
            q_values = self.Q[state]  # Shape: [n_actions]
            return int(torch.argmax(q_values).item())
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update rule.
        
        Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state s
            action: Action taken a
            reward: Reward received r
            next_state: Next state s'
            done: Episode termination flag
        
        Returns:
            TD error for monitoring
        """
        # Current Q-value
        q_current = self.Q[state, action].item()
        
        # Target value
        if done:
            # Terminal state: no future rewards
            q_target = reward
        else:
            # Bootstrap from next state
            q_next_max = torch.max(self.Q[next_state]).item()
            q_target = reward + self.gamma * q_next_max
        
        # TD error
        td_error = q_target - q_current
        
        # Q-learning update
        self.Q[state, action] += self.alpha * td_error
        
        # Store statistics
        self.td_errors.append(td_error)
        self.q_updates.append({
            'state': state,
            'action': action,
            'q_old': q_current,
            'q_new': self.Q[state, action].item(),
            'td_error': td_error
        })
        
        return td_error

def demonstrate_single_update():
    """Demonstrate a single Q-learning update step."""
    print("="*50)
    print("Single Q-Learning Update")
    print("="*50)
    
    # Create simple agent
    agent = TabularQLearning(n_states=16, n_actions=4, learning_rate=0.5, gamma=0.9)
    
    # Manually set some Q-values for demonstration
    agent.Q[5, 2] = 0.3  # Q(s=5, a=2) = 0.3
    agent.Q[6, 0] = 0.7  # Q(s=6, a=0) = 0.7
    agent.Q[6, 1] = 0.5  # Q(s=6, a=1) = 0.5
    
    print("Initial Q-values:")
    print(f"  Q(5, 2) = {agent.Q[5, 2].item():.3f}")
    print(f"  Q(6, 0) = {agent.Q[6, 0].item():.3f}")
    print(f"  Q(6, 1) = {agent.Q[6, 1].item():.3f}")
    print()
    
    # Perform update: (s=5, a=2, r=0.1, s'=6, done=False)
    state, action, reward, next_state, done = 5, 2, 0.1, 6, False
    
    print("Transition: s=5, a=2, r=0.1, s'=6, done=False")
    print(f"Learning rate (α) = {agent.alpha}")
    print(f"Discount factor (γ) = {agent.gamma}")
    print()
    
    # Calculate components
    q_current = agent.Q[state, action].item()
    q_next_max = torch.max(agent.Q[next_state]).item()
    q_target = reward + agent.gamma * q_next_max
    td_error = q_target - q_current
    
    print("Update calculation:")
    print(f"  Q(s,a) = Q(5,2) = {q_current:.3f}")
    print(f"  max_a' Q(s',a') = max Q(6,:) = {q_next_max:.3f}")
    print(f"  Target = r + γ * max Q(s',a') = {reward} + {agent.gamma} * {q_next_max:.3f}")
    print(f"  Target = {q_target:.3f}")
    print(f"  TD error = {td_error:.3f}")
    print(f"  Update = α * TD_error = {agent.alpha} * {td_error:.3f} = {agent.alpha * td_error:.3f}")
    
    # Perform update
    agent.update(state, action, reward, next_state, done)
    
    print(f"\nAfter update:")
    print(f"  Q(5, 2) = {agent.Q[5, 2].item():.3f}")

def run_episode(env, agent, epsilon=0.1, max_steps=100, render=False):
    """Run a single episode with Q-learning."""
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        # Select action
        action = agent.get_action(state, epsilon)
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update Q-table
        td_error = agent.update(state, action, reward, next_state, done)
        
        if render and step < 5:  # Show first few steps
            print(f"  Step {step}: s={state}, a={action}, r={reward:.1f}, "
                  f"s'={next_state}, TD={td_error:.3f}")
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    return total_reward, steps

def train_simple_agent(episodes=500):
    """Train a Q-learning agent on FrozenLake."""
    print("="*50)
    print("Training Q-Learning Agent")
    print("="*50)
    
    # Create environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Create agent
    agent = TabularQLearning(n_states, n_actions, learning_rate=0.1, gamma=0.99)
    
    # Training parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    # Track performance
    returns = []
    successes = []
    epsilon_values = []
    
    epsilon = epsilon_start
    
    for episode in range(episodes):
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        epsilon_values.append(epsilon)
        
        # Run episode
        reward, steps = run_episode(env, agent, epsilon)
        returns.append(reward)
        successes.append(1 if reward > 0 else 0)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_return = np.mean(returns[-100:])
            avg_success = np.mean(successes[-100:])
            print(f"Episode {episode+1}: Avg return={avg_return:.3f}, "
                  f"Success rate={avg_success:.2%}, ε={epsilon:.3f}")
    
    env.close()
    
    # Analyze final Q-table
    print("\nFinal Q-table analysis:")
    print(f"  Max Q-value: {torch.max(agent.Q).item():.3f}")
    print(f"  Min Q-value: {torch.min(agent.Q).item():.3f}")
    print(f"  Mean Q-value: {torch.mean(agent.Q).item():.3f}")
    
    # Show Q-values for starting state
    print(f"\nQ-values for starting state (s=0):")
    for a in range(n_actions):
        print(f"  Q(0, {a}) = {agent.Q[0, a].item():.3f}")
    print(f"  Best action: {torch.argmax(agent.Q[0]).item()}")
    
    return agent, returns, successes, epsilon_values

def visualize_q_table(agent):
    """Visualize the Q-table as a heatmap."""
    print("="*50)
    print("Q-Table Visualization")
    print("="*50)
    
    Q_numpy = agent.Q.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(Q_numpy, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Q-value')
    plt.xlabel('Action (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)')
    plt.ylabel('State')
    plt.title('Q-Table Heatmap')
    
    # Mark max action for each state
    for s in range(agent.n_states):
        best_a = np.argmax(Q_numpy[s])
        plt.plot(best_a, s, 'g*', markersize=8)
    
    plt.tight_layout()
    plt.show()

def analyze_td_errors(agent):
    """Analyze TD error evolution during training."""
    print("="*50)
    print("TD Error Analysis")
    print("="*50)
    
    if len(agent.td_errors) == 0:
        print("No TD errors recorded")
        return
    
    td_errors = agent.td_errors
    
    # Statistics
    print(f"Total updates: {len(td_errors)}")
    print(f"Mean TD error: {np.mean(td_errors):.4f}")
    print(f"Std TD error: {np.std(td_errors):.4f}")
    print(f"Max TD error: {np.max(td_errors):.4f}")
    print(f"Min TD error: {np.min(td_errors):.4f}")
    
    # Show TD error evolution (moving average)
    window = 100
    if len(td_errors) > window:
        ma_td = np.convolve(td_errors, np.ones(window)/window, mode='valid')
        print(f"\nMoving average TD error (window={window}):")
        print(f"  First {window} updates: {ma_td[0]:.4f}")
        print(f"  Last {window} updates: {ma_td[-1]:.4f}")
        print(f"  Reduction: {(ma_td[0] - ma_td[-1])/abs(ma_td[0]):.1%}")

def compare_initialization_strategies():
    """Compare different Q-table initialization strategies."""
    print("="*50)
    print("Q-Table Initialization Comparison")
    print("="*50)
    
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    strategies = [
        ("Zero", 0.0),
        ("Small positive", 0.01),
        ("Optimistic", 1.0),
        ("Pessimistic", -1.0)
    ]
    
    for name, init_value in strategies:
        # Create agent with specific initialization
        agent = TabularQLearning(n_states, n_actions)
        agent.Q.fill_(init_value)
        
        # Train briefly
        returns = []
        epsilon = 0.1
        
        for _ in range(100):
            reward, _ = run_episode(env, agent, epsilon)
            returns.append(reward)
        
        avg_return = np.mean(returns)
        success_rate = np.mean([1 if r > 0 else 0 for r in returns])
        
        print(f"{name} (init={init_value:+.1f}): "
              f"Avg return={avg_return:.3f}, Success={success_rate:.2%}")
    
    env.close()

def main():
    print("="*50)
    print("Experiment 03: Tabular Q-Learning Basics")
    print("="*50)
    
    # 1. Demonstrate single update
    demonstrate_single_update()
    print()
    
    # 2. Train simple agent
    agent, returns, successes, epsilons = train_simple_agent(episodes=500)
    print()
    
    # 3. Analyze TD errors
    analyze_td_errors(agent)
    print()
    
    # 4. Compare initialization strategies
    compare_initialization_strategies()
    
    # 5. Final performance
    print("\n" + "="*50)
    print("Final Performance Summary")
    print("="*50)
    final_success = np.mean(successes[-100:]) if len(successes) >= 100 else np.mean(successes)
    print(f"Final success rate: {final_success:.2%}")
    print(f"Total TD errors recorded: {len(agent.td_errors)}")
    print(f"Final epsilon: {epsilons[-1]:.3f}")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()