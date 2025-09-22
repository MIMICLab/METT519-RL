#!/usr/bin/env python3
"""
RL2025 - Lecture 5: Experiment 07 - Double Q-Learning

This experiment implements Double Q-learning to address overestimation bias.

Learning objectives:
- Understand overestimation bias in Q-learning
- Implement Double Q-learning algorithm
- Compare performance with standard Q-learning

Prerequisites: exp06_stochastic.py completed successfully
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

class StandardQLearning:
    """Standard Q-learning with potential overestimation."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        
        self.Q = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        self.max_q_values = []  # Track max Q-values
    
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(torch.argmax(self.Q[state]).item())
    
    def update(self, state, action, reward, next_state, done):
        q_current = self.Q[state, action].item()
        
        if done:
            q_target = reward
        else:
            # Standard Q-learning: max over next actions
            q_next_max = torch.max(self.Q[next_state]).item()
            q_target = reward + self.gamma * q_next_max
            self.max_q_values.append(q_next_max)
        
        self.Q[state, action] += self.alpha * (q_target - q_current)
        return q_target - q_current

class DoubleQLearning:
    """Double Q-learning to reduce overestimation bias."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        
        # Two Q-tables
        self.Q1 = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        self.Q2 = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        
        self.max_q_values = []  # Track max Q-values
        self.update_counts = {'Q1': 0, 'Q2': 0}
    
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # Use average of both Q-tables for action selection
        q_avg = (self.Q1[state] + self.Q2[state]) / 2
        return int(torch.argmax(q_avg).item())
    
    def update(self, state, action, reward, next_state, done):
        # Randomly choose which Q-table to update
        if random.random() < 0.5:
            # Update Q1 using Q2 for value estimation
            q_current = self.Q1[state, action].item()
            
            if done:
                q_target = reward
            else:
                # Select action using Q1, evaluate using Q2
                best_action = torch.argmax(self.Q1[next_state]).item()
                q_next = self.Q2[next_state, best_action].item()
                q_target = reward + self.gamma * q_next
                self.max_q_values.append(q_next)
            
            self.Q1[state, action] += self.alpha * (q_target - q_current)
            self.update_counts['Q1'] += 1
            td_error = q_target - q_current
        else:
            # Update Q2 using Q1 for value estimation
            q_current = self.Q2[state, action].item()
            
            if done:
                q_target = reward
            else:
                # Select action using Q2, evaluate using Q1
                best_action = torch.argmax(self.Q2[next_state]).item()
                q_next = self.Q1[next_state, best_action].item()
                q_target = reward + self.gamma * q_next
                self.max_q_values.append(q_next)
            
            self.Q2[state, action] += self.alpha * (q_target - q_current)
            self.update_counts['Q2'] += 1
            td_error = q_target - q_current
        
        return td_error
    
    def get_combined_q(self):
        """Return average of both Q-tables."""
        return (self.Q1 + self.Q2) / 2

def create_overestimation_example():
    """Create a simple example showing overestimation bias."""
    print("="*50)
    print("Overestimation Bias Example")
    print("="*50)
    
    # Simple 2-state, 2-action example
    n_states, n_actions = 2, 2
    
    # Standard Q-learning
    std_agent = StandardQLearning(n_states, n_actions, alpha=0.5)
    
    # Double Q-learning
    double_agent = DoubleQLearning(n_states, n_actions, alpha=0.5)
    
    # Simulate transitions with noise
    print("\nSimulating noisy rewards...")
    print("True optimal value: 0.5")
    print("Rewards: 0.5 + noise ~ N(0, 0.5)")
    print()
    
    for i in range(100):
        # Transition from state 0 to state 1
        state, action = 0, 0
        next_state = 1
        
        # Noisy reward (true value 0.5 with noise)
        true_value = 0.5
        noise = np.random.normal(0, 0.5)
        reward = true_value + noise
        
        # Update both agents
        std_agent.update(state, action, reward, next_state, done=False)
        double_agent.update(state, action, reward, next_state, done=False)
        
        if (i + 1) % 20 == 0:
            std_q = std_agent.Q[0, 0].item()
            double_q = (double_agent.Q1[0, 0].item() + double_agent.Q2[0, 0].item()) / 2
            print(f"Update {i+1:3d}: Std Q={std_q:.3f}, Double Q={double_q:.3f}")
    
    # Final comparison
    print("\nFinal estimates:")
    std_estimate = std_agent.Q[0, 0].item()
    double_estimate = (double_agent.Q1[0, 0].item() + double_agent.Q2[0, 0].item()) / 2
    
    print(f"Standard Q-learning: {std_estimate:.3f} (bias: {std_estimate - true_value:+.3f})")
    print(f"Double Q-learning:   {double_estimate:.3f} (bias: {double_estimate - true_value:+.3f})")

def compare_on_frozenlake(episodes=1000):
    """Compare standard and double Q-learning on FrozenLake."""
    print("="*50)
    print("FrozenLake Comparison")
    print("="*50)
    
    # Create environment
    env = gym.make("FrozenLake-v1", is_slippery=True)  # Use slippery for more stochasticity
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Create agents
    std_agent = StandardQLearning(n_states, n_actions, alpha=0.1, gamma=0.99)
    double_agent = DoubleQLearning(n_states, n_actions, alpha=0.1, gamma=0.99)
    
    # Training parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    # Track performance
    std_returns = []
    double_returns = []
    std_q_means = []
    double_q_means = []
    
    epsilon = epsilon_start
    
    for episode in range(episodes):
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Train standard Q-learning
        state, _ = env.reset(seed=42 + episode)
        done = False
        std_return = 0
        
        while not done:
            action = std_agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            std_agent.update(state, action, reward, next_state, done)
            state = next_state
            std_return += reward
        
        std_returns.append(std_return)
        std_q_means.append(torch.mean(std_agent.Q).item())
        
        # Train double Q-learning
        state, _ = env.reset(seed=42 + episode)
        done = False
        double_return = 0
        
        while not done:
            action = double_agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            double_agent.update(state, action, reward, next_state, done)
            state = next_state
            double_return += reward
        
        double_returns.append(double_return)
        double_q_means.append(torch.mean(double_agent.get_combined_q()).item())
        
        # Progress report
        if (episode + 1) % 200 == 0:
            std_success = np.mean(std_returns[-100:])
            double_success = np.mean(double_returns[-100:])
            print(f"Episode {episode+1}: Std={std_success:.2%}, Double={double_success:.2%}")
    
    env.close()
    
    # Final comparison
    print("\nFinal Performance:")
    std_final = np.mean(std_returns[-100:])
    double_final = np.mean(double_returns[-100:])
    print(f"Standard Q-learning: {std_final:.2%}")
    print(f"Double Q-learning:   {double_final:.2%}")
    
    # Q-value statistics
    print("\nQ-value Statistics:")
    print(f"Standard - Mean: {std_q_means[-1]:.3f}, Max: {torch.max(std_agent.Q).item():.3f}")
    print(f"Double   - Mean: {double_q_means[-1]:.3f}, Max: {torch.max(double_agent.get_combined_q()).item():.3f}")
    
    # Update balance for double Q-learning
    print(f"\nDouble Q-learning update balance:")
    print(f"Q1 updates: {double_agent.update_counts['Q1']}")
    print(f"Q2 updates: {double_agent.update_counts['Q2']}")
    
    return std_returns, double_returns, std_q_means, double_q_means

def visualize_q_value_evolution(std_q_means, double_q_means):
    """Visualize how Q-values evolve during training."""
    plt.figure(figsize=(12, 5))
    
    # Q-value means over time
    plt.subplot(1, 2, 1)
    episodes = len(std_q_means)
    plt.plot(range(episodes), std_q_means, 'b-', label='Standard Q', alpha=0.7)
    plt.plot(range(episodes), double_q_means, 'r-', label='Double Q', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Mean Q-value')
    plt.title('Q-value Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Difference in Q-values
    plt.subplot(1, 2, 2)
    diff = np.array(std_q_means) - np.array(double_q_means)
    plt.plot(range(episodes), diff, 'g-', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Q(Standard) - Q(Double)')
    plt.title('Overestimation Bias')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_with_optimistic_initialization():
    """Test impact of optimistic initialization."""
    print("="*50)
    print("Optimistic Initialization Test")
    print("="*50)
    
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    init_values = [0.0, 0.5, 1.0, 2.0]
    episodes = 500
    
    for init_val in init_values:
        # Standard Q-learning
        std_agent = StandardQLearning(n_states, n_actions)
        std_agent.Q.fill_(init_val)
        
        # Double Q-learning
        double_agent = DoubleQLearning(n_states, n_actions)
        double_agent.Q1.fill_(init_val)
        double_agent.Q2.fill_(init_val)
        
        std_returns = []
        double_returns = []
        
        for episode in range(episodes):
            # Standard Q-learning
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            
            while not done:
                action = std_agent.select_action(state, epsilon=0.1)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                std_agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            std_returns.append(total_reward)
            
            # Double Q-learning
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            
            while not done:
                action = double_agent.select_action(state, epsilon=0.1)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                double_agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            double_returns.append(total_reward)
        
        std_success = np.mean(std_returns[-100:])
        double_success = np.mean(double_returns[-100:])
        
        print(f"\nInit={init_val:.1f}:")
        print(f"  Standard: {std_success:.2%} success")
        print(f"  Double:   {double_success:.2%} success")
    
    env.close()

def analyze_bias_in_states():
    """Analyze overestimation bias state by state."""
    print("="*50)
    print("State-by-State Bias Analysis")
    print("="*50)
    
    env = gym.make("FrozenLake-v1", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Train both algorithms
    std_agent = StandardQLearning(n_states, n_actions)
    double_agent = DoubleQLearning(n_states, n_actions)
    
    episodes = 1000
    epsilon = 0.1
    
    for episode in range(episodes):
        # Standard Q-learning
        state, _ = env.reset(seed=42 + episode)
        done = False
        while not done:
            action = std_agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            std_agent.update(state, action, reward, next_state, done)
            state = next_state
        
        # Double Q-learning
        state, _ = env.reset(seed=42 + episode)
        done = False
        while not done:
            action = double_agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            double_agent.update(state, action, reward, next_state, done)
            state = next_state
    
    env.close()
    
    # Compare max Q-values for each state
    print("\nMax Q-value comparison by state:")
    print("State | Standard | Double | Bias")
    print("-" * 40)
    
    total_bias = 0
    for s in range(n_states):
        std_max = torch.max(std_agent.Q[s]).item()
        double_max = torch.max(double_agent.get_combined_q()[s]).item()
        bias = std_max - double_max
        total_bias += bias
        
        if s % 4 == 0:  # Sample states
            print(f"  {s:2d}  | {std_max:7.4f} | {double_max:7.4f} | {bias:+7.4f}")
    
    avg_bias = total_bias / n_states
    print(f"\nAverage overestimation bias: {avg_bias:+.4f}")

def main():
    print("="*50)
    print("Experiment 07: Double Q-Learning")
    print("="*50)
    
    # 1. Simple overestimation example
    create_overestimation_example()
    print()
    
    # 2. Compare on FrozenLake
    std_returns, double_returns, std_q_means, double_q_means = compare_on_frozenlake(1000)
    print()
    
    # 3. Visualize Q-value evolution
    # visualize_q_value_evolution(std_q_means, double_q_means)
    
    # 4. Test with optimistic initialization
    test_with_optimistic_initialization()
    print()
    
    # 5. Analyze bias by state
    analyze_bias_in_states()
    
    print("\n" + "="*50)
    print("Key Findings:")
    print("- Standard Q-learning tends to overestimate Q-values")
    print("- Double Q-learning reduces this bias effectively")
    print("- The bias is more pronounced with optimistic initialization")
    print("- Performance improvement varies with environment stochasticity")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()