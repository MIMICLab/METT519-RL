#!/usr/bin/env python3
"""
RL2025 - Lecture 5: Experiment 04 - Exploration Strategies

This experiment compares different exploration strategies for Q-learning.

Learning objectives:
- Implement epsilon-greedy, Boltzmann, and UCB exploration
- Compare exploration-exploitation trade-offs
- Analyze convergence under different strategies

Prerequisites: exp03_tabular_q.py completed successfully
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import gymnasium as gym
import matplotlib.pyplot as plt
import math

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

class ExplorationQLearning:
    """Q-learning with various exploration strategies."""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99, 
                 exploration_strategy="epsilon_greedy"):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy
        
        # Q-table
        self.Q = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        
        # Visit counts for UCB
        self.N_sa = torch.zeros((n_states, n_actions), dtype=torch.int32, device=device)
        self.N_s = torch.zeros(n_states, dtype=torch.int32, device=device)
        
        # Statistics
        self.action_counts = torch.zeros(n_actions, dtype=torch.int32)
        self.exploration_history = []
    
    def epsilon_greedy_action(self, state, epsilon=0.1):
        """Standard epsilon-greedy exploration."""
        if random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
            self.exploration_history.append(True)
        else:
            q_values = self.Q[state]
            action = int(torch.argmax(q_values).item())
            self.exploration_history.append(False)
        return action
    
    def boltzmann_action(self, state, temperature=1.0):
        """Boltzmann (softmax) exploration."""
        q_values = self.Q[state].cpu().numpy()
        
        # Compute probabilities with temperature
        # Subtract max for numerical stability
        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values / max(temperature, 0.01))
        probs = exp_q / np.sum(exp_q)
        
        # Sample action
        action = np.random.choice(self.n_actions, p=probs)
        
        # Track if this was exploration (not greedy)
        greedy_action = np.argmax(q_values)
        self.exploration_history.append(action != greedy_action)
        
        return int(action)
    
    def ucb_action(self, state, c=2.0):
        """Upper Confidence Bound (UCB) exploration."""
        q_values = self.Q[state].cpu().numpy()
        n_s = self.N_s[state].item()
        
        if n_s == 0:
            # First visit to state: random action
            action = random.randint(0, self.n_actions - 1)
            self.exploration_history.append(True)
        else:
            # UCB formula: Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
            ucb_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                n_sa = self.N_sa[state, a].item()
                if n_sa == 0:
                    # Unvisited action: infinite UCB
                    ucb_values[a] = float('inf')
                else:
                    exploration_bonus = c * math.sqrt(math.log(n_s) / n_sa)
                    ucb_values[a] = q_values[a] + exploration_bonus
            
            action = int(np.argmax(ucb_values))
            
            # Track if this was exploration
            greedy_action = np.argmax(q_values)
            self.exploration_history.append(action != greedy_action)
        
        return action
    
    def decaying_epsilon_greedy(self, state, episode, total_episodes):
        """Epsilon-greedy with various decay schedules."""
        # Linear decay
        epsilon = 1.0 - (0.99 * episode / total_episodes)
        return self.epsilon_greedy_action(state, epsilon)
    
    def get_action(self, state, **kwargs):
        """Select action based on exploration strategy."""
        if self.exploration_strategy == "epsilon_greedy":
            action = self.epsilon_greedy_action(state, kwargs.get('epsilon', 0.1))
        elif self.exploration_strategy == "boltzmann":
            action = self.boltzmann_action(state, kwargs.get('temperature', 1.0))
        elif self.exploration_strategy == "ucb":
            action = self.ucb_action(state, kwargs.get('c', 2.0))
        elif self.exploration_strategy == "decaying_epsilon":
            action = self.decaying_epsilon_greedy(
                state, kwargs.get('episode', 0), kwargs.get('total_episodes', 1000))
        else:
            # Default to greedy
            action = int(torch.argmax(self.Q[state]).item())
            self.exploration_history.append(False)
        
        # Update counts
        self.N_sa[state, action] += 1
        self.N_s[state] += 1
        self.action_counts[action] += 1
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update."""
        q_current = self.Q[state, action].item()
        
        if done:
            q_target = reward
        else:
            q_next_max = torch.max(self.Q[next_state]).item()
            q_target = reward + self.gamma * q_next_max
        
        td_error = q_target - q_current
        self.Q[state, action] += self.alpha * td_error
        
        return td_error

def compare_exploration_strategies(episodes=1000):
    """Compare different exploration strategies."""
    print("="*50)
    print("Exploration Strategy Comparison")
    print("="*50)
    
    strategies = [
        ("epsilon_greedy", {"epsilon": 0.1}),
        ("boltzmann", {"temperature": 1.0}),
        ("ucb", {"c": 2.0}),
        ("decaying_epsilon", {"total_episodes": episodes})
    ]
    
    results = {}
    
    for strategy_name, params in strategies:
        print(f"\nTesting {strategy_name}...")
        
        # Create environment and agent
        env = gym.make("FrozenLake-v1", is_slippery=False)
        agent = ExplorationQLearning(
            env.observation_space.n, 
            env.action_space.n,
            exploration_strategy=strategy_name
        )
        
        returns = []
        successes = []
        
        for episode in range(episodes):
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            
            # Add episode number for decaying strategies
            if strategy_name == "decaying_epsilon":
                params['episode'] = episode
            
            while not done:
                action = agent.get_action(state, **params)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            returns.append(total_reward)
            successes.append(1 if total_reward > 0 else 0)
        
        env.close()
        
        # Calculate metrics
        exploration_rate = np.mean(agent.exploration_history) if agent.exploration_history else 0
        final_success = np.mean(successes[-100:]) if len(successes) >= 100 else np.mean(successes)
        
        results[strategy_name] = {
            'returns': returns,
            'successes': successes,
            'exploration_rate': exploration_rate,
            'final_success': final_success,
            'action_distribution': agent.action_counts.cpu().numpy()
        }
        
        print(f"  Final success rate: {final_success:.2%}")
        print(f"  Exploration rate: {exploration_rate:.2%}")
        print(f"  Action distribution: {agent.action_counts.cpu().numpy()}")
    
    return results

def analyze_exploration_decay():
    """Analyze different epsilon decay schedules."""
    print("="*50)
    print("Epsilon Decay Analysis")
    print("="*50)
    
    episodes = 1000
    t = np.arange(episodes)
    
    # Different decay schedules
    schedules = {
        'Linear': lambda e: 1.0 - 0.99 * (e / episodes),
        'Exponential': lambda e: 0.01 + 0.99 * np.exp(-5 * e / episodes),
        'Inverse': lambda e: 1.0 / (1.0 + e / 100),
        'Cosine': lambda e: 0.01 + 0.495 * (1 + np.cos(np.pi * e / episodes))
    }
    
    plt.figure(figsize=(10, 6))
    for name, schedule in schedules.items():
        epsilons = [schedule(e) for e in t]
        plt.plot(t, epsilons, label=name, linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Test each schedule
    print("\nPerformance with different decay schedules:")
    
    for name, schedule in schedules.items():
        env = gym.make("FrozenLake-v1", is_slippery=False)
        agent = ExplorationQLearning(env.observation_space.n, env.action_space.n)
        
        successes = []
        for episode in range(500):  # Shorter test
            epsilon = schedule(episode)
            
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            
            while not done:
                action = agent.epsilon_greedy_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            successes.append(1 if total_reward > 0 else 0)
        
        env.close()
        
        final_success = np.mean(successes[-100:])
        print(f"  {name}: {final_success:.2%} success rate")

def test_temperature_schedules():
    """Test different temperature schedules for Boltzmann exploration."""
    print("="*50)
    print("Boltzmann Temperature Schedules")
    print("="*50)
    
    episodes = 500
    
    # Temperature schedules
    temp_schedules = {
        'Constant (T=1.0)': lambda e: 1.0,
        'Linear decay': lambda e: max(0.1, 2.0 - 1.9 * e / episodes),
        'Exponential decay': lambda e: 0.1 + 1.9 * np.exp(-5 * e / episodes),
        'Inverse decay': lambda e: 2.0 / (1.0 + e / 50)
    }
    
    for name, schedule in temp_schedules.items():
        env = gym.make("FrozenLake-v1", is_slippery=False)
        agent = ExplorationQLearning(
            env.observation_space.n, 
            env.action_space.n,
            exploration_strategy="boltzmann"
        )
        
        successes = []
        temps = []
        
        for episode in range(episodes):
            temperature = schedule(episode)
            temps.append(temperature)
            
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            
            while not done:
                action = agent.boltzmann_action(state, temperature)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            successes.append(1 if total_reward > 0 else 0)
        
        env.close()
        
        final_success = np.mean(successes[-100:])
        final_temp = temps[-1]
        print(f"{name}: {final_success:.2%} success, final T={final_temp:.3f}")

def analyze_ucb_parameters():
    """Analyze UCB exploration with different c values."""
    print("="*50)
    print("UCB Parameter Analysis")
    print("="*50)
    
    c_values = [0.5, 1.0, 2.0, 5.0]
    episodes = 500
    
    for c in c_values:
        env = gym.make("FrozenLake-v1", is_slippery=False)
        agent = ExplorationQLearning(
            env.observation_space.n,
            env.action_space.n,
            exploration_strategy="ucb"
        )
        
        successes = []
        total_explorations = 0
        
        for episode in range(episodes):
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            
            while not done:
                action = agent.ucb_action(state, c)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            successes.append(1 if total_reward > 0 else 0)
        
        env.close()
        
        exploration_rate = np.mean(agent.exploration_history)
        final_success = np.mean(successes[-100:])
        
        print(f"c={c:.1f}: Success={final_success:.2%}, Exploration={exploration_rate:.2%}")

def main():
    print("="*50)
    print("Experiment 04: Exploration Strategies")
    print("="*50)
    
    # 1. Compare main strategies
    results = compare_exploration_strategies(episodes=1000)
    
    # 2. Analyze epsilon decay schedules
    analyze_exploration_decay()
    
    # 3. Test temperature schedules for Boltzmann
    test_temperature_schedules()
    
    # 4. Analyze UCB parameters
    analyze_ucb_parameters()
    
    # 5. Summary
    print("\n" + "="*50)
    print("Exploration Strategy Summary")
    print("="*50)
    
    for strategy, data in results.items():
        print(f"{strategy}:")
        print(f"  Final success: {data['final_success']:.2%}")
        print(f"  Exploration rate: {data['exploration_rate']:.2%}")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()