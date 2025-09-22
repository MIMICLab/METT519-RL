#!/usr/bin/env python3
"""
RL2025 - Lecture 5: Experiment 06 - Q-Learning in Stochastic Environments

This experiment explores Q-learning behavior in stochastic environments.

Learning objectives:
- Compare Q-learning vs SARSA in stochastic settings
- Analyze impact of environment slipperiness
- Understand optimistic vs conservative value estimates

Prerequisites: exp05_schedules.py completed successfully
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
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

class ActionSlipWrapper(gym.Wrapper):
    """Wrapper that adds action slippage to any environment."""
    def __init__(self, env, slip_prob=0.1):
        super().__init__(env)
        self.slip_prob = slip_prob
    
    def step(self, action):
        if random.random() < self.slip_prob:
            # Slip: execute random action instead
            action = self.action_space.sample()
        return self.env.step(action)

class QLearningAgent:
    """Standard Q-learning (off-policy)."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        self.update_count = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(torch.argmax(self.Q[state]).item())
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning: use max over next actions."""
        q_current = self.Q[state, action].item()
        
        if done:
            q_target = reward
        else:
            # Off-policy: use max Q-value
            q_target = reward + self.gamma * torch.max(self.Q[next_state]).item()
        
        self.Q[state, action] += self.alpha * (q_target - q_current)
        self.update_count += 1
        
        return q_target - q_current  # TD error

class SARSAAgent:
    """SARSA (on-policy)."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        self.update_count = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(torch.argmax(self.Q[state]).item())
    
    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA: use actual next action."""
        q_current = self.Q[state, action].item()
        
        if done:
            q_target = reward
        else:
            # On-policy: use Q-value of actual next action
            q_target = reward + self.gamma * self.Q[next_state, next_action].item()
        
        self.Q[state, action] += self.alpha * (q_target - q_current)
        self.update_count += 1
        
        return q_target - q_current  # TD error

def compare_qlearning_sarsa(slip_prob=0.0, episodes=1000):
    """Compare Q-learning and SARSA in stochastic environment."""
    print(f"\nComparing with slip_prob={slip_prob:.1f}")
    print("-" * 40)
    
    # Create environments
    env_q = gym.make("FrozenLake-v1", is_slippery=False)
    env_s = gym.make("FrozenLake-v1", is_slippery=False)
    
    if slip_prob > 0:
        env_q = ActionSlipWrapper(env_q, slip_prob)
        env_s = ActionSlipWrapper(env_s, slip_prob)
    
    n_states = env_q.observation_space.n
    n_actions = env_q.action_space.n
    
    # Create agents
    q_agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)
    sarsa_agent = SARSAAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    # Track performance
    q_returns = []
    sarsa_returns = []
    
    for episode in range(episodes):
        # Q-learning episode
        state, _ = env_q.reset(seed=42 + episode)
        done = False
        q_return = 0
        
        while not done:
            action = q_agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env_q.step(action)
            done = terminated or truncated
            q_agent.update(state, action, reward, next_state, done)
            state = next_state
            q_return += reward
        
        q_returns.append(q_return)
        
        # SARSA episode
        state, _ = env_s.reset(seed=42 + episode)
        action = sarsa_agent.select_action(state)
        done = False
        sarsa_return = 0
        
        while not done:
            next_state, reward, terminated, truncated, _ = env_s.step(action)
            done = terminated or truncated
            next_action = sarsa_agent.select_action(next_state)
            sarsa_agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            sarsa_return += reward
        
        sarsa_returns.append(sarsa_return)
    
    env_q.close()
    env_s.close()
    
    # Compare results
    q_success = np.mean(q_returns[-100:])
    sarsa_success = np.mean(sarsa_returns[-100:])
    
    print(f"Q-learning success rate: {q_success:.2%}")
    print(f"SARSA success rate: {sarsa_success:.2%}")
    
    # Compare Q-values
    q_mean = torch.mean(q_agent.Q).item()
    sarsa_mean = torch.mean(sarsa_agent.Q).item()
    
    print(f"Q-learning mean Q-value: {q_mean:.3f}")
    print(f"SARSA mean Q-value: {sarsa_mean:.3f}")
    
    return q_returns, sarsa_returns, q_agent.Q, sarsa_agent.Q

def analyze_slippage_impact():
    """Analyze impact of increasing slippage."""
    print("="*50)
    print("Impact of Environment Slippage")
    print("="*50)
    
    slip_probs = [0.0, 0.1, 0.2, 0.3, 0.4]
    episodes = 1000
    
    q_performances = []
    sarsa_performances = []
    
    for slip_prob in slip_probs:
        q_returns, sarsa_returns, _, _ = compare_qlearning_sarsa(slip_prob, episodes)
        
        q_performances.append(np.mean(q_returns[-100:]))
        sarsa_performances.append(np.mean(sarsa_returns[-100:]))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(slip_probs, q_performances, 'b-o', label='Q-learning', linewidth=2)
    plt.plot(slip_probs, sarsa_performances, 'r-s', label='SARSA', linewidth=2)
    plt.xlabel('Action Slip Probability')
    plt.ylabel('Success Rate')
    plt.title('Q-learning vs SARSA under Stochasticity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nPerformance Summary:")
    print("Slip Prob | Q-learning | SARSA")
    print("-" * 35)
    for i, slip in enumerate(slip_probs):
        print(f"  {slip:.1f}    |   {q_performances[i]:.2%}    | {sarsa_performances[i]:.2%}")

def test_native_slippery():
    """Test with FrozenLake's native slippery dynamics."""
    print("="*50)
    print("Native Slippery Dynamics")
    print("="*50)
    
    episodes = 1000
    
    for is_slippery in [False, True]:
        print(f"\nis_slippery={is_slippery}:")
        print("-" * 30)
        
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        # Train Q-learning
        q_agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)
        q_returns = []
        
        for episode in range(episodes):
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            
            while not done:
                action = q_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                q_agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            q_returns.append(total_reward)
        
        env.close()
        
        success_rate = np.mean(q_returns[-100:])
        print(f"Success rate: {success_rate:.2%}")
        print(f"Mean Q-value: {torch.mean(q_agent.Q).item():.3f}")
        print(f"Max Q-value: {torch.max(q_agent.Q).item():.3f}")

def analyze_value_estimates():
    """Analyze optimistic vs pessimistic value estimates."""
    print("="*50)
    print("Value Estimate Analysis")
    print("="*50)
    
    slip_prob = 0.2
    episodes = 500
    
    # Create environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = ActionSlipWrapper(env, slip_prob)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Train both algorithms
    q_agent = QLearningAgent(n_states, n_actions)
    sarsa_agent = SARSAAgent(n_states, n_actions)
    
    for episode in range(episodes):
        # Q-learning
        state, _ = env.reset(seed=42 + episode)
        done = False
        while not done:
            action = q_agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            q_agent.update(state, action, reward, next_state, done)
            state = next_state
        
        # SARSA
        state, _ = env.reset(seed=42 + episode)
        action = sarsa_agent.select_action(state)
        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = sarsa_agent.select_action(next_state)
            sarsa_agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
    
    env.close()
    
    # Compare Q-values for specific states
    print("\nQ-value comparison for selected states:")
    print("State | Q-learning | SARSA | Difference")
    print("-" * 45)
    
    for state in [0, 5, 10, 15]:  # Sample states
        q_max = torch.max(q_agent.Q[state]).item()
        sarsa_max = torch.max(sarsa_agent.Q[state]).item()
        diff = q_max - sarsa_max
        
        print(f"  {state:2d}  |   {q_max:6.3f}   | {sarsa_max:6.3f} |  {diff:+6.3f}")
    
    # Overall statistics
    q_values = q_agent.Q.cpu().numpy().flatten()
    sarsa_values = sarsa_agent.Q.cpu().numpy().flatten()
    
    print("\nOverall Q-value statistics:")
    print(f"Q-learning: mean={np.mean(q_values):.3f}, std={np.std(q_values):.3f}")
    print(f"SARSA:      mean={np.mean(sarsa_values):.3f}, std={np.std(sarsa_values):.3f}")
    
    # Count optimistic estimates
    optimistic = np.sum(q_values > sarsa_values)
    total = len(q_values)
    print(f"\nOptimistic estimates: {optimistic}/{total} ({optimistic/total:.1%})")

def test_different_maps():
    """Test on maps with different difficulty levels."""
    print("="*50)
    print("Performance on Different Maps")
    print("="*50)
    
    map_configs = [
        (4, 0.1, "Easy (4x4, 10% holes)"),
        (4, 0.3, "Medium (4x4, 30% holes)"),
        (8, 0.2, "Large (8x8, 20% holes)"),
        (8, 0.4, "Hard (8x8, 40% holes)")
    ]
    
    slip_prob = 0.1
    episodes = 500
    
    for size, hole_prob, description in map_configs:
        print(f"\n{description}:")
        print("-" * 30)
        
        # Generate map
        safe_prob = 1.0 - hole_prob
        desc = generate_random_map(size=size, p=safe_prob, seed=42)
        
        # Create environment
        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
        env = ActionSlipWrapper(env, slip_prob)
        
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        # Train Q-learning
        agent = QLearningAgent(n_states, n_actions, alpha=0.2, gamma=0.99, epsilon=0.2)
        returns = []
        
        for episode in range(episodes):
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 200:  # Limit steps for large maps
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            
            returns.append(total_reward)
        
        env.close()
        
        success_rate = np.mean([1 if r > 0 else 0 for r in returns[-100:]])
        print(f"Success rate: {success_rate:.2%}")
        print(f"Updates performed: {agent.update_count}")

def main():
    print("="*50)
    print("Experiment 06: Q-Learning in Stochastic Environments")
    print("="*50)
    
    # 1. Compare Q-learning and SARSA
    print("\n1. Basic Comparison:")
    compare_qlearning_sarsa(slip_prob=0.2, episodes=1000)
    
    # 2. Analyze slippage impact
    print("\n2. Slippage Impact:")
    analyze_slippage_impact()
    
    # 3. Test native slippery dynamics
    print("\n3. Native Slippery:")
    test_native_slippery()
    
    # 4. Analyze value estimates
    print("\n4. Value Estimates:")
    analyze_value_estimates()
    
    # 5. Test different maps
    print("\n5. Different Maps:")
    test_different_maps()
    
    print("\n" + "="*50)
    print("Key Findings:")
    print("- Q-learning tends to be more optimistic than SARSA")
    print("- SARSA performs better in highly stochastic environments")
    print("- Both algorithms struggle with large maps and high hole density")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()