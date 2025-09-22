#!/usr/bin/env python3
"""
RL2025 - Lecture 5: Experiment 08 - Hyperparameter Optimization

This experiment explores hyperparameter optimization for Q-learning.

Learning objectives:
- Systematic hyperparameter search
- Performance sensitivity analysis
- Optimal configuration discovery

Prerequisites: exp07_double_q.py completed successfully
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import itertools
import json
from dataclasses import dataclass, asdict
import hashlib

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

@dataclass
class HyperParams:
    """Hyperparameter configuration for Q-learning."""
    alpha: float = 0.1          # Learning rate
    gamma: float = 0.99         # Discount factor
    epsilon_start: float = 1.0  # Initial exploration
    epsilon_end: float = 0.01   # Final exploration
    epsilon_decay: float = 0.995 # Exploration decay rate
    
    def get_id(self):
        """Generate unique ID for this configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class OptimizedQLearning:
    """Q-learning agent with configurable hyperparameters."""
    def __init__(self, n_states, n_actions, params: HyperParams):
        self.n_states = n_states
        self.n_actions = n_actions
        self.params = params
        
        self.Q = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        self.epsilon = params.epsilon_start
        
        # Statistics
        self.td_errors = []
        self.q_values = []
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(torch.argmax(self.Q[state]).item())
    
    def update(self, state, action, reward, next_state, done):
        q_current = self.Q[state, action].item()
        
        if done:
            q_target = reward
        else:
            q_target = reward + self.params.gamma * torch.max(self.Q[next_state]).item()
        
        td_error = q_target - q_current
        self.Q[state, action] += self.params.alpha * td_error
        
        self.td_errors.append(abs(td_error))
        self.q_values.append(torch.mean(self.Q).item())
        
        return td_error
    
    def decay_epsilon(self):
        self.epsilon = max(self.params.epsilon_end, 
                          self.epsilon * self.params.epsilon_decay)

def evaluate_params(params: HyperParams, env_name="FrozenLake-v1", 
                    episodes=500, runs=5, verbose=False):
    """Evaluate a hyperparameter configuration."""
    all_returns = []
    all_successes = []
    
    for run in range(runs):
        env = gym.make(env_name, is_slippery=False)
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        agent = OptimizedQLearning(n_states, n_actions, params)
        
        returns = []
        for episode in range(episodes):
            state, _ = env.reset(seed=42 + run * 1000 + episode)
            done = False
            total_reward = 0
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            agent.decay_epsilon()
            returns.append(total_reward)
        
        env.close()
        
        # Calculate metrics for this run
        final_returns = returns[-100:] if len(returns) >= 100 else returns
        all_returns.extend(final_returns)
        all_successes.append(np.mean(final_returns))
        
        if verbose and run == 0:
            print(f"  Run 1: Success={np.mean(final_returns):.2%}")
    
    # Aggregate metrics
    mean_success = np.mean(all_successes)
    std_success = np.std(all_successes)
    
    return {
        'mean_success': mean_success,
        'std_success': std_success,
        'all_successes': all_successes
    }

def grid_search():
    """Perform grid search over hyperparameters."""
    print("="*50)
    print("Grid Search for Optimal Hyperparameters")
    print("="*50)
    
    # Define search space
    search_space = {
        'alpha': [0.01, 0.05, 0.1, 0.2, 0.5],
        'gamma': [0.9, 0.95, 0.99],
        'epsilon_decay': [0.99, 0.995, 0.999]
    }
    
    # Generate all combinations
    keys = search_space.keys()
    values = search_space.values()
    combinations = list(itertools.product(*values))
    
    print(f"Testing {len(combinations)} configurations...")
    print()
    
    results = []
    best_config = None
    best_score = -float('inf')
    
    for i, combo in enumerate(combinations):
        config_dict = dict(zip(keys, combo))
        params = HyperParams(
            alpha=config_dict['alpha'],
            gamma=config_dict['gamma'],
            epsilon_decay=config_dict['epsilon_decay']
        )
        
        # Evaluate configuration
        metrics = evaluate_params(params, episodes=300, runs=3)
        
        results.append({
            'params': config_dict,
            'score': metrics['mean_success'],
            'std': metrics['std_success']
        })
        
        # Track best
        if metrics['mean_success'] > best_score:
            best_score = metrics['mean_success']
            best_config = config_dict
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(combinations)} configurations tested")
    
    # Sort results
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nTop 5 Configurations:")
    print("-" * 50)
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Score: {result['score']:.2%} ± {result['std']:.2%}")
        print(f"   α={result['params']['alpha']:.2f}, "
              f"γ={result['params']['gamma']:.2f}, "
              f"ε_decay={result['params']['epsilon_decay']:.3f}")
    
    return best_config, results

def sensitivity_analysis():
    """Analyze sensitivity to each hyperparameter."""
    print("="*50)
    print("Hyperparameter Sensitivity Analysis")
    print("="*50)
    
    # Base configuration
    base_params = HyperParams(
        alpha=0.1,
        gamma=0.99,
        epsilon_decay=0.995
    )
    
    # Test ranges
    param_ranges = {
        'alpha': np.logspace(-2, 0, 10),  # 0.01 to 1.0
        'gamma': np.linspace(0.8, 0.999, 10),
        'epsilon_decay': np.linspace(0.98, 0.999, 10)
    }
    
    sensitivity_results = {}
    
    for param_name, values in param_ranges.items():
        print(f"\nTesting {param_name}...")
        scores = []
        
        for value in values:
            # Create modified params
            test_params = HyperParams(
                alpha=base_params.alpha,
                gamma=base_params.gamma,
                epsilon_decay=base_params.epsilon_decay
            )
            setattr(test_params, param_name, value)
            
            # Evaluate
            metrics = evaluate_params(test_params, episodes=300, runs=3)
            scores.append(metrics['mean_success'])
        
        sensitivity_results[param_name] = {
            'values': values.tolist(),
            'scores': scores
        }
        
        # Find optimal value
        best_idx = np.argmax(scores)
        best_value = values[best_idx]
        best_score = scores[best_idx]
        
        print(f"  Optimal {param_name}: {best_value:.3f} (score: {best_score:.2%})")
    
    return sensitivity_results

def test_environment_variations():
    """Test optimal parameters on different environments."""
    print("="*50)
    print("Testing on Environment Variations")
    print("="*50)
    
    # Use previously found good parameters
    optimal_params = HyperParams(
        alpha=0.1,
        gamma=0.99,
        epsilon_decay=0.995
    )
    
    # Test different environments
    env_configs = [
        ("4x4 Easy", 4, 0.1, False),
        ("4x4 Medium", 4, 0.2, False),
        ("4x4 Slippery", 4, 0.2, True),
        ("8x8 Easy", 8, 0.1, False),
        ("8x8 Hard", 8, 0.3, False)
    ]
    
    for name, size, hole_prob, is_slippery in env_configs:
        print(f"\n{name} (size={size}, holes={hole_prob:.0%}, slip={is_slippery}):")
        
        # Create custom environment
        if size != 4 or hole_prob != 0.2:
            safe_prob = 1.0 - hole_prob
            desc = generate_random_map(size=size, p=safe_prob, seed=42)
            env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)
        else:
            env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
        
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        # Train agent
        agent = OptimizedQLearning(n_states, n_actions, optimal_params)
        
        returns = []
        max_steps = 200 if size == 8 else 100
        
        for episode in range(500):
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            
            agent.decay_epsilon()
            returns.append(total_reward)
        
        env.close()
        
        success_rate = np.mean(returns[-100:])
        convergence_episode = None
        
        # Find convergence point (first episode with >50% success in 50-episode window)
        for i in range(50, len(returns)):
            if np.mean(returns[i-50:i]) > 0.5:
                convergence_episode = i
                break
        
        print(f"  Success rate: {success_rate:.2%}")
        if convergence_episode:
            print(f"  Converged at episode: {convergence_episode}")
        else:
            print(f"  Did not converge to 50% success")

def adaptive_hyperparameters():
    """Test adaptive hyperparameter schedules."""
    print("="*50)
    print("Adaptive Hyperparameter Schedules")
    print("="*50)
    
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    strategies = {
        "Fixed": lambda ep, td: (0.1, 0.995),  # Fixed alpha and epsilon_decay
        "Linear decay": lambda ep, td: (max(0.01, 0.5 - 0.49 * ep / 500), 0.995),
        "TD-based": lambda ep, td: (min(0.5, 0.1 + 0.4 * td), 0.995),
        "Episode-based": lambda ep, td: (0.1, 0.99 + 0.009 * min(1, ep / 500))
    }
    
    for name, strategy in strategies.items():
        print(f"\n{name} strategy:")
        
        agent = OptimizedQLearning(n_states, n_actions, HyperParams())
        returns = []
        
        for episode in range(500):
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            episode_td_errors = []
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                td_error = agent.update(state, action, reward, next_state, done)
                episode_td_errors.append(abs(td_error))
                
                state = next_state
                total_reward += reward
            
            # Adapt hyperparameters
            avg_td = np.mean(episode_td_errors) if episode_td_errors else 0
            new_alpha, new_decay = strategy(episode, avg_td)
            agent.params.alpha = new_alpha
            agent.params.epsilon_decay = new_decay
            agent.decay_epsilon()
            
            returns.append(total_reward)
        
        success_rate = np.mean(returns[-100:])
        print(f"  Final success rate: {success_rate:.2%}")
        print(f"  Final alpha: {agent.params.alpha:.3f}")
        print(f"  Final epsilon: {agent.epsilon:.3f}")
    
    env.close()

def save_best_config(config, filename="best_qlearning_config.json"):
    """Save best configuration to file."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nBest configuration saved to {filename}")

def main():
    print("="*50)
    print("Experiment 08: Hyperparameter Optimization")
    print("="*50)
    
    # 1. Grid search
    best_config, all_results = grid_search()
    print(f"\nBest configuration found:")
    print(f"  α={best_config['alpha']:.2f}")
    print(f"  γ={best_config['gamma']:.2f}")
    print(f"  ε_decay={best_config['epsilon_decay']:.3f}")
    
    # 2. Sensitivity analysis
    sensitivity = sensitivity_analysis()
    
    # 3. Test on variations
    test_environment_variations()
    
    # 4. Adaptive hyperparameters
    adaptive_hyperparameters()
    
    # 5. Save best configuration
    # save_best_config(best_config)
    
    print("\n" + "="*50)
    print("Optimization Summary:")
    print(f"- Best alpha: {best_config['alpha']:.2f}")
    print(f"- Best gamma: {best_config['gamma']:.2f}")
    print(f"- Best epsilon decay: {best_config['epsilon_decay']:.3f}")
    print("- Adaptive schedules can improve convergence")
    print("- Optimal parameters vary with environment complexity")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()