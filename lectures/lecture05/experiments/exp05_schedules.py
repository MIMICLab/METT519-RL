#!/usr/bin/env python3
"""
RL2025 - Lecture 5: Experiment 05 - Learning Rate and Schedule Analysis

This experiment analyzes the impact of learning rate schedules on Q-learning.

Learning objectives:
- Implement various learning rate schedules
- Understand Robbins-Monro conditions for convergence
- Compare schedule performance empirically

Prerequisites: exp04_exploration.py completed successfully
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

class ScheduledQLearning:
    """Q-learning with configurable schedules for alpha and epsilon."""
    
    def __init__(self, n_states, n_actions, gamma=0.99,
                 alpha_schedule="constant", eps_schedule="constant",
                 alpha_params=None, eps_params=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Q-table
        self.Q = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        
        # Schedule configurations
        self.alpha_schedule = alpha_schedule
        self.eps_schedule = eps_schedule
        self.alpha_params = alpha_params or {}
        self.eps_params = eps_params or {}
        
        # Visit counts for state-action dependent schedules
        self.N_sa = torch.zeros((n_states, n_actions), dtype=torch.int32, device=device)
        self.global_step = 0
        
        # Track schedule values
        self.alpha_history = []
        self.eps_history = []
        self.td_errors = []
    
    def get_alpha(self, state=None, action=None):
        """Get current learning rate based on schedule."""
        if self.alpha_schedule == "constant":
            alpha = self.alpha_params.get('value', 0.1)
        
        elif self.alpha_schedule == "linear":
            start = self.alpha_params.get('start', 1.0)
            end = self.alpha_params.get('end', 0.01)
            steps = self.alpha_params.get('steps', 10000)
            progress = min(1.0, self.global_step / steps)
            alpha = start + (end - start) * progress
        
        elif self.alpha_schedule == "exponential":
            start = self.alpha_params.get('start', 1.0)
            end = self.alpha_params.get('end', 0.01)
            tau = self.alpha_params.get('tau', 5000)
            alpha = end + (start - end) * math.exp(-self.global_step / tau)
        
        elif self.alpha_schedule == "one_over_t":
            start = self.alpha_params.get('start', 1.0)
            tau = self.alpha_params.get('tau', 100)
            alpha = start / (1.0 + self.global_step / tau)
        
        elif self.alpha_schedule == "one_over_sqrt_t":
            start = self.alpha_params.get('start', 1.0)
            tau = self.alpha_params.get('tau', 100)
            alpha = start / math.sqrt(1.0 + self.global_step / tau)
        
        elif self.alpha_schedule == "state_action_dependent":
            # Alpha depends on visit count of (s,a)
            if state is not None and action is not None:
                n_sa = self.N_sa[state, action].item()
                start = self.alpha_params.get('start', 1.0)
                alpha = start / (1.0 + n_sa)
            else:
                alpha = 0.1
        
        else:
            alpha = 0.1
        
        self.alpha_history.append(alpha)
        return alpha
    
    def get_epsilon(self):
        """Get current exploration rate based on schedule."""
        if self.eps_schedule == "constant":
            eps = self.eps_params.get('value', 0.1)
        
        elif self.eps_schedule == "linear":
            start = self.eps_params.get('start', 1.0)
            end = self.eps_params.get('end', 0.01)
            steps = self.eps_params.get('steps', 10000)
            progress = min(1.0, self.global_step / steps)
            eps = start + (end - start) * progress
        
        elif self.eps_schedule == "exponential":
            start = self.eps_params.get('start', 1.0)
            end = self.eps_params.get('end', 0.01)
            tau = self.eps_params.get('tau', 5000)
            eps = end + (start - end) * math.exp(-self.global_step / tau)
        
        elif self.eps_schedule == "one_over_t":
            start = self.eps_params.get('start', 1.0)
            tau = self.eps_params.get('tau', 100)
            eps = max(0.01, start / (1.0 + self.global_step / tau))
        
        else:
            eps = 0.1
        
        self.eps_history.append(eps)
        return eps
    
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            q_values = self.Q[state]
            return int(torch.argmax(q_values).item())
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update with scheduled learning rate."""
        # Get state-action dependent alpha
        alpha = self.get_alpha(state, action)
        
        # Current Q-value
        q_current = self.Q[state, action].item()
        
        # Target
        if done:
            q_target = reward
        else:
            q_next_max = torch.max(self.Q[next_state]).item()
            q_target = reward + self.gamma * q_next_max
        
        # TD error
        td_error = q_target - q_current
        self.td_errors.append(abs(td_error))
        
        # Update
        self.Q[state, action] += alpha * td_error
        
        # Update counts
        self.N_sa[state, action] += 1
        self.global_step += 1
        
        return td_error

def verify_robbins_monro_conditions():
    """Verify which schedules satisfy Robbins-Monro conditions."""
    print("="*50)
    print("Robbins-Monro Conditions Verification")
    print("="*50)
    print("Conditions: sum(alpha_t) = ∞ and sum(alpha_t^2) < ∞")
    print()
    
    T = 100000  # Number of steps to check
    schedules = {
        'Constant (α=0.1)': lambda t: 0.1,
        'Linear decay': lambda t: max(0.01, 1.0 - 0.99 * t / T),
        'Exponential decay': lambda t: 0.01 + 0.99 * math.exp(-t / 5000),
        '1/t': lambda t: 1.0 / (1.0 + t),
        '1/sqrt(t)': lambda t: 1.0 / math.sqrt(1.0 + t),
        '1/t^0.75': lambda t: 1.0 / ((1.0 + t) ** 0.75),
        '1/t^2': lambda t: 1.0 / ((1.0 + t) ** 2)
    }
    
    for name, schedule in schedules.items():
        alphas = [schedule(t) for t in range(T)]
        sum_alpha = sum(alphas)
        sum_alpha_sq = sum(a**2 for a in alphas)
        
        # Check conditions (approximately)
        cond1 = "✓" if sum_alpha > T/10 else "✗"  # Grows with T
        cond2 = "✓" if sum_alpha_sq < 100 else "✗"  # Bounded
        
        print(f"{name:20s}: sum(α)={sum_alpha:8.1f} {cond1}, "
              f"sum(α²)={sum_alpha_sq:8.1f} {cond2}")

def compare_alpha_schedules(episodes=1000):
    """Compare different learning rate schedules."""
    print("="*50)
    print("Learning Rate Schedule Comparison")
    print("="*50)
    
    schedules = [
        ("constant", {"value": 0.1}),
        ("linear", {"start": 1.0, "end": 0.01, "steps": episodes * 50}),
        ("exponential", {"start": 1.0, "end": 0.01, "tau": episodes * 10}),
        ("one_over_t", {"start": 1.0, "tau": 100}),
        ("one_over_sqrt_t", {"start": 1.0, "tau": 100}),
        ("state_action_dependent", {"start": 1.0})
    ]
    
    results = {}
    
    for alpha_schedule, alpha_params in schedules:
        print(f"\nTesting {alpha_schedule}...")
        
        env = gym.make("FrozenLake-v1", is_slippery=False)
        agent = ScheduledQLearning(
            env.observation_space.n,
            env.action_space.n,
            alpha_schedule=alpha_schedule,
            alpha_params=alpha_params,
            eps_schedule="exponential",
            eps_params={"start": 1.0, "end": 0.01, "tau": episodes * 10}
        )
        
        returns = []
        successes = []
        
        for episode in range(episodes):
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            returns.append(total_reward)
            successes.append(1 if total_reward > 0 else 0)
        
        env.close()
        
        # Calculate metrics
        final_success = np.mean(successes[-100:]) if len(successes) >= 100 else np.mean(successes)
        avg_td_error = np.mean(agent.td_errors[-1000:]) if len(agent.td_errors) >= 1000 else np.mean(agent.td_errors)
        
        results[alpha_schedule] = {
            'returns': returns,
            'successes': successes,
            'final_success': final_success,
            'avg_td_error': avg_td_error,
            'alpha_history': agent.alpha_history
        }
        
        print(f"  Final success rate: {final_success:.2%}")
        print(f"  Average TD error (last 1000): {avg_td_error:.4f}")
        print(f"  Final alpha: {agent.alpha_history[-1]:.4f}")
    
    return results

def analyze_convergence_speed(results):
    """Analyze how quickly different schedules converge."""
    print("="*50)
    print("Convergence Speed Analysis")
    print("="*50)
    
    target_success = 0.7  # Target 70% success rate
    window = 50  # Moving average window
    
    for schedule, data in results.items():
        successes = data['successes']
        
        # Calculate moving average
        ma_success = []
        for i in range(window, len(successes)):
            ma_success.append(np.mean(successes[i-window:i]))
        
        # Find first episode reaching target
        convergence_episode = None
        for i, success_rate in enumerate(ma_success):
            if success_rate >= target_success:
                convergence_episode = i + window
                break
        
        if convergence_episode:
            print(f"{schedule}: Reached {target_success:.0%} at episode {convergence_episode}")
        else:
            print(f"{schedule}: Never reached {target_success:.0%}")

def test_combined_schedules():
    """Test combinations of alpha and epsilon schedules."""
    print("="*50)
    print("Combined Schedule Testing")
    print("="*50)
    
    combinations = [
        ("constant", "constant", "Fixed α, Fixed ε"),
        ("constant", "exponential", "Fixed α, Decaying ε"),
        ("one_over_t", "constant", "Decaying α, Fixed ε"),
        ("one_over_t", "exponential", "Decaying α, Decaying ε"),
        ("exponential", "exponential", "Exp. decay α, Exp. decay ε"),
        ("one_over_t", "one_over_t", "1/t α, 1/t ε")
    ]
    
    episodes = 500
    
    for alpha_sched, eps_sched, description in combinations:
        env = gym.make("FrozenLake-v1", is_slippery=False)
        
        alpha_params = {"start": 1.0, "tau": 100} if "over" in alpha_sched else {"start": 1.0, "end": 0.01, "tau": 5000}
        eps_params = {"start": 1.0, "tau": 100} if "over" in eps_sched else {"start": 1.0, "end": 0.01, "tau": 5000}
        
        if alpha_sched == "constant":
            alpha_params = {"value": 0.1}
        if eps_sched == "constant":
            eps_params = {"value": 0.1}
        
        agent = ScheduledQLearning(
            env.observation_space.n,
            env.action_space.n,
            alpha_schedule=alpha_sched,
            eps_schedule=eps_sched,
            alpha_params=alpha_params,
            eps_params=eps_params
        )
        
        successes = []
        
        for episode in range(episodes):
            state, _ = env.reset(seed=42 + episode)
            done = False
            total_reward = 0
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            successes.append(1 if total_reward > 0 else 0)
        
        env.close()
        
        final_success = np.mean(successes[-100:])
        print(f"{description:30s}: {final_success:.2%} success")

def visualize_schedule_evolution():
    """Visualize how schedules evolve over time."""
    print("="*50)
    print("Schedule Evolution Visualization")
    print("="*50)
    
    steps = 10000
    t = np.arange(steps)
    
    # Define schedules
    schedules = {
        'Constant': lambda t: 0.1,
        'Linear': lambda t: max(0.01, 1.0 - 0.99 * t / steps),
        'Exponential': lambda t: 0.01 + 0.99 * math.exp(-t / 2000),
        '1/t': lambda t: 1.0 / (1.0 + t / 100),
        '1/√t': lambda t: 1.0 / math.sqrt(1.0 + t / 100)
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot schedules
    for name, schedule in schedules.items():
        values = [schedule(i) for i in t]
        ax1.plot(t[:1000], values[:1000], label=name, linewidth=2)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Learning Rate (α)')
    ax1.set_title('Learning Rate Schedules (First 1000 steps)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative sums
    for name, schedule in schedules.items():
        values = [schedule(i) for i in t]
        cumsum = np.cumsum(values)
        ax2.plot(t, cumsum, label=name, linewidth=2)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Sum of α')
    ax2.set_title('Cumulative Learning Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("="*50)
    print("Experiment 05: Learning Rate Schedules")
    print("="*50)
    
    # 1. Verify Robbins-Monro conditions
    verify_robbins_monro_conditions()
    print()
    
    # 2. Compare alpha schedules
    results = compare_alpha_schedules(episodes=1000)
    print()
    
    # 3. Analyze convergence speed
    analyze_convergence_speed(results)
    print()
    
    # 4. Test combined schedules
    test_combined_schedules()
    
    # 5. Summary
    print("\n" + "="*50)
    print("Schedule Performance Summary")
    print("="*50)
    
    for schedule, data in results.items():
        print(f"{schedule}:")
        print(f"  Final success: {data['final_success']:.2%}")
        print(f"  Avg TD error: {data['avg_td_error']:.4f}")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()