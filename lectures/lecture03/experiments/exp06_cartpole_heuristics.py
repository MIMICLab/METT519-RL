#!/usr/bin/env python3
"""
RL2025 - Lecture 3: Experiment 06 - CartPole Heuristics and Policy Design

This experiment implements and compares different heuristic policies for CartPole-v1,
demonstrating how domain knowledge can be used to create effective policies before
learning more sophisticated approaches.

Learning objectives:
- Design heuristic policies using domain knowledge
- Understand CartPole-v1 physics and dynamics
- Compare different control strategies
- Analyze policy robustness and failure modes

Prerequisites: Experiments 01-05 completed successfully
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch

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

import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Any, Tuple
import json
from dataclasses import dataclass

@dataclass
class PolicyAnalysis:
    """Container for policy analysis results"""
    name: str
    returns: List[float]
    episode_lengths: List[int]
    mean_return: float
    std_return: float
    success_rate: float  # Episodes with return >= 475 (95% of max)
    failure_modes: Dict[str, int]

def make_env(env_id: str = "CartPole-v1", seed: int = 42) -> gym.Env:
    """Create and initialize environment with proper seeding"""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def analyze_episode(env: gym.Env, policy: Callable, max_steps: int = 500) -> Tuple[float, int, Dict[str, Any]]:
    """
    Run episode and collect detailed analysis
    
    Returns:
        (total_reward, episode_length, episode_info)
    """
    obs, _ = env.reset()
    
    # Track trajectory
    trajectory = {
        'observations': [obs.copy()],
        'actions': [],
        'rewards': []
    }
    
    total_reward = 0.0
    steps = 0
    
    for step in range(max_steps):
        action = policy(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['observations'].append(next_obs.copy())
        
        total_reward += reward
        steps += 1
        obs = next_obs
        
        if terminated or truncated:
            # Analyze failure mode
            failure_info = analyze_failure_mode(trajectory, terminated, truncated)
            break
    else:
        failure_info = {'type': 'success', 'reason': 'max_steps_reached'}
    
    return total_reward, steps, {
        'trajectory': trajectory,
        'failure_info': failure_info,
        'final_obs': obs
    }

def analyze_failure_mode(trajectory: Dict[str, List], terminated: bool, truncated: bool) -> Dict[str, str]:
    """Analyze why an episode ended"""
    if truncated:
        return {'type': 'truncated', 'reason': 'time_limit'}
    
    if not terminated:
        return {'type': 'unknown', 'reason': 'unclear'}
    
    # Check final observation to determine failure mode
    final_obs = trajectory['observations'][-1]
    x, x_dot, theta, theta_dot = final_obs
    
    # CartPole-v1 termination conditions:
    # |x| > 2.4, |theta| > 0.2095 (about 12 degrees)
    
    if abs(x) > 2.4:
        return {'type': 'position_limit', 'reason': f'cart_position_{x:.3f}'}
    elif abs(theta) > 0.2095:
        return {'type': 'angle_limit', 'reason': f'pole_angle_{theta:.3f}'}
    else:
        return {'type': 'unknown_termination', 'reason': 'unclear_termination'}

# =============================================================================
# HEURISTIC POLICIES
# =============================================================================

def random_policy(obs: np.ndarray) -> int:
    """Baseline: pure random policy"""
    return np.random.randint(0, 2)

def simple_angle_policy(obs: np.ndarray) -> int:
    """Push cart in direction of pole lean (simplest heuristic)"""
    x, x_dot, theta, theta_dot = obs
    return 1 if theta > 0 else 0

def pd_controller_policy(obs: np.ndarray, kp: float = 1.0, kd: float = 0.5) -> int:
    """PD controller using angle and angular velocity"""
    x, x_dot, theta, theta_dot = obs
    
    # PD control signal
    control = kp * theta + kd * theta_dot
    
    return 1 if control > 0 else 0

def enhanced_pd_policy(obs: np.ndarray) -> int:
    """Enhanced PD controller considering both position and angle"""
    x, x_dot, theta, theta_dot = obs
    
    # Weighted combination of position and angle control
    position_control = 0.1 * x + 0.05 * x_dot  # Keep cart centered
    angle_control = 1.0 * theta + 0.6 * theta_dot  # Balance pole
    
    total_control = position_control + angle_control
    
    return 1 if total_control > 0 else 0

def adaptive_policy(obs: np.ndarray) -> int:
    """Adaptive policy that changes strategy based on situation"""
    x, x_dot, theta, theta_dot = obs
    
    # Emergency recovery if pole is falling fast
    if abs(theta_dot) > 2.0:
        return 1 if theta_dot > 0 else 0
    
    # Emergency recovery if cart is moving too fast toward edge
    if abs(x) > 1.5:
        if x > 0 and x_dot > 0:  # Moving right toward right edge
            return 0  # Push left
        elif x < 0 and x_dot < 0:  # Moving left toward left edge
            return 1  # Push right
    
    # Normal operation: enhanced PD control
    return enhanced_pd_policy(obs)

def nonlinear_policy(obs: np.ndarray) -> int:
    """Nonlinear policy using trigonometric and energy considerations"""
    x, x_dot, theta, theta_dot = obs
    
    # Energy-based approach: consider kinetic and potential energy
    # Approximation of pole energy
    pole_energy = 0.5 * theta_dot**2 + (1 - np.cos(theta))
    
    # Position penalty (quadratic)
    position_penalty = x**2
    
    # Control signal combining energy and position
    if abs(theta) < 0.05:  # Pole nearly upright
        control = 0.5 * x + 0.3 * x_dot  # Focus on centering
    else:
        control = 2.0 * theta + 0.8 * theta_dot - 0.2 * position_penalty
    
    return 1 if control > 0 else 0

def create_tuned_policy(kp_theta: float = 1.0, kd_theta: float = 0.6, 
                       kp_pos: float = 0.1, kd_pos: float = 0.05) -> Callable:
    """Factory function for tunable PD policy"""
    def tuned_policy(obs: np.ndarray) -> int:
        x, x_dot, theta, theta_dot = obs
        
        position_control = kp_pos * x + kd_pos * x_dot
        angle_control = kp_theta * theta + kd_theta * theta_dot
        
        control = position_control + angle_control
        return 1 if control > 0 else 0
    
    return tuned_policy

# =============================================================================
# EVALUATION AND ANALYSIS
# =============================================================================

def evaluate_policy_detailed(policy: Callable, 
                            policy_name: str,
                            num_episodes: int = 50, 
                            seed: int = 42) -> PolicyAnalysis:
    """Comprehensive policy evaluation"""
    
    setup_seed(seed)
    env = make_env("CartPole-v1", seed)
    
    returns = []
    episode_lengths = []
    failure_modes = {}
    
    print(f"Evaluating {policy_name} over {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        env.reset(seed=seed + ep)
        total_reward, length, episode_info = analyze_episode(env, policy)
        
        returns.append(total_reward)
        episode_lengths.append(length)
        
        # Track failure modes
        failure_type = episode_info['failure_info']['type']
        failure_modes[failure_type] = failure_modes.get(failure_type, 0) + 1
        
        if ep < 5:  # Show details for first few episodes
            print(f"  Episode {ep+1}: reward={total_reward}, length={length}, "
                  f"failure={failure_type}")
    
    env.close()
    
    # Calculate statistics
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    success_rate = sum(1 for r in returns if r >= 475) / len(returns)  # 95% of max
    
    return PolicyAnalysis(
        name=policy_name,
        returns=returns,
        episode_lengths=episode_lengths,
        mean_return=mean_return,
        std_return=std_return,
        success_rate=success_rate,
        failure_modes=failure_modes
    )

def compare_all_policies(num_episodes: int = 30) -> Dict[str, PolicyAnalysis]:
    """Compare all heuristic policies"""
    
    policies = {
        "Random": random_policy,
        "Simple Angle": simple_angle_policy,
        "PD Controller": pd_controller_policy,
        "Enhanced PD": enhanced_pd_policy,
        "Adaptive": adaptive_policy,
        "Nonlinear": nonlinear_policy,
        "Tuned PD": create_tuned_policy(kp_theta=1.2, kd_theta=0.8, 
                                       kp_pos=0.15, kd_pos=0.1)
    }
    
    results = {}
    
    for name, policy in policies.items():
        results[name] = evaluate_policy_detailed(policy, name, num_episodes)
    
    return results

def visualize_policy_comparison(results: Dict[str, PolicyAnalysis], save_path: str = None):
    """Create comprehensive visualization of policy performance"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CartPole Heuristic Policy Comparison', fontsize=16)
    
    policy_names = list(results.keys())
    
    # 1. Mean returns with error bars
    means = [results[name].mean_return for name in policy_names]
    stds = [results[name].std_return for name in policy_names]
    
    bars = axes[0, 0].bar(range(len(policy_names)), means, yerr=stds, 
                         capsize=5, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Mean Episode Return')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_xticks(range(len(policy_names)))
    axes[0, 0].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{mean:.0f}', ha='center', va='bottom')
    
    # 2. Success rates
    success_rates = [results[name].success_rate * 100 for name in policy_names]
    bars = axes[0, 1].bar(range(len(policy_names)), success_rates, 
                         alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Success Rate (Return ≥ 475)')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_xticks(range(len(policy_names)))
    axes[0, 1].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)
    
    # 3. Return distributions (violin plot)
    return_data = [results[name].returns for name in policy_names]
    parts = axes[0, 2].violinplot(return_data, positions=range(len(policy_names)), 
                                  showmeans=True, showmedians=True)
    axes[0, 2].set_title('Return Distributions')
    axes[0, 2].set_ylabel('Episode Return')
    axes[0, 2].set_xticks(range(len(policy_names)))
    axes[0, 2].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Episode length comparison
    mean_lengths = [np.mean(results[name].episode_lengths) for name in policy_names]
    axes[1, 0].bar(range(len(policy_names)), mean_lengths, 
                   alpha=0.7, color='orange')
    axes[1, 0].set_title('Mean Episode Length')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_xticks(range(len(policy_names)))
    axes[1, 0].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Failure mode analysis (for best few policies)
    top_policies = sorted(policy_names, key=lambda x: results[x].mean_return, reverse=True)[:4]
    
    failure_types = set()
    for name in top_policies:
        failure_types.update(results[name].failure_modes.keys())
    
    failure_types = sorted(list(failure_types))
    x_pos = np.arange(len(failure_types))
    
    width = 0.2
    for i, policy in enumerate(top_policies):
        counts = [results[policy].failure_modes.get(ft, 0) for ft in failure_types]
        axes[1, 1].bar(x_pos + i*width, counts, width, 
                      label=policy, alpha=0.7)
    
    axes[1, 1].set_title('Failure Modes (Top 4 Policies)')
    axes[1, 1].set_xlabel('Failure Type')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks(x_pos + width * 1.5)
    axes[1, 1].set_xticklabels(failure_types, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance scatter: mean vs std
    axes[1, 2].scatter(means, stds, s=100, alpha=0.7)
    for i, name in enumerate(policy_names):
        axes[1, 2].annotate(name, (means[i], stds[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    axes[1, 2].set_title('Return Mean vs Variability')
    axes[1, 2].set_xlabel('Mean Return')
    axes[1, 2].set_ylabel('Standard Deviation')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy comparison plot saved to: {save_path}")
    elif plt.isinteractive():
        plt.show()
    else:
        print("Warning: Matplotlib running in non-interactive mode; skipping figure display.")

    plt.close(fig)

def hyperparameter_tuning_demo():
    """Demonstrate hyperparameter tuning for PD controller"""
    print("\n--- Hyperparameter Tuning Demo ---")
    
    # Grid search over PD parameters
    kp_values = [0.5, 1.0, 1.5, 2.0]
    kd_values = [0.3, 0.6, 0.9, 1.2]
    
    best_params = None
    best_performance = -float('inf')
    
    results = {}
    
    for kp in kp_values:
        for kd in kd_values:
            policy = create_tuned_policy(kp_theta=kp, kd_theta=kd)
            
            # Quick evaluation with fewer episodes
            setup_seed(42)
            env = make_env("CartPole-v1", 42)
            
            returns = []
            for ep in range(10):
                env.reset(seed=42 + ep)
                total_reward, _, _ = analyze_episode(env, policy)
                returns.append(total_reward)
            
            env.close()
            
            mean_return = np.mean(returns)
            results[(kp, kd)] = mean_return
            
            print(f"kp={kp:.1f}, kd={kd:.1f}: mean_return={mean_return:.1f}")
            
            if mean_return > best_performance:
                best_performance = mean_return
                best_params = (kp, kd)
    
    print(f"\nBest parameters: kp={best_params[0]}, kd={best_params[1]}")
    print(f"Best performance: {best_performance:.1f}")
    
    return best_params, results

def main():
    """Run CartPole heuristics experiment"""
    print("="*60)
    print("Experiment 06: CartPole Heuristics and Policy Design")
    print("="*60)
    
    # Compare all policies
    print("Comparing all heuristic policies...")
    results = compare_all_policies(num_episodes=50)
    
    # Print detailed results
    print("\n" + "="*60)
    print("POLICY PERFORMANCE SUMMARY")
    print("="*60)
    
    # Sort by mean return
    sorted_policies = sorted(results.items(), key=lambda x: x[1].mean_return, reverse=True)
    
    print(f"{'Policy':<15} {'Mean Return':<12} {'Success Rate':<12} {'Std Dev':<10}")
    print("-" * 60)
    
    for name, analysis in sorted_policies:
        print(f"{name:<15} {analysis.mean_return:<12.1f} {analysis.success_rate*100:<12.1f}% {analysis.std_return:<10.1f}")
    
    # Detailed analysis of top 3 policies
    print(f"\n--- Top 3 Policy Detailed Analysis ---")
    for i, (name, analysis) in enumerate(sorted_policies[:3]):
        print(f"\n{i+1}. {name}:")
        print(f"   Mean return: {analysis.mean_return:.1f} ± {analysis.std_return:.1f}")
        print(f"   Success rate: {analysis.success_rate*100:.1f}%")
        print(f"   Return range: [{min(analysis.returns):.0f}, {max(analysis.returns):.0f}]")
        print(f"   Failure modes: {analysis.failure_modes}")
    
    # Hyperparameter tuning demonstration
    best_params, tuning_results = hyperparameter_tuning_demo()
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    # Convert results to JSON-serializable format
    json_results = {}
    for name, analysis in results.items():
        json_results[name] = {
            'mean_return': analysis.mean_return,
            'std_return': analysis.std_return,
            'success_rate': analysis.success_rate,
            'returns': analysis.returns,
            'episode_lengths': analysis.episode_lengths,
            'failure_modes': analysis.failure_modes
        }
    
    with open("results/heuristic_policies.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    # Save tuning results
    tuning_json = {str(k): v for k, v in tuning_results.items()}
    with open("results/hyperparameter_tuning.json", "w") as f:
        json.dump(tuning_json, f, indent=2)
    
    print(f"\nResults saved to:")
    print("  - results/heuristic_policies.json")
    print("  - results/hyperparameter_tuning.json")
    
    # Create visualizations
    try:
        visualize_policy_comparison(results, "results/policy_comparison.png")
    except ImportError:
        print("Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("1. Simple heuristics can work surprisingly well")
    print("2. PD controllers are effective for CartPole")
    print("3. Considering both angle and position improves performance")
    print("4. Adaptive strategies can handle edge cases better")
    print("5. Hyperparameter tuning can significantly improve performance")
    print("6. Understanding failure modes helps improve policies")
    
    print("\nExperiment 06 completed successfully!")
    return True

if __name__ == "__main__":
    main()
