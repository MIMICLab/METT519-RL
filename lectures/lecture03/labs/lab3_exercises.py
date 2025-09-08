"""
RL2025 - Lecture 3 Take-Home Lab Exercises
The World of Reinforcement Learning

Complete these exercises to reinforce today's learning.
Submit your solutions by uploading to the course repository.

Total Points: 100
Due: Before next lecture (Week 4)

Instructions:
1. Complete each function according to its docstring
2. Run the main() function to test your implementations
3. Submit the completed file with your solutions
4. Include a brief report (comments) explaining your observations

Topics Covered:
- Agent-environment interaction
- Returns and discounting
- Exploration vs exploitation
- Epsilon-greedy policies
- Statistical analysis
- Policy evaluation
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
from typing import List, Tuple, Callable, Dict, Any
import json

# =============================================================================
# Exercise 1: Environment Interaction (15 points)
# =============================================================================

def exercise1_environment_basics():
    """
    Exercise 1: Understanding CartPole-v1 Environment
    
    Tasks:
    1. Create CartPole-v1 environment with seed 42
    2. Reset environment and examine observation space
    3. Take 20 random actions and collect trajectory
    4. Analyze the observation values and action effects
    
    Returns:
    - Dictionary with environment information and trajectory analysis
    
    Points: 15
    """
    # TODO: Implement environment interaction
    # Hints:
    # - Use gym.make() to create environment
    # - Use env.reset(seed=42) for reproducibility
    # - Collect observations, actions, rewards in lists
    # - Calculate basic statistics (mean, std, range) of observations
    
    result = {
        'env_id': None,  # Should be "CartPole-v1"
        'obs_space_shape': None,  # Shape of observation space
        'action_space_n': None,  # Number of possible actions
        'trajectory': {
            'observations': [],
            'actions': [],
            'rewards': [],
            'total_reward': 0.0,
            'episode_length': 0
        },
        'observation_stats': {
            'position_mean': 0.0,
            'velocity_mean': 0.0,
            'angle_mean': 0.0,
            'angular_velocity_mean': 0.0,
            'position_range': [0.0, 0.0],  # [min, max]
            'angle_range': [0.0, 0.0]      # [min, max]
        }
    }
    
    # YOUR CODE HERE
    pass
    
    return result

# =============================================================================
# Exercise 2: Returns and Discounting (20 points)
# =============================================================================

def exercise2_returns_calculation(rewards: List[float], gamma: float = 0.99):
    """
    Exercise 2: Implement Returns Calculation
    
    Implement the discounted returns calculation from the RL formula:
    G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k+1}
    
    Tasks:
    1. Implement both forward and backward calculation methods
    2. Verify both methods give identical results
    3. Analyze how different gamma values affect returns
    4. Handle edge cases (empty rewards, gamma=0, gamma=1)
    
    Args:
        rewards: List of rewards [r_1, r_2, ..., r_T]
        gamma: Discount factor
    
    Returns:
        Dictionary with returns and analysis
    
    Points: 20
    """
    # TODO: Implement returns calculation
    # Hints:
    # - Forward method: calculate each G_t directly from definition
    # - Backward method: G_t = r_{t+1} + gamma * G_{t+1}
    # - Test with different gamma values: [0.0, 0.5, 0.9, 0.99, 1.0]
    
    result = {
        'returns_forward': [],      # Returns calculated forward
        'returns_backward': [],     # Returns calculated backward
        'methods_match': False,     # Do both methods give same results?
        'gamma_analysis': {},       # Returns for different gamma values
        'undiscounted_return': 0.0, # Sum of all rewards (gamma=1)
        'immediate_only': 0.0,      # First reward only (gamma=0)
    }
    
    # YOUR CODE HERE
    pass
    
    return result

# =============================================================================
# Exercise 3: Policy Implementation (25 points)
# =============================================================================

def exercise3_policy_design():
    """
    Exercise 3: Design and Compare Policies
    
    Tasks:
    1. Implement three policies: random, heuristic, and epsilon-greedy
    2. Test each policy on CartPole-v1 for 20 episodes
    3. Compare their performance using appropriate metrics
    4. Analyze failure modes and success patterns
    
    Returns:
        Dictionary with policy implementations and performance comparison
    
    Points: 25
    """
    
    def random_policy(obs):
        """Pure random policy"""
        # YOUR CODE HERE
        pass
    
    def heuristic_policy(obs):
        """
        Design a heuristic policy for CartPole
        Hint: Use pole angle and angular velocity to decide action
        """
        # YOUR CODE HERE
        pass
    
    def epsilon_greedy_policy(obs, epsilon=0.1):
        """
        Epsilon-greedy policy combining heuristic with exploration
        """
        # YOUR CODE HERE
        pass
    
    # Test all policies
    env = gym.make("CartPole-v1")
    policies = {
        'random': random_policy,
        'heuristic': heuristic_policy,
        'epsilon_greedy': lambda obs: epsilon_greedy_policy(obs, 0.1)
    }
    
    results = {}
    
    for policy_name, policy in policies.items():
        env.reset(seed=42)
        episode_returns = []
        episode_lengths = []
        failure_modes = {'terminated': 0, 'truncated': 0}
        
        for episode in range(20):
            # TODO: Run episode with policy and collect data
            # - Reset environment with seed 42 + episode
            # - Run episode until termination/truncation
            # - Record total reward, length, and failure mode
            pass
        
        results[policy_name] = {
            'returns': episode_returns,
            'lengths': episode_lengths,
            'mean_return': np.mean(episode_returns) if episode_returns else 0,
            'std_return': np.std(episode_returns) if episode_returns else 0,
            'success_rate': 0,  # TODO: Calculate success rate (return >= 400)
            'failure_modes': failure_modes
        }
    
    env.close()
    
    # TODO: Add analysis comparing the policies
    analysis = {
        'best_policy': '',  # Name of best performing policy
        'performance_ranking': [],  # Policies ranked by mean return
        'insights': ''  # Your observations about policy performance
    }
    
    return {'policy_results': results, 'analysis': analysis}

# =============================================================================
# Exercise 4: Exploration Analysis (20 points)
# =============================================================================

def exercise4_exploration_analysis():
    """
    Exercise 4: Exploration vs Exploitation Analysis
    
    Tasks:
    1. Test epsilon-greedy with different epsilon values
    2. Analyze the exploration-exploitation tradeoff
    3. Find optimal epsilon value for CartPole
    4. Study the effect of epsilon on performance variance
    
    Returns:
        Analysis of exploration strategies
    
    Points: 20
    """
    
    def create_epsilon_greedy(epsilon):
        """Create epsilon-greedy policy with given epsilon"""
        def policy(obs):
            # TODO: Implement epsilon-greedy decision
            # With probability epsilon: random action
            # With probability 1-epsilon: heuristic action
            pass
        return policy
    
    # TODO: Test different epsilon values
    epsilon_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    env = gym.make("CartPole-v1")
    
    results = {}
    
    for epsilon in epsilon_values:
        policy = create_epsilon_greedy(epsilon)
        returns = []
        
        # TODO: Evaluate policy over multiple episodes
        for episode in range(30):
            # Run episode and collect return
            pass
        
        results[epsilon] = {
            'mean_return': np.mean(returns) if returns else 0,
            'std_return': np.std(returns) if returns else 0,
            'returns': returns
        }
    
    env.close()
    
    # TODO: Find optimal epsilon
    optimal_epsilon = 0.1  # TODO: Determine based on results
    
    analysis = {
        'epsilon_results': results,
        'optimal_epsilon': optimal_epsilon,
        'insights': {
            'pure_exploration_performance': 0,  # epsilon=1.0 mean return
            'pure_exploitation_performance': 0,  # epsilon=0.0 mean return
            'best_performance': 0,  # Best mean return achieved
            'exploration_exploitation_tradeoff': ''  # Your analysis
        }
    }
    
    return analysis

# =============================================================================
# Exercise 5: Statistical Analysis (20 points)
# =============================================================================

def exercise5_statistical_significance():
    """
    Exercise 5: Statistical Significance Testing
    
    Tasks:
    1. Compare two policies with proper statistical methods
    2. Calculate confidence intervals for performance metrics
    3. Perform significance tests (t-test)
    4. Calculate effect size (Cohen's d)
    5. Determine if differences are statistically meaningful
    
    Returns:
        Statistical analysis comparing policy performances
    
    Points: 20
    """
    
    # TODO: Collect data for two policies
    # Policy A: Heuristic (epsilon=0.0)
    # Policy B: Epsilon-greedy (epsilon=0.1)
    
    env = gym.make("CartPole-v1")
    
    # Collect data for Policy A
    policy_a_returns = []
    # TODO: Run 50 episodes with heuristic policy
    
    # Collect data for Policy B  
    policy_b_returns = []
    # TODO: Run 50 episodes with epsilon-greedy policy
    
    env.close()
    
    # TODO: Calculate statistics
    def calculate_confidence_interval(data, confidence=0.95):
        """Calculate confidence interval for mean"""
        # TODO: Implement using t-distribution
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        # Use t-distribution for small samples
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin = t_val * std / np.sqrt(n)
        return mean, mean - margin, mean + margin
    
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size"""
        # TODO: Implement Cohen's d calculation
        return 0.0
    
    def perform_t_test(group1, group2):
        """Perform independent t-test"""
        # TODO: Use scipy.stats.ttest_ind
        from scipy import stats
        return stats.ttest_ind(group1, group2)
    
    # TODO: Perform statistical analysis
    stats_a = calculate_confidence_interval(policy_a_returns)
    stats_b = calculate_confidence_interval(policy_b_returns)
    effect_size = cohens_d(policy_a_returns, policy_b_returns)
    t_stat, p_value = perform_t_test(policy_a_returns, policy_b_returns)
    
    result = {
        'policy_a_stats': {
            'mean': stats_a[0],
            'ci_lower': stats_a[1], 
            'ci_upper': stats_a[2],
            'returns': policy_a_returns
        },
        'policy_b_stats': {
            'mean': stats_b[0],
            'ci_lower': stats_b[1],
            'ci_upper': stats_b[2], 
            'returns': policy_b_returns
        },
        'comparison': {
            'effect_size_cohens_d': effect_size,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_005': p_value < 0.05,
            'significant_at_001': p_value < 0.01,
            'practical_significance': abs(effect_size) > 0.5,  # Medium effect
        },
        'interpretation': {
            'mean_difference': stats_b[0] - stats_a[0],
            'effect_interpretation': 'small' if abs(effect_size) < 0.5 else 'medium' if abs(effect_size) < 0.8 else 'large',
            'conclusion': ''  # TODO: Your statistical conclusion
        }
    }
    
    return result

# =============================================================================
# BONUS Exercise: Advanced Policy Design (10 points)
# =============================================================================

def bonus_exercise_advanced_policy():
    """
    BONUS Exercise: Design an Advanced Policy
    
    Tasks:
    1. Design a more sophisticated policy using additional features
    2. Consider position, velocity, angle, and angular velocity
    3. Implement adaptive behavior based on state
    4. Compare against baseline policies
    
    Returns:
        Advanced policy implementation and evaluation
    
    Points: 10 (Bonus)
    """
    
    def advanced_policy(obs):
        """
        Design your own advanced policy for CartPole
        
        Ideas to try:
        - Weighted combination of state variables
        - Non-linear transformations
        - Adaptive thresholds
        - Emergency recovery behaviors
        """
        x, x_dot, theta, theta_dot = obs
        
        # TODO: Implement your advanced policy
        # Be creative but explain your reasoning!
        
        return 0  # Placeholder
    
    # TODO: Evaluate your advanced policy
    env = gym.make("CartPole-v1")
    
    # Compare against baseline
    policies = {
        'random': lambda obs: np.random.randint(0, 2),
        'heuristic': lambda obs: 1 if obs[2] + 0.5 * obs[3] > 0 else 0,
        'advanced': advanced_policy
    }
    
    results = {}
    for name, policy in policies.items():
        returns = []
        for episode in range(30):
            env.reset(seed=42 + episode)
            total_reward = 0
            obs, _ = env.reset()
            
            for _ in range(500):  # Max steps
                action = policy(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            returns.append(total_reward)
        
        results[name] = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'returns': returns
        }
    
    env.close()
    
    return {
        'policy_description': '',  # TODO: Describe your policy design
        'results': results,
        'improvement': results['advanced']['mean_return'] - results['heuristic']['mean_return'],
        'explanation': ''  # TODO: Explain why your policy works
    }

# =============================================================================
# TESTING AND GRADING FUNCTIONS
# =============================================================================

def grade_exercise(exercise_func, exercise_name, max_points):
    """Grade an individual exercise"""
    print(f"\n--- Grading {exercise_name} ---")
    
    try:
        result = exercise_func()
        
        if result is None:
            print(f"❌ {exercise_name}: No implementation (0/{max_points} points)")
            return 0
        
        # Basic checks - students get partial credit for attempting
        points = 0
        
        # Check if result has expected structure
        if isinstance(result, dict) and len(result) > 0:
            points += max_points * 0.2  # 20% for basic structure
            
            # Additional checks based on exercise
            if 'trajectory' in result or 'returns' in result or 'policy_results' in result:
                points += max_points * 0.3  # 30% for having main components
                
            # Check for analysis/insights
            if any(key in result for key in ['analysis', 'insights', 'interpretation']):
                points += max_points * 0.2  # 20% for analysis
                
            # Remaining 30% based on implementation quality (basic check)
            points += max_points * 0.3
        
        points = min(points, max_points)  # Cap at max points
        print(f"✓ {exercise_name}: {points:.0f}/{max_points} points")
        return points
        
    except Exception as e:
        print(f"❌ {exercise_name}: Error - {str(e)} (0/{max_points} points)")
        return 0

def create_visualization_report(results):
    """Create visualization of results"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Lab 3 Exercise Results Summary', fontsize=14)
        
        # This is a placeholder - students should customize based on their results
        
        # Plot 1: Policy comparison (if exercise 3 completed)
        axes[0, 0].text(0.5, 0.5, 'Policy Comparison\n(Complete Exercise 3)', 
                       ha='center', va='center')
        axes[0, 0].set_title('Policy Performance')
        
        # Plot 2: Exploration analysis (if exercise 4 completed)
        axes[0, 1].text(0.5, 0.5, 'Exploration Analysis\n(Complete Exercise 4)', 
                       ha='center', va='center')
        axes[0, 1].set_title('Epsilon vs Performance')
        
        # Plot 3: Returns analysis (if exercise 2 completed)
        axes[1, 0].text(0.5, 0.5, 'Returns Analysis\n(Complete Exercise 2)', 
                       ha='center', va='center')
        axes[1, 0].set_title('Discount Factor Effects')
        
        # Plot 4: Statistical analysis (if exercise 5 completed)
        axes[1, 1].text(0.5, 0.5, 'Statistical Analysis\n(Complete Exercise 5)', 
                       ha='center', va='center')
        axes[1, 1].set_title('Significance Testing')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('lab_results', exist_ok=True)
        plt.savefig('lab_results/lab3_summary.png', dpi=150, bbox_inches='tight')
        print("📊 Results visualization saved to: lab_results/lab3_summary.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")

def main():
    """
    Run all exercises and provide feedback
    
    This function will test your implementations and provide a grade.
    Make sure to complete each exercise before running this!
    """
    print("="*60)
    print("RL2025 - Lecture 3 Lab Exercise Grading")
    print("="*60)
    print(f"Student: [Your Name Here]")
    print(f"Date: {np.datetime64('today')}")
    
    # Define exercises
    exercises = [
        (exercise1_environment_basics, "Exercise 1: Environment Basics", 15),
        (exercise2_returns_calculation, "Exercise 2: Returns Calculation", 20),
        (exercise3_policy_design, "Exercise 3: Policy Design", 25),
        (exercise4_exploration_analysis, "Exercise 4: Exploration Analysis", 20),
        (exercise5_statistical_significance, "Exercise 5: Statistical Analysis", 20),
        (bonus_exercise_advanced_policy, "BONUS: Advanced Policy", 10)
    ]
    
    # Grade each exercise
    total_points = 0
    max_total = 100  # Excluding bonus
    
    results = {}
    
    for exercise_func, name, max_points in exercises:
        points = grade_exercise(exercise_func, name, max_points)
        total_points += points
        
        # Store results for analysis
        try:
            result = exercise_func()
            results[name] = result
        except:
            results[name] = None
    
    # Calculate final grade
    final_grade = (total_points / max_total) * 100
    
    print("\n" + "="*60)
    print("FINAL GRADE REPORT")
    print("="*60)
    print(f"Total Points: {total_points:.0f}/{max_total}")
    print(f"Final Grade: {final_grade:.1f}%")
    
    if final_grade >= 90:
        print("🎉 Excellent work! You've mastered the RL fundamentals.")
    elif final_grade >= 80:
        print("👍 Good job! You have a solid understanding of the concepts.")
    elif final_grade >= 70:
        print("👌 Satisfactory work. Consider reviewing weaker areas.")
    elif final_grade >= 60:
        print("📚 Passing grade. Please review the material and improve implementations.")
    else:
        print("❌ Below passing. Please revisit the lecture materials and seek help.")
    
    # Save results
    os.makedirs('lab_results', exist_ok=True)
    
    grade_report = {
        'student_name': '[Your Name Here]',
        'submission_date': str(np.datetime64('today')),
        'total_points': float(total_points),
        'max_points': max_total,
        'final_grade': float(final_grade),
        'exercise_results': {name: result for name, result in results.items()}
    }
    
    with open('lab_results/lab3_grade_report.json', 'w') as f:
        json.dump(grade_report, f, indent=2, default=str)
    
    print(f"\n📄 Grade report saved to: lab_results/lab3_grade_report.json")
    
    # Create visualizations
    create_visualization_report(results)
    
    print("\n" + "="*60)
    print("SUBMISSION CHECKLIST")
    print("="*60)
    print("□ All functions implemented (no 'pass' statements)")
    print("□ Code runs without errors")
    print("□ Results saved to lab_results/ directory")
    print("□ Brief report/comments added explaining observations")
    print("□ File renamed to: lab3_exercises_[YourName].py")
    
    print("\n📤 Ready to submit? Upload your completed file to the course repository.")
    
    return final_grade

if __name__ == "__main__":
    main()