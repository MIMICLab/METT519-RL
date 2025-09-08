#!/usr/bin/env python3
"""
RL2025 - Lecture 3: Experiment 08 - Statistical Analysis and Significance Testing

This experiment demonstrates proper statistical analysis for RL experiments,
including confidence intervals, hypothesis testing, and effect size analysis
for comparing different policies.

Learning objectives:
- Perform proper statistical analysis of RL experiments
- Calculate confidence intervals and effect sizes
- Test for statistical significance between policies
- Understand the importance of multiple runs and error bars

Prerequisites: Experiments 01-07 completed successfully
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
from typing import Dict, List, Any, Callable, Tuple
import json
from dataclasses import dataclass
from scipy import stats
import pandas as pd

@dataclass
class StatisticalResult:
    """Container for statistical analysis results"""
    mean: float
    std: float
    sem: float  # Standard error of mean
    ci_lower: float  # 95% confidence interval lower bound
    ci_upper: float  # 95% confidence interval upper bound
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    min_val: float
    max_val: float
    n_samples: int

def make_env(env_id: str = "CartPole-v1", seed: int = 42) -> gym.Env:
    """Create and initialize environment with proper seeding"""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def rollout_episode(env: gym.Env, policy: Callable, max_steps: int = 500) -> float:
    """Run single episode and return total reward"""
    obs, _ = env.reset()
    total_reward = 0.0
    
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return total_reward

def collect_policy_data(policy: Callable, 
                       policy_name: str,
                       num_episodes: int = 100, 
                       num_runs: int = 5,
                       seed: int = 42) -> List[List[float]]:
    """
    Collect multiple independent runs of policy evaluation
    
    Args:
        policy: Policy function
        policy_name: Name for logging
        num_episodes: Episodes per run
        num_runs: Number of independent runs
        seed: Base random seed
    
    Returns:
        List of lists, where each inner list contains returns from one run
    """
    print(f"Collecting data for {policy_name}: {num_runs} runs x {num_episodes} episodes")
    
    all_runs_data = []
    
    for run in range(num_runs):
        # Use different seed for each run to ensure independence
        run_seed = seed + run * 1000
        setup_seed(run_seed)
        
        env = make_env("CartPole-v1", run_seed)
        run_returns = []
        
        for episode in range(num_episodes):
            # Different seed for each episode within the run
            env.reset(seed=run_seed + episode)
            total_reward = rollout_episode(env, policy)
            run_returns.append(total_reward)
        
        env.close()
        all_runs_data.append(run_returns)
        
        print(f"  Run {run+1}: mean={np.mean(run_returns):.1f}, std={np.std(run_returns):.1f}")
    
    return all_runs_data

def calculate_statistics(data: List[float], confidence_level: float = 0.95) -> StatisticalResult:
    """Calculate comprehensive statistics for a dataset"""
    
    data = np.array(data)
    n = len(data)
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    sem = std / np.sqrt(n)  # Standard error of mean
    
    # Confidence interval using t-distribution
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    ci_lower = mean - t_critical * sem
    ci_upper = mean + t_critical * sem
    
    # Percentiles
    median = np.median(data)
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    
    return StatisticalResult(
        mean=float(mean),
        std=float(std),
        sem=float(sem),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        median=float(median),
        q25=float(q25),
        q75=float(q75),
        min_val=float(np.min(data)),
        max_val=float(np.max(data)),
        n_samples=int(n)
    )

def compare_policies_statistical(policy_a_data: List[float], 
                               policy_b_data: List[float],
                               policy_a_name: str = "Policy A",
                               policy_b_name: str = "Policy B") -> Dict[str, Any]:
    """
    Perform statistical comparison between two policies
    
    Returns:
        Dictionary with comparison results
    """
    data_a = np.array(policy_a_data)
    data_b = np.array(policy_b_data)
    
    # Basic statistics
    stats_a = calculate_statistics(policy_a_data)
    stats_b = calculate_statistics(policy_b_data)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(data_a) - 1) * stats_a.std**2 + 
                         (len(data_b) - 1) * stats_b.std**2) / 
                        (len(data_a) + len(data_b) - 2))
    cohens_d = (stats_a.mean - stats_b.mean) / pooled_std
    
    # Statistical tests
    # 1. Welch's t-test (unequal variances)
    t_stat, p_value_welch = stats.ttest_ind(data_a, data_b, equal_var=False)
    
    # 2. Student's t-test (equal variances)
    t_stat_student, p_value_student = stats.ttest_ind(data_a, data_b, equal_var=True)
    
    # 3. Mann-Whitney U test (non-parametric)
    u_stat, p_value_mannwhitney = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
    
    # 4. Levene's test for equal variances
    levene_stat, p_value_levene = stats.levene(data_a, data_b)
    
    # 5. Normality tests
    _, p_norm_a = stats.shapiro(data_a) if len(data_a) <= 5000 else stats.kstest(data_a, 'norm')
    _, p_norm_b = stats.shapiro(data_b) if len(data_b) <= 5000 else stats.kstest(data_b, 'norm')
    
    # Effect size interpretation
    effect_size_interpretation = "negligible"
    if abs(cohens_d) >= 0.8:
        effect_size_interpretation = "large"
    elif abs(cohens_d) >= 0.5:
        effect_size_interpretation = "medium"
    elif abs(cohens_d) >= 0.2:
        effect_size_interpretation = "small"
    
    return {
        'policy_a_name': policy_a_name,
        'policy_b_name': policy_b_name,
        'stats_a': stats_a,
        'stats_b': stats_b,
        'mean_difference': stats_a.mean - stats_b.mean,
        'cohens_d': cohens_d,
        'effect_size_interpretation': effect_size_interpretation,
        'welch_t_test': {
            't_statistic': t_stat,
            'p_value': p_value_welch,
            'significant_05': p_value_welch < 0.05,
            'significant_01': p_value_welch < 0.01
        },
        'student_t_test': {
            't_statistic': t_stat_student,
            'p_value': p_value_student,
            'significant_05': p_value_student < 0.05,
            'significant_01': p_value_student < 0.01
        },
        'mann_whitney_test': {
            'u_statistic': u_stat,
            'p_value': p_value_mannwhitney,
            'significant_05': p_value_mannwhitney < 0.05,
            'significant_01': p_value_mannwhitney < 0.01
        },
        'levene_test': {
            'statistic': levene_stat,
            'p_value': p_value_levene,
            'equal_variances': p_value_levene > 0.05
        },
        'normality_tests': {
            'policy_a_normal': p_norm_a > 0.05,
            'policy_b_normal': p_norm_b > 0.05,
            'p_value_a': p_norm_a,
            'p_value_b': p_norm_b
        }
    }

def bootstrap_confidence_interval(data: List[float], 
                                statistic: Callable = np.mean,
                                confidence_level: float = 0.95,
                                n_bootstrap: int = 10000) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval
    
    Returns:
        (statistic_value, ci_lower, ci_upper)
    """
    data = np.array(data)
    n = len(data)
    
    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate percentiles for confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha/2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
    
    original_stat = statistic(data)
    
    return original_stat, ci_lower, ci_upper

def power_analysis(effect_size: float, alpha: float = 0.05, n_per_group: int = 50) -> float:
    """
    Calculate statistical power for two-sample t-test
    
    Args:
        effect_size: Cohen's d effect size
        alpha: Type I error rate
        n_per_group: Sample size per group
        
    Returns:
        Statistical power (0-1)
    """
    from scipy.stats import norm
    
    # Critical value for two-tailed test
    t_critical = stats.t.ppf(1 - alpha/2, df=2*n_per_group-2)
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n_per_group / 2)
    
    # Power calculation (approximation)
    power = 1 - stats.t.cdf(t_critical, df=2*n_per_group-2, loc=ncp) + \
            stats.t.cdf(-t_critical, df=2*n_per_group-2, loc=ncp)
    
    return power

def create_statistical_plots(results: Dict[str, Any], save_path: str = None):
    """Create comprehensive statistical visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistical Analysis Results', fontsize=16)
    
    # Extract data
    policy_names = list(results.keys())
    
    # 1. Mean comparisons with confidence intervals
    means = [results[name]['overall_stats'].mean for name in policy_names]
    ci_lowers = [results[name]['overall_stats'].ci_lower for name in policy_names]
    ci_uppers = [results[name]['overall_stats'].ci_upper for name in policy_names]
    
    x_pos = np.arange(len(policy_names))
    axes[0, 0].errorbar(x_pos, means, 
                       yerr=[np.array(means) - np.array(ci_lowers),
                            np.array(ci_uppers) - np.array(means)],
                       fmt='o', capsize=5, capthick=2, markersize=8)
    axes[0, 0].set_title('Mean Returns with 95% Confidence Intervals')
    axes[0, 0].set_ylabel('Mean Return')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plots
    all_data = [np.concatenate(results[name]['run_data']) for name in policy_names]
    box_parts = axes[0, 1].boxplot(all_data, labels=policy_names, patch_artist=True)
    axes[0, 1].set_title('Return Distributions')
    axes[0, 1].set_ylabel('Episode Return')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(box_parts['boxes'])))
    for patch, color in zip(box_parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 3. Standard error comparison
    sems = [results[name]['overall_stats'].sem for name in policy_names]
    axes[0, 2].bar(x_pos, sems, alpha=0.7, color='orange')
    axes[0, 2].set_title('Standard Error of Mean')
    axes[0, 2].set_ylabel('Standard Error')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Run-to-run variability
    run_means = []
    run_stds = []
    for name in policy_names:
        run_data = results[name]['run_data']
        means_per_run = [np.mean(run) for run in run_data]
        run_means.append(np.mean(means_per_run))
        run_stds.append(np.std(means_per_run))
    
    axes[1, 0].errorbar(x_pos, run_means, yerr=run_stds, 
                       fmt='s', capsize=5, capthick=2, markersize=8, color='red')
    axes[1, 0].set_title('Run-to-Run Variability')
    axes[1, 0].set_ylabel('Mean Return Across Runs')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Sample size effect (if available)
    if len(policy_names) >= 2:
        # Show how confidence intervals change with sample size
        sample_data = all_data[0]  # Use first policy's data
        sample_sizes = [10, 25, 50, 100, 200, 500]
        ci_widths = []
        
        for n in sample_sizes:
            if n <= len(sample_data):
                subset = np.random.choice(sample_data, size=n, replace=False)
                stats_subset = calculate_statistics(subset)
                ci_width = stats_subset.ci_upper - stats_subset.ci_lower
                ci_widths.append(ci_width)
            else:
                ci_widths.append(np.nan)
        
        valid_indices = ~np.isnan(ci_widths)
        axes[1, 1].plot(np.array(sample_sizes)[valid_indices], 
                       np.array(ci_widths)[valid_indices], 
                       'o-', linewidth=2, markersize=6)
        axes[1, 1].set_title('Confidence Interval Width vs Sample Size')
        axes[1, 1].set_xlabel('Sample Size')
        axes[1, 1].set_ylabel('CI Width')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log')
    
    # 6. Effect sizes (if we have comparisons)
    if hasattr(results, 'comparisons') and results.get('comparisons'):
        comparison_names = list(results['comparisons'].keys())
        effect_sizes = [results['comparisons'][name]['cohens_d'] 
                       for name in comparison_names]
        
        bars = axes[1, 2].barh(range(len(comparison_names)), effect_sizes, alpha=0.7)
        axes[1, 2].set_title('Effect Sizes (Cohen\'s d)')
        axes[1, 2].set_xlabel('Effect Size')
        axes[1, 2].set_yticks(range(len(comparison_names)))
        axes[1, 2].set_yticklabels(comparison_names)
        axes[1, 2].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 2].axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='Small')
        axes[1, 2].axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
        axes[1, 2].axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Large')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Color bars based on effect size magnitude
        for bar, effect_size in zip(bars, effect_sizes):
            if abs(effect_size) >= 0.8:
                bar.set_color('green')
            elif abs(effect_size) >= 0.5:
                bar.set_color('orange')
            elif abs(effect_size) >= 0.2:
                bar.set_color('yellow')
            else:
                bar.set_color('lightgray')
    else:
        axes[1, 2].text(0.5, 0.5, 'No comparisons available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Effect Sizes')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistical plots saved to: {save_path}")
    else:
        plt.show()

# Define policies for testing
def heuristic_action(obs: np.ndarray) -> int:
    """Simple CartPole heuristic"""
    x, x_dot, theta, theta_dot = obs
    control_signal = theta + 0.5 * theta_dot
    return 1 if control_signal > 0.0 else 0

def random_policy(obs: np.ndarray) -> int:
    """Random policy"""
    return np.random.randint(0, 2)

def epsilon_greedy_policy(epsilon: float):
    """Create epsilon-greedy policy"""
    def policy(obs: np.ndarray) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        return heuristic_action(obs)
    return policy

def main():
    """Run statistical analysis experiment"""
    print("="*60)
    print("Experiment 08: Statistical Analysis and Significance Testing")
    print("="*60)
    
    # Define policies to compare
    policies = {
        "Random": random_policy,
        "Heuristic": heuristic_action,
        "ε-greedy (0.1)": epsilon_greedy_policy(0.1),
        "ε-greedy (0.2)": epsilon_greedy_policy(0.2)
    }
    
    # Collect data for each policy
    print("Collecting policy performance data...")
    results = {}
    
    for name, policy in policies.items():
        print(f"\n--- Collecting data for {name} ---")
        run_data = collect_policy_data(
            policy=policy, 
            policy_name=name,
            num_episodes=50,  # Episodes per run
            num_runs=5,       # Independent runs
            seed=42
        )
        
        # Calculate overall statistics
        all_data = np.concatenate(run_data)
        overall_stats = calculate_statistics(all_data)
        
        # Bootstrap confidence interval for median
        median_val, median_ci_lower, median_ci_upper = bootstrap_confidence_interval(
            all_data, statistic=np.median, n_bootstrap=5000
        )
        
        results[name] = {
            'run_data': run_data,
            'overall_stats': overall_stats,
            'bootstrap_median_ci': {
                'median': median_val,
                'ci_lower': median_ci_lower,
                'ci_upper': median_ci_upper
            }
        }
        
        print(f"Overall: {overall_stats.mean:.1f} ± {overall_stats.std:.1f} "
              f"(95% CI: [{overall_stats.ci_lower:.1f}, {overall_stats.ci_upper:.1f}])")
    
    # Perform pairwise comparisons
    print("\n--- Pairwise Statistical Comparisons ---")
    policy_names = list(policies.keys())
    comparisons = {}
    
    for i in range(len(policy_names)):
        for j in range(i+1, len(policy_names)):
            name_a, name_b = policy_names[i], policy_names[j]
            data_a = np.concatenate(results[name_a]['run_data'])
            data_b = np.concatenate(results[name_b]['run_data'])
            
            comparison = compare_policies_statistical(
                data_a, data_b, name_a, name_b
            )
            
            comp_key = f"{name_a} vs {name_b}"
            comparisons[comp_key] = comparison
            
            print(f"\n{comp_key}:")
            print(f"  Mean difference: {comparison['mean_difference']:.2f}")
            print(f"  Effect size (Cohen's d): {comparison['cohens_d']:.3f} "
                  f"({comparison['effect_size_interpretation']})")
            print(f"  Welch's t-test p-value: {comparison['welch_t_test']['p_value']:.6f} "
                  f"({'significant' if comparison['welch_t_test']['significant_05'] else 'not significant'} at α=0.05)")
            print(f"  Mann-Whitney U p-value: {comparison['mann_whitney_test']['p_value']:.6f}")
    
    # Add comparisons to results for plotting
    results['comparisons'] = comparisons
    
    # Power analysis example
    print("\n--- Power Analysis Example ---")
    example_effect_sizes = [0.2, 0.5, 0.8]
    sample_sizes = [20, 50, 100, 200]
    
    print("Statistical Power for Different Effect Sizes and Sample Sizes:")
    print("Effect Size | n=20  | n=50  | n=100 | n=200")
    print("-" * 45)
    
    for effect_size in example_effect_sizes:
        power_row = f"{effect_size:10.1f} |"
        for n in sample_sizes:
            power = power_analysis(effect_size, alpha=0.05, n_per_group=n)
            power_row += f" {power:5.3f} |"
        print(power_row)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    # Convert results to JSON-serializable format
    json_results = {}
    for name, result in results.items():
        if name != 'comparisons':
            json_results[name] = {
                'overall_stats': {
                    'mean': result['overall_stats'].mean,
                    'std': result['overall_stats'].std,
                    'sem': result['overall_stats'].sem,
                    'ci_lower': result['overall_stats'].ci_lower,
                    'ci_upper': result['overall_stats'].ci_upper,
                    'median': result['overall_stats'].median,
                    'n_samples': result['overall_stats'].n_samples
                },
                'run_means': [float(np.mean(run)) for run in result['run_data']],
                'bootstrap_median_ci': result['bootstrap_median_ci']
            }
    
    # Add comparisons
    json_comparisons = {}
    for comp_name, comp_result in comparisons.items():
        json_comparisons[comp_name] = {
            'mean_difference': comp_result['mean_difference'],
            'cohens_d': comp_result['cohens_d'],
            'effect_size_interpretation': comp_result['effect_size_interpretation'],
            'welch_p_value': comp_result['welch_t_test']['p_value'],
            'welch_significant_05': bool(comp_result['welch_t_test']['significant_05']),
            'mann_whitney_p_value': comp_result['mann_whitney_test']['p_value'],
            'mann_whitney_significant_05': bool(comp_result['mann_whitney_test']['significant_05'])
        }
    
    json_results['comparisons'] = json_comparisons
    
    with open("results/statistical_analysis.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nStatistical analysis results saved to: results/statistical_analysis.json")
    
    # Create visualizations
    try:
        create_statistical_plots(results, "results/statistical_analysis.png")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\n" + "="*60)
    print("KEY STATISTICAL INSIGHTS:")
    print("="*60)
    print("1. Always report confidence intervals, not just means")
    print("2. Use multiple independent runs for robust results")
    print("3. Check for statistical significance AND effect size")
    print("4. Consider both parametric and non-parametric tests")
    print("5. Bootstrap methods provide robust confidence intervals")
    print("6. Power analysis helps determine required sample sizes")
    print("7. Effect sizes are more interpretable than p-values alone")
    
    print("\nExperiment 08 completed successfully!")
    return True

if __name__ == "__main__":
    try:
        from scipy import stats
    except ImportError:
        print("This experiment requires scipy. Please install with: pip install scipy")
        exit(1)
    
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, some features may be limited")
    
    main()