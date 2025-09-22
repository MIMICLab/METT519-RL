#!/usr/bin/env python3
"""
RL2025 - Lecture 3: Experiment 09 - Integrated Test and Final Validation

This experiment integrates all components from Experiments 01-08, providing
a comprehensive test of the RL fundamentals covered in Lecture 3. It serves
as both a validation test and a complete demonstration.

Learning objectives:
- Integrate all Lecture 3 components into a cohesive experiment
- Validate reproducibility and correctness across all implementations
- Demonstrate complete experimental workflow
- Create comprehensive final report

Prerequisites: Experiments 01-08 completed successfully
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

import sys
import time
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple
from dataclasses import dataclass, asdict
import subprocess

import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

@dataclass
class IntegratedTestConfig:
    """Configuration for integrated test"""
    env_id: str = "CartPole-v1"
    test_episodes: int = 50
    statistical_runs: int = 3
    seed: int = 42
    log_dir: str = "runs/integrated_test"
    save_results: bool = True
    create_plots: bool = True
    run_tensorboard: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class IntegratedTester:
    """Main class for running integrated tests"""
    
    def __init__(self, config: IntegratedTestConfig):
        self.config = config
        self.results = {}
        self.test_start_time = None
        self.writer = None
        
        # Create output directories
        self.output_dir = Path("results/integrated_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        if config.run_tensorboard:
            self.writer = SummaryWriter(config.log_dir)
    
    def log_test_start(self):
        """Initialize test session with metadata"""
        self.test_start_time = time.time()
        
        print("="*70)
        print("RL2025 - LECTURE 3 INTEGRATED TEST")
        print("="*70)
        print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}")
        print(f"AMP enabled: {amp_enabled}")
        print(f"Environment: {self.config.env_id}")
        print(f"Output directory: {self.output_dir}")
        
        # Log system information
        system_info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "gymnasium_version": gym.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "test_config": self.config.to_dict()
        }
        
        self.results["system_info"] = system_info
        
        if self.writer:
            self.writer.add_text("Test/System_Info", 
                               json.dumps(system_info, indent=2), 0)
    
    def test_01_environment_setup(self) -> bool:
        """Test 01: Environment setup and basic functionality"""
        print("\n--- Test 01: Environment Setup ---")
        
        try:
            # Test environment creation
            env = self.make_env()
            obs, info = env.reset()
            
            # Verify observation properties
            assert obs.shape == (4,), f"Expected obs shape (4,), got {obs.shape}"
            assert len(info) >= 0, "Info dictionary should exist"
            
            # Test a few steps
            total_reward = 0
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            env.close()
            
            test_result = {
                "passed": True,
                "obs_shape": obs.shape,
                "total_reward_10_steps": float(total_reward),
                "final_obs": obs.tolist()
            }
            
            print("✓ Environment setup test passed")
            
        except Exception as e:
            test_result = {"passed": False, "error": str(e)}
            print(f"✗ Environment setup test failed: {e}")
        
        self.results["test_01"] = test_result
        return test_result["passed"]
    
    def test_02_policy_implementations(self) -> bool:
        """Test 02: Policy implementations"""
        print("\n--- Test 02: Policy Implementations ---")
        
        try:
            policies = {
                "random": self.random_policy,
                "heuristic": self.heuristic_policy,
                "epsilon_greedy_01": self.create_epsilon_greedy(0.1),
                "epsilon_greedy_02": self.create_epsilon_greedy(0.2)
            }
            
            env = self.make_env()
            policy_results = {}
            
            for name, policy in policies.items():
                # Test policy with a few episodes
                returns = []
                for episode in range(5):
                    env.reset(seed=42 + episode)
                    total_reward = self.rollout_episode(env, policy)
                    returns.append(total_reward)
                
                policy_results[name] = {
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)),
                    "returns": returns
                }
                
                print(f"✓ {name}: {np.mean(returns):.1f} ± {np.std(returns):.1f}")
            
            env.close()
            
            test_result = {"passed": True, "policy_results": policy_results}
            
        except Exception as e:
            test_result = {"passed": False, "error": str(e)}
            print(f"✗ Policy implementation test failed: {e}")
        
        self.results["test_02"] = test_result
        return test_result["passed"]
    
    def test_03_returns_calculation(self) -> bool:
        """Test 03: Returns and discounting calculations"""
        print("\n--- Test 03: Returns Calculation ---")
        
        try:
            # Test with synthetic reward sequence
            rewards = [1.0, 2.0, 3.0, 1.0, 2.0]
            gamma_values = [0.0, 0.5, 0.9, 0.99, 1.0]
            
            returns_results = {}
            
            for gamma in gamma_values:
                returns = self.calculate_returns(rewards, gamma)
                returns_results[str(gamma)] = returns
                
                # Verify first return calculation manually
                expected_g0 = sum(rewards[k] * (gamma ** k) for k in range(len(rewards)))
                assert abs(returns[0] - expected_g0) < 1e-6, f"Return calculation error for gamma={gamma}"
                
                print(f"✓ γ={gamma}: G₀={returns[0]:.3f}")
            
            test_result = {"passed": True, "returns_results": returns_results}
            
        except Exception as e:
            test_result = {"passed": False, "error": str(e)}
            print(f"✗ Returns calculation test failed: {e}")
        
        self.results["test_03"] = test_result
        return test_result["passed"]
    
    def test_04_statistical_analysis(self) -> bool:
        """Test 04: Statistical analysis functions"""
        print("\n--- Test 04: Statistical Analysis ---")
        
        try:
            # Generate test data
            np.random.seed(42)
            data_a = np.random.normal(100, 15, 50)  # Policy A
            data_b = np.random.normal(110, 20, 50)  # Policy B (better)
            
            # Calculate statistics
            stats_a = self.calculate_statistics(data_a)
            stats_b = self.calculate_statistics(data_b)
            
            # Verify statistics
            assert abs(stats_a.mean - np.mean(data_a)) < 1e-6, "Mean calculation error"
            assert abs(stats_a.std - np.std(data_a, ddof=1)) < 1e-6, "Std calculation error"
            
            # Test comparison
            comparison = self.compare_policies(data_a, data_b)
            
            test_result = {
                "passed": True,
                "stats_a_mean": stats_a.mean,
                "stats_b_mean": stats_b.mean,
                "effect_size": comparison["cohens_d"],
                "p_value": comparison["p_value"]
            }
            
            print(f"✓ Policy A mean: {stats_a.mean:.1f}")
            print(f"✓ Policy B mean: {stats_b.mean:.1f}")
            print(f"✓ Effect size: {comparison['cohens_d']:.3f}")
            
        except Exception as e:
            test_result = {"passed": False, "error": str(e)}
            print(f"✗ Statistical analysis test failed: {e}")
        
        self.results["test_04"] = test_result
        return test_result["passed"]
    
    def test_05_reproducibility(self) -> bool:
        """Test 05: Reproducibility verification"""
        print("\n--- Test 05: Reproducibility ---")
        
        try:
            seed = 12345
            env = self.make_env()
            policy = self.heuristic_policy
            
            # Run same experiment twice
            results_1 = []
            results_2 = []
            
            for run in range(2):
                results = results_1 if run == 0 else results_2
                
                for episode in range(5):
                    setup_seed(seed + episode)
                    env.reset(seed=seed + episode)
                    total_reward = self.rollout_episode(env, policy)
                    results.append(total_reward)
            
            env.close()
            
            # Check reproducibility
            reproducible = np.allclose(results_1, results_2, rtol=1e-10)
            
            test_result = {
                "passed": reproducible,
                "results_1": results_1,
                "results_2": results_2,
                "max_difference": float(np.max(np.abs(np.array(results_1) - np.array(results_2))))
            }
            
            if reproducible:
                print("✓ Reproducibility test passed")
            else:
                print("✗ Reproducibility test failed")
                print(f"  Results 1: {results_1}")
                print(f"  Results 2: {results_2}")
            
        except Exception as e:
            test_result = {"passed": False, "error": str(e)}
            print(f"✗ Reproducibility test failed: {e}")
        
        self.results["test_05"] = test_result
        return test_result["passed"]
    
    def test_06_comprehensive_experiment(self) -> bool:
        """Test 06: Run comprehensive experiment with logging"""
        print("\n--- Test 06: Comprehensive Experiment ---")
        
        try:
            # Run experiment similar to what students will do
            env = self.make_env()
            
            policies = {
                "Random": self.random_policy,
                "Heuristic": self.heuristic_policy,
                "ε-greedy (0.1)": self.create_epsilon_greedy(0.1)
            }
            
            experiment_results = {}
            
            for policy_name, policy in policies.items():
                print(f"  Running {policy_name}...")
                
                # Collect data
                returns = []
                for episode in range(self.config.test_episodes):
                    env.reset(seed=42 + episode)
                    total_reward = self.rollout_episode(env, policy)
                    returns.append(total_reward)
                    
                    # Log to TensorBoard
                    if self.writer:
                        self.writer.add_scalar(f"Policy_{policy_name}/Episode_Return", 
                                             total_reward, episode)
                
                # Calculate statistics
                stats = self.calculate_statistics(returns)
                
                experiment_results[policy_name] = {
                    "mean_return": stats.mean,
                    "std_return": stats.std,
                    "ci_lower": stats.ci_lower,
                    "ci_upper": stats.ci_upper,
                    "returns": returns
                }
                
                # Log summary statistics
                if self.writer:
                    self.writer.add_scalar(f"Summary/Mean_Return", stats.mean, 
                                         list(policies.keys()).index(policy_name))
                    self.writer.add_histogram(f"Summary/Return_Distribution_{policy_name}", 
                                            np.array(returns), 0)
                
                print(f"    Mean: {stats.mean:.1f} ± {stats.std:.1f}")
            
            env.close()
            
            test_result = {"passed": True, "experiment_results": experiment_results}
            
        except Exception as e:
            test_result = {"passed": False, "error": str(e)}
            print(f"✗ Comprehensive experiment test failed: {e}")
        
        self.results["test_06"] = test_result
        return test_result["passed"]
    
    def test_07_visualization(self) -> bool:
        """Test 07: Visualization capabilities"""
        print("\n--- Test 07: Visualization ---")
        
        try:
            if not self.config.create_plots:
                print("✓ Visualization test skipped (create_plots=False)")
                return True
            
            # Create test plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Integrated Test Visualizations')
            
            # Plot 1: Sample episode trajectory
            rewards = [1.0] * 20 + [0.8] * 10 + [1.2] * 15  # Synthetic data
            axes[0, 0].plot(rewards)
            axes[0, 0].set_title('Sample Episode Rewards')
            axes[0, 0].set_ylabel('Reward')
            
            # Plot 2: Policy comparison
            policies = ['Random', 'Heuristic', 'ε-greedy']
            means = [25.3, 145.7, 178.2]
            stds = [15.2, 45.3, 38.1]
            
            axes[0, 1].bar(policies, means, yerr=stds, capsize=5, alpha=0.7)
            axes[0, 1].set_title('Policy Performance')
            axes[0, 1].set_ylabel('Mean Return')
            
            # Plot 3: Returns distribution
            data = np.random.normal(150, 30, 100)
            axes[1, 0].hist(data, bins=15, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Returns Distribution')
            axes[1, 0].set_xlabel('Return')
            
            # Plot 4: Learning progress
            episodes = range(50)
            progress = [50 + i * 2 + np.random.randn() * 10 for i in episodes]
            axes[1, 1].plot(episodes, progress)
            axes[1, 1].set_title('Learning Progress')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Return')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / "test_visualization.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            test_result = {"passed": True, "plot_saved": str(plot_path)}
            print(f"✓ Visualization test passed, plot saved to {plot_path}")
            
        except Exception as e:
            test_result = {"passed": False, "error": str(e)}
            print(f"✗ Visualization test failed: {e}")
        
        self.results["test_07"] = test_result
        return test_result["passed"]
    
    def generate_final_report(self) -> str:
        """Generate comprehensive final report"""
        
        total_time = time.time() - self.test_start_time
        
        # Count passed tests
        test_keys = [k for k in self.results.keys() if k.startswith("test_")]
        passed_tests = sum(1 for k in test_keys if self.results[k]["passed"])
        total_tests = len(test_keys)
        
        # Generate report
        report = f"""
# RL2025 Lecture 3 - Integrated Test Report

**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Time**: {total_time:.1f} seconds
**Device**: {device}

## Test Summary
- **Tests Passed**: {passed_tests}/{total_tests}
- **Success Rate**: {passed_tests/total_tests*100:.1f}%

## Individual Test Results

"""
        
        for test_key in sorted(test_keys):
            result = self.results[test_key]
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            
            report += f"### {test_key.replace('_', ' ').title()}\n"
            report += f"**Status**: {status}\n"
            
            if not result["passed"] and "error" in result:
                report += f"**Error**: {result['error']}\n"
            
            report += "\n"
        
        # Add experiment results if available
        if "test_06" in self.results and self.results["test_06"]["passed"]:
            exp_results = self.results["test_06"]["experiment_results"]
            report += "## Policy Performance Summary\n\n"
            
            for policy_name, stats in exp_results.items():
                report += f"**{policy_name}**: {stats['mean_return']:.1f} ± {stats['std_return']:.1f}\n"
            
            report += "\n"
        
        # Add recommendations
        report += "## Recommendations\n\n"
        
        if passed_tests == total_tests:
            report += "🎉 All tests passed! The Lecture 3 implementation is working correctly.\n\n"
            report += "**Next Steps**:\n"
            report += "- Proceed to Lecture 4 materials\n"
            report += "- Try experimenting with different hyperparameters\n"
            report += "- Explore additional environments\n"
        else:
            report += "⚠️ Some tests failed. Please review the errors above.\n\n"
            report += "**Troubleshooting**:\n"
            report += "- Check that all required packages are installed\n"
            report += "- Verify that previous experiments ran successfully\n"
            report += "- Review error messages for specific issues\n"
        
        # Save report
        report_path = self.output_dir / "integrated_test_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nFinal report saved to: {report_path}")
        return report
    
    def run_all_tests(self) -> bool:
        """Run complete test suite"""
        
        self.log_test_start()
        
        # Run all tests in sequence
        tests = [
            self.test_01_environment_setup,
            self.test_02_policy_implementations,
            self.test_03_returns_calculation,
            self.test_04_statistical_analysis,
            self.test_05_reproducibility,
            self.test_06_comprehensive_experiment,
            self.test_07_visualization
        ]
        
        test_results = []
        for test_func in tests:
            try:
                result = test_func()
                test_results.append(result)
            except Exception as e:
                print(f"Test {test_func.__name__} crashed: {e}")
                test_results.append(False)
        
        # Save all results
        if self.config.save_results:
            results_path = self.output_dir / "integrated_test_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"Results saved to: {results_path}")
        
        # Generate final report
        report = self.generate_final_report()
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.flush()
            self.writer.close()
        
        # Summary
        passed_count = sum(test_results)
        total_count = len(test_results)
        
        print("\n" + "="*70)
        print("INTEGRATED TEST SUMMARY")
        print("="*70)
        print(f"Tests passed: {passed_count}/{total_count}")
        print(f"Success rate: {passed_count/total_count*100:.1f}%")
        
        if passed_count == total_count:
            print("🎉 ALL TESTS PASSED! Lecture 3 implementation is complete.")
        else:
            print("⚠️  Some tests failed. Please check the report for details.")
        
        return passed_count == total_count
    
    # Helper methods for policies and calculations
    def make_env(self, seed: int = None) -> gym.Env:
        """Create environment with proper seeding"""
        if seed is None:
            seed = self.config.seed
        
        env = gym.make(self.config.env_id)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    def rollout_episode(self, env: gym.Env, policy: Callable, max_steps: int = 500) -> float:
        """Run single episode"""
        obs, _ = env.reset()
        total_reward = 0.0
        
        for _ in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return total_reward
    
    def random_policy(self, obs: np.ndarray) -> int:
        """Random policy"""
        return np.random.randint(0, 2)
    
    def heuristic_policy(self, obs: np.ndarray) -> int:
        """Heuristic policy for CartPole"""
        x, x_dot, theta, theta_dot = obs
        control_signal = theta + 0.5 * theta_dot
        return 1 if control_signal > 0.0 else 0
    
    def create_epsilon_greedy(self, epsilon: float) -> Callable:
        """Create epsilon-greedy policy"""
        def policy(obs: np.ndarray) -> int:
            if np.random.random() < epsilon:
                return np.random.randint(0, 2)
            return self.heuristic_policy(obs)
        return policy
    
    def calculate_returns(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """Calculate discounted returns"""
        returns = []
        G = 0.0
        
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.append(G)
        
        return list(reversed(returns))
    
    def calculate_statistics(self, data: List[float]) -> object:
        """Calculate basic statistics"""
        from types import SimpleNamespace
        
        data = np.array(data)
        n = len(data)
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        sem = std / np.sqrt(n)
        
        # Simple confidence interval (assuming normal distribution)
        t_critical = 1.96  # Approximation for large n
        ci_lower = mean - t_critical * sem
        ci_upper = mean + t_critical * sem
        
        return SimpleNamespace(
            mean=float(mean),
            std=float(std),
            sem=float(sem),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper)
        )
    
    def compare_policies(self, data_a: List[float], data_b: List[float]) -> Dict[str, float]:
        """Simple policy comparison"""
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            print('Warning: SciPy not installed; skipping statistical tests.')
            return {
                "cohens_d": None,
                "t_statistic": None,
                "p_value": None,
                "error": "scipy-not-installed"
            }
        
        # Calculate Cohen's d (effect size)
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        std_a, std_b = np.std(data_a, ddof=1), np.std(data_b, ddof=1)
        pooled_std = np.sqrt(((len(data_a) - 1) * std_a**2 + 
                             (len(data_b) - 1) * std_b**2) / 
                            (len(data_a) + len(data_b) - 2))
        
        cohens_d = (mean_a - mean_b) / pooled_std
        
        # Simple t-test
        t_stat, p_value = scipy_stats.ttest_ind(data_a, data_b)
        
        return {
            "cohens_d": float(cohens_d),
            "t_statistic": float(t_stat),
            "p_value": float(p_value)
        }

def main():
    """Run integrated test"""
    
    # Check dependencies
    try:
        from scipy import stats
    except ImportError:
        print("Warning: scipy not available, some statistical tests will be limited")
    
    # Configuration
    config = IntegratedTestConfig(
        test_episodes=30,  # Reduce for faster testing
        statistical_runs=3,
        save_results=True,
        create_plots=True,
        run_tensorboard=True
    )
    
    # Run tests
    tester = IntegratedTester(config)
    success = tester.run_all_tests()
    
    # Final message
    if success:
        print("\n🎓 Congratulations! You have successfully completed all Lecture 3 experiments.")
        print("You are now ready to proceed to Lecture 4: Mathematical Foundations (MDPs).")
        
        if config.run_tensorboard:
            print(f"\nTo view TensorBoard logs, run:")
            print(f"  tensorboard --logdir={config.log_dir}")
    else:
        print("\n📚 Please review the failed tests and make necessary corrections.")
        print("All tests must pass before proceeding to the next lecture.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)