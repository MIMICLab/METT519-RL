#!/usr/bin/env python3
"""
RL2025 - Lecture 3: Experiment 07 - TensorBoard Logging and Experiment Tracking

This experiment demonstrates comprehensive experiment logging using TensorBoard,
including scalar metrics, histograms, images, and text logging for RL experiments.

Learning objectives:
- Set up TensorBoard logging for RL experiments
- Log different types of data (scalars, histograms, text, hyperparameters)
- Visualize episode trajectories and policy performance
- Create reproducible experiment tracking

Prerequisites: Experiments 01-06 completed successfully
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
import time
import datetime
from dataclasses import dataclass, asdict
import io
import base64

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriterImpl
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

    class _StubSummaryWriter:
        """Fallback writer that disables TensorBoard logging gracefully."""

        _warned = False

        def __init__(self, *args, **kwargs):
            if not _StubSummaryWriter._warned:
                print("Warning: TensorBoard not installed; logging calls will be no-ops.")
                _StubSummaryWriter._warned = True

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def add_image(self, *args, **kwargs):
            pass

        def add_hparams(self, *args, **kwargs):
            pass

        def close(self):
            pass

    _SummaryWriterImpl = _StubSummaryWriter

SummaryWriter = _SummaryWriterImpl

@dataclass
class ExperimentConfig:
    """Configuration for RL experiment"""
    env_id: str = "CartPole-v1"
    num_episodes: int = 100
    max_steps: int = 500
    seed: int = 42
    log_interval: int = 10
    policy_name: str = "epsilon_greedy"
    epsilon: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def make_env(env_id: str = "CartPole-v1", seed: int = 42) -> gym.Env:
    """Create and initialize environment with proper seeding"""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def heuristic_action(obs: np.ndarray) -> int:
    """Simple CartPole heuristic"""
    x, x_dot, theta, theta_dot = obs
    control_signal = theta + 0.5 * theta_dot
    return 1 if control_signal > 0.0 else 0

def epsilon_greedy_policy(epsilon: float):
    """Create epsilon-greedy policy"""
    def policy(obs: np.ndarray) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        return heuristic_action(obs)
    return policy

def collect_episode_with_logging(env: gym.Env, 
                                policy: Callable, 
                                writer: SummaryWriter,
                                episode_num: int,
                                log_trajectory: bool = False) -> Tuple[float, int, Dict[str, Any]]:
    """
    Collect episode with comprehensive logging
    
    Returns:
        (total_reward, episode_length, episode_data)
    """
    obs, _ = env.reset()
    
    # Episode data collection
    episode_data = {
        'observations': [obs.copy()],
        'actions': [],
        'rewards': [],
        'q_values': [],  # For demonstration (would be actual Q-values in DQN)
        'step_times': [time.time()]
    }
    
    total_reward = 0.0
    steps = 0
    
    for step in range(500):  # CartPole max steps
        # Record action decision
        action = policy(obs)
        episode_data['actions'].append(action)
        
        # Simulate Q-values for logging (in real DQN, these would be actual values)
        fake_q_values = np.random.randn(2) + [0.5, 0.3]  # Slight bias toward action 0
        episode_data['q_values'].append(fake_q_values)
        
        # Environment step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        episode_data['observations'].append(next_obs.copy())
        episode_data['rewards'].append(reward)
        episode_data['step_times'].append(time.time())
        
        total_reward += reward
        steps += 1
        obs = next_obs
        
        # Log step-level metrics
        if log_trajectory:
            writer.add_scalar(f'Episode_{episode_num}/Step_Reward', reward, step)
            writer.add_scalar(f'Episode_{episode_num}/Position', obs[0], step)
            writer.add_scalar(f'Episode_{episode_num}/Velocity', obs[1], step)
            writer.add_scalar(f'Episode_{episode_num}/Angle', obs[2], step)
            writer.add_scalar(f'Episode_{episode_num}/Angular_Velocity', obs[3], step)
        
        if terminated or truncated:
            break
    
    # Log episode summary
    writer.add_scalar('Episode/Return', total_reward, episode_num)
    writer.add_scalar('Episode/Length', steps, episode_num)
    writer.add_scalar('Episode/Final_Position', abs(obs[0]), episode_num)
    writer.add_scalar('Episode/Final_Angle', abs(obs[2]), episode_num)
    
    # Log action distribution
    if episode_data['actions']:
        action_counts = np.bincount(episode_data['actions'], minlength=2)
        action_ratio = action_counts[1] / max(len(episode_data['actions']), 1)
        writer.add_scalar('Episode/Action_1_Ratio', action_ratio, episode_num)
    
    # Log Q-value statistics (simulated)
    if episode_data['q_values']:
        q_array = np.array(episode_data['q_values'])
        writer.add_scalar('Episode/Mean_Q_Value', q_array.mean(), episode_num)
        writer.add_scalar('Episode/Q_Value_Std', q_array.std(), episode_num)
        writer.add_histogram('Episode/Q_Values', q_array, episode_num)
    
    return total_reward, steps, episode_data

def log_episode_trajectory(writer: SummaryWriter, 
                          episode_data: Dict[str, Any], 
                          episode_num: int):
    """Create and log trajectory visualization"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Episode {episode_num} Trajectory', fontsize=14)
        
        observations = np.array(episode_data['observations'][:-1])  # Exclude final obs
        actions = np.array(episode_data['actions'])
        time_steps = range(len(observations))
        
        # Position and velocity
        axes[0, 0].plot(time_steps, observations[:, 0], label='Position', color='blue')
        axes[0, 0].axhline(y=2.4, color='red', linestyle='--', alpha=0.7, label='Limit')
        axes[0, 0].axhline(y=-2.4, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_ylabel('Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Angle and angular velocity
        axes[0, 1].plot(time_steps, observations[:, 2], label='Angle', color='orange')
        axes[0, 1].axhline(y=0.2095, color='red', linestyle='--', alpha=0.7, label='Limit')
        axes[0, 1].axhline(y=-0.2095, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_ylabel('Angle (rad)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Actions
        axes[1, 0].step(time_steps, actions, where='post', color='green', linewidth=2)
        axes[1, 0].set_ylabel('Action')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylim(-0.1, 1.1)
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_yticklabels(['Left (0)', 'Right (1)'])
        axes[1, 0].grid(True, alpha=0.3)
        
        # State space trajectory
        axes[1, 1].scatter(observations[:, 0], observations[:, 2], 
                          c=time_steps, cmap='viridis', alpha=0.7, s=20)
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Angle (rad)')
        axes[1, 1].set_title('State Space Trajectory')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add boundaries
        axes[1, 1].axvline(x=2.4, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].axvline(x=-2.4, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].axhline(y=0.2095, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].axhline(y=-0.2095, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Convert plot to image and log to TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to PIL Image for TensorBoard
        import PIL.Image
        image = PIL.Image.open(buf)
        image_array = np.array(image)
        
        writer.add_image(f'Trajectories/Episode_{episode_num}', 
                        image_array, episode_num, dataformats='HWC')
        
        plt.close(fig)
        buf.close()
        
    except Exception as e:
        print(f"Failed to create trajectory plot: {e}")

def log_hyperparameters(writer: SummaryWriter, config: ExperimentConfig, results: Dict[str, float]):
    """Log hyperparameters and results for hyperparameter tuning analysis"""
    
    hparams = config.to_dict()
    metrics = {
        'hparam/mean_return': results['mean_return'],
        'hparam/std_return': results['std_return'],
        'hparam/success_rate': results['success_rate'],
        'hparam/mean_length': results['mean_length']
    }
    
    writer.add_hparams(hparams, metrics)

def log_environment_info(writer: SummaryWriter, env: gym.Env):
    """Log environment specifications and metadata"""
    
    env_info = {
        'env_id': env.spec.id,
        'max_episode_steps': env.spec.max_episode_steps,
        'reward_threshold': getattr(env.spec, 'reward_threshold', None),
        'observation_space': str(env.observation_space),
        'action_space': str(env.action_space),
        'obs_shape': env.observation_space.shape,
        'n_actions': env.action_space.n if hasattr(env.action_space, 'n') else None
    }
    
    # Convert to readable text format
    env_text = '\n'.join([f'{k}: {v}' for k, v in env_info.items()])
    writer.add_text('Environment/Specifications', env_text, 0)

def run_logged_experiment(config: ExperimentConfig, 
                         log_dir: str = None) -> Dict[str, Any]:
    """
    Run complete experiment with comprehensive logging
    
    Returns:
        Dictionary with experiment results
    """
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/exp07_{config.policy_name}_{timestamp}"
    
    print(f"Starting experiment: {config.policy_name}")
    print(f"Logging to: {log_dir}")
    
    # Setup
    setup_seed(config.seed)
    writer = SummaryWriter(log_dir=log_dir)
    env = make_env(config.env_id, config.seed)
    
    # Log environment info
    log_environment_info(writer, env)
    
    # Create policy
    if config.policy_name == "epsilon_greedy":
        policy = epsilon_greedy_policy(config.epsilon)
    elif config.policy_name == "random":
        policy = lambda obs: np.random.randint(0, 2)
    elif config.policy_name == "heuristic":
        policy = heuristic_action
    else:
        raise ValueError(f"Unknown policy: {config.policy_name}")
    
    # Log experiment configuration
    config_text = '\n'.join([f'{k}: {v}' for k, v in config.to_dict().items()])
    writer.add_text('Experiment/Configuration', config_text, 0)
    
    # Run episodes
    returns = []
    lengths = []
    start_time = time.time()
    
    for episode in range(config.num_episodes):
        # Reset environment with different seed for variety
        env.reset(seed=config.seed + episode)
        
        # Collect episode with logging
        log_trajectory = (episode % (config.num_episodes // 5) == 0)  # Log 5 trajectory plots
        total_reward, episode_length, episode_data = collect_episode_with_logging(
            env, policy, writer, episode, log_trajectory
        )
        
        returns.append(total_reward)
        lengths.append(episode_length)
        
        # Log trajectory visualization for selected episodes
        if log_trajectory:
            log_episode_trajectory(writer, episode_data, episode)
        
        # Periodic logging
        if (episode + 1) % config.log_interval == 0:
            recent_returns = returns[-config.log_interval:]
            writer.add_scalar('Progress/Mean_Return_Recent', np.mean(recent_returns), episode)
            writer.add_scalar('Progress/Std_Return_Recent', np.std(recent_returns), episode)
            
            print(f"Episode {episode + 1}/{config.num_episodes}: "
                  f"Recent mean return: {np.mean(recent_returns):.1f}")
    
    # Calculate final statistics
    results = {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns)),
        'mean_length': float(np.mean(lengths)),
        'success_rate': float(sum(1 for r in returns if r >= 475) / len(returns)),
        'total_time': time.time() - start_time,
        'episodes': len(returns)
    }
    
    # Log final statistics
    for key, value in results.items():
        writer.add_scalar(f'Final/{key}', value, 0)
    
    # Log return distribution
    writer.add_histogram('Final/Return_Distribution', np.array(returns), 0)
    writer.add_histogram('Final/Length_Distribution', np.array(lengths), 0)
    
    # Log hyperparameters and results
    log_hyperparameters(writer, config, results)
    
    # Create summary text
    summary_text = f"""
    Experiment Summary:
    Policy: {config.policy_name}
    Episodes: {config.num_episodes}
    Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}
    Success Rate: {results['success_rate']*100:.1f}%
    Total Time: {results['total_time']:.1f}s
    """
    writer.add_text('Experiment/Summary', summary_text, 0)
    
    print(f"\nExperiment completed!")
    print(f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"Success rate: {results['success_rate']*100:.1f}%")
    
    # Close resources
    writer.flush()
    writer.close()
    env.close()
    
    return results

def compare_policies_with_logging():
    """Compare multiple policies with separate TensorBoard logs"""
    
    base_config = ExperimentConfig(
        num_episodes=50,
        log_interval=5
    )
    
    policies_to_test = [
        ("random", {"policy_name": "random"}),
        ("heuristic", {"policy_name": "heuristic"}),
        ("epsilon_0.05", {"policy_name": "epsilon_greedy", "epsilon": 0.05}),
        ("epsilon_0.1", {"policy_name": "epsilon_greedy", "epsilon": 0.1}),
        ("epsilon_0.2", {"policy_name": "epsilon_greedy", "epsilon": 0.2}),
    ]
    
    comparison_results = {}
    
    for name, config_updates in policies_to_test:
        config = ExperimentConfig(**{**base_config.to_dict(), **config_updates})
        results = run_logged_experiment(config, f"runs/comparison_{name}")
        comparison_results[name] = results
        time.sleep(1)  # Brief pause between experiments
    
    return comparison_results

def demonstrate_advanced_logging():
    """Demonstrate advanced TensorBoard logging features"""
    
    print("\n--- Advanced Logging Demo ---")
    writer = SummaryWriter("runs/advanced_logging_demo")
    
    # 1. Custom scalars with different smoothing
    for i in range(100):
        # Noisy signal
        noisy_value = np.sin(i * 0.1) + 0.1 * np.random.randn()
        writer.add_scalar('Demo/Noisy_Signal', noisy_value, i)
        
        # Smooth signal
        smooth_value = np.sin(i * 0.1)
        writer.add_scalar('Demo/Smooth_Signal', smooth_value, i)
        
        # Custom layout grouping
        writer.add_scalars('Demo/Combined_Signals', {
            'noisy': noisy_value,
            'smooth': smooth_value,
            'difference': abs(noisy_value - smooth_value)
        }, i)
    
    # 2. Parameter histograms over time
    for epoch in range(10):
        # Simulate parameter evolution
        weights = torch.randn(1000) + epoch * 0.1
        biases = torch.randn(100) * (1 + epoch * 0.05)
        
        writer.add_histogram('Parameters/Weights', weights, epoch)
        writer.add_histogram('Parameters/Biases', biases, epoch)
    
    # 3. Custom images and figures
    try:
        # Generate sample confusion matrix-style plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        data = np.random.rand(10, 10)
        im = ax.imshow(data, cmap='Blues')
        ax.set_title('Sample Heatmap')
        plt.colorbar(im)
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        import PIL.Image
        image = PIL.Image.open(buf)
        image_array = np.array(image)
        
        writer.add_image('Demo/Custom_Heatmap', image_array, 0, dataformats='HWC')
        plt.close(fig)
        buf.close()
        
    except Exception as e:
        print(f"Image logging failed: {e}")
    
    # 4. Text and metadata
    writer.add_text('Demo/Information', 'This is a demonstration of TensorBoard capabilities', 0)
    writer.add_text('Demo/Markdown', '# This is a heading\n- Bullet point 1\n- Bullet point 2', 0)
    
    print("Advanced logging demo completed. Check runs/advanced_logging_demo/")
    writer.close()

def main():
    """Run TensorBoard logging experiment"""
    print("="*60)
    print("Experiment 07: TensorBoard Logging and Experiment Tracking")
    print("="*60)
    
    # 1. Single experiment with detailed logging
    print("\n1. Running detailed single experiment...")
    config = ExperimentConfig(
        policy_name="epsilon_greedy",
        epsilon=0.1,
        num_episodes=30,
        log_interval=5
    )
    results = run_logged_experiment(config)
    
    # 2. Compare multiple policies
    print("\n2. Running policy comparison with logging...")
    comparison_results = compare_policies_with_logging()
    
    # 3. Demonstrate advanced features
    demonstrate_advanced_logging()
    
    # Print comparison summary
    print("\n" + "="*60)
    print("POLICY COMPARISON RESULTS")
    print("="*60)
    
    for policy_name, results in comparison_results.items():
        print(f"{policy_name:15s}: Return={results['mean_return']:6.1f}±{results['std_return']:5.1f}, "
              f"Success={results['success_rate']*100:5.1f}%")
    
    # Save comparison results
    os.makedirs("results", exist_ok=True)
    with open("results/tensorboard_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nComparison results saved to: results/tensorboard_comparison.json")
    
    print("\n" + "="*60)
    print("TENSORBOARD INSTRUCTIONS")
    print("="*60)
    print("To view the results, run:")
    print("  tensorboard --logdir=runs")
    print("Then open: http://localhost:6006")
    print()
    print("Available log directories:")
    if os.path.exists("runs"):
        for item in sorted(os.listdir("runs")):
            if os.path.isdir(f"runs/{item}"):
                print(f"  - runs/{item}")
    
    print("\n" + "="*60)
    print("KEY TENSORBOARD FEATURES DEMONSTRATED:")
    print("="*60)
    print("1. Scalar logging (returns, lengths, statistics)")
    print("2. Histogram logging (Q-values, distributions)")
    print("3. Image logging (trajectory plots)")
    print("4. Text logging (configurations, summaries)")
    print("5. Hyperparameter tracking")
    print("6. Custom scalar grouping")
    print("7. Experiment comparison")
    
    print("\nExperiment 07 completed successfully!")
    return True

if __name__ == "__main__":
    main()
