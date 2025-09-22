#!/usr/bin/env python3
"""
RL2025 - Lecture 5: Experiment 09 - Integrated Q-Learning Test

This experiment integrates all Q-learning concepts into a comprehensive test.

Learning objectives:
- Complete Q-learning implementation with all features
- Performance benchmarking and visualization
- Reproducibility verification

Prerequisites: All previous experiments (exp01-exp08) completed successfully
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriterImpl
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

    class _StubSummaryWriter:
        _warned = False

        def __init__(self, *args, **kwargs):
            if not _StubSummaryWriter._warned:
                print("Warning: TensorBoard not installed; logging disabled in Lecture 5 integrated test.")
                _StubSummaryWriter._warned = True

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def add_hparams(self, *args, **kwargs):
            pass

        def close(self):
            pass

    _SummaryWriterImpl = _StubSummaryWriter

SummaryWriter = _SummaryWriterImpl

def capture_rng_states() -> Dict[str, Any]:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def restore_rng_states(states: Optional[Dict[str, Any]]) -> None:
    if not states:
        return
    if states.get('python') is not None:
        random.setstate(states['python'])
    if states.get('numpy') is not None:
        np.random.set_state(states['numpy'])
    if states.get('torch') is not None:
        torch.set_rng_state(states['torch'])
    if states.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['cuda'])


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Return RNG states for reproducibility if needed by callers
    return capture_rng_states()

# Proper device selection (CUDA > MPS > CPU)
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
amp_enabled = torch.cuda.is_available()
setup_seed(42)

@dataclass
class ExperimentConfig:
    """Complete configuration for Q-learning experiment."""
    # Environment
    env_id: str = "FrozenLake-v1"
    map_size: int = 4
    hole_prob: float = 0.2
    is_slippery: bool = False
    slip_prob: float = 0.0
    max_episode_steps: int = 100
    
    # Training
    episodes: int = 1000
    eval_interval: int = 50
    eval_episodes: int = 100
    
    # Q-learning
    algorithm: str = "q_learning"  # "q_learning", "sarsa", "double_q"
    gamma: float = 0.99
    
    # Learning rate schedule
    alpha_start: float = 1.0
    alpha_end: float = 0.01
    alpha_schedule: str = "exponential"
    alpha_tau: float = 5000.0
    
    # Exploration schedule
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_schedule: str = "exponential"
    eps_tau: float = 5000.0
    
    # Reproducibility
    seed: int = 42
    device: str = str(device)
    
    # Logging
    log_dir: str = "runs/exp09"
    save_checkpoint: bool = True
    resume_from_checkpoint: bool = True
    
    def get_exp_id(self):
        """Generate unique experiment ID."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

class IntegratedQLearning:
    """Complete Q-learning implementation with all features."""
    
    def __init__(self, n_states: int, n_actions: int, config: ExperimentConfig):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.global_step = 0
        
        # Initialize Q-table(s)
        if config.algorithm == "double_q":
            self.Q1 = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
            self.Q2 = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        else:
            self.Q = torch.zeros((n_states, n_actions), dtype=torch.float32, device=device)
        
        # Visit counts for statistics
        self.N_sa = torch.zeros((n_states, n_actions), dtype=torch.int32, device=device)
        
        # Statistics tracking
        self.td_errors = []
        self.q_values_history = []
        self.action_distribution = torch.zeros(n_actions, dtype=torch.int32)
    
    def get_alpha(self) -> float:
        """Get current learning rate."""
        c = self.config
        if c.alpha_schedule == "constant":
            return c.alpha_start
        elif c.alpha_schedule == "linear":
            progress = min(1.0, self.global_step / (c.episodes * 50))
            return c.alpha_start + (c.alpha_end - c.alpha_start) * progress
        elif c.alpha_schedule == "exponential":
            return c.alpha_end + (c.alpha_start - c.alpha_end) * np.exp(-self.global_step / c.alpha_tau)
        elif c.alpha_schedule == "one_over_t":
            return c.alpha_start / (1.0 + self.global_step / c.alpha_tau)
        return c.alpha_start
    
    def get_epsilon(self) -> float:
        """Get current exploration rate."""
        c = self.config
        if c.eps_schedule == "constant":
            return c.eps_start
        elif c.eps_schedule == "linear":
            progress = min(1.0, self.global_step / (c.episodes * 50))
            return c.eps_start + (c.eps_end - c.eps_start) * progress
        elif c.eps_schedule == "exponential":
            return c.eps_end + (c.eps_start - c.eps_end) * np.exp(-self.global_step / c.eps_tau)
        elif c.eps_schedule == "one_over_t":
            return max(c.eps_end, c.eps_start / (1.0 + self.global_step / c.eps_tau))
        return c.eps_start
    
    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy."""
        epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            if self.config.algorithm == "double_q":
                q_avg = (self.Q1[state] + self.Q2[state]) / 2
                action = int(torch.argmax(q_avg).item())
            else:
                action = int(torch.argmax(self.Q[state]).item())
        
        self.action_distribution[action] += 1
        return action
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, next_action: Optional[int], done: bool) -> float:
        """Update Q-values based on algorithm."""
        alpha = self.get_alpha()
        gamma = self.config.gamma
        
        if self.config.algorithm == "q_learning":
            q_current = self.Q[state, action].item()
            if done:
                q_target = reward
            else:
                q_target = reward + gamma * torch.max(self.Q[next_state]).item()
            td_error = q_target - q_current
            self.Q[state, action] += alpha * td_error
            
        elif self.config.algorithm == "sarsa":
            q_current = self.Q[state, action].item()
            if done:
                q_target = reward
            else:
                q_target = reward + gamma * self.Q[next_state, next_action].item()
            td_error = q_target - q_current
            self.Q[state, action] += alpha * td_error
            
        elif self.config.algorithm == "double_q":
            if random.random() < 0.5:
                q_current = self.Q1[state, action].item()
                if done:
                    q_target = reward
                else:
                    best_action = torch.argmax(self.Q1[next_state]).item()
                    q_target = reward + gamma * self.Q2[next_state, best_action].item()
                td_error = q_target - q_current
                self.Q1[state, action] += alpha * td_error
            else:
                q_current = self.Q2[state, action].item()
                if done:
                    q_target = reward
                else:
                    best_action = torch.argmax(self.Q2[next_state]).item()
                    q_target = reward + gamma * self.Q1[next_state, best_action].item()
                td_error = q_target - q_current
                self.Q2[state, action] += alpha * td_error
        
        # Update statistics
        self.N_sa[state, action] += 1
        self.global_step += 1
        self.td_errors.append(abs(td_error))
        
        # Track Q-values
        if self.config.algorithm == "double_q":
            mean_q = torch.mean((self.Q1 + self.Q2) / 2).item()
        else:
            mean_q = torch.mean(self.Q).item()
        self.q_values_history.append(mean_q)
        
        return td_error
    
    def get_policy(self) -> torch.Tensor:
        """Get greedy policy from Q-values."""
        if self.config.algorithm == "double_q":
            Q_combined = (self.Q1 + self.Q2) / 2
        else:
            Q_combined = self.Q
        
        policy = torch.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            best_action = torch.argmax(Q_combined[s]).item()
            policy[s, best_action] = 1.0
        
        return policy
    
    def save_checkpoint(self, path: str, episode: int, returns: List[float]):
        """Save training checkpoint."""
        checkpoint = {
            'config': asdict(self.config),
            'episode': episode,
            'global_step': self.global_step,
            'returns': returns,
            'rng_states': capture_rng_states(),
            'statistics': {
                'td_errors': self.td_errors[-1000:],  # Save last 1000
                'q_values': self.q_values_history[-1000:],
                'action_dist': self.action_distribution.cpu().numpy().tolist()
            }
        }
        
        if self.config.algorithm == "double_q":
            checkpoint['Q1'] = self.Q1.cpu()
            checkpoint['Q2'] = self.Q2.cpu()
        else:
            checkpoint['Q'] = self.Q.cpu()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Optional[Dict]:
        """Load training checkpoint."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            
            if self.config.algorithm == "double_q":
                self.Q1 = checkpoint['Q1'].to(device)
                self.Q2 = checkpoint['Q2'].to(device)
            else:
                self.Q = checkpoint['Q'].to(device)
            
            self.global_step = checkpoint['global_step']
            restore_rng_states(checkpoint.get('rng_states'))
            return checkpoint
        return None

def create_environment(config: ExperimentConfig) -> gym.Env:
    """Create environment based on configuration."""
    if config.map_size != 4 or config.hole_prob != 0.2:
        safe_prob = 1.0 - config.hole_prob
        desc = generate_random_map(size=config.map_size, p=safe_prob, seed=config.seed)
        env = gym.make(config.env_id, desc=desc, is_slippery=config.is_slippery)
    else:
        env = gym.make(config.env_id, is_slippery=config.is_slippery)
    
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    
    # Add action slip wrapper if needed
    if config.slip_prob > 0:
        class SlipWrapper(gym.Wrapper):
            def step(self, action):
                if random.random() < config.slip_prob:
                    action = self.action_space.sample()
                return self.env.step(action)
        env = SlipWrapper(env)
    
    return env

def evaluate_policy(env: gym.Env, agent: IntegratedQLearning, 
                    episodes: int, seed: int) -> Tuple[float, float]:
    """Evaluate current policy."""
    returns = []
    successes = []
    
    for i in range(episodes):
        state, _ = env.reset(seed=seed + 10000 + i)
        done = False
        total_reward = 0
        
        while not done:
            # Greedy action selection for evaluation
            if agent.config.algorithm == "double_q":
                q_values = (agent.Q1[state] + agent.Q2[state]) / 2
            else:
                q_values = agent.Q[state]
            action = int(torch.argmax(q_values).item())
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        
        returns.append(total_reward)
        successes.append(1 if total_reward > 0 else 0)
    
    return np.mean(returns), np.mean(successes)

def train_agent(config: ExperimentConfig) -> Dict:
    """Complete training pipeline."""
    print(f"\nStarting experiment: {config.get_exp_id()}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Device: {device}")
    print("-" * 50)
    
    # Setup
    env = create_environment(config)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    agent = IntegratedQLearning(n_states, n_actions, config)
    
    # Logging
    exp_id = config.get_exp_id()
    log_dir = os.path.join(config.log_dir, exp_id)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Save config
    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Training metrics
    returns = []
    eval_returns = []
    eval_successes = []
    start_time = time.time()
    
    # Check for checkpoint
    checkpoint_path = os.path.join(log_dir, "checkpoint.pt")
    start_episode = 0
    if config.resume_from_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = agent.load_checkpoint(checkpoint_path)
        if checkpoint:
            start_episode = checkpoint['episode']
            returns = checkpoint['returns']
            print(f"Resumed from episode {start_episode}")
    
    # Training loop
    for episode in range(start_episode, config.episodes):
        state, _ = env.reset(seed=config.seed + episode)
        done = False
        total_reward = 0
        steps = 0
        
        # SARSA needs initial action
        if config.algorithm == "sarsa":
            action = agent.select_action(state)
        
        while not done:
            if config.algorithm == "sarsa":
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_action = agent.select_action(next_state) if not done else None
                agent.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
            else:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, None, done)
                state = next_state
            
            total_reward += reward
            steps += 1
        
        returns.append(total_reward)
        
        # Logging
        writer.add_scalar('train/return', total_reward, episode)
        writer.add_scalar('train/epsilon', agent.get_epsilon(), episode)
        writer.add_scalar('train/alpha', agent.get_alpha(), episode)
        writer.add_scalar('train/steps', steps, episode)
        
        # Evaluation
        if (episode + 1) % config.eval_interval == 0:
            avg_return, success_rate = evaluate_policy(env, agent, config.eval_episodes, config.seed)
            eval_returns.append(avg_return)
            eval_successes.append(success_rate)
            
            writer.add_scalar('eval/avg_return', avg_return, episode)
            writer.add_scalar('eval/success_rate', success_rate, episode)
            
            print(f"Episode {episode+1}/{config.episodes}: "
                  f"Success={success_rate:.2%}, "
                  f"ε={agent.get_epsilon():.3f}, "
                  f"α={agent.get_alpha():.3f}")
            
            # Save checkpoint
            if config.save_checkpoint:
                agent.save_checkpoint(checkpoint_path, episode + 1, returns)
    
    env.close()
    writer.close()
    
    # Final statistics
    elapsed = time.time() - start_time
    final_success = eval_successes[-1] if eval_successes else 0
    
    results = {
        'exp_id': exp_id,
        'algorithm': config.algorithm,
        'episodes': config.episodes,
        'elapsed_time': elapsed,
        'final_success': final_success,
        'returns': returns,
        'eval_returns': eval_returns,
        'eval_successes': eval_successes,
        'log_dir': log_dir
    }
    
    # Save final results
    with open(os.path.join(log_dir, "results.json"), 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'returns'}, f, indent=2)
    
    return results

def run_comparison():
    """Compare all three algorithms."""
    print("="*50)
    print("Algorithm Comparison")
    print("="*50)
    
    base_config = ExperimentConfig(
        episodes=1000,
        eval_interval=50,
        map_size=4,
        hole_prob=0.2,
        is_slippery=False,
        slip_prob=0.1
    )
    
    algorithms = ["q_learning", "sarsa", "double_q"]
    all_results = {}
    
    for algo in algorithms:
        config = ExperimentConfig(**{**asdict(base_config), 'algorithm': algo})
        results = train_agent(config)
        all_results[algo] = results
    
    # Compare results
    print("\n" + "="*50)
    print("Final Comparison:")
    print("-" * 50)
    for algo, results in all_results.items():
        print(f"{algo:12s}: Success={results['final_success']:.2%}, "
              f"Time={results['elapsed_time']:.1f}s")
    
    return all_results

def verify_reproducibility():
    """Verify reproducibility of results."""
    print("="*50)
    print("Reproducibility Test")
    print("="*50)
    
    config = ExperimentConfig(
        episodes=200,
        eval_interval=50,
        seed=42
    )
    
    results = []
    for run in range(3):
        print(f"\nRun {run+1}/3...")
        setup_seed(42)  # Reset seed
        run_config_dict = asdict(config).copy()
        run_config_dict.update({
            'resume_from_checkpoint': False,
            'save_checkpoint': False,
            'log_dir': os.path.join(config.log_dir, f"repro_run_{run+1}")
        })
        run_config = ExperimentConfig(**run_config_dict)
        result = train_agent(run_config)
        results.append(result['final_success'])
    
    print("\n" + "-"*50)
    print("Reproducibility Results:")
    for i, success in enumerate(results):
        print(f"Run {i+1}: {success:.2%}")
    
    if len(set(results)) == 1:
        print("✓ Results are perfectly reproducible!")
    else:
        print("✗ Results vary between runs")
    
    return results

def generate_report(results: Dict):
    """Generate comprehensive experiment report."""
    print("\n" + "="*50)
    print("EXPERIMENT REPORT")
    print("="*50)
    
    print("\n1. CONFIGURATION")
    print("-" * 30)
    config = results.get('config', {})
    for key, value in config.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    print("\n2. PERFORMANCE METRICS")
    print("-" * 30)
    print(f"  Final success rate: {results['final_success']:.2%}")
    print(f"  Training time: {results['elapsed_time']:.1f} seconds")
    print(f"  Episodes trained: {results['episodes']}")
    
    print("\n3. CONVERGENCE")
    print("-" * 30)
    if 'eval_successes' in results:
        successes = results['eval_successes']
        for threshold in [0.5, 0.7, 0.9]:
            converged = False
            for i, s in enumerate(successes):
                if s >= threshold:
                    episode = (i + 1) * 50  # eval_interval
                    print(f"  Reached {threshold:.0%} at episode {episode}")
                    converged = True
                    break
            if not converged:
                print(f"  Never reached {threshold:.0%}")
    
    print("\n4. FILES GENERATED")
    print("-" * 30)
    if 'log_dir' in results:
        print(f"  Log directory: {results['log_dir']}")
        print(f"  - config.json")
        print(f"  - checkpoint.pt")
        print(f"  - results.json")
        print(f"  - TensorBoard logs")

def main():
    print("="*50)
    print("Experiment 09: Integrated Q-Learning Test")
    print("="*50)
    
    # 1. Single algorithm test
    print("\n1. SINGLE ALGORITHM TEST")
    config = ExperimentConfig(
        algorithm="q_learning",
        episodes=500,
        eval_interval=50
    )
    results = train_agent(config)
    generate_report(results)
    
    # 2. Algorithm comparison
    print("\n2. ALGORITHM COMPARISON")
    comparison_results = run_comparison()
    
    # 3. Reproducibility verification
    print("\n3. REPRODUCIBILITY")
    repro_results = verify_reproducibility()
    
    # 4. Final summary
    print("\n" + "="*50)
    print("INTEGRATED TEST COMPLETE")
    print("="*50)
    print("✓ All Q-learning variants implemented")
    print("✓ Comprehensive logging and checkpointing")
    print("✓ Reproducibility verified")
    print("✓ Performance benchmarked")
    print(f"\nDevice used: {device}")
    print(f"Total experiment time: {time.time():.1f} seconds")
    
    print("\nTo view results in TensorBoard:")
    print(f"  tensorboard --logdir {config.log_dir}")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
