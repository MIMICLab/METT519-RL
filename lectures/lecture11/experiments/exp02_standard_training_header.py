#!/usr/bin/env python3
"""
RL2025 - Lecture 11: Experiment 02 - Standard Training Header Module

Implements a comprehensive training utilities module with device management,
checkpointing, logging, and evaluation scaffolding for modern RL methods.

Learning objectives:
- Create reusable training infrastructure
- Implement proper device management and AMP support
- Build checkpoint save/resume functionality
- Set up TensorBoard logging and metrics tracking

Prerequisites: PyTorch 2.x, tensorboard
"""

import os
import json
import yaml
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

# PyTorch 2.x Standard Practice Header
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    """Lightweight experiment configuration."""
    # Model parameters
    model_name: str = "default"
    hidden_size: int = 256
    num_layers: int = 2
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    
    # RL-specific parameters
    gamma: float = 0.99
    epsilon: float = 0.1
    beta: float = 0.01  # For DPO/RLHF
    c_puct: float = 1.0  # For MCTS
    
    # System parameters
    device: str = str(device)
    amp_enabled: bool = amp_enabled
    compile_model: bool = False
    seed: int = 42
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    def save(self, path: str):
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str):
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

class CheckpointManager:
    """Handles model checkpointing with torch.compile support."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                       epoch: int, metrics: Dict[str, float], config: ExperimentConfig):
        """Save model checkpoint."""
        # Handle compiled models
        model_state = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'config': asdict(config),
            'torch_version': torch.__version__,
            'device': str(device)
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                       checkpoint_path: Optional[str] = None) -> Tuple[int, Dict[str, float]]:
        """Load model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        
        if not Path(checkpoint_path).exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return 0, {}
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle compiled models
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['metrics']

class ExperimentLogger:
    """Unified logging with TensorBoard support."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir / experiment_name))
        self.metrics = {}
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / f'{experiment_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log scalar value."""
        self.writer.add_scalar(name, value, step)
        self.metrics[name] = value
    
    def log_scalars(self, metrics: Dict[str, float], step: int):
        """Log multiple scalar values."""
        for name, value in metrics.items():
            self.log_scalar(name, value, step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log histogram of tensor values."""
        self.writer.add_histogram(name, values, step)
    
    def log_text(self, name: str, text: str, step: int):
        """Log text."""
        self.writer.add_text(name, text, step)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def close(self):
        """Close logger."""
        self.writer.close()

class TimingUtils:
    """Timing utilities for performance measurement."""
    
    @staticmethod
    @contextmanager
    def timer(name: str):
        """Context manager for timing operations."""
        start = time.time()
        yield
        end = time.time()
        print(f"{name}: {end - start:.4f}s")
    
    @staticmethod
    def measure_throughput(model: nn.Module, input_shape: Tuple[int, ...], 
                          num_iterations: int = 100, warmup: int = 10) -> float:
        """Measure model throughput."""
        model.eval()
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        
        throughput = num_iterations / (end - start)
        return throughput

class GymUtils:
    """Gymnasium environment utilities."""
    
    @staticmethod
    def rollout(env: gym.Env, policy: callable, max_steps: int = 1000, 
                render: bool = False) -> Tuple[List[np.ndarray], List[int], List[float], bool]:
        """Perform environment rollout."""
        observations, actions, rewards = [], [], []
        
        obs, info = env.reset()
        observations.append(obs)
        total_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()
            
            # Get action from policy
            action = policy(obs)
            actions.append(action)
            
            # Environment step
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return observations, actions, rewards, terminated

class MinimalDQN(nn.Module):
    """Minimal DQN for reference implementation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: [B, state_dim], Output: [B, action_dim]"""
        return self.network(x)

def dqn_update_step(model: nn.Module, target_model: nn.Module, optimizer: torch.optim.Optimizer,
                   states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                   next_states: torch.Tensor, dones: torch.Tensor, gamma: float = 0.99) -> float:
    """Single DQN update step."""
    # Current Q values: [B, action_dim] -> [B]
    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Next Q values: [B]
    with torch.no_grad():
        next_q = target_model(next_states).max(1)[0]
        target_q = rewards + gamma * next_q * (1 - dones.float())
    
    # Loss computation
    loss = F.mse_loss(current_q, target_q)
    
    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def setup_cli_args():
    """Setup command line arguments for VS Code/Colab compatibility."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL2025 Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--experiment-name", type=str, default="experiment", help="Experiment name")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()

def main():
    print("="*60)
    print("Experiment 02: Standard Training Header Module")
    print("="*60)
    
    # Test configuration management
    print("Testing configuration management...")
    config = ExperimentConfig(learning_rate=1e-3, batch_size=64)
    config.save("test_config.yaml")
    loaded_config = ExperimentConfig.load("test_config.yaml")
    print(f"  Config save/load: {'✓' if loaded_config.learning_rate == 1e-3 else '✗'}")
    
    # Test checkpoint manager
    print("Testing checkpoint manager...")
    model = MinimalDQN(state_dim=4, action_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint_manager = CheckpointManager("./test_checkpoints")
    
    # Save checkpoint
    metrics = {"loss": 0.5, "reward": 100.0}
    checkpoint_path = checkpoint_manager.save_checkpoint(model, optimizer, 1, metrics, config)
    print(f"  Checkpoint save: {'✓' if Path(checkpoint_path).exists() else '✗'}")
    
    # Load checkpoint
    epoch, loaded_metrics = checkpoint_manager.load_checkpoint(model, optimizer)
    print(f"  Checkpoint load: {'✓' if epoch == 1 and loaded_metrics['loss'] == 0.5 else '✗'}")
    
    # Test logging
    print("Testing logging system...")
    logger = ExperimentLogger("./test_logs", "test_experiment")
    logger.log_scalar("test_metric", 42.0, 0)
    logger.info("Test log message")
    print(f"  Logging: ✓")
    logger.close()
    
    # Test timing utilities
    print("Testing timing utilities...")
    with TimingUtils.timer("Test operation"):
        time.sleep(0.01)  # Simulate work
    
    throughput = TimingUtils.measure_throughput(model, (1, 4), num_iterations=10, warmup=2)
    print(f"  Model throughput: {throughput:.1f} inferences/sec")
    
    # Test DQN update
    print("Testing DQN update...")
    target_model = MinimalDQN(state_dim=4, action_dim=2).to(device)
    target_model.load_state_dict(model.state_dict())
    
    # Sample batch: [B, state_dim], [B], [B], [B, state_dim], [B]
    batch_size = 8
    states = torch.randn(batch_size, 4).to(device)
    actions = torch.randint(0, 2, (batch_size,)).to(device)
    rewards = torch.randn(batch_size).to(device)
    next_states = torch.randn(batch_size, 4).to(device)
    dones = torch.randint(0, 2, (batch_size,)).bool().to(device)
    
    loss = dqn_update_step(model, target_model, optimizer, states, actions, rewards, next_states, dones)
    print(f"  DQN update: {'✓' if isinstance(loss, float) else '✗'} (loss: {loss:.4f})")
    
    # Cleanup test files
    import shutil
    for path in ["test_config.yaml", "./test_checkpoints", "./test_logs"]:
        if Path(path).exists():
            if Path(path).is_file():
                Path(path).unlink()
            else:
                shutil.rmtree(path)
    
    print("\nTraining header module verification complete!")
    print("All utilities ready for advanced RL implementations.")

if __name__ == "__main__":
    main()