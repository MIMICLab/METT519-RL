#!/usr/bin/env python3
"""
Experiment 5: Standard Code Header Implementation
Slides: 46-55 (Standard code header for the course)
Time: 1:25-1:40 (15 minutes)

This experiment implements and tests the standard code header (v1) that will
be used throughout the course, including seeding, device management, AMP,
and basic utilities.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import json
import platform

# ============================================================================
# STANDARD CODE HEADER V1 - Core Module
# ============================================================================

def setup_seed(seed=42, deterministic=True):
    """
    Set seeds for reproducibility across all RNGs.
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

def get_device():
    """
    Get the best available device (CUDA > MPS > CPU).
    Standard device selection for the course.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class AutocastContext:
    """Context manager for automatic mixed precision"""
    def __init__(self, enabled=True, dtype=torch.float16):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        
    def __enter__(self):
        if self.enabled:
            from torch.cuda.amp import autocast
            self.context = autocast(dtype=self.dtype)
            return self.context.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and hasattr(self, 'context'):
            return self.context.__exit__(exc_type, exc_val, exc_tb)

def get_scaler(enabled=True):
    """Get GradScaler for AMP training"""
    if enabled and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        return GradScaler()
    else:
        # Dummy scaler for CPU/MPS
        class DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        return DummyScaler()

def _torch_compile_supported() -> bool:
    if not hasattr(torch, "compile"):
        return False
    if platform.system().lower().startswith("win"):
        try:
            import triton  # type: ignore
        except ImportError:
            return False
    return True


def compile_if_available(module, mode='default'):
    """Optionally compile module with torch.compile (PyTorch 2.x)"""
    if not _torch_compile_supported():
        if hasattr(torch, 'compile'):
            print("Warning: torch.compile not supported on this platform (missing Triton backend)")
        return module

    try:
        return torch.compile(module, mode=mode)
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")
        return module

# ============================================================================
# Training Utilities
# ============================================================================

def dqn_td_step(q_net, target_q_net, batch, gamma=0.99, huber_delta=1.0, 
                optimizer=None, scaler=None, amp_enabled=False):
    """
    Minimal DQN training step with TD learning.
    
    Input shapes:
        states: [B, state_dim]
        actions: [B]
        rewards: [B]
        next_states: [B, state_dim]
        dones: [B]
    
    Returns:
        loss: scalar float
    """
    states, actions, rewards, next_states, dones = batch
    
    # Get device from states
    device = states.device
    
    # Q-values for current states
    q_values = q_net(states)                           # [B, state_dim] -> [B, num_actions]
    q_values = q_values.gather(1, actions.unsqueeze(1))  # [B, num_actions] -> [B, 1]
    q_values = q_values.squeeze(1)                     # [B, 1] -> [B]
    
    # Target Q-values
    with torch.no_grad():
        next_q_values = target_q_net(next_states)      # [B, state_dim] -> [B, num_actions]
        next_q_values = next_q_values.max(dim=1)[0]    # [B, num_actions] -> [B]
        targets = rewards + gamma * (1 - dones) * next_q_values  # [B]
    
    # Huber loss
    loss = F.smooth_l1_loss(q_values, targets, beta=huber_delta)  # [] scalar
    
    # Optimization step if optimizer provided
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate_policy(policy_fn, env_fn, episodes=10, max_steps=1000):
    """
    Evaluate a policy over multiple episodes.
    
    Args:
        policy_fn: Function that takes state and returns action
        env_fn: Function that returns environment
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        mean_return: Average episode return
        std_return: Standard deviation of returns
        mean_length: Average episode length
    """
    returns = []
    lengths = []
    
    for ep in range(episodes):
        try:
            env = env_fn()
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1
            
            returns.append(total_reward)
            lengths.append(steps)
            
        except Exception as e:
            print(f"Warning: Environment not available - {e}")
            return 0.0, 0.0, 0.0
    
    mean_return = np.mean(returns) if returns else 0.0
    std_return = np.std(returns) if returns else 0.0
    mean_length = np.mean(lengths) if lengths else 0.0
    
    return mean_return, std_return, mean_length

# ============================================================================
# Test Functions
# ============================================================================

def test_seeding():
    """Test reproducibility with seeding"""
    print("\n" + "="*60)
    print("Testing Seeding and Reproducibility")
    print("="*60)
    
    results = []
    for run in range(3):
        setup_seed(42)
        
        # Generate random numbers
        py_rand = [random.random() for _ in range(3)]
        np_rand = np.random.randn(3)
        torch_rand = torch.randn(3)
        
        results.append((py_rand, np_rand, torch_rand))
        
        print(f"  Run {run+1}:")
        print(f"    Python: {[f'{x:.4f}' for x in py_rand]}")
        print(f"    NumPy:  {[f'{x:.4f}' for x in np_rand]}")
        print(f"    Torch:  {[f'{x:.4f}' for x in torch_rand.numpy()]}")
    
    # Check reproducibility
    all_same = all(
        np.allclose(results[0][1], r[1]) and 
        torch.allclose(results[0][2], r[2])
        for r in results[1:]
    )
    
    print(f"  Reproducible: {'✓' if all_same else '✗'}")
    return all_same

def test_device_management():
    """Test device selection and operations"""
    print("\n" + "="*60)
    print("Testing Device Management")
    print("="*60)
    
    device = get_device()
    print(f"  Selected device: {device}")
    
    # Test tensor operations on device
    x = torch.randn(10, 20, device=device)  # [10, 20]
    y = torch.randn(20, 30, device=device)  # [20, 30]
    z = torch.matmul(x, y)                  # [10, 30]
    
    print(f"  Tensor operation: [{x.shape[0]}, {x.shape[1]}] @ [{y.shape[0]}, {y.shape[1]}] -> {list(z.shape)}")
    print(f"  Result device: {z.device}")
    
    return True

def test_amp_context():
    """Test AMP context manager"""
    print("\n" + "="*60)
    print("Testing AMP Context")
    print("="*60)
    
    device = get_device()
    amp_enabled = device.type == 'cuda'
    
    print(f"  AMP enabled: {amp_enabled}")
    
    # Simple model
    model = nn.Linear(10, 5).to(device)
    x = torch.randn(32, 10, device=device)  # [32, 10]
    
    # Test with AMP context
    with AutocastContext(enabled=amp_enabled):
        output = model(x)  # [32, 10] -> [32, 5]
        print(f"  Output dtype with AMP: {output.dtype}")
    
    # Test without AMP
    with AutocastContext(enabled=False):
        output = model(x)  # [32, 10] -> [32, 5]
        print(f"  Output dtype without AMP: {output.dtype}")
    
    return True

def test_dqn_step():
    """Test DQN training step"""
    print("\n" + "="*60)
    print("Testing DQN Training Step")
    print("="*60)
    
    device = get_device()
    
    # Create Q-networks
    class SimpleQNet(nn.Module):
        def __init__(self, state_dim=4, action_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, action_dim)
        
        def forward(self, x):
            # x: [B, state_dim] -> [B, action_dim]
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    
    q_net = SimpleQNet().to(device)
    target_q_net = SimpleQNet().to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    scaler = get_scaler(enabled=device.type == 'cuda')
    
    # Create dummy batch
    batch_size = 32
    states = torch.randn(batch_size, 4, device=device)       # [32, 4]
    actions = torch.randint(0, 2, (batch_size,), device=device)  # [32]
    rewards = torch.randn(batch_size, device=device)          # [32]
    next_states = torch.randn(batch_size, 4, device=device)   # [32, 4]
    dones = torch.zeros(batch_size, device=device)            # [32]
    
    batch = (states, actions, rewards, next_states, dones)
    
    # Run training step
    loss = dqn_td_step(
        q_net, target_q_net, batch,
        gamma=0.99, huber_delta=1.0,
        optimizer=optimizer, scaler=scaler,
        amp_enabled=device.type == 'cuda'
    )
    
    print(f"  Loss: {loss:.4f}")
    print(f"  Q-net grad norm: {torch.nn.utils.clip_grad_norm_(q_net.parameters(), float('inf')):.4f}")
    
    return True

def test_compilation():
    """Test model compilation"""
    print("\n" + "="*60)
    print("Testing Model Compilation")
    print("="*60)
    
    device = get_device()
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)
    
    # Try to compile
    compiled_model = compile_if_available(model)
    
    # Test forward pass
    x = torch.randn(32, 10, device=device)  # [32, 10]
    
    # Original model
    output1 = model(x)  # [32, 10] -> [32, 5]
    
    # Compiled model
    output2 = compiled_model(x)  # [32, 10] -> [32, 5]
    
    print(f"  Model compiled: {compiled_model is not model}")
    print(f"  Output shape: {list(output2.shape)}")
    print(f"  Outputs match: {torch.allclose(output1, output2, atol=1e-5)}")
    
    return True

def save_test_results(results):
    """Save test results"""
    results_dir = Path("runs") / "header_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"results_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Test results saved to {filename}")
    return filename

def main():
    """Main execution"""
    print("="*60)
    print("RL2025 - Lecture 1: Standard Code Header Implementation")
    print("="*60)
    
    # Display header info
    print("\nStandard Code Header v1 Components:")
    print("  • Seeding and reproducibility")
    print("  • Device management (CUDA > MPS > CPU)")
    print("  • AMP context and scaler")
    print("  • DQN training step")
    print("  • Policy evaluation")
    print("  • Model compilation")
    
    # Run tests
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "device": str(get_device()),
        "tests": {}
    }
    
    tests = [
        ("seeding", test_seeding),
        ("device_management", test_device_management),
        ("amp_context", test_amp_context),
        ("dqn_step", test_dqn_step),
        ("compilation", test_compilation)
    ]
    
    all_passed = True
    for name, test_fn in tests:
        try:
            passed = test_fn()
            test_results["tests"][name] = {"passed": passed}
        except Exception as e:
            print(f"  Test {name} failed: {e}")
            test_results["tests"][name] = {"passed": False, "error": str(e)}
            all_passed = False
    
    # Save results
    save_test_results(test_results)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, result in test_results["tests"].items():
        status = "✓ PASSED" if result["passed"] else "✗ FAILED"
        print(f"  {name:20} {status}")
    
    if all_passed:
        print("\n✅ All tests passed! Standard header ready for use.")
    else:
        print("\n⚠️  Some tests failed. Check implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
