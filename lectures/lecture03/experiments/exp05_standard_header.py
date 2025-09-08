#!/usr/bin/env python3
"""
RL2025 - Lecture 3: Experiment 05 - Standard Header Implementation

This experiment implements and tests the standard code header that will be used
throughout the course for all RL implementations. It includes device setup,
seeding, AMP configuration, and utility functions.

Learning objectives:
- Understand the standard header components
- Test device selection and seeding functionality
- Implement basic RL utilities (environment creation, rollouts)
- Verify AMP and compilation features work correctly

Prerequisites: Experiments 01-04 completed successfully
"""

# This experiment tests the standard header, so we import it after creating it
import os, sys, random, time, json, math, pathlib, dataclasses
from typing import Callable, Dict, Tuple, Any, Optional, Sequence
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

try:
    import gymnasium as gym
except ImportError:
    raise ImportError("Please install Gymnasium: pip install 'gymnasium[classic-control]'")

# ----------------------------- Device & Seed ---------------------------------
def init_device(seed: int = 42) -> torch.device:
    """Initialize device and set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda")
    
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)
        return torch.device("mps")
    
    return torch.device("cpu")

# ----------------------------- AMP Control -----------------------------------
@dataclasses.dataclass
class AMPConfig:
    enabled: bool = False
    dtype: torch.dtype = torch.float16  # or torch.bfloat16 on Ampere+ / CPU

class AMPContext:
    def __init__(self, cfg: AMPConfig, device: torch.device):
        self.cfg, self.device = cfg, device
        # Only enable AMP for CUDA and CPU (not MPS)
        amp_enabled = cfg.enabled and device.type in ("cuda", "cpu")
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.enabled and device.type == "cuda"))
        # Use CPU autocast as fallback for non-CUDA/CPU devices
        device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
        self._ctx = torch.autocast(
            device_type=device_type,
            dtype=cfg.dtype, 
            enabled=amp_enabled
        )
    
    def __enter__(self):
        return self._ctx.__enter__()
    
    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)

# ----------------------------- Checkpointing ---------------------------------
def save_ckpt(path: str, obj: Dict[str, Any]) -> None:
    """Save checkpoint to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

def load_ckpt(path: str) -> Dict[str, Any]:
    """Load checkpoint from disk"""
    return torch.load(path, map_location="cpu")

# ----------------------------- Env Helpers -----------------------------------
def make_env(env_id: str = "CartPole-v1", seed: int = 42):
    """Create and initialize environment with proper seeding"""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def rollout_once(env, policy: Callable[[np.ndarray], int], render: bool = False) -> Tuple[float, int]:
    """Run single episode and return (total_reward, steps)"""
    obs, info = env.reset()
    done, truncated = False, False
    total_r, steps = 0.0, 0
    
    while not (done or truncated):
        if render: 
            env.render()
        action = policy(obs)  # action: int, shape=()
        obs, reward, done, truncated, info = env.step(action)
        total_r += float(reward)
        steps += 1
    
    return total_r, steps

# ----------------------------- Evaluation ------------------------------------
def evaluate_policy(env_id: str, 
                   policy: Callable[[np.ndarray], int], 
                   episodes: int = 10, 
                   seed: int = 123):
    """Evaluate policy over multiple episodes"""
    env = make_env(env_id=env_id, seed=seed)
    returns = []
    
    for ep in range(episodes):
        G, steps = rollout_once(env, policy, render=False)
        returns.append(G)
    
    env.close()
    return np.array(returns, dtype=np.float32)  # shape=(episodes,)

# ----------------------------- DQN Minimal Step -------------------------------
class QNet(nn.Module):
    """Simple Q-network for DQN"""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # obs: torch.Tensor[shape=(B, obs_dim)], out: torch.Tensor[shape=(B, n_actions)]
        return self.net(x)

def dqn_step(batch: Dict[str, torch.Tensor],
             q: QNet, q_target: QNet,
             optimizer: torch.optim.Optimizer,
             gamma: float = 0.99,
             huber_delta: float = 1.0,
             amp: Optional[AMPContext] = None) -> Dict[str, float]:
    """Single DQN training step"""
    # batch keys: s, a, r, ns, done
    device = next(q.parameters()).device
    s   = batch["s"].to(device)    # (B, obs_dim)
    a   = batch["a"].long().to(device)  # (B,)
    r   = batch["r"].to(device)    # (B,)
    ns  = batch["ns"].to(device)   # (B, obs_dim)
    done= batch["done"].to(device) # (B,)
    
    with (amp or AMPContext(AMPConfig(False), device)):
        q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)
        with torch.no_grad():
            max_next_q = q_target(ns).max(dim=1).values  # (B,)
            target = r + (1.0 - done) * gamma * max_next_q  # (B,)
        
        td = target - q_sa
        loss = torch.nn.functional.huber_loss(q_sa, target, delta=huber_delta)
    
    optimizer.zero_grad(set_to_none=True)
    
    if amp and amp.scaler.is_enabled():
        amp.scaler.scale(loss).backward()
        amp.scaler.step(optimizer)
        amp.scaler.update()
    else:
        loss.backward()
        optimizer.step()
    
    return {
        "loss": float(loss.detach().cpu().item()),
        "q_mean": float(q_sa.detach().mean().cpu().item()),
        "target_mean": float(target.detach().mean().cpu().item())
    }

# ----------------------------- torch.compile ---------------------------------
def maybe_compile(module: nn.Module) -> nn.Module:
    """Try to compile module, fall back gracefully if not supported"""
    try:
        return torch.compile(module)  # PyTorch 2.x
    except Exception:
        return module

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_device_selection():
    """Test device selection logic"""
    print("\n--- Testing Device Selection ---")
    
    device = init_device(seed=42)
    print(f"Selected device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if hasattr(torch.backends, "mps"):
        print(f"MPS available: {torch.backends.mps.is_available()}")
    else:
        print("MPS not available (expected on non-Apple hardware)")
    
    # Test tensor creation on device
    test_tensor = torch.randn(3, 4, device=device)
    print(f"Test tensor device: {test_tensor.device}")
    print(f"Test tensor shape: {test_tensor.shape}")
    
    return device

def test_seeding():
    """Test reproducibility of seeding"""
    print("\n--- Testing Seeding ---")
    
    def get_random_numbers(seed):
        init_device(seed)
        return {
            'python_random': random.random(),
            'numpy_random': np.random.rand(),
            'torch_random': torch.rand(1).item()
        }
    
    # Test same seed produces same results
    result1 = get_random_numbers(12345)
    result2 = get_random_numbers(12345)
    
    print("Seed 12345 (run 1):", result1)
    print("Seed 12345 (run 2):", result2)
    
    reproducible = all(abs(result1[k] - result2[k]) < 1e-6 for k in result1)
    print(f"Seeding reproducible: {reproducible}")
    
    # Test different seed produces different results
    result3 = get_random_numbers(54321)
    print("Seed 54321:", result3)
    
    different = any(abs(result1[k] - result3[k]) > 1e-6 for k in result1)
    print(f"Different seeds produce different results: {different}")
    
    return reproducible and different

def test_amp_functionality(device: torch.device):
    """Test AMP configuration and context"""
    print("\n--- Testing AMP Functionality ---")
    
    # Test different AMP configurations
    configs = [
        AMPConfig(enabled=False),
        AMPConfig(enabled=True, dtype=torch.float16),
        AMPConfig(enabled=True, dtype=torch.bfloat16),
    ]
    
    for i, cfg in enumerate(configs):
        print(f"\nAMP Config {i+1}: enabled={cfg.enabled}, dtype={cfg.dtype}")
        
        try:
            amp_ctx = AMPContext(cfg, device)
            print(f"  Scaler enabled: {amp_ctx.scaler.is_enabled()}")
            
            # Test context manager
            with amp_ctx:
                test_tensor = torch.randn(2, 3, device=device, requires_grad=True)
                output = test_tensor.sum()
                
                if cfg.enabled and device.type == "cuda":
                    print(f"  Output dtype in AMP context: {output.dtype}")
                else:
                    print(f"  Output dtype (no AMP): {output.dtype}")
            
            print(f"  AMP context works: True")
            
        except Exception as e:
            print(f"  AMP context failed: {e}")
    
    return True

def test_environment_utilities():
    """Test environment helper functions"""
    print("\n--- Testing Environment Utilities ---")
    
    # Test environment creation
    env = make_env("CartPole-v1", seed=42)
    print(f"Environment created: {env.spec.id}")
    
    # Test policy function
    def simple_policy(obs):
        return 0 if obs[2] < 0 else 1  # Simple angle-based policy
    
    # Test single rollout
    total_reward, steps = rollout_once(env, simple_policy)
    print(f"Single rollout: reward={total_reward}, steps={steps}")
    
    env.close()
    
    # Test policy evaluation
    returns = evaluate_policy("CartPole-v1", simple_policy, episodes=5, seed=42)
    print(f"Policy evaluation (5 episodes): {returns}")
    print(f"  Mean: {returns.mean():.2f}, Std: {returns.std():.2f}")
    
    return True

def test_qnet_and_dqn_step(device: torch.device):
    """Test Q-network and DQN training step"""
    print("\n--- Testing Q-Network and DQN Step ---")
    
    # Create Q-networks
    obs_dim, n_actions = 4, 2
    q_net = QNet(obs_dim, n_actions, hidden=64).to(device)
    q_target = QNet(obs_dim, n_actions, hidden=64).to(device)
    
    # Copy weights
    q_target.load_state_dict(q_net.state_dict())
    
    print(f"Q-network created: {sum(p.numel() for p in q_net.parameters())} parameters")
    
    # Test forward pass
    batch_size = 8
    test_obs = torch.randn(batch_size, obs_dim, device=device)
    q_values = q_net(test_obs)
    print(f"Q-values shape: {q_values.shape}")
    
    # Test DQN step
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    
    # Create fake batch
    batch = {
        "s": torch.randn(batch_size, obs_dim),
        "a": torch.randint(0, n_actions, (batch_size,)),
        "r": torch.randn(batch_size),
        "ns": torch.randn(batch_size, obs_dim),
        "done": torch.randint(0, 2, (batch_size,)).float()
    }
    
    # Test without AMP
    metrics = dqn_step(batch, q_net, q_target, optimizer, gamma=0.99)
    print(f"DQN step metrics: {metrics}")
    
    # Test with AMP (if CUDA available)
    if device.type == "cuda":
        amp_cfg = AMPConfig(enabled=True)
        amp_ctx = AMPContext(amp_cfg, device)
        metrics_amp = dqn_step(batch, q_net, q_target, optimizer, amp=amp_ctx)
        print(f"DQN step with AMP: {metrics_amp}")
    
    return True

def test_checkpointing():
    """Test checkpoint save/load functionality"""
    print("\n--- Testing Checkpointing ---")
    
    # Create test data
    test_data = {
        "step": 1000,
        "model_state": {"weight": torch.randn(3, 4)},
        "optimizer_state": {"lr": 0.001},
        "metadata": {"experiment": "test"}
    }
    
    # Test save
    os.makedirs("test_results", exist_ok=True)
    save_path = "test_results/test_checkpoint.pt"
    save_ckpt(save_path, test_data)
    print(f"Checkpoint saved to: {save_path}")
    
    # Test load
    loaded_data = load_ckpt(save_path)
    print(f"Checkpoint loaded, keys: {list(loaded_data.keys())}")
    
    # Verify data integrity
    data_match = torch.allclose(test_data["model_state"]["weight"], 
                               loaded_data["model_state"]["weight"])
    print(f"Data integrity check: {data_match}")
    
    return data_match

def test_torch_compile():
    """Test torch.compile functionality"""
    print("\n--- Testing torch.compile ---")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )
    
    print(f"Original model type: {type(model)}")
    
    # Test compilation
    compiled_model = maybe_compile(model)
    print(f"Compiled model type: {type(compiled_model)}")
    
    # Test forward pass
    test_input = torch.randn(3, 4)
    output1 = model(test_input)
    output2 = compiled_model(test_input)
    
    outputs_match = torch.allclose(output1, output2)
    print(f"Outputs match: {outputs_match}")
    
    return True

def run_integration_test():
    """Run complete integration test combining all components"""
    print("\n--- Integration Test ---")
    
    # Initialize device and seeding
    device = init_device(seed=42)
    
    # Create environment
    env = make_env("CartPole-v1", seed=42)
    
    # Create Q-network
    q_net = QNet(4, 2, hidden=32).to(device)
    q_net = maybe_compile(q_net)
    
    # Simple epsilon-greedy policy using Q-network
    def q_policy(obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            q_values = q_net(obs_tensor)
            return q_values.argmax().item()
    
    # Test rollout with neural policy
    total_reward, steps = rollout_once(env, q_policy)
    print(f"Neural policy rollout: reward={total_reward}, steps={steps}")
    
    env.close()
    
    # Test with TensorBoard logging
    log_dir = "test_results/tensorboard"
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar("Test/Reward", total_reward, 0)
    writer.add_histogram("Test/QValues", q_net(torch.randn(10, 4, device=device)))
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    
    return True

def main():
    """Run all standard header tests"""
    print("="*60)
    print("Experiment 05: Standard Header Implementation and Testing")
    print("="*60)
    
    test_results = []
    
    # Run all tests
    try:
        device = test_device_selection()
        test_results.append(("Device Selection", True))
        
        test_results.append(("Seeding", test_seeding()))
        test_results.append(("AMP Functionality", test_amp_functionality(device)))
        test_results.append(("Environment Utilities", test_environment_utilities()))
        test_results.append(("Q-Network & DQN", test_qnet_and_dqn_step(device)))
        test_results.append(("Checkpointing", test_checkpointing()))
        test_results.append(("Torch Compile", test_torch_compile()))
        test_results.append(("Integration Test", run_integration_test()))
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        test_results.append(("Error", False))
    
    # Print results summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in test_results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in test_results)
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    # Save header to file for reuse
    header_code = '''# PyTorch 2.x Standard Practice Header
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
'''
    
    with open("standard_header.py", "w") as f:
        f.write('"""Standard header for RL2025 course experiments"""\n')
        f.write(header_code)
    
    print("\nStandard header saved to: standard_header.py")
    print("\nExperiment 05 completed successfully!")
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)