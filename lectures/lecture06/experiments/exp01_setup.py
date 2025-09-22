#!/usr/bin/env python3
"""
RL2025 - Lecture 6: Experiment 01 - Environment Setup and Verification

This experiment verifies the development environment and tests basic
Gymnasium functionality for DQN implementation.

Learning objectives:
- Verify PyTorch and Gymnasium installation
- Test GPU/CPU device selection
- Understand environment interaction API
- Verify reproducibility with proper seeding (supports deterministic mode)

Prerequisites: PyTorch 2.x, Gymnasium installed
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch, platform
import gymnasium as gym
import sys

DETERMINISTIC_ENV_FLAG = "RL_DETERMINISTIC"
_DETERMINISTIC_TRUE_VALUES = {"1", "true", "yes", "on"}

def deterministic_mode_enabled() -> bool:
    """Check whether deterministic mode is requested via environment."""
    return os.environ.get(DETERMINISTIC_ENV_FLAG, "").strip().lower() in _DETERMINISTIC_TRUE_VALUES

def configure_deterministic_behavior() -> bool:
    """Configure PyTorch and CUDA backends for deterministic execution."""
    if not deterministic_mode_enabled():
        return False

    # Ensure cuBLAS uses deterministic workspace size if not already set
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if torch.cuda.is_available() and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Enforce deterministic behavior where supported
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some backends (e.g., MPS) may not fully support deterministic algorithms yet
        pass

    return True

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_mode_enabled():
        configure_deterministic_behavior()


def is_torch_compile_supported() -> bool:
    if not hasattr(torch, "compile"):
        return False
    if platform.system().lower().startswith("win"):
        try:
            import triton  # type: ignore
        except ImportError:
            return False
    return True

# Proper device selection (CUDA > MPS > CPU)
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
amp_enabled = torch.cuda.is_available() and not deterministic_mode_enabled()
setup_seed(42)

def should_use_amp(request_amp: bool = True) -> bool:
    """Return whether AMP should be enabled under the current mode."""
    return request_amp and torch.cuda.is_available() and not deterministic_mode_enabled()

def should_use_compile(request_compile: bool = True) -> bool:
    """Return whether torch.compile should be used under the current mode."""
    return request_compile and not deterministic_mode_enabled() and is_torch_compile_supported()

def main():
    print("="*50)
    print("Experiment 01: Environment Setup and Verification")
    print("="*50)
    
    # 1. System Information
    print("\n1. System Information:")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Gymnasium version: {gym.__version__}")
    print(f"   NumPy version: {np.__version__}")
    
    # 2. Device Information
    print("\n2. Device Configuration:")
    print(f"   Device: {device}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    print(f"   MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"   AMP enabled: {amp_enabled}")
    
    # 3. Test Gymnasium Environment
    print("\n3. Testing Gymnasium Environment (CartPole-v1):")
    env = gym.make("CartPole-v1")
    
    # Get environment info
    print(f"   Observation space: {env.observation_space}")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Number of actions: {env.action_space.n}")
    
    # 4. Test Environment Interaction
    print("\n4. Testing Environment Interaction:")
    
    # Reset with seed for reproducibility
    obs, info = env.reset(seed=42)
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Initial observation: {obs}")
    
    # Try seeding action space (may not be supported in all versions)
    try:
        env.action_space.seed(42)
        print("   Action space seeding: Supported")
    except:
        print("   Action space seeding: Not supported (this is OK)")
    
    # Take a few random steps
    print("\n5. Taking 5 Random Steps:")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"   Step {step+1}: action={action}, reward={reward:.2f}, done={done}")
        if done:
            print("   Episode terminated early")
            break
    
    # 6. Test Tensor Operations
    print("\n6. Testing PyTorch Tensor Operations:")
    
    # Create test tensors
    test_tensor = torch.randn(4, 128, device=device)  # Batch of 4, hidden size 128
    print(f"   Created tensor shape: {test_tensor.shape}")
    print(f"   Tensor device: {test_tensor.device}")
    
    # Test basic neural network operation
    linear = torch.nn.Linear(128, env.action_space.n).to(device)
    with torch.no_grad():
        output = linear(test_tensor)
    print(f"   Linear layer output shape: {output.shape}")
    
    # Test gradient computation
    test_input = torch.randn(1, 4, device=device, requires_grad=True)
    test_output = test_input.sum()
    test_output.backward()
    print(f"   Gradient computation: {'Success' if test_input.grad is not None else 'Failed'}")
    
    # 7. Test Reproducibility
    print("\n7. Testing Reproducibility:")
    
    # Reset environment twice with same seed
    obs1, _ = env.reset(seed=100)
    actions = []
    for _ in range(10):
        action = env.action_space.sample()
        actions.append(action)
        env.step(action)
    
    obs2, _ = env.reset(seed=100)
    reproducible = True
    for i, expected_action in enumerate(actions):
        action = env.action_space.sample()
        if action != expected_action:
            reproducible = False
            break
        env.step(action)
    
    print(f"   Same seed produces same trajectory: {reproducible}")
    print(f"   Initial observations match: {np.allclose(obs1, obs2)}")
    
    # 8. Memory Test
    print("\n8. Testing Memory Allocation:")
    
    # Allocate a replay buffer-like structure
    buffer_size = 10000
    obs_dim = env.observation_space.shape[0]
    
    try:
        buffer = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        print(f"   Allocated buffer shape: {buffer.shape}")
        print(f"   Memory size: {buffer.nbytes / 1024 / 1024:.2f} MB")
        del buffer
        print("   Memory allocation: Success")
    except MemoryError:
        print("   Memory allocation: Failed (insufficient memory)")
    
    env.close()
    
    print("\n" + "="*50)
    print("Environment setup verification completed successfully!")
    print("All components are ready for DQN implementation.")
    print("="*50)

if __name__ == "__main__":
    main()
