#!/usr/bin/env python3
"""
RL2025 - Lecture 6: Experiment 08 - Training Optimizations

This experiment demonstrates modern PyTorch optimizations for DQN:
Automatic Mixed Precision (AMP) and torch.compile for faster training.

Learning objectives:
- Implement AMP for memory efficiency
- Use torch.compile for JIT optimization
- Benchmark performance improvements
- Understand when to use each optimization

Prerequisites: Complete DQN implementation from previous experiments
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import time
import sys

from exp01_setup import (
    setup_seed,
    device,
    amp_enabled,
    deterministic_mode_enabled,
    should_use_amp,
    should_use_compile,
    is_torch_compile_supported,
    configure_deterministic_behavior,
)

setup_seed(42)
if deterministic_mode_enabled():
    configure_deterministic_behavior()

class QNetwork(nn.Module):
    """Q-Network for value function approximation"""
    
    def __init__(self, obs_dim, n_actions, hidden_sizes=(128, 128)):
        super(QNetwork, self).__init__()
        
        layers = []
        input_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, n_actions))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def push(self, obs, action, reward, next_obs, done):
        idx = self.position
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_obs
        self.dones[idx] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        obs = torch.FloatTensor(self.observations[indices]).to(device)
        actions = torch.LongTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_obs = torch.FloatTensor(self.next_observations[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return self.size

class OptimizedDQNAgent:
    """DQN agent with modern PyTorch optimizations"""
    
    def __init__(self, env, learning_rate=1e-3, gamma=0.99,
                 buffer_size=10000, batch_size=32,
                 use_amp=False, use_compile=False):

        self.env = env
        if deterministic_mode_enabled():
            configure_deterministic_behavior()
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_amp = should_use_amp(use_amp)
        self.use_compile = should_use_compile(use_compile)

        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        # Initialize networks
        self.q_network = QNetwork(self.obs_dim, self.n_actions).to(device)
        self.target_network = QNetwork(self.obs_dim, self.n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Apply torch.compile if available and requested
        if self.use_compile and hasattr(torch, 'compile'):
            print("   Compiling networks with torch.compile()...")
            self.q_network = torch.compile(self.q_network)
            self.target_network = torch.compile(self.target_network)
        elif use_compile and not self.use_compile:
            if not hasattr(torch, 'compile'):
                print("   torch.compile not available (requires PyTorch 2.0+)")
            elif deterministic_mode_enabled():
                print("   Deterministic mode active: skipping torch.compile()")
            elif not is_torch_compile_supported():
                print("   torch.compile not supported on this platform (missing Triton backend)")
            else:
                print("   torch.compile disabled for current configuration")

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize AMP scaler if using mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("   AMP (Automatic Mixed Precision) enabled")
        elif use_amp and not self.use_amp:
            print("   Deterministic mode active: AMP disabled")
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, self.obs_dim)
        
        # Tracking
        self.update_times = []
        self.forward_times = []
        self.backward_times = []
        self.memory_usage = []
    
    def update_with_amp(self, batch):
        """Update with Automatic Mixed Precision"""
        obs, actions, rewards, next_obs, dones = batch
        
        # Mixed precision context
        with torch.cuda.amp.autocast():
            # Current Q-values
            current_q_values = self.q_network(obs)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Double DQN target
            with torch.no_grad():
                next_q_online = self.q_network(next_obs)
                next_actions = next_q_online.argmax(1, keepdim=True)
                
                next_q_target = self.target_network(next_obs)
                next_q_selected = next_q_target.gather(1, next_actions).squeeze()
                
                targets = rewards + self.gamma * (1 - dones) * next_q_selected
            
            loss = F.huber_loss(current_q_values, targets)
        
        # Backward pass with scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        
        # Optimizer step with scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def update_standard(self, batch):
        """Standard update without AMP"""
        obs, actions, rewards, next_obs, dones = batch
        
        # Current Q-values
        current_q_values = self.q_network(obs)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN target
        with torch.no_grad():
            next_q_online = self.q_network(next_obs)
            next_actions = next_q_online.argmax(1, keepdim=True)
            
            next_q_target = self.target_network(next_obs)
            next_q_selected = next_q_target.gather(1, next_actions).squeeze()
            
            targets = rewards + self.gamma * (1 - dones) * next_q_selected
        
        loss = F.huber_loss(current_q_values, targets)
        
        # Standard backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        return loss.item()
    
    def benchmark_update(self):
        """Benchmark a single update step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Time the update
        start_time = time.perf_counter()
        
        if self.use_amp:
            loss = self.update_with_amp(batch)
        else:
            loss = self.update_standard(batch)
        
        update_time = time.perf_counter() - start_time
        self.update_times.append(update_time * 1000)  # Convert to ms
        
        return loss

def demonstrate_amp():
    """Demonstrate Automatic Mixed Precision"""
    
    print("\n1. Automatic Mixed Precision (AMP):")
    
    print("\n   What is AMP?")
    print("   - Uses float16 for forward/backward passes")
    print("   - Keeps master weights in float32")
    print("   - Automatic loss scaling to prevent underflow")
    print("   - Reduces memory usage and increases speed")
    
    if not torch.cuda.is_available():
        print("\n   Note: CUDA not available, AMP demonstration limited")
        return
    
    # Compare memory usage
    print("\n   Memory usage comparison:")
    
    # Float32 tensors
    size = (1024, 1024)
    tensor_fp32 = torch.randn(size, dtype=torch.float32, device=device)
    memory_fp32 = tensor_fp32.element_size() * tensor_fp32.nelement() / (1024**2)
    
    # Float16 tensors
    tensor_fp16 = torch.randn(size, dtype=torch.float16, device=device)
    memory_fp16 = tensor_fp16.element_size() * tensor_fp16.nelement() / (1024**2)
    
    print(f"   Float32 tensor ({size}): {memory_fp32:.2f} MB")
    print(f"   Float16 tensor ({size}): {memory_fp16:.2f} MB")
    print(f"   Memory reduction: {(1 - memory_fp16/memory_fp32)*100:.1f}%")
    
    # Speed comparison
    print("\n   Speed comparison (matrix multiplication):")
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(tensor_fp32, tensor_fp32)
        _ = torch.matmul(tensor_fp16, tensor_fp16)
    
    # Float32 timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(tensor_fp32, tensor_fp32)
    torch.cuda.synchronize()
    time_fp32 = time.perf_counter() - start
    
    # Float16 timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(tensor_fp16, tensor_fp16)
    torch.cuda.synchronize()
    time_fp16 = time.perf_counter() - start
    
    print(f"   Float32 time: {time_fp32*1000:.2f} ms")
    print(f"   Float16 time: {time_fp16*1000:.2f} ms")
    print(f"   Speedup: {time_fp32/time_fp16:.2f}x")

def demonstrate_torch_compile():
    """Demonstrate torch.compile optimization"""
    
    print("\n2. Torch Compile (JIT Compilation):")
    
    print("\n   What is torch.compile?")
    print("   - JIT compilation of PyTorch models")
    print("   - Fuses operations for efficiency")
    print("   - Reduces Python overhead")
    print("   - Available in PyTorch 2.0+")
    
    if deterministic_mode_enabled():
        print("\n   Deterministic mode active: skipping torch.compile demonstration")
        return

    if not should_use_compile(True):
        if not hasattr(torch, 'compile'):
            print("\n   Note: torch.compile not available (requires PyTorch 2.0+)")
        elif not is_torch_compile_supported():
            print("\n   torch.compile not supported on this platform (missing Triton backend)")
        else:
            print("\n   torch.compile disabled for current configuration")
        return

    # Create test network
    test_net = QNetwork(4, 2, (64, 64)).to(device)
    test_input = torch.randn(32, 4).to(device)
    
    # Warmup
    for _ in range(10):
        _ = test_net(test_input)
    
    # Time uncompiled
    start = time.perf_counter()
    for _ in range(100):
        _ = test_net(test_input)
    time_uncompiled = time.perf_counter() - start
    
    # Compile network
    print("   Compiling network...")
    compiled_net = torch.compile(test_net)
    
    # Warmup compiled (first run is slow due to compilation)
    for _ in range(10):
        _ = compiled_net(test_input)
    
    # Time compiled
    start = time.perf_counter()
    for _ in range(100):
        _ = compiled_net(test_input)
    time_compiled = time.perf_counter() - start
    
    print(f"\n   Uncompiled time: {time_uncompiled*1000:.2f} ms")
    print(f"   Compiled time: {time_compiled*1000:.2f} ms")
    print(f"   Speedup: {time_uncompiled/time_compiled:.2f}x")

def benchmark_configurations():
    """Benchmark different optimization configurations"""
    
    print("\n3. Benchmarking Different Configurations:")
    
    env = gym.make("CartPole-v1")
    configs = [
        ("Baseline", False, False),
        ("AMP only", True, False),
        ("Compile only", False, True),
        ("AMP + Compile", True, True)
    ]
    
    results = []
    
    for name, use_amp, use_compile in configs:
        if use_amp and not torch.cuda.is_available():
            print(f"\n   Skipping {name} (CUDA required)")
            continue
        
        if use_compile and not is_torch_compile_supported():
            reason = "torch.compile not available"
            if hasattr(torch, 'compile'):
                reason = "torch.compile not supported on this platform (missing Triton backend)"
            print(f"\n   Skipping {name} ({reason})")
            continue
        
        print(f"\n   Testing: {name}")
        
        agent = OptimizedDQNAgent(
            env,
            use_amp=use_amp,
            use_compile=use_compile,
            buffer_size=1000,
            batch_size=32
        )
        
        # Fill buffer
        obs, _ = env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs
        
        # Benchmark updates
        n_updates = 100
        start_time = time.perf_counter()
        
        for _ in range(n_updates):
            agent.benchmark_update()
        
        total_time = time.perf_counter() - start_time
        avg_time = (total_time / n_updates) * 1000  # ms
        
        results.append((name, avg_time))
        print(f"      Average update time: {avg_time:.3f} ms")
    
    env.close()
    
    # Summary
    if results:
        print("\n   Summary:")
        baseline_time = results[0][1] if results else 1
        for name, avg_time in results:
            speedup = baseline_time / avg_time
            print(f"      {name:15s}: {avg_time:6.3f} ms (speedup: {speedup:.2f}x)")

def optimization_guidelines():
    """Provide guidelines for using optimizations"""
    
    print("\n4. Optimization Guidelines:")
    
    print("\n   When to use AMP:")
    print("   + GPU with Tensor Cores (V100, RTX series)")
    print("   + Large batch sizes")
    print("   + Memory-constrained scenarios")
    print("   - Not beneficial on CPU")
    print("   - May reduce precision (usually negligible)")
    
    print("\n   When to use torch.compile:")
    print("   + PyTorch 2.0+ available")
    print("   + Stable model architecture")
    print("   + Many forward passes")
    print("   - First compilation is slow")
    print("   - May not work with dynamic graphs")
    
    print("\n   Best practices:")
    print("   1. Profile first to identify bottlenecks")
    print("   2. Start with torch.compile (easier)")
    print("   3. Add AMP if using compatible GPU")
    print("   4. Monitor accuracy - ensure no degradation")
    print("   5. Larger models benefit more")

def memory_profiling():
    """Profile memory usage with optimizations"""
    
    print("\n5. Memory Profiling:")
    
    if not torch.cuda.is_available():
        print("   CUDA not available for memory profiling")
        return
    
    print("\n   Comparing memory usage:")
    
    # Standard precision
    torch.cuda.reset_peak_memory_stats()
    net_fp32 = QNetwork(4, 2, (256, 256)).to(device)
    opt_fp32 = optim.Adam(net_fp32.parameters())
    
    for _ in range(10):
        batch = torch.randn(128, 4).to(device)
        out = net_fp32(batch)
        loss = out.mean()
        loss.backward()
        opt_fp32.step()
        opt_fp32.zero_grad()
    
    memory_fp32 = torch.cuda.max_memory_allocated() / (1024**2)
    
    # Mixed precision
    torch.cuda.reset_peak_memory_stats()
    net_fp16 = QNetwork(4, 2, (256, 256)).to(device)
    opt_fp16 = optim.Adam(net_fp16.parameters())
    scaler = torch.cuda.amp.GradScaler()
    
    for _ in range(10):
        batch = torch.randn(128, 4).to(device)
        with torch.cuda.amp.autocast():
            out = net_fp16(batch)
            loss = out.mean()
        
        scaler.scale(loss).backward()
        scaler.step(opt_fp16)
        scaler.update()
        opt_fp16.zero_grad()
    
    memory_fp16 = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f"   Standard precision: {memory_fp32:.2f} MB")
    print(f"   Mixed precision: {memory_fp16:.2f} MB")
    print(f"   Memory saved: {memory_fp32 - memory_fp16:.2f} MB ({(1-memory_fp16/memory_fp32)*100:.1f}%)")

def main():
    print("="*50)
    print("Experiment 08: Training Optimizations")
    print("="*50)
    
    print(f"\nSystem Information:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Device: {device}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    if not hasattr(torch, 'compile'):
        print("   torch.compile available: False (requires PyTorch 2.0+)")
    elif is_torch_compile_supported():
        print("   torch.compile available: True")
    else:
        print("   torch.compile available: False (backend not supported on this platform)")
    
    # Demonstrate AMP
    demonstrate_amp()
    
    # Demonstrate torch.compile
    demonstrate_torch_compile()
    
    # Benchmark configurations
    benchmark_configurations()
    
    # Optimization guidelines
    optimization_guidelines()
    
    # Memory profiling
    memory_profiling()
    
    print("\n" + "="*50)
    print("Training optimizations completed!")
    print("Next: Complete integrated DQN test")
    print("="*50)

if __name__ == "__main__":
    main()
