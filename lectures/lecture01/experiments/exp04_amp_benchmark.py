#!/usr/bin/env python3
"""
Experiment 4: AMP and Compilation Benchmarks
Slides: 36-45 (Performance optimization with AMP and torch.compile)
Time: 1:00-1:20 (20 minutes with 5 min break)

This experiment benchmarks Automatic Mixed Precision (AMP) and torch.compile
across different configurations to demonstrate performance improvements.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from contextlib import contextmanager
import platform

# Setup device
def get_device():
    """Get best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# torch.compile support detection -------------------------------------------------

def can_use_torch_compile() -> bool:
    """Return True when torch.compile is available and its backend is usable."""
    if not hasattr(torch, "compile"):
        return False

    if platform.system().lower().startswith("win"):
        try:
            import triton  # type: ignore
        except ImportError:
            return False

    return True

# Timer context manager
@contextmanager
def timer(name="Operation", record=None):
    """Simple timer context manager that can record elapsed milliseconds."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    elapsed_ms = elapsed * 1000
    if record is not None:
        record(elapsed_ms)
    print(f"  {name}: {elapsed_ms:.2f} ms")

class BenchmarkModel(nn.Module):
    """Model for benchmarking - simulates a typical deep network"""
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=256, num_layers=4):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(dims[i+1]))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [B, input_dim] -> [B, output_dim]
        return self.network(x)

def benchmark_standard_training(model, device, batch_size=128, num_steps=50):
    """Benchmark standard training without optimizations"""
    print("\n  Standard Training (FP32, no compile):")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Dummy data
    torch.manual_seed(42)
    X = torch.randn(batch_size, 512, device=device)  # [B, 512]
    y = torch.randn(batch_size, 256, device=device)  # [B, 256]
    
    model.train()
    losses = []
    
    # Warmup
    for _ in range(5):
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Actual benchmark
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        # Forward pass
        output = model(X)                    # [B, 512] -> [B, 256]
        loss = criterion(output, y)          # [] scalar
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start_time
    avg_step_time = total_time / num_steps * 1000  # ms
    
    print(f"    Total time: {total_time:.2f} s")
    print(f"    Avg step: {avg_step_time:.2f} ms")
    print(f"    Final loss: {losses[-1]:.4f}")
    
    return avg_step_time, losses

def benchmark_amp_training(model, device, batch_size=128, num_steps=50):
    """Benchmark training with Automatic Mixed Precision"""
    print("\n  AMP Training (FP16/BF16, no compile):")
    
    # Check if AMP is available
    amp_enabled = device.type == 'cuda'
    
    if not amp_enabled:
        print("    AMP not available on this device (CPU/MPS)")
        return None, None
    
    from torch.cuda.amp import autocast, GradScaler
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    # Dummy data
    torch.manual_seed(42)
    X = torch.randn(batch_size, 512, device=device)  # [B, 512]
    y = torch.randn(batch_size, 256, device=device)  # [B, 256]
    
    model.train()
    losses = []
    
    # Warmup
    for _ in range(5):
        with autocast():
            output = model(X)
            loss = criterion(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Actual benchmark
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        # Forward pass with autocast
        with autocast():
            output = model(X)                # [B, 512] -> [B, 256]
            loss = criterion(output, y)      # [] scalar
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        losses.append(loss.item())
    
    torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start_time
    avg_step_time = total_time / num_steps * 1000  # ms
    
    print(f"    Total time: {total_time:.2f} s")
    print(f"    Avg step: {avg_step_time:.2f} ms")
    print(f"    Final loss: {losses[-1]:.4f}")
    
    return avg_step_time, losses

def benchmark_compiled_training(model, device, batch_size=128, num_steps=50):
    """Benchmark training with torch.compile"""
    print("\n  Compiled Training (FP32, torch.compile):")
    
    # Check if torch.compile is available (PyTorch 2.0+)
    if not can_use_torch_compile():
        if hasattr(torch, 'compile'):
            print("    torch.compile not supported on this platform (Triton backend missing)")
        else:
            print("    torch.compile not available (requires PyTorch 2.0+)")
        return None, None

    model = model.to(device)
    
    # Compile the model
    try:
        compiled_model = torch.compile(model, mode='default')
    except Exception as e:
        print(f"    Compilation failed: {e}")
        return None, None
    
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Dummy data
    torch.manual_seed(42)
    X = torch.randn(batch_size, 512, device=device)  # [B, 512]
    y = torch.randn(batch_size, 256, device=device)  # [B, 256]
    
    compiled_model.train()
    losses = []
    
    # Warmup (important for compiled models)
    print("    Warming up compiled model...")
    for _ in range(10):
        output = compiled_model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Actual benchmark
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        # Forward pass
        output = compiled_model(X)           # [B, 512] -> [B, 256]
        loss = criterion(output, y)          # [] scalar
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start_time
    avg_step_time = total_time / num_steps * 1000  # ms
    
    print(f"    Total time: {total_time:.2f} s")
    print(f"    Avg step: {avg_step_time:.2f} ms")
    print(f"    Final loss: {losses[-1]:.4f}")
    
    return avg_step_time, losses

def benchmark_amp_compiled_training(model, device, batch_size=128, num_steps=50):
    """Benchmark training with both AMP and torch.compile"""
    print("\n  AMP + Compiled Training (FP16/BF16, torch.compile):")
    
    # Check requirements
    if device.type != 'cuda':
        print("    AMP not available on this device")
        return None, None
    
    if not can_use_torch_compile():
        if hasattr(torch, 'compile'):
            print("    torch.compile not supported on this platform (Triton backend missing)")
        else:
            print("    torch.compile not available")
        return None, None

    from torch.cuda.amp import autocast, GradScaler

    model = model.to(device)
    
    # Compile the model
    try:
        compiled_model = torch.compile(model, mode='default')
    except Exception as e:
        print(f"    Compilation failed: {e}")
        return None, None
    
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    # Dummy data
    torch.manual_seed(42)
    X = torch.randn(batch_size, 512, device=device)  # [B, 512]
    y = torch.randn(batch_size, 256, device=device)  # [B, 256]
    
    compiled_model.train()
    losses = []
    
    # Warmup
    print("    Warming up compiled model with AMP...")
    for _ in range(10):
        with autocast():
            output = compiled_model(X)
            loss = criterion(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Actual benchmark
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        # Forward pass with autocast
        with autocast():
            output = compiled_model(X)        # [B, 512] -> [B, 256]
            loss = criterion(output, y)      # [] scalar
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        losses.append(loss.item())
    
    torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start_time
    avg_step_time = total_time / num_steps * 1000  # ms
    
    print(f"    Total time: {total_time:.2f} s")
    print(f"    Avg step: {avg_step_time:.2f} ms")
    print(f"    Final loss: {losses[-1]:.4f}")
    
    return avg_step_time, losses

def benchmark_matmul_operations(device):
    """Benchmark matrix multiplication with different optimizations"""
    print("\n" + "="*60)
    print("Matrix Multiplication Benchmarks")
    print("="*60)
    
    sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
    results = {}
    
    for size in sizes:
        print(f"\n  Matrix size: {size[0]}x{size[1]}")
        size_key = f"{size[0]}x{size[1]}"
        results[size_key] = {}
        
        # Create matrices
        A = torch.randn(size, device=device)
        B = torch.randn(size, device=device)
        
        # Standard matmul
        with timer("Standard (FP32)", lambda ms, key=size_key: results[key].__setitem__('fp32_ms', ms)):
            for _ in range(10):
                C = torch.matmul(A, B)
        
        # With TF32 (if available)
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            with timer("TF32 precision", lambda ms, key=size_key: results[key].__setitem__('tf32_ms', ms)):
                for _ in range(10):
                    C = torch.matmul(A, B)
            torch.set_float32_matmul_precision('highest')
        
        # With AMP (if available)
        if device.type == 'cuda':
            from torch.cuda.amp import autocast
            with timer("AMP (FP16)", lambda ms, key=size_key: results[key].__setitem__('amp_ms', ms)):
                with autocast():
                    for _ in range(10):
                        C = torch.matmul(A, B)

    return results

def save_benchmark_results(results):
    """Save benchmark results to file"""
    results_dir = Path("runs") / "benchmarks"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = results_dir / f"amp_compile_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Benchmark results saved to {filename}")
    return filename

def main():
    """Main execution"""
    print("="*60)
    print("RL2025 - Lecture 1: AMP and Compilation Benchmarks")
    print("="*60)
    
    # Get device
    device = get_device()
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Matrix multiplication benchmarks
    matmul_results = benchmark_matmul_operations(device)
    
    # Model training benchmarks
    print("\n" + "="*60)
    print("Training Benchmarks (4-layer network)")
    print("="*60)
    
    results = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "timestamp": datetime.now().isoformat(),
        "matmul": matmul_results
    }
    
    # Benchmark different configurations
    configurations = [
        ("standard", benchmark_standard_training),
        ("amp", benchmark_amp_training),
        ("compiled", benchmark_compiled_training),
        ("amp_compiled", benchmark_amp_compiled_training)
    ]
    
    for name, benchmark_fn in configurations:
        # Create fresh model for each benchmark
        torch.manual_seed(42)
        model = BenchmarkModel()
        
        avg_time, losses = benchmark_fn(model, device)
        
        if avg_time is not None:
            results[name] = {
                "avg_step_ms": avg_time,
                "final_loss": losses[-1] if losses else None
            }
    
    # Calculate speedups
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    
    if "standard" in results and results["standard"].get("avg_step_ms"):
        baseline = results["standard"]["avg_step_ms"]
        print(f"  Baseline (FP32): {baseline:.2f} ms/step")
        
        for name in ["amp", "compiled", "amp_compiled"]:
            if name in results and results[name] and results[name].get("avg_step_ms"):
                speedup = baseline / results[name]["avg_step_ms"]
                print(f"  {name:15} {results[name]['avg_step_ms']:.2f} ms/step (speedup: {speedup:.2f}x)")
    
    # Save results
    save_benchmark_results(results)
    
    print("\n✅ Experiment 4 completed successfully!")
    
    # Recommendations
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)
    if device.type == 'cuda':
        print("  ✓ Use AMP for memory efficiency and speed")
        print("  ✓ Use torch.compile for additional optimizations")
        print("  ✓ Combine both for maximum performance")
    else:
        print("  ℹ Running on CPU/MPS - limited optimization options")
        print("  ℹ Consider using CUDA device for best performance")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
