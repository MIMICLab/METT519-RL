#!/usr/bin/env python3
"""
Experiment 2: PyTorch Basics and Device Detection
Slides: 16-25 (PyTorch fundamentals and device management)
Time: 0:30-0:45 (15 minutes)

This experiment introduces PyTorch basics, demonstrates device detection
(CUDA > MPS > CPU), and performs basic tensor operations.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import json
from pathlib import Path

def check_pytorch_installation():
    """Check PyTorch installation and version"""
    print("="*60)
    print("PyTorch Installation Check")
    print("="*60)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Check for MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS Available: {mps_available}")
    
    if mps_available:
        print("  Apple Silicon GPU detected")
    
    return True

def get_device():
    """
    Get the best available device (CUDA > MPS > CPU)
    This is the standard device selection logic for the course.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU device")
    
    return device

def demonstrate_tensor_basics(device):
    """Demonstrate basic tensor operations with shape annotations"""
    print("\n" + "="*60)
    print("Tensor Operations Demo")
    print("="*60)
    
    # Create tensors with different methods
    print("\n1. Creating Tensors:")
    
    # From Python list
    x_list = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(f"  From list: shape {list(x_list.shape)} = [2, 3]")
    
    # From NumPy
    np_array = np.random.randn(3, 4).astype(np.float32)
    x_numpy = torch.from_numpy(np_array)
    print(f"  From NumPy: shape {list(x_numpy.shape)} = [3, 4]")
    
    # Random tensors
    x_rand = torch.randn(2, 3, 4, device=device)  # [B=2, C=3, H=4]
    print(f"  Random normal: shape {list(x_rand.shape)} = [2, 3, 4]")
    
    # Zeros and ones
    x_zeros = torch.zeros(5, 5, device=device)  # [5, 5]
    x_ones = torch.ones(3, 3, device=device)    # [3, 3]
    print(f"  Zeros: shape {list(x_zeros.shape)} = [5, 5]")
    print(f"  Ones: shape {list(x_ones.shape)} = [3, 3]")
    
    print("\n2. Tensor Operations:")
    
    # Matrix multiplication
    A = torch.randn(10, 20, device=device)  # [10, 20]
    B = torch.randn(20, 30, device=device)  # [20, 30]
    C = torch.matmul(A, B)                  # [10, 20] @ [20, 30] -> [10, 30]
    print(f"  MatMul: [{A.shape[0]}, {A.shape[1]}] @ [{B.shape[0]}, {B.shape[1]}] -> {list(C.shape)}")
    
    # Broadcasting
    x = torch.randn(5, 1, device=device)    # [5, 1]
    y = torch.randn(1, 3, device=device)    # [1, 3]
    z = x + y                                # [5, 1] + [1, 3] -> [5, 3]
    print(f"  Broadcasting: {list(x.shape)} + {list(y.shape)} -> {list(z.shape)}")
    
    # Reshape operations
    x = torch.randn(2, 3, 4, device=device)  # [2, 3, 4]
    x_reshaped = x.reshape(6, 4)             # [6, 4]
    x_viewed = x.view(-1, 12)                # [2, 12]
    x_flattened = x.flatten()                # [24]
    print(f"  Reshape: {list(x.shape)} -> reshape {list(x_reshaped.shape)}")
    print(f"  View: {list(x.shape)} -> view {list(x_viewed.shape)}")
    print(f"  Flatten: {list(x.shape)} -> flatten {list(x_flattened.shape)}")
    
    return True

def benchmark_device_performance(device):
    """Simple benchmark comparing CPU vs selected device"""
    print("\n" + "="*60)
    print("Device Performance Benchmark")
    print("="*60)
    
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        # Create random matrices
        A_cpu = torch.randn(size, size)
        B_cpu = torch.randn(size, size)
        
        A_device = A_cpu.to(device)
        B_device = B_cpu.to(device)
        
        # CPU timing
        import time
        start = time.perf_counter()
        for _ in range(10):
            C_cpu = torch.matmul(A_cpu, B_cpu)
        cpu_time = (time.perf_counter() - start) / 10
        
        # Device timing
        if device.type != 'cpu':
            # Synchronize for accurate GPU timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(10):
                C_device = torch.matmul(A_device, B_device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            device_time = (time.perf_counter() - start) / 10
            speedup = cpu_time / device_time
            
            print(f"  Size [{size}, {size}] x [{size}, {size}]:")
            print(f"    CPU: {cpu_time*1000:.2f} ms")
            print(f"    {device.type.upper()}: {device_time*1000:.2f} ms")
            print(f"    Speedup: {speedup:.2f}x")
        else:
            print(f"  Size [{size}, {size}]: {cpu_time*1000:.2f} ms (CPU only)")
    
    return True

def demonstrate_autograd():
    """Demonstrate automatic differentiation"""
    print("\n" + "="*60)
    print("Automatic Differentiation Demo")
    print("="*60)
    
    # Simple gradient computation
    x = torch.tensor([2.0], requires_grad=True)  # [1]
    y = x ** 2                                    # [1]
    z = 2 * y + 3                                 # [1]
    
    print(f"  x = {x.item():.2f} (requires_grad=True)")
    print(f"  y = x^2 = {y.item():.2f}")
    print(f"  z = 2*y + 3 = {z.item():.2f}")
    
    # Compute gradients
    z.backward()
    print(f"  dz/dx = {x.grad.item():.2f} (analytical: 4*x = {4*x.item():.2f})")
    
    # Multi-dimensional example
    print("\n  Multi-dimensional gradient:")
    X = torch.randn(3, 4, requires_grad=True)    # [3, 4]
    W = torch.randn(4, 5, requires_grad=True)    # [4, 5]
    Y = torch.matmul(X, W)                       # [3, 4] @ [4, 5] -> [3, 5]
    loss = Y.sum()                                # [] scalar
    
    loss.backward()
    
    print(f"    X shape: {list(X.shape)}")
    print(f"    W shape: {list(W.shape)}")
    print(f"    Y shape: {list(Y.shape)}")
    print(f"    X.grad shape: {list(X.grad.shape)}")
    print(f"    W.grad shape: {list(W.grad.shape)}")
    
    return True

def save_device_info(device):
    """Save device information to file"""
    device_info_dir = Path("runs") / "device_info"
    device_info_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    info = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count()
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        })
    
    filename = device_info_dir / f"device_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Device info saved to {filename}")
    return filename

def main():
    """Main execution"""
    print("="*60)
    print("RL2025 - Lecture 1: PyTorch Basics & Device Detection")
    print("="*60)
    
    # Check PyTorch installation
    check_pytorch_installation()
    
    # Get best available device
    device = get_device()
    
    # Demonstrate tensor operations
    demonstrate_tensor_basics(device)
    
    # Benchmark performance
    benchmark_device_performance(device)
    
    # Demonstrate autograd
    demonstrate_autograd()
    
    # Save device information
    save_device_info(device)
    
    print("\n✅ Experiment 2 completed successfully!")
    print(f"   Device used: {device}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)