#!/usr/bin/env python3
"""
Experiment 3: Reproducibility and Seeding
Slides: 26-35 (Reproducibility in ML experiments)
Time: 0:45-1:00 (15 minutes)

This experiment demonstrates proper seeding methodology for reproducible
experiments across random, numpy, and torch, including CUDA considerations.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime

def setup_seed(seed=42, deterministic=True):
    """
    Set seeds for all random number generators.
    This is the standard seeding function for the course.
    
    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic algorithms (may be slower)
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA seeds (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Deterministic algorithms
    if deterministic:
        torch.use_deterministic_algorithms(True)
        # May need to set environment variable for some operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Disable benchmark mode for consistent results
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print(f"✓ Seeds set to {seed}")
    print(f"  Deterministic mode: {deterministic}")
    print(f"  CUDA deterministic: {torch.backends.cudnn.deterministic if torch.cuda.is_available() else 'N/A'}")
    
    return seed

def test_python_random_reproducibility():
    """Test Python's random module reproducibility"""
    print("\n" + "="*60)
    print("Testing Python random reproducibility")
    print("="*60)
    
    results = []
    
    for run in range(3):
        random.seed(42)
        values = [random.random() for _ in range(5)]
        results.append(values)
        print(f"  Run {run+1}: {[f'{v:.4f}' for v in values]}")
    
    # Check if all runs are identical
    all_same = all(results[0] == r for r in results[1:])
    print(f"  Reproducible: {'✓' if all_same else '✗'}")
    
    return all_same

def test_numpy_reproducibility():
    """Test NumPy reproducibility"""
    print("\n" + "="*60)
    print("Testing NumPy reproducibility")
    print("="*60)
    
    results = []
    
    for run in range(3):
        np.random.seed(42)
        array = np.random.randn(2, 3)  # [2, 3]
        results.append(array.copy())
        print(f"  Run {run+1} shape {list(array.shape)}:")
        print(f"    First row: {[f'{v:.4f}' for v in array[0]]}")
    
    # Check if all runs are identical
    all_same = all(np.allclose(results[0], r) for r in results[1:])
    print(f"  Reproducible: {'✓' if all_same else '✗'}")
    
    return all_same

def test_torch_reproducibility(device):
    """Test PyTorch reproducibility on given device"""
    print("\n" + "="*60)
    print(f"Testing PyTorch reproducibility on {device}")
    print("="*60)
    
    results = []
    
    for run in range(3):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create random tensor
        tensor = torch.randn(2, 3, device=device)  # [2, 3]
        results.append(tensor.cpu().clone())
        print(f"  Run {run+1} shape {list(tensor.shape)}:")
        print(f"    First row: {[f'{v:.4f}' for v in tensor[0].cpu().numpy()]}")
    
    # Check if all runs are identical
    all_same = all(torch.allclose(results[0], r) for r in results[1:])
    print(f"  Reproducible: {'✓' if all_same else '✗'}")
    
    return all_same

def test_neural_network_reproducibility(device):
    """Test neural network training reproducibility"""
    print("\n" + "="*60)
    print("Testing Neural Network reproducibility")
    print("="*60)
    
    class SimpleNet(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)   # [B, 10] -> [B, 20]
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)  # [B, 20] -> [B, 5]
        
        def forward(self, x):
            # x: [B, 10]
            x = self.fc1(x)     # [B, 10] -> [B, 20]
            x = self.relu(x)    # [B, 20] -> [B, 20]
            x = self.fc2(x)     # [B, 20] -> [B, 5]
            return x            # [B, 5]
    
    losses = []
    
    for run in range(3):
        # Reset all seeds
        setup_seed(42, deterministic=True)
        
        # Create model and optimizer
        model = SimpleNet().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Training data (fixed seed)
        torch.manual_seed(42)
        X = torch.randn(32, 10, device=device)  # [B=32, input_dim=10]
        y = torch.randn(32, 5, device=device)   # [B=32, output_dim=5]
        
        # Training steps
        run_losses = []
        model.train()
        for step in range(5):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(X)                    # [32, 10] -> [32, 5]
            loss = criterion(output, y)          # [] scalar
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            run_losses.append(loss.item())
        
        losses.append(run_losses)
        print(f"  Run {run+1} losses: {[f'{l:.4f}' for l in run_losses]}")
    
    # Check reproducibility
    all_same = all(
        all(abs(losses[0][i] - losses[j][i]) < 1e-6 
            for i in range(len(losses[0])))
        for j in range(1, len(losses))
    )
    print(f"  Reproducible: {'✓' if all_same else '✗'}")
    
    return all_same

def test_dataloader_reproducibility(device):
    """Test DataLoader reproducibility with workers"""
    print("\n" + "="*60)
    print("Testing DataLoader reproducibility")
    print("="*60)
    
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Generate data based on index
            torch.manual_seed(idx)  # Seed based on index for reproducibility
            x = torch.randn(10)     # [10]
            y = torch.randn(5)      # [5]
            return x, y
    
    def worker_init_fn(worker_id):
        """Initialize worker with unique seed"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    results = []
    
    for run in range(3):
        # Set global seed
        setup_seed(42)
        
        # Create dataset and loader
        dataset = SimpleDataset(size=20)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for reproducibility (can test with 2)
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(42)  # Important for shuffle
        )
        
        # Collect first batch
        first_batch = next(iter(loader))
        x_batch, y_batch = first_batch  # [4, 10], [4, 5]
        
        results.append(x_batch[0].clone())  # Store first sample
        print(f"  Run {run+1} first sample: {[f'{v:.4f}' for v in x_batch[0][:3]]}")
    
    # Check reproducibility
    all_same = all(torch.allclose(results[0], r) for r in results[1:])
    print(f"  Reproducible: {'✓' if all_same else '✗'}")
    print(f"  Note: Set num_workers=0 for perfect reproducibility")
    
    return all_same

def save_reproducibility_report(results):
    """Save reproducibility test results"""
    report_dir = Path("runs") / "reproducibility"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "results": results,
        "all_tests_passed": all(results.values()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark if torch.cuda.is_available() else None
    }
    
    filename = report_dir / f"report_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Reproducibility report saved to {filename}")
    return filename

def main():
    """Main execution"""
    print("="*60)
    print("RL2025 - Lecture 1: Reproducibility and Seeding")
    print("="*60)
    
    # Get device
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Setup seeds with deterministic mode
    setup_seed(42, deterministic=True)
    
    # Run reproducibility tests
    results = {
        "python_random": test_python_random_reproducibility(),
        "numpy": test_numpy_reproducibility(),
        "torch": test_torch_reproducibility(device),
        "neural_network": test_neural_network_reproducibility(device),
        "dataloader": test_dataloader_reproducibility(device)
    }
    
    # Save report
    save_reproducibility_report(results)
    
    # Summary
    print("\n" + "="*60)
    print("REPRODUCIBILITY TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All reproducibility tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check deterministic settings.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)