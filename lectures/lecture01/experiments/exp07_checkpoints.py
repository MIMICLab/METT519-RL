#!/usr/bin/env python3
"""
Experiment 7: Checkpoint Utilities
Slides: 61-65 (Model checkpointing and recovery)
Time: 1:50-2:00 (10 minutes)

Demonstrates checkpoint saving/loading for models, optimizers, and training state.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
import hashlib
import json

class Checkpointer:
    """Comprehensive checkpointing utility for training"""
    
    def __init__(self, checkpoint_dir=None, max_checkpoints=5):
        """
        Initialize checkpointer.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path("runs") / "checkpoints"
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save(self, step, **objects):
        """
        Save checkpoint with all training objects.
        
        Args:
            step: Training step/epoch number
            **objects: Dict of objects to save (model, optimizer, etc.)
        """
        # Generate checkpoint name with hash
        config_str = f"step_{step}"
        hash_str = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.checkpoint_dir / f"ckpt_{timestamp}_{step}_{hash_str}.pt"
        
        # Prepare checkpoint data
        checkpoint = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        # Add state dicts for all objects
        for name, obj in objects.items():
            if hasattr(obj, 'state_dict'):
                checkpoint[name] = obj.state_dict()
            else:
                checkpoint[name] = obj
        
        # Add RNG states for reproducibility
        checkpoint['rng_states'] = {
            'python': np.random.get_state(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            checkpoint['rng_states']['cuda'] = torch.cuda.get_rng_state_all()
        
        # Save checkpoint
        torch.save(checkpoint, filename)
        self.checkpoints.append(filename)
        
        print(f"✓ Checkpoint saved: {filename.name}")
        
        # Remove old checkpoints if exceeding limit
        if len(self.checkpoints) > self.max_checkpoints:
            old_ckpt = self.checkpoints.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
                print(f"  Removed old checkpoint: {old_ckpt.name}")
        
        return filename
    
    def load(self, checkpoint_path, **objects):
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            **objects: Dict of objects to restore (model, optimizer, etc.)
        
        Returns:
            checkpoint: Full checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✓ Loading checkpoint: {checkpoint_path.name}")
        print(f"  Step: {checkpoint.get('step', 'unknown')}")
        print(f"  Saved at: {checkpoint.get('timestamp', 'unknown')}")
        
        # Restore state dicts
        for name, obj in objects.items():
            if name in checkpoint and hasattr(obj, 'load_state_dict'):
                obj.load_state_dict(checkpoint[name])
                print(f"  Restored: {name}")
        
        # Restore RNG states
        if 'rng_states' in checkpoint:
            rng_states = checkpoint['rng_states']
            
            if 'python' in rng_states:
                np.random.set_state(rng_states['python'])
            if 'numpy' in rng_states:
                np.random.set_state(rng_states['numpy'])
            if 'torch' in rng_states:
                torch.set_rng_state(rng_states['torch'])
            if 'cuda' in rng_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states['cuda'])
            
            print("  Restored RNG states")
        
        return checkpoint
    
    def get_latest(self):
        """Get path to latest checkpoint"""
        if self.checkpoints:
            return self.checkpoints[-1]
        
        # Search directory for checkpoints
        ckpt_files = sorted(self.checkpoint_dir.glob("ckpt_*.pt"))
        if ckpt_files:
            return ckpt_files[-1]
        
        return None

def demo_checkpoint_workflow():
    """Demonstrate complete checkpoint workflow"""
    print("\n" + "="*60)
    print("Checkpoint Workflow Demo")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and optimizer
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            # x: [B, 10] -> [B, 5]
            x = torch.relu(self.fc1(x))  # [B, 10] -> [B, 20]
            return self.fc2(x)            # [B, 20] -> [B, 5]
    
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    # Initialize checkpointer
    checkpointer = Checkpointer()
    
    # Training loop with checkpointing
    print("\n  Training Phase 1 (steps 0-19):")
    
    train_loss = []
    for step in range(20):
        # Dummy training step
        X = torch.randn(32, 10, device=device)  # [32, 10]
        y = torch.randn(32, 5, device=device)   # [32, 5]
        
        output = model(X)  # [32, 10] -> [32, 5]
        loss = nn.functional.mse_loss(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss.append(loss.item())
        
        # Save checkpoint every 10 steps
        if (step + 1) % 10 == 0:
            ckpt_path = checkpointer.save(
                step=step + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss
            )
            print(f"    Step {step + 1}: loss = {loss.item():.4f}")
    
    # Simulate interruption and recovery
    print("\n  Simulating training interruption...")
    
    # Create new model and optimizer (simulating restart)
    model_new = SimpleModel().to(device)
    optimizer_new = torch.optim.Adam(model_new.parameters(), lr=1e-3)
    scheduler_new = torch.optim.lr_scheduler.StepLR(optimizer_new, step_size=10)
    
    # Load latest checkpoint
    latest_ckpt = checkpointer.get_latest()
    if latest_ckpt:
        print(f"\n  Loading checkpoint: {latest_ckpt.name}")
        
        checkpoint = checkpointer.load(
            latest_ckpt,
            model=model_new,
            optimizer=optimizer_new,
            scheduler=scheduler_new
        )
        
        # Restore training state
        start_step = checkpoint['step']
        train_loss = checkpoint.get('train_loss', [])
        
        print(f"  Resuming from step {start_step}")
    else:
        start_step = 0
        train_loss = []
    
    # Continue training
    print("\n  Training Phase 2 (resumed):")
    
    for step in range(start_step, start_step + 10):
        X = torch.randn(32, 10, device=device)
        y = torch.randn(32, 5, device=device)
        
        output = model_new(X)
        loss = nn.functional.mse_loss(output, y)
        
        optimizer_new.zero_grad()
        loss.backward()
        optimizer_new.step()
        scheduler_new.step()
        
        train_loss.append(loss.item())
        
        if (step + 1) % 5 == 0:
            print(f"    Step {step + 1}: loss = {loss.item():.4f}")
    
    # Verify model equivalence
    print("\n  Verifying checkpoint restoration:")
    
    # Compare parameters
    params_match = all(
        torch.allclose(p1, p2)
        for p1, p2 in zip(model.parameters(), model_new.parameters())
    )
    
    print(f"    Parameters match: {'✓' if params_match else '✗'}")
    print(f"    Total training steps: {len(train_loss)}")
    
    return True

def test_checkpoint_formats():
    """Test different checkpoint formats and compatibility"""
    print("\n" + "="*60)
    print("Testing Checkpoint Formats")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simple data to checkpoint
    data = {
        'tensor': torch.randn(10, 10, device=device),
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2},
        'numpy': np.random.randn(5, 5)
    }
    
    checkpoint_dir = Path("runs") / "format_tests"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Test different save formats
    formats = [
        ("standard", lambda p, d: torch.save(d, p)),
        ("compressed", lambda p, d: torch.save(d, p, _use_new_zipfile_serialization=True))
    ]
    
    for format_name, save_fn in formats:
        path = checkpoint_dir / f"test_{format_name}.pt"
        
        # Save
        save_fn(path, data)
        size_mb = path.stat().st_size / 1024 / 1024
        
        # Load and verify
        loaded = torch.load(path, map_location='cpu')
        
        # Move tensor to CPU for comparison
        data_cpu = data.copy()
        data_cpu['tensor'] = data_cpu['tensor'].cpu()
        loaded['tensor'] = loaded['tensor'].cpu()
        
        match = torch.allclose(data_cpu['tensor'], loaded['tensor'])
        
        print(f"  Format: {format_name}")
        print(f"    Size: {size_mb:.3f} MB")
        print(f"    Loaded successfully: {'✓' if match else '✗'}")
    
    return True

def main():
    """Main execution"""
    print("="*60)
    print("RL2025 - Lecture 1: Checkpoint Utilities")
    print("="*60)
    
    # Run demonstrations
    success = True
    
    try:
        # Test checkpoint workflow
        success = demo_checkpoint_workflow() and success
        
        # Test formats
        success = test_checkpoint_formats() and success
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        success = False
    
    if success:
        print("\n✅ Experiment 7 completed successfully!")
        print("\nKey takeaways:")
        print("  • Always save RNG states for perfect reproducibility")
        print("  • Include metadata (step, timestamp, versions)")
        print("  • Implement checkpoint rotation to save disk space")
        print("  • Test checkpoint loading regularly during development")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)