#!/usr/bin/env python3
"""
Experiment 6: Logging and TensorBoard Setup
Slides: 56-60 (Logging and visualization)
Time: 1:40-1:50 (10 minutes)

Demonstrates logging setup with TensorBoard for experiment tracking.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """Wrapper for TensorBoard logging with automatic system info"""
    
    def __init__(self, log_dir=None, comment=""):
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("runs") / f"exp_{timestamp}_{comment}"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except Exception as e:
            print(f"Warning: TensorBoard not available: {e}")
            self.writer = None
            self.enabled = False
        
        # Log system info
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system and library information"""
        if not self.enabled:
            return
        
        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "timestamp": datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
        
        # Save as JSON
        with open(self.log_dir / "system_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        # Log to TensorBoard
        self.writer.add_text("system/info", json.dumps(info, indent=2), 0)
    
    def log_scalar(self, tag, value, step):
        """Log scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram of values"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_hyperparams(self, hparams, metrics=None):
        """Log hyperparameters"""
        if self.enabled:
            self.writer.add_hparams(hparams, metrics or {})
    
    def log_model(self, model, input_shape):
        """Log model graph"""
        if self.enabled:
            try:
                dummy_input = torch.randn(1, *input_shape)
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                print(f"Warning: Could not log model graph: {e}")
    
    def close(self):
        """Close the writer"""
        if self.enabled and self.writer:
            self.writer.close()

def demo_training_with_logging():
    """Demonstrate training with comprehensive logging"""
    print("\n" + "="*60)
    print("Training with Logging Demo")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = Logger(comment="demo")
    
    # Model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Log hyperparameters
    hparams = {
        "lr": 1e-3,
        "batch_size": 32,
        "hidden_dim": 64,
        "optimizer": "Adam"
    }
    logger.log_hyperparams(hparams)
    
    # Log model graph
    logger.log_model(model.cpu(), (10,))
    model.to(device)
    
    # Training loop
    print("  Training for 100 steps...")
    for step in range(100):
        # Dummy data
        X = torch.randn(32, 10, device=device)  # [32, 10]
        y = torch.randn(32, 1, device=device)   # [32, 1]
        
        # Forward pass
        output = model(X)  # [32, 10] -> [32, 1]
        loss = nn.functional.mse_loss(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        logger.log_scalar("train/loss", loss.item(), step)
        
        if step % 20 == 0:
            # Log gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.log_histogram(f"gradients/{name}", param.grad.cpu(), step)
            
            print(f"    Step {step}: loss = {loss.item():.4f}")
    
    # Final metrics
    final_metrics = {"final_loss": loss.item()}
    logger.log_hyperparams(hparams, final_metrics)
    
    logger.close()
    print(f"\n  ✓ Logs saved to {logger.log_dir}")
    print(f"  Run: tensorboard --logdir {logger.log_dir.parent}")
    
    return True

def main():
    """Main execution"""
    print("="*60)
    print("RL2025 - Lecture 1: Logging and TensorBoard")
    print("="*60)
    
    # Check TensorBoard availability
    try:
        import tensorboard
        print(f"TensorBoard version: {tensorboard.__version__}")
    except ImportError:
        print("TensorBoard not installed")
        return False
    
    # Run demo
    success = demo_training_with_logging()
    
    if success:
        print("\n✅ Experiment 6 completed successfully!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)