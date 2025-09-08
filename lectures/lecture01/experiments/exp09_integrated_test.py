#!/usr/bin/env python3
"""
Experiment 9: Integrated Smoke Test
Slides: 71-75 (Complete integration test)
Time: 2:10-2:25 (15 minutes)

Final integrated test combining all components from experiments 1-8.
This serves as a complete validation of the course infrastructure.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from datetime import datetime
import json
import time

# Import components from previous experiments (simulated here)
from exp05_standard_header import (
    setup_seed, get_device, AutocastContext, 
    get_scaler, compile_if_available, dqn_td_step
)

class IntegratedTest:
    """Complete integration test for all course components"""
    
    def __init__(self):
        self.results = {}
        self.device = None
        self.passed_tests = 0
        self.total_tests = 0
    
    def test_environment(self):
        """Test 1: Environment setup"""
        print("\n" + "="*60)
        print("Test 1: Environment Setup")
        print("="*60)
        
        self.total_tests += 1
        
        try:
            # Check Python version
            version = sys.version_info
            assert (3, 10) <= (version.major, version.minor) <= (3, 12), \
                f"Python 3.10-3.12 required, got {version.major}.{version.minor}"
            
            # Check PyTorch
            import torch
            print(f"  PyTorch version: {torch.__version__}")
            
            # Check device
            self.device = get_device()
            print(f"  Device: {self.device}")
            
            self.results["environment"] = {
                "python": f"{version.major}.{version.minor}.{version.micro}",
                "pytorch": torch.__version__,
                "device": str(self.device)
            }
            
            print("  ✓ Environment test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            print(f"  ✗ Environment test failed: {e}")
            return False
    
    def test_reproducibility(self):
        """Test 2: Reproducibility"""
        print("\n" + "="*60)
        print("Test 2: Reproducibility")
        print("="*60)
        
        self.total_tests += 1
        
        try:
            results = []
            
            for run in range(3):
                setup_seed(42)
                
                # Generate random values
                torch_tensor = torch.randn(5, device=self.device)
                np_array = np.random.randn(5)
                py_random = [random.random() for _ in range(5)]
                
                results.append({
                    'torch': torch_tensor.cpu().numpy(),
                    'numpy': np_array,
                    'python': py_random
                })
            
            # Check consistency
            torch_consistent = all(
                np.allclose(results[0]['torch'], r['torch'])
                for r in results[1:]
            )
            numpy_consistent = all(
                np.allclose(results[0]['numpy'], r['numpy'])
                for r in results[1:]
            )
            
            assert torch_consistent and numpy_consistent, \
                "Reproducibility check failed"
            
            self.results["reproducibility"] = {
                "torch_consistent": torch_consistent,
                "numpy_consistent": numpy_consistent
            }
            
            print("  ✓ Reproducibility test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            print(f"  ✗ Reproducibility test failed: {e}")
            return False
    
    def test_model_training(self):
        """Test 3: Model training with AMP"""
        print("\n" + "="*60)
        print("Test 3: Model Training")
        print("="*60)
        
        self.total_tests += 1
        
        try:
            setup_seed(42)
            
            # Simple model
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            ).to(self.device)
            
            # Try compilation
            model = compile_if_available(model)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scaler = get_scaler(enabled=self.device.type == 'cuda')
            
            # Training loop
            losses = []
            amp_enabled = self.device.type == 'cuda'
            
            for step in range(10):
                X = torch.randn(32, 10, device=self.device)  # [32, 10]
                y = torch.randn(32, 5, device=self.device)   # [32, 5]
                
                optimizer.zero_grad()
                
                with AutocastContext(enabled=amp_enabled):
                    output = model(X)  # [32, 10] -> [32, 5]
                    loss = nn.functional.mse_loss(output, y)
                
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                losses.append(loss.item())
            
            # Check training progress
            assert losses[-1] < losses[0], "Model didn't improve"
            
            self.results["training"] = {
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "amp_enabled": amp_enabled
            }
            
            print(f"  Initial loss: {losses[0]:.4f}")
            print(f"  Final loss: {losses[-1]:.4f}")
            print("  ✓ Model training test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            print(f"  ✗ Model training test failed: {e}")
            return False
    
    def test_dqn_components(self):
        """Test 4: DQN components"""
        print("\n" + "="*60)
        print("Test 4: DQN Components")
        print("="*60)
        
        self.total_tests += 1
        
        try:
            setup_seed(42)
            
            # Q-networks
            class QNetwork(nn.Module):
                def __init__(self, state_dim=4, action_dim=2):
                    super().__init__()
                    self.fc1 = nn.Linear(state_dim, 64)
                    self.fc2 = nn.Linear(64, action_dim)
                
                def forward(self, x):
                    # x: [B, state_dim] -> [B, action_dim]
                    x = torch.relu(self.fc1(x))
                    return self.fc2(x)
            
            q_net = QNetwork().to(self.device)
            target_net = QNetwork().to(self.device)
            target_net.load_state_dict(q_net.state_dict())
            
            optimizer = torch.optim.Adam(q_net.parameters())
            
            # Create batch
            batch_size = 16
            states = torch.randn(batch_size, 4, device=self.device)
            actions = torch.randint(0, 2, (batch_size,), device=self.device)
            rewards = torch.randn(batch_size, device=self.device)
            next_states = torch.randn(batch_size, 4, device=self.device)
            dones = torch.zeros(batch_size, device=self.device)
            
            batch = (states, actions, rewards, next_states, dones)
            
            # DQN step
            loss = dqn_td_step(
                q_net, target_net, batch,
                optimizer=optimizer,
                scaler=get_scaler(False),
                amp_enabled=False
            )
            
            assert isinstance(loss, float), "DQN step should return float loss"
            
            self.results["dqn"] = {
                "td_loss": loss,
                "batch_size": batch_size
            }
            
            print(f"  TD Loss: {loss:.4f}")
            print("  ✓ DQN components test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            print(f"  ✗ DQN components test failed: {e}")
            return False
    
    def test_checkpointing(self):
        """Test 5: Checkpointing"""
        print("\n" + "="*60)
        print("Test 5: Checkpointing")
        print("="*60)
        
        self.total_tests += 1
        
        try:
            setup_seed(42)
            
            # Create model
            model = nn.Linear(10, 5).to(self.device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
            # Save initial state
            initial_weight = model.weight.clone()
            
            # Train for a few steps
            for _ in range(5):
                x = torch.randn(32, 10, device=self.device)
                y = torch.randn(32, 5, device=self.device)
                
                output = model(x)
                loss = nn.functional.mse_loss(output, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Save checkpoint
            ckpt_dir = Path("runs") / "test_checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            ckpt_path = ckpt_dir / "test.pt"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': 5
            }, ckpt_path)
            
            # Create new model and load
            new_model = nn.Linear(10, 5).to(self.device)
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
            
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            new_model.load_state_dict(checkpoint['model'])
            new_optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Verify weights match
            assert torch.allclose(model.weight, new_model.weight), \
                "Checkpoint restoration failed"
            
            self.results["checkpointing"] = {
                "checkpoint_size_kb": ckpt_path.stat().st_size / 1024,
                "restoration_success": True
            }
            
            print(f"  Checkpoint size: {ckpt_path.stat().st_size / 1024:.2f} KB")
            print("  ✓ Checkpointing test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            print(f"  ✗ Checkpointing test failed: {e}")
            return False
    
    def test_logging(self):
        """Test 6: Logging infrastructure"""
        print("\n" + "="*60)
        print("Test 6: Logging Infrastructure")
        print("="*60)
        
        self.total_tests += 1
        
        try:
            # Create log directory
            log_dir = Path("runs") / "test_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "experiment": "integrated_test",
                "device": str(self.device),
                "results": self.results
            }
            
            # Save as JSON
            log_file = log_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Verify file was created
            assert log_file.exists(), "Log file not created"
            
            self.results["logging"] = {
                "log_file": str(log_file),
                "size_bytes": log_file.stat().st_size
            }
            
            print(f"  Log file: {log_file.name}")
            print("  ✓ Logging test passed")
            self.passed_tests += 1
            return True
            
        except Exception as e:
            print(f"  ✗ Logging test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("="*60)
        print("INTEGRATED SMOKE TEST")
        print("="*60)
        
        start_time = time.time()
        
        # Run tests in sequence
        tests = [
            self.test_environment,
            self.test_reproducibility,
            self.test_model_training,
            self.test_dqn_components,
            self.test_checkpointing,
            self.test_logging
        ]
        
        for test_fn in tests:
            test_fn()
        
        elapsed = time.time() - start_time
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"  Tests passed: {self.passed_tests}/{self.total_tests}")
        print(f"  Success rate: {100 * self.passed_tests / self.total_tests:.1f}%")
        print(f"  Total time: {elapsed:.2f} seconds")
        
        # Save final results
        results_file = Path("runs") / "integration_test_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "passed": self.passed_tests,
            "total": self.total_tests,
            "success_rate": self.passed_tests / self.total_tests,
            "elapsed_seconds": elapsed,
            "device": str(self.device),
            "test_results": self.results
        }
        
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n  Results saved to: {results_file}")
        
        return self.passed_tests == self.total_tests

def main():
    """Main execution"""
    print("="*60)
    print("RL2025 - Lecture 1: Integrated Smoke Test")
    print("="*60)
    print("\nThis test validates all course infrastructure components:")
    print("  • Environment setup and dependencies")
    print("  • Reproducibility and seeding")
    print("  • Model training with AMP")
    print("  • DQN components")
    print("  • Checkpointing system")
    print("  • Logging infrastructure")
    
    # Run integrated test
    tester = IntegratedTest()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All integration tests passed!")
        print("\nYour environment is ready for the RL2025 course.")
        print("\nNext steps:")
        print("  1. Review experiment files (exp01-exp09)")
        print("  2. Try modifying parameters")
        print("  3. Run experiments in Jupyter/Colab")
        print("  4. Check TensorBoard logs: tensorboard --logdir runs")
    else:
        print("\n⚠️  Some tests failed.")
        print("\nTroubleshooting:")
        print("  1. Check Python version (3.10-3.12)")
        print("  2. Install missing packages: pip install -r requirements.txt")
        print("  3. Verify CUDA installation (optional)")
        print("  4. Review failed test details above")
    
    return success

if __name__ == "__main__":
    # For standalone execution, import from exp05
    if 'exp05_standard_header' not in sys.modules:
        # Add experiments directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Try importing
        try:
            from exp05_standard_header import (
                setup_seed, get_device, AutocastContext,
                get_scaler, compile_if_available, dqn_td_step
            )
        except ImportError:
            print("Warning: Could not import from exp05_standard_header")
            print("Using fallback implementations...")
            
            # Fallback implementations
            def setup_seed(seed=42, deterministic=True):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            
            def get_device():
                if torch.cuda.is_available():
                    return torch.device('cuda')
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return torch.device('mps')
                else:
                    return torch.device('cpu')
            
            class AutocastContext:
                def __init__(self, enabled=True, dtype=torch.float16):
                    self.enabled = False
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            
            def get_scaler(enabled=True):
                class DummyScaler:
                    def scale(self, loss): return loss
                    def step(self, optimizer): optimizer.step()
                    def update(self): pass
                return DummyScaler()
            
            def compile_if_available(module, mode='default'):
                return module
            
            def dqn_td_step(q_net, target_q_net, batch, gamma=0.99, huber_delta=1.0,
                          optimizer=None, scaler=None, amp_enabled=False):
                states, actions, rewards, next_states, dones = batch
                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_q_net(next_states).max(dim=1)[0]
                    targets = rewards + gamma * (1 - dones) * next_q_values
                loss = nn.functional.smooth_l1_loss(q_values, targets)
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                return loss.item()
    
    success = main()
    sys.exit(0 if success else 1)