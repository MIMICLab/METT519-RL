"""
RL2025 - Lecture 1 Take-Home Lab Exercises
Course Overview and Environment Setup

Complete these exercises to reinforce today's learning.
Submit your solutions by uploading to the course repository.

Total Points: 100
Due: Before next lecture (Week 2)
"""

import torch
import torch.nn as nn
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter


# Exercise 1: Environment Verification (15 points)
def exercise1_environment_check():
    """
    Verify your development environment is properly configured.
    
    Tasks:
    1. Check Python version (must be 3.10-3.12)
    2. Verify PyTorch installation and version
    3. Test CUDA/MPS availability
    4. Create a simple tensor and move it to the optimal device
    
    Return a dictionary with all the system information.
    """
    
    # TODO: Implement environment verification
    # Hint: Use sys.version_info, torch.__version__, torch.cuda.is_available()
    
    info = {
        'python_version': None,  # (major, minor) tuple
        'pytorch_version': None,  # string
        'cuda_available': None,   # boolean
        'mps_available': None,    # boolean
        'device': None,          # optimal device string
        'test_tensor_device': None  # device where test tensor was placed
    }
    
    # Your code here
    pass
    
    return info


# Exercise 2: Reproducibility Implementation (20 points)
def exercise2_reproducibility():
    """
    Implement a comprehensive seeding function and test reproducibility.
    
    Tasks:
    1. Create a seeding function that handles all RNG sources
    2. Test that the same seed produces identical results
    3. Verify reproducibility across multiple runs
    4. Handle both CPU and GPU scenarios
    
    Returns:
    - seeding_function: Your implemented function
    - test_results: Dictionary showing reproducibility test results
    """
    
    def setup_seed(seed=42, deterministic=True):
        """
        Set all random seeds for reproducible experiments.
        
        Args:
            seed (int): Random seed value
            deterministic (bool): Whether to use deterministic algorithms
        """
        # TODO: Implement comprehensive seeding
        # Hint: Cover random, np.random, torch, torch.cuda
        pass
    
    # Test reproducibility
    def test_reproducibility():
        """Test that seeding produces identical results."""
        results = {}
        
        # Test 1: Python random
        setup_seed(42)
        python_rand1 = [random.random() for _ in range(5)]
        setup_seed(42)
        python_rand2 = [random.random() for _ in range(5)]
        results['python_random'] = python_rand1 == python_rand2
        
        # TODO: Add tests for NumPy random, PyTorch random
        # TODO: Test tensor operations give same results
        
        return results
    
    return setup_seed, test_reproducibility()


# Exercise 3: Device Management (15 points)
def exercise3_device_management():
    """
    Implement intelligent device selection and performance testing.
    
    Tasks:
    1. Create a device selection function following course standards
    2. Benchmark tensor operations on different devices (if available)
    3. Handle device compatibility issues gracefully
    4. Implement device-aware tensor creation utilities
    
    Returns:
    - get_device_function: Your device selection implementation
    - benchmark_results: Performance comparison results
    """
    
    def get_optimal_device():
        """
        Select the best available device following CUDA > MPS > CPU priority.
        
        Returns:
            torch.device: The optimal device for computations
        """
        # TODO: Implement device selection logic
        # Hint: Check CUDA first, then MPS, then fallback to CPU
        pass
    
    def benchmark_devices():
        """Benchmark matrix operations on available devices."""
        results = {}
        
        # TODO: Implement benchmarking
        # Create large tensors and time matrix multiplication
        # Test on all available devices
        # Return timing results
        
        return results
    
    return get_optimal_device, benchmark_devices()


# Exercise 4: Performance Optimization (25 points)
def exercise4_performance_optimization():
    """
    Implement and benchmark AMP and compilation optimizations.
    
    Tasks:
    1. Create a simple neural network for benchmarking
    2. Implement training loop with and without AMP
    3. Test torch.compile() if available
    4. Compare performance across different configurations
    5. Handle compatibility issues (older PyTorch versions, CPU-only)
    
    Returns:
    - benchmark_results: Performance comparison across configurations
    """
    
    class BenchmarkModel(nn.Module):
        """Simple model for performance testing."""
        def __init__(self, input_size=1024, hidden_size=2048):
            super().__init__()
            # TODO: Implement a simple feedforward network
            # 3-4 layers, with ReLU activations
            pass
        
        def forward(self, x):
            # TODO: Implement forward pass
            pass
    
    def benchmark_configuration(model, use_amp=False, use_compile=False, num_iterations=100):
        """
        Benchmark a specific configuration.
        
        Args:
            model: PyTorch model to benchmark
            use_amp: Whether to use Automatic Mixed Precision
            use_compile: Whether to compile the model
            num_iterations: Number of training steps to benchmark
            
        Returns:
            dict: Timing and performance results
        """
        # TODO: Implement benchmarking logic
        # Create dummy data, optimizer, loss function
        # Time the training loop
        # Return results including total time, steps/second
        pass
    
    # TODO: Run benchmarks for all configurations
    benchmark_results = {}
    
    return benchmark_results


# Exercise 5: Logging and Visualization (15 points)
def exercise5_logging_system():
    """
    Implement a comprehensive logging system for RL experiments.
    
    Tasks:
    1. Set up TensorBoard logging
    2. Log scalar metrics (loss, accuracy, etc.)
    3. Log histograms of model parameters
    4. Create utility functions for common logging patterns
    5. Implement experiment metadata logging
    
    Returns:
    - logger_class: Your logging implementation
    - demo_logs: Example logs created during testing
    """
    
    class ExperimentLogger:
        """Comprehensive logging for RL experiments."""
        
        def __init__(self, log_dir='runs/lab1_exercise', metadata=None):
            """
            Initialize logger.
            
            Args:
                log_dir: Directory for TensorBoard logs
                metadata: Dictionary of experiment metadata
            """
            # TODO: Initialize TensorBoard writer
            # TODO: Log experiment metadata
            pass
        
        def log_scalar(self, name, value, step):
            """Log a scalar value."""
            # TODO: Implement scalar logging
            pass
        
        def log_histogram(self, name, values, step):
            """Log a histogram of values."""
            # TODO: Implement histogram logging
            pass
        
        def log_model_parameters(self, model, step):
            """Log histograms of all model parameters."""
            # TODO: Iterate through model parameters and log histograms
            pass
        
        def close(self):
            """Close the logger."""
            # TODO: Implement cleanup
            pass
    
    # Demo usage
    def create_demo_logs():
        """Create sample logs to test the logging system."""
        # TODO: Create logger instance
        # TODO: Generate some fake training data and log it
        # TODO: Return summary of what was logged
        pass
    
    return ExperimentLogger, create_demo_logs()


# Exercise 6: Checkpoint Management (10 points)
def exercise6_checkpoint_system():
    """
    Implement a robust checkpointing system.
    
    Tasks:
    1. Save complete training state (model, optimizer, RNG states)
    2. Load and restore training state
    3. Handle different device scenarios (save on GPU, load on CPU)
    4. Implement checkpoint validation
    
    Returns:
    - checkpoint_manager: Your checkpoint management class
    """
    
    class CheckpointManager:
        """Manage model checkpoints and training state."""
        
        def __init__(self, checkpoint_dir='checkpoints'):
            """Initialize checkpoint manager."""
            # TODO: Set up checkpoint directory
            pass
        
        def save_checkpoint(self, model, optimizer, epoch, loss, filepath=None):
            """
            Save complete training state.
            
            Args:
                model: PyTorch model
                optimizer: Optimizer instance
                epoch: Current epoch number
                loss: Current loss value
                filepath: Optional custom filepath
            """
            # TODO: Create comprehensive checkpoint dictionary
            # TODO: Include RNG states for reproducibility
            # TODO: Save to file
            pass
        
        def load_checkpoint(self, filepath, model, optimizer=None, device=None):
            """
            Load training state from checkpoint.
            
            Args:
                filepath: Path to checkpoint file
                model: Model to load state into
                optimizer: Optional optimizer to restore
                device: Device to load tensors to
                
            Returns:
                dict: Loaded checkpoint information
            """
            # TODO: Load checkpoint file
            # TODO: Handle device mapping
            # TODO: Restore RNG states
            # TODO: Return checkpoint info
            pass
        
        def list_checkpoints(self):
            """List available checkpoint files."""
            # TODO: Return list of checkpoint files with metadata
            pass
    
    return CheckpointManager


# Bonus Exercise: Integration Test (Extra Credit - 10 points)
def bonus_integration_test():
    """
    Create a comprehensive test that validates your entire setup.
    
    Tasks:
    1. Test all previous exercises work together
    2. Create a mini training loop that uses all components
    3. Verify reproducibility across the entire pipeline
    4. Generate a report of all test results
    
    Returns:
    - test_results: Comprehensive test report
    """
    
    def run_integration_test():
        """Run comprehensive integration test."""
        test_results = {
            'environment': None,
            'reproducibility': None,
            'device_management': None,
            'performance': None,
            'logging': None,
            'checkpointing': None,
            'overall_score': 0
        }
        
        # TODO: Run all previous exercises
        # TODO: Test integration between components
        # TODO: Score each component
        # TODO: Generate overall score
        
        return test_results
    
    return run_integration_test()


# Main execution and grading function
def main():
    """
    Run all exercises and generate a submission report.
    
    This function will:
    1. Execute all exercises
    2. Collect results and scores
    3. Generate a submission report
    4. Save results for grading
    """
    print("=== RL2025 Lecture 1 - Lab Exercises ===\n")
    
    results = {}
    total_score = 0
    
    # Exercise 1
    print("Exercise 1: Environment Verification")
    try:
        env_info = exercise1_environment_check()
        results['exercise1'] = env_info
        # TODO: Implement scoring logic
        score1 = 0  # Replace with actual scoring
        total_score += score1
        print(f"Score: {score1}/15\n")
    except Exception as e:
        print(f"Error: {e}\n")
        results['exercise1'] = {'error': str(e)}
    
    # TODO: Run remaining exercises following same pattern
    
    # Generate submission report
    print(f"\n=== FINAL SCORE: {total_score}/100 ===")
    print("\nSubmission Report:")
    print(f"- Environment Check: {'✓' if 'exercise1' in results else '✗'}")
    print(f"- Reproducibility: {'✓' if 'exercise2' in results else '✗'}")
    print(f"- Device Management: {'✓' if 'exercise3' in results else '✗'}")
    print(f"- Performance Optimization: {'✓' if 'exercise4' in results else '✗'}")
    print(f"- Logging System: {'✓' if 'exercise5' in results else '✗'}")
    print(f"- Checkpoint Management: {'✓' if 'exercise6' in results else '✗'}")
    
    # Save results for instructor grading
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"lab1_results_{timestamp}.json"
    
    # TODO: Save results to JSON file
    
    print(f"\nResults saved to: {results_file}")
    print("Submit this file along with your completed code.")
    
    return results


if __name__ == "__main__":
    # Run the lab exercises
    main()
    
    print("\n" + "="*50)
    print("Lab 1 Exercises Complete!")
    print("="*50)
    print("\nNext Steps:")
    print("1. Review your solutions and test thoroughly")
    print("2. Ensure all code runs without errors") 
    print("3. Submit your completed file to the course repository")
    print("4. Prepare for Week 2: Deep Learning Essentials")