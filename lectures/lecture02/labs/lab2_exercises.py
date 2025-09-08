"""
RL2025 - Lecture 2 Take-Home Lab Exercises
Deep Learning Essentials

Complete these exercises to reinforce today's learning.
Submit your solutions by uploading to the course repository.

Total Points: 100
Due: Before next lecture (Week 3)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random, numpy as np


def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# Exercise 1: Tensor Mastery (15 points)
def exercise1_tensor_mastery():
    """
    Implement a function that takes an input tensor x[B,D] and:
    1. Normalizes each feature to zero mean and unit variance.
    2. Returns both the normalized tensor and the per-feature mean/std.

    Returns:
    - x_norm[B,D], mean[D], std[D]
    """
    # TODO: Implement
    pass


# Exercise 2: Autograd Sanity (15 points)
def exercise2_autograd_sanity():
    """
    Create a scalar-valued function f(x) = x^3 + 2x and compute df/dx
    at x=2 using autograd. Return the gradient as a Python float.
    """
    # TODO: Implement
    pass


# Exercise 3: Custom Module (20 points)
def exercise3_custom_module():
    """
    Implement an nn.Module named SmallMLP with constructor (in_dim=2,
    hidden=16, out_dim=2). Forward should produce logits[B, out_dim].
    Return an instance and a dummy forward pass on random input.

    Returns:
    - model: nn.Module
    - logits: torch.Tensor of shape [5, out_dim]
    """
    # TODO: Implement
    pass


# Exercise 4: Training Loop (20 points)
def exercise4_training_loop():
    """
    Train SmallMLP from Exercise 3 for 5 epochs on a synthetic dataset
    (binary classification). Use CrossEntropyLoss and Adam. Return the
    final validation accuracy as a float in [0,1].
    """
    # TODO: Implement
    pass


# Exercise 5: Regularization and Init (20 points)
def exercise5_regularization_init():
    """
    Apply Kaiming initialization and weight decay to the model from
    Exercise 3. Train for 5 epochs and compare validation accuracy with
    and without weight decay (e.g., 0 vs 1e-4). Return a dict:

    {"no_wd": acc0, "wd": acc1}
    """
    # TODO: Implement
    pass


# Exercise 6: AMP or Scheduler (10 points)
def exercise6_amp_or_scheduler():
    """
    If CUDA is available, enable AMP and run a few training steps,
    returning True if it completes without error. Otherwise, set up a
    StepLR scheduler and run 3 epochs, returning the list of lrs used.
    """
    # TODO: Implement
    pass


# Bonus: Tiny-batch Overfit (10 points)
def bonus_tiny_batch_overfit():
    """
    Demonstrate overfitting a tiny batch (e.g., 8 samples) to near 100%
    training accuracy within 50 iterations. Return the final training
    accuracy as a float.
    """
    # TODO: Implement
    pass


def main():
    setup_seed(42)
    results = {}
    try:
        results['ex1'] = 'OK' if exercise1_tensor_mastery() is None else 'OK'
    except Exception as e:
        results['ex1'] = f'ERROR: {e}'
    try:
        results['ex2'] = exercise2_autograd_sanity()
    except Exception as e:
        results['ex2'] = f'ERROR: {e}'
    try:
        results['ex3'] = 'OK' if exercise3_custom_module() is None else 'OK'
    except Exception as e:
        results['ex3'] = f'ERROR: {e}'
    try:
        results['ex4'] = exercise4_training_loop()
    except Exception as e:
        results['ex4'] = f'ERROR: {e}'
    try:
        results['ex5'] = exercise5_regularization_init()
    except Exception as e:
        results['ex5'] = f'ERROR: {e}'
    try:
        results['ex6'] = exercise6_amp_or_scheduler()
    except Exception as e:
        results['ex6'] = f'ERROR: {e}'
    try:
        results['bonus'] = bonus_tiny_batch_overfit()
    except Exception as e:
        results['bonus'] = f'ERROR: {e}'

    print("Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()

