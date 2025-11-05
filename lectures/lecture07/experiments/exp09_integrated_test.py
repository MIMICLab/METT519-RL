#!/usr/bin/env python3
"""
RL2025 - Lecture 7: Experiment 09 - Integrated Test (Smoke Check)

This script provides a fast, end-to-end smoke test for Lecture 7 code.
It validates that core components run without errors and basic invariants
hold (tensor shapes, loss computation, replay buffer sampling, short train).

Learning objectives:
- Verify imports and device selection work
- Exercise DQN forward/loss/optimization for a few steps
- Confirm Gymnasium API alignment and seeding

Prerequisites: PyTorch 2.x, Gymnasium, NumPy
"""

from __future__ import annotations

import os
import sys
import time
import random
from pathlib import Path
from typing import Tuple

os.environ.setdefault("KMP_CONSTRUCT_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

BOOL_TYPES = (bool, np.bool_) + ((np.bool8,) if hasattr(np, "bool8") else ())


def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)


def import_exp05() -> Tuple[object, object]:
    """Import DQN and ReplayBuffer from exp05_training_loop.py.
    Returns (DQN, ReplayBuffer) classes.
    """
    exp_dir = Path(__file__).resolve().parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    mod = __import__('exp05_training_loop')
    return mod.DQN, mod.ReplayBuffer


def tiny_train(env_name: str = "CartPole-v1", steps: int = 1000) -> None:
    """Run a very short DQN training to check runtime health."""
    setup_seed(42)
    DQN, ReplayBuffer = import_exp05()

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = DQN(state_dim, action_dim).to(device)
    target = DQN(state_dim, action_dim).to(device)
    target.load_state_dict(policy.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    memory = ReplayBuffer(capacity=2000)

    epsilon = 1.0
    gamma = 0.99
    batch_size = 32

    state, _ = env.reset(seed=42)
    ep_reward = 0.0
    updates = 0
    start = time.time()

    for t in range(steps):
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy(s).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward

        # One update step when ready
        if len(memory) >= batch_size:
            states, actions, rewards, next_states, dones = memory.sample(batch_size)

            current_q = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target(next_states).max(1)[0]
                targets = rewards + gamma * next_q * (1 - dones)

            assert current_q.shape == targets.shape == (batch_size,), "Q/target shape mismatch"

            loss = F.smooth_l1_loss(current_q, targets)
            assert torch.isfinite(loss), "Loss not finite"

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            updates += 1

        if done:
            state, _ = env.reset()
            ep_reward = 0.0
            # Simple epsilon decay per episode
            epsilon = max(0.05, epsilon * 0.995)

        # Periodic hard target sync
        if t > 0 and t % 200 == 0:
            target.load_state_dict(policy.state_dict())

    env.close()
    elapsed = time.time() - start
    print(f"Integrated test: {steps} steps, {updates} updates, time {elapsed:.2f}s, device={device}")


def main() -> None:
    print('=' * 50)
    print('Lecture 7 - Integrated Test (Smoke Check)')
    print('=' * 50)
    print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {device}")

    # Quick API checks
    env = gym.make('CartPole-v1')
    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray) and obs.shape == (4,), "Unexpected observation shape"
    ns, r, term, trunc, info = env.step(env.action_space.sample())
    assert isinstance(term, BOOL_TYPES) and isinstance(trunc, BOOL_TYPES), "Gymnasium step API mismatch"
    env.close()
    print('Gymnasium API check: OK')

    # Tiny training
    tiny_train(steps=600)


if __name__ == "__main__":
    main()
