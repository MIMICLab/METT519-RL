# METT519 Reinforcement Learning (Fall 2025)
_Sogang University MIMIC Lab_

This repository hosts the lecture slides and hands-on experiments for the Fall 2025 offering of METT519 Reinforcement Learning. The focus is on reproducible PyTorch implementations that track the lecture flow.

## What's Here
- Eleven lecture modules (`lecture01`–`lecture11`) with synchronized slides and runnable experiments
- Integrated tests (`exp09_integrated_test.py`) that validate each lecture's code bundle
- Repository workflow guidelines in `lectures/AGENTS.md`
- Python dependency lock-in via `requirements.txt`

## Lecture Progress (Fall 2025)
| Week | Topic | Materials | Notes |
|------|-------|-----------|-------|
| 1 | Course Overview & Environment Setup | Slides, experiments | Reproducible environments, logging, checkpoints |
| 2 | Deep Learning Essentials | Slides, experiments | PyTorch tensors, autograd, training loops |
| 3 | RL Fundamentals | Slides, experiments | Agent loop, Gymnasium primer, policy/value functions |
| 4 | Mathematical Foundations | Slides, experiments | MDP formalism, Bellman equations, policy/value iteration |
| 5 | Value-Based Learning I | Slides, experiments | Tabular Q-learning, exploration schedules, integrated test bench |
| 6 | Deep Q-Networks | Slides, experiments | Replay buffers, target networks, Double DQN, AMP toggles |
| 7 | DQN Project (CartPole Case Study) | Slides, experiments | Advanced DQN variants, debugging, integrated DQN project |
| 8 | Policy Gradient Methods — REINFORCE | Slides, experiments | Baselines, reward-to-go, entropy regularization |
| 9 | Actor-Critic Methods | Slides, experiments | Advantage estimation (GAE), A2C/A3C, entropy tuning |
| 10 | Proximal Policy Optimization (PPO) | Slides, experiments | Clipped objective, GAE, tuning and diagnostics |
| 11 | Modern Directions & Capstone: RLHF, DPO, MCTS, AlphaZero | Slides, experiments | RLHF pipeline, DPO, PUCT/MCTS, AlphaZero self-play |


## Directory Layout
```
.
├── lectures/
│   ├── lecture01/  # Setup, reproducibility, tooling
│   ├── lecture02/  # PyTorch workflow
│   ├── lecture03/  # RL fundamentals
│   ├── lecture04/  # MDPs and dynamic programming
│   ├── lecture05/  # Tabular Q-learning
│   ├── lecture06/  # Deep Q-Networks
│   ├── lecture07/  # DQN project (CartPole)
│   ├── lecture08/  # REINFORCE
│   ├── lecture09/  # Actor-Critic (A2C/A3C, GAE)
│   ├── lecture10/  # PPO
│   └── lecture11/  # Modern RL: RLHF, DPO, MCTS, AlphaZero
├── LICENSE
├── requirements.txt
└── README.md
```

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Verify each lecture with `python lectures/lectureXX/experiments/exp09_integrated_test.py` (replace `XX`).
- For DQN stability (Lectures 6–7), export `RL_DETERMINISTIC=1` before running experiments to disable `torch.compile` and AMP.
- Slides are built with `pdflatex lectures/lectureXX/slides/lectureXX.tex`; rebuild after editing.

## Working Notes
- GPU acceleration is optional; CPU-only execution is supported throughout Lecture 6 with smaller batch sizes.

## License
Released under the MIT License. See `LICENSE` for details.

_Last updated: 2025-11-16_
