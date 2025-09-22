# METT519 Reinforcement Learning (Fall 2025)
_Sogang University MIMIC Lab_

This repository hosts the lecture slides and hands-on experiments for the Fall 2025 offering of METT519 Reinforcement Learning. The focus is on reproducible PyTorch implementations that track the lecture flow.

## What's Here
- Six lecture modules (`lecture01`–`lecture06`) with synchronized slides and runnable experiments
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
| 7–13 | In Development | — | Content will be uploaded after slide/code alignment |

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
│   └── ...
├── requirements.txt
└── README.md
```

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Verify the toolchain with `python lectures/lecture01/experiments/exp09_integrated_test.py`.
- For Lecture 6 determinism, export `RL_DETERMINISTIC=1` before running experiments to disable `torch.compile` and AMP.
- Slides are built with `pdflatex lecture0X/slides/lectureX.tex`; rebuild after editing.

## Working Notes
- GPU acceleration is optional; CPU-only execution is supported throughout Lecture 6 with smaller batch sizes.

## License
Released under the MIT License. See `LICENSE` for details.

_Last updated: 2025-09-23_
