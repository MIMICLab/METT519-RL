# METT519 - Reinforcement Learning
**Sogang University MIMIC Lab**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Course Overview

This repository contains the complete course materials for **METT519 Reinforcement Learning**, a comprehensive 13-week graduate-level course that bridges theory and practice in modern reinforcement learning.

### Course Philosophy
- **30% Theory, 70% Practice**: Strong emphasis on hands-on implementation
- **PyTorch-First**: All implementations use PyTorch 2.x with CUDA/MPS support
- **Reproducible Research**: Fixed seeds, documented environments, version control
- **Industry Standards**: Professional coding practices and project structure

### Key Features
- 🎯 **13 Weekly Lectures** with 3-hour format (180 minutes each)
- 📊 **Progressive Difficulty**: From basics to state-of-the-art methods
- 🔬 **Hands-on Experiments**: 9-10 experiments per lecture
- 📝 **Take-Home Labs**: Weekly exercises with solutions
- 💻 **Cross-Platform**: Works on Windows, macOS, and Linux

## Repository Structure

```
METT519-Reinforcement Learning/
├── lectures/                    # Weekly lecture materials
│   ├── lecture01/              # Week 1: Introduction & Setup
│   │   ├── slides/             # LaTeX Beamer slides
│   │   ├── experiments/        # 9 hands-on experiments
│   │   ├── labs/               # Take-home exercises
│   │   └── solutions/          # Lab solutions (when available)
│   ├── lecture02/              # Week 2: Deep Learning Essentials
│   └── ...                     # Additional lectures
├── resources/                  # Additional materials
└── requirements.txt           # Python dependencies
```

## Available Lectures

- **Lecture 01**: Introduction & Environment Setup ✅ Available
- **Lecture 02**: Deep Learning Essentials ✅ Available
- **Lecture 03**: RL Fundamentals ✅ Available

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/METT519-RL-Official.git
cd METT519-RL-Official

# Create conda environment
conda create -n mett519 python=3.10
conda activate mett519

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
# Test PyTorch and CUDA setup
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run first experiment
cd lectures/lecture01/experiments
python exp01_setup.py
```

## Course Structure (13 Weeks)

| Week | Topic | Focus Areas |
|------|--------|-------------|
| 1 | Introduction & Setup | Environment, PyTorch, Reproducibility |
| 2 | Deep Learning Essentials | Tensors, Autograd, Neural Networks |
| 3 | RL Fundamentals | Agent-Environment, States, Actions, Rewards |
| 4 | Mathematical Foundations | MDP, Bellman Equations, Value Functions |
| 5 | Q-Learning | Tabular Methods, Exploration Strategies |
| 6 | Deep Q-Networks (DQN) | Function Approximation, Experience Replay |
| 7 | DQN Project | Complete Implementation, Optimization |
| 8 | Policy Gradient Methods | REINFORCE, Policy Gradient Theorem |
| 9 | Actor-Critic Methods | A2C, Advantage Functions |
| 10 | Proximal Policy Optimization | PPO, Trust Regions, GAE |
| 11 | Advanced Topics | RLHF, DPO, MCTS, AlphaZero |
| 12 | Project Development | Debugging, Optimization, Configuration |
| 13 | Final Presentations | Project Demos, Reproducibility |

## Learning Path

### For Beginners
1. Start with **Lecture 01** (Introduction & Setup)
2. Complete all experiments in order (exp01 → exp09)
3. Work through lab exercises
4. Move to **Lecture 02** (Deep Learning Essentials)

### For Advanced Students
1. Review **Lecture 03** (RL Fundamentals) for context
2. Jump to specific topics of interest
3. Focus on advanced experiments (exp07-exp09)
4. Contribute improvements via pull requests

## Technical Requirements

### Hardware
- **CPU**: Multi-core processor (Intel/AMD)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 5GB free space

### Software
- **Python**: 3.10 or higher
- **PyTorch**: 2.0 or higher with CUDA/MPS support
- **LaTeX**: For compiling slides (optional)
- **Git**: For version control

### Tested Environments
- ✅ Ubuntu 20.04/22.04 with CUDA 11.8/12.1
- ✅ macOS 12+ with Apple Silicon (MPS)
- ✅ Windows 10/11 with WSL2 + CUDA
- ✅ Google Colab (CPU/GPU runtimes)

## Course Materials

### Slides
- **Format**: LaTeX Beamer (academic standard)
- **Length**: 70-80 slides per 3-hour lecture

### Experiments
- **Structure**: 9 progressive experiments per lecture
- **Pattern**: Setup → Basics → Core → Advanced → Integration
- **Standards**: Reproducible, documented, cross-platform

### Lab Exercises
- **Format**: Take-home assignments (100 points each)
- **Deadline**: Before next lecture
- **Solutions**: Provided after submission deadline

## Contributing

We welcome contributions from students and educators! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/improvement`
3. **Commit** your changes: `git commit -am 'Add improvement'`
4. **Push** to the branch: `git push origin feature/improvement`
5. **Submit** a pull request

### Types of Contributions
- 🐛 Bug fixes in code or documentation
- 📚 Additional examples and explanations
- 🔧 Performance improvements
- 🌐 Translations (Korean ↔ English)
- 🎨 Visualization enhancements

## Support and Community

- **Issues**: [GitHub Issues](https://github.com/MIMICLab/METT519-RL/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MIMICLab/METT519-RL/discussions)
- **Email**: taehoonkim@sogang.ac.kr

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
If you use these materials in your research or teaching, please cite:
```bibtex
@misc{mett519_rl_2025,
  title={METT519 Reinforcement Learning Course Materials},
  author={Kim, Taehoon},
  year={2025},
  institution={Sogang University MIMIC Lab},
  url={https://github.com/yourusername/METT519-RL-Official}
}
```

## Acknowledgments

- **Sogang University MIMIC Lab** for providing resources and support
- **PyTorch Team** for the excellent deep learning framework  
- **OpenAI Gymnasium** for standardized RL environments
- **Contributors** who help improve these materials

---

**Course Instructor**: Taehoon Kim  
**Institution**: Sogang University MIMIC Lab  
**Website**: [https://mimic-lab.com](https://mimic-lab.com)  
**Last Updated**: 2025-09-07
