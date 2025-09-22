#!/usr/bin/env python3
"""
Experiment 1: Environment Verification and Setup
Slides: 5-15 (Environment setup and course overview)
Time: 0:10-0:30 (20 minutes)

This experiment verifies the Python environment, checks package installations,
and ensures all required dependencies are available for the course.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path
import json
from datetime import datetime

def check_python_version():
    """Check Python version meets requirements (3.10-3.12)"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    if not (3, 10) <= (version.major, version.minor) <= (3, 12):
        print("⚠️  Warning: Python 3.10-3.12 recommended for this course")
        return False
    else:
        print("✓ Python version OK")
        return True

def check_package_installations():
    """Check if required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'tqdm': 'Progress bars',
        'tensorboard': 'TensorBoard',
        'jupyter': 'Jupyter',
        'ipykernel': 'IPython kernel'
    }
    
    optional_packages = {
        'gymnasium': 'Gymnasium (will be installed in Lecture 3)',
        'torchvision': 'TorchVision',
        'torchaudio': 'TorchAudio'
    }
    
    print("\n" + "="*60)
    print("Required Packages:")
    print("="*60)
    
    missing_required = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            module = sys.modules[package]
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {description:20} ({package}=={version})")
        except ImportError:
            print(f"✗ {description:20} ({package}) - NOT INSTALLED")
            missing_required.append(package)
    
    print("\n" + "="*60)
    print("Optional Packages:")
    print("="*60)
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            module = sys.modules[package]
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {description:30} ({package}=={version})")
        except ImportError:
            print(f"ℹ {description:30} ({package}) - Not installed (optional)")
    
    return len(missing_required) == 0, missing_required

def create_environment_files():
    """Create conda environment.yml and requirements.txt files"""
    
    # Create envs directory
    envs_dir = Path("envs")
    envs_dir.mkdir(exist_ok=True)
    
    # environment.yml for conda
    environment_yml = """name: rl2025
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.0
  - torchvision
  - torchaudio
  - numpy
  - matplotlib
  - pandas
  - tensorboard
  - tqdm
  - pyyaml
  - ipykernel
  - jupyterlab
  - pip
  - pip:
    - rich
    # - gymnasium  # Will be installed in Lecture 3
"""
    
    with open(envs_dir / "environment.yml", "w") as f:
        f.write(environment_yml)
    print(f"\n✓ Created {envs_dir}/environment.yml")
    
    # requirements.txt for pip
    requirements_txt = """torch>=2.0.0
torchvision
torchaudio
numpy>=1.20.0
matplotlib>=3.3.0
pandas>=1.3.0
tensorboard>=2.8.0
tqdm>=4.60.0
pyyaml>=5.4.0
rich>=10.0.0
ipykernel>=6.0.0
jupyterlab>=3.0.0
# gymnasium  # Will be installed in Lecture 3
"""
    
    with open(envs_dir / "requirements.txt", "w") as f:
        f.write(requirements_txt)
    print(f"✓ Created {envs_dir}/requirements.txt")

def save_system_info():
    """Save system information to JSON file"""
    
    # Create runs/sysinfo directory
    sysinfo_dir = Path("runs") / "sysinfo"
    sysinfo_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect system information
    sysinfo = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count()
    }
    
    # Try to get pip list
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"],
                              capture_output=True, text=True, check=True)
        sysinfo["packages"] = json.loads(result.stdout)
    except Exception as e:
        sysinfo["packages"] = f"Error getting package list: {e}"
    
    # Save to timestamped file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = sysinfo_dir / f"sysinfo_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(sysinfo, f, indent=2)
    
    print(f"\n✓ System info saved to {filename}")
    return filename

def print_setup_instructions():
    """Print setup instructions for students"""
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS")
    print("="*60)
    print("""
1. Create conda environment:
   conda env create -f envs/environment.yml
   conda activate rl2025

2. Or use pip:
   python -m pip install -r envs/requirements.txt

3. Register Jupyter kernel:
   python -m ipykernel install --user --name rl2025

4. Verify installation:
   python exp01_setup.py

5. For Google Colab:
   - Upload this notebook
   - Run the bootstrap cell (see exp02)
   - All experiments should work on Colab
""")

def main():
    """Main execution"""
    print("="*60)
    print("RL2025 - Lecture 1: Environment Verification")
    print("="*60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check package installations
    packages_ok, missing = check_package_installations()
    
    # Create environment files
    create_environment_files()
    
    # Save system information
    save_system_info()
    
    # Print setup instructions if needed
    if not packages_ok:
        print_setup_instructions()
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Please install missing packages before proceeding.")
    else:
        print("\n✅ All required packages are installed!")
        print("You're ready to proceed with the course.")
    
    return python_ok and packages_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
