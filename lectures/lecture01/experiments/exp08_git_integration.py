#!/usr/bin/env python3
"""
Experiment 8: Git Integration and Version Tracking
Slides: 66-70 (Version control for experiments)
Time: 2:00-2:10 (10 minutes)

Demonstrates Git integration for experiment tracking and reproducibility.
"""

import os
import sys
import subprocess
import hashlib
import json
from pathlib import Path
from datetime import datetime

def run_command(cmd, capture=True):
    """Run shell command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True, check=True
        )
        return result.stdout.strip() if capture else True
    except subprocess.CalledProcessError as e:
        if capture:
            return None
        return False

def init_git_repo():
    """Initialize Git repository with proper configuration"""
    print("\n" + "="*60)
    print("Initializing Git Repository")
    print("="*60)
    
    # Check if already in a git repo
    if run_command("git rev-parse --git-dir", capture=True):
        print("  Already in a Git repository")
        return True
    
    # Initialize new repo
    if run_command("git init"):
        print("  ✓ Git repository initialized")
    
    # Set user info (if not set)
    user_name = run_command("git config user.name")
    if not user_name:
        run_command("git config user.name 'RL Student'")
        print("  ✓ Set default user name")
    
    user_email = run_command("git config user.email")
    if not user_email:
        run_command("git config user.email 'student@rl2025.course'")
        print("  ✓ Set default user email")
    
    return True

def create_gitignore():
    """Create comprehensive .gitignore for ML projects"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
.venv

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# PyTorch
*.pt
*.pth
*.pkl
runs/
checkpoints/
logs/
*.ckpt

# TensorBoard
events.out.tfevents.*

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Data (customize as needed)
data/
datasets/
*.csv
*.h5
*.hdf5

# Temporary
tmp/
temp/
*.tmp
*.bak
*.log

# Keep specific files
!requirements.txt
!environment.yml
!*.md
"""
    
    gitignore_path = Path(".gitignore")
    
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content)
        print("\n  ✓ Created .gitignore file")
        return True
    else:
        print("\n  .gitignore already exists")
        return False

def capture_environment():
    """Capture complete environment information"""
    print("\n" + "="*60)
    print("Capturing Environment Information")
    print("="*60)
    
    env_dir = Path("runs") / "meta"
    env_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd()
    }
    
    # Git information
    git_info = {}
    git_info["branch"] = run_command("git branch --show-current") or "unknown"
    git_info["commit"] = run_command("git rev-parse HEAD") or "unknown"
    git_info["status"] = run_command("git status --short") or ""
    git_info["remote"] = run_command("git remote -v") or ""
    
    env_info["git"] = git_info
    
    # Installed packages
    pip_list = run_command("pip list --format=json")
    if pip_list:
        try:
            env_info["packages"] = json.loads(pip_list)
        except:
            env_info["packages"] = []
    
    # PyTorch specific
    try:
        import torch
        env_info["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
    except ImportError:
        env_info["pytorch"] = "Not installed"
    
    # Save to file
    filename = env_dir / f"env_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(env_info, f, indent=2)
    
    print(f"  ✓ Environment captured to {filename}")
    
    # Display summary
    print(f"\n  Git branch: {git_info['branch']}")
    print(f"  Git commit: {git_info['commit'][:8] if git_info['commit'] != 'unknown' else 'unknown'}")
    
    if git_info["status"]:
        print(f"  Uncommitted changes:")
        for line in git_info["status"].split('\n')[:5]:
            print(f"    {line}")
    
    return filename

def hash_config(config_dict):
    """Generate stable hash from configuration dictionary"""
    # Convert to sorted JSON string for consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True)
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:12]

def create_experiment_record(name, config, results=None):
    """Create comprehensive experiment record"""
    print("\n" + "="*60)
    print("Creating Experiment Record")
    print("="*60)
    
    # Generate hash
    config_hash = hash_config(config)
    
    # Create record
    record = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "config_hash": config_hash,
        "results": results or {}
    }
    
    # Add git info
    record["git"] = {
        "commit": run_command("git rev-parse HEAD") or "unknown",
        "branch": run_command("git branch --show-current") or "unknown",
        "dirty": bool(run_command("git status --short"))
    }
    
    # Save record
    record_dir = Path("runs") / "experiments"
    record_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = record_dir / f"{name}_{timestamp}_{config_hash}.json"
    
    with open(filename, "w") as f:
        json.dump(record, f, indent=2)
    
    print(f"  ✓ Experiment record: {filename.name}")
    print(f"  Config hash: {config_hash}")
    
    return filename, config_hash

def demo_reproducible_experiment():
    """Demonstrate reproducible experiment workflow"""
    print("\n" + "="*60)
    print("Reproducible Experiment Demo")
    print("="*60)
    
    # Define experiment configuration
    config = {
        "model": "SimpleNet",
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "seed": 42,
        "optimizer": "Adam"
    }
    
    # Create experiment record
    record_file, exp_hash = create_experiment_record(
        name="demo_training",
        config=config
    )
    
    # Create results directory with hash
    results_dir = Path("runs") / f"results_{exp_hash}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Results directory: {results_dir}")
    
    # Simulate training with results
    print("\n  Running experiment...")
    
    results = {
        "final_loss": 0.0234,
        "best_accuracy": 0.956,
        "training_time": 120.5
    }
    
    # Update record with results
    with open(record_file, "r") as f:
        record = json.load(f)
    
    record["results"] = results
    
    with open(record_file, "w") as f:
        json.dump(record, f, indent=2)
    
    print(f"  ✓ Results updated in {record_file.name}")
    
    return True

def create_commit_template():
    """Create Git commit message template"""
    template = """# Experiment: <name>
# Config hash: <hash>
# Results: <brief summary>

# Detailed changes:
# - 
# - 

# Performance metrics:
# - Loss: 
# - Accuracy: 
# - Time: 
"""
    
    template_path = Path(".gitmessage")
    
    if not template_path.exists():
        with open(template_path, "w") as f:
            f.write(template)
        
        # Set as default template
        run_command("git config commit.template .gitmessage")
        
        print("\n  ✓ Created commit template: .gitmessage")
        return True
    
    return False

def demo_automated_commit():
    """Demonstrate automated experiment commit"""
    print("\n" + "="*60)
    print("Automated Commit Demo")
    print("="*60)
    
    # Check for changes
    status = run_command("git status --short")
    
    if not status:
        # Create a dummy file to commit
        dummy_file = Path("runs") / "dummy_experiment.txt"
        dummy_file.parent.mkdir(parents=True, exist_ok=True)
        dummy_file.write_text(f"Experiment run at {datetime.now()}")
        print(f"  Created {dummy_file} (not staged). Use `git add` manually if desired.")
    
    # Generate commit message
    config_hash = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    
    commit_msg = f"""Experiment: demo_run_{config_hash}

Config: lr=0.001, batch_size=32, epochs=10
Results: loss=0.0234, acc=0.956
Hash: {config_hash}
"""
    
    # Show what would be committed
    print("  Commit message:")
    for line in commit_msg.split('\n'):
        print(f"    {line}")
    
    print("\n  (Commit not executed - demonstration only)")
    
    return True

def main():
    """Main execution"""
    print("="*60)
    print("RL2025 - Lecture 1: Git Integration and Version Tracking")
    print("="*60)
    
    success = True
    
    try:
        # Initialize Git repo
        init_git_repo()
        
        # Create .gitignore
        create_gitignore()
        
        # Capture environment
        capture_environment()
        
        # Demo experiment recording
        demo_reproducible_experiment()
        
        # Create commit template
        create_commit_template()
        
        # Demo automated commit
        demo_automated_commit()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        success = False
    
    if success:
        print("\n✅ Experiment 8 completed successfully!")
        print("\nKey practices:")
        print("  • Always capture environment before experiments")
        print("  • Use config hashes in directory/file names")
        print("  • Commit code before running experiments")
        print("  • Include hashes in commit messages")
        print("  • Keep experiment records with Git info")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
