#!/usr/bin/env python3
"""
RL2025 - Lecture 11: Experiment 01 - Setup and Environment Verification

Verifies that all required libraries and hardware setup are working correctly
for advanced RL topics: RLHF, DPO, MCTS, and AlphaZero implementations.

Learning objectives:
- Verify PyTorch installation with CUDA/MPS support
- Test transformer model loading capabilities
- Validate NumPy and game environment dependencies
- Confirm mixed precision training support

Prerequisites: Python 3.10+, PyTorch 2.x
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import __version__ as transformers_version

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Proper device selection (CUDA > MPS > CPU)
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
amp_enabled = torch.cuda.is_available()
setup_seed(42)

def verify_pytorch_setup():
    """Verify PyTorch installation and capabilities."""
    print("PyTorch Setup Verification:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Device: {device}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  AMP enabled: {amp_enabled}")
    
    # Test basic tensor operations
    x = torch.randn(10, 10, device=device)
    y = torch.mm(x, x.T)
    print(f"  Basic tensor ops: {'✓' if y.shape == (10, 10) else '✗'}")
    
    return True

def verify_transformer_support():
    """Verify transformers library support."""
    print(f"\nTransformers Library:")
    print(f"  Version: {transformers_version}")
    
    try:
        # Test tokenizer creation
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
        test_text = "Hello world"
        tokens = tokenizer(test_text, return_tensors='pt')
        print(f"  Tokenization test: {'✓' if tokens['input_ids'].shape[1] > 0 else '✗'}")
        return True
    except Exception as e:
        print(f"  Tokenization test: ✗ ({e})")
        return False

def verify_game_environment():
    """Verify game environment capabilities for Gomoku."""
    print(f"\nGame Environment Test:")
    
    # Simple 5x5 board representation
    board = np.zeros((5, 5), dtype=np.int8)
    board[2, 2] = 1  # Place a stone
    board[2, 3] = -1  # Opponent stone
    
    # Convert to tensor
    board_tensor = torch.from_numpy(board).unsqueeze(0)  # [1, 5, 5]
    print(f"  Board tensor shape: {board_tensor.shape}")
    
    # Test legal moves
    legal_mask = (board == 0).flatten()
    legal_actions = np.where(legal_mask)[0]
    print(f"  Legal actions: {len(legal_actions)}/25")
    print(f"  Board representation: {'✓' if len(legal_actions) == 23 else '✗'}")
    
    return len(legal_actions) == 23

def verify_mixed_precision():
    """Verify mixed precision training support."""
    print(f"\nMixed Precision Test:")
    
    if not amp_enabled:
        print("  AMP not available on this device")
        return True
    
    try:
        # Create simple model
        model = nn.Linear(10, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.GradScaler()
        
        # Test forward pass with autocast
        x = torch.randn(5, 10, device=device)
        with torch.autocast(device_type='cuda'):
            y = model(x)
            loss = F.mse_loss(y, torch.randn_like(y))
        
        # Test backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"  Mixed precision training: ✓")
        return True
    except Exception as e:
        print(f"  Mixed precision training: ✗ ({e})")
        return False

def verify_compilation_support():
    """Verify torch.compile support."""
    print(f"\nCompilation Support:")
    
    if torch.__version__ < '2.0':
        print("  torch.compile requires PyTorch 2.0+")
        return False
    
    try:
        @torch.compile
        def simple_function(x):
            return x * 2 + 1
        
        x = torch.randn(10, device=device)
        y = simple_function(x)
        print(f"  torch.compile: {'✓' if y.shape == x.shape else '✗'}")
        return True
    except Exception as e:
        print(f"  torch.compile: ✗ ({e})")
        return False

def main():
    print("="*60)
    print("Experiment 01: Setup and Environment Verification")
    print("="*60)
    
    results = []
    
    # Run all verification tests
    results.append(verify_pytorch_setup())
    results.append(verify_transformer_support())
    results.append(verify_game_environment())
    results.append(verify_mixed_precision())
    results.append(verify_compilation_support())
    
    # Summary
    print(f"\nVerification Summary:")
    print(f"  Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print(f"  Status: ✓ All systems ready for Lecture 11")
    else:
        print(f"  Status: ⚠ Some issues detected - check output above")
    
    # Environment info for reproducibility
    print(f"\nReproducibility Info:")
    print(f"  Python: {os.sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  Device: {device}")
    print(f"  Seed: 42")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()