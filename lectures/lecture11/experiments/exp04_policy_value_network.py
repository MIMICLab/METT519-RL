#!/usr/bin/env python3
"""
RL2025 - Lecture 11: Experiment 04 - Policy-Value Network for Gomoku

Implements a small residual CNN that outputs both policy logits and value
estimates for the Gomoku 5×5 environment. Supports torch.compile and 
mixed precision training.

Learning objectives:
- Design policy-value network architecture
- Implement proper masking for illegal moves
- Support mixed precision and compilation
- Handle batch processing and tensor shapes

Prerequisites: PyTorch 2.x, Gomoku environment
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

from exp02_standard_training_header import TimingUtils as _TimingUtils

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


class TimingUtils(_TimingUtils):
    """Re-export timing utilities for policy-value benchmarks."""

    pass

class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Input: [B, C, H, W]
        Output: [B, C, H, W]
        """
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        
        return out

class PolicyValueNet(nn.Module):
    """
    Policy-Value Network for Gomoku 5×5.
    
    Architecture:
    - Input: [B, 2, 5, 5] (current player stones, opponent stones)
    - Stem: Conv2d to expand channels
    - Body: Several residual blocks
    - Policy head: Conv2d + Linear -> [B, 25] logits (masked for illegal moves)
    - Value head: Conv2d + Linear -> [B, 1] value in tanh
    """
    
    def __init__(self, input_channels: int = 2, hidden_channels: int = 64, 
                 num_residual_blocks: int = 4, board_size: int = 5):
        super().__init__()
        
        self.board_size = board_size
        self.action_size = board_size * board_size
        
        # Stem: Expand input channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Body: Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_linear = nn.Linear(2 * board_size * board_size, self.action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_linear1 = nn.Linear(board_size * board_size, hidden_channels)
        self.value_linear2 = nn.Linear(hidden_channels, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [B, 2, 5, 5]
            legal_mask: Legal moves mask [B, 25] (True for legal moves)
            
        Returns:
            policy_logits: [B, 25] policy logits (illegal moves masked to -inf)
            value: [B, 1] value estimate in tanh
        """
        batch_size = x.shape[0]
        
        # Stem: [B, 2, 5, 5] -> [B, hidden_channels, 5, 5]
        x = self.stem(x)
        
        # Body: Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head: [B, hidden_channels, 5, 5] -> [B, 25]
        policy = F.relu(self.policy_bn(self.policy_conv(x)))  # [B, 2, 5, 5]
        policy = policy.view(batch_size, -1)  # [B, 2*5*5]
        policy_logits = self.policy_linear(policy)  # [B, 25]
        
        # Apply legal move masking
        if legal_mask is not None:
            legal_mask = legal_mask.to(policy_logits.device)
            # Set illegal moves to -inf
            policy_logits = torch.where(legal_mask, policy_logits, torch.tensor(-float('inf'), device=policy_logits.device, dtype=policy_logits.dtype))
        
        # Value head: [B, hidden_channels, 5, 5] -> [B, 1]
        value = F.relu(self.value_bn(self.value_conv(x)))  # [B, 1, 5, 5]
        value = value.view(batch_size, -1)  # [B, 1*5*5]
        value = F.relu(self.value_linear1(value))  # [B, hidden_channels]
        value = torch.tanh(self.value_linear2(value))  # [B, 1]
        
        return policy_logits, value

class PolicyValueTrainer:
    """Training utilities for PolicyValueNet."""
    
    def __init__(self, model: PolicyValueNet, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scaler = torch.cuda.GradScaler() if amp_enabled else None
        
        # Compile model if supported (disabled on MPS for compatibility)
        if hasattr(torch, 'compile') and device.type == 'cuda':
            try:
                self.model = torch.compile(self.model)
                print("Model compiled successfully")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        else:
            print("Model compilation skipped (not on CUDA or not supported)")
    
    def train_step(self, states: torch.Tensor, policies: torch.Tensor, 
                   values: torch.Tensor, legal_masks: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            states: [B, 2, 5, 5] board states
            policies: [B, 25] target policy distributions
            values: [B, 1] target values
            legal_masks: [B, 25] legal move masks
            
        Returns:
            Dictionary of losses and metrics
        """
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        legal_masks = legal_masks.to(device)

        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if amp_enabled and self.scaler is not None:
            with torch.autocast(device_type='cuda'):
                policy_logits, pred_values = self.model(states, legal_masks)
                loss_dict = self._compute_losses(policy_logits, pred_values, policies, values)
            
            # Backward pass
            self.scaler.scale(loss_dict['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, pred_values = self.model(states, legal_masks)
            loss_dict = self._compute_losses(policy_logits, pred_values, policies, values)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            self.optimizer.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def _compute_losses(self, policy_logits: torch.Tensor, pred_values: torch.Tensor,
                       target_policies: torch.Tensor, target_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute policy and value losses.
        
        Args:
            policy_logits: [B, 25] predicted policy logits
            pred_values: [B, 1] predicted values
            target_policies: [B, 25] target policy distributions
            target_values: [B, 1] target values
            
        Returns:
            Dictionary of computed losses
        """
        # Policy loss: Cross-entropy between target policy and predicted logits
        policy_loss = -torch.sum(target_policies * F.log_softmax(policy_logits, dim=1), dim=1).mean()
        
        # Value loss: MSE between predicted and target values
        value_loss = F.mse_loss(pred_values, target_values)
        
        # Total loss with weighting
        total_loss = policy_loss + value_loss
        
        # Additional metrics
        policy_entropy = -torch.sum(F.softmax(policy_logits, dim=1) * F.log_softmax(policy_logits, dim=1), dim=1).mean()
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_entropy': policy_entropy
        }
    
    @torch.no_grad()
    def evaluate(self, states: torch.Tensor, legal_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate model on batch of states.
        
        Args:
            states: [B, 2, 5, 5] board states
            legal_masks: [B, 25] legal move masks
            
        Returns:
            policy_probs: [B, 25] policy probabilities
            values: [B, 1] value estimates
        """
        states = states.to(device)
        legal_masks = legal_masks.to(device)

        self.model.eval()
        
        policy_logits, values = self.model(states, legal_masks)
        policy_probs = F.softmax(policy_logits, dim=1)
        
        return policy_probs, values

def test_network_architecture():
    """Test network architecture and shapes."""
    print("Testing network architecture...")
    
    model = PolicyValueNet()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    states = torch.randn(batch_size, 2, 5, 5)
    legal_masks = torch.randint(0, 2, (batch_size, 25)).bool()
    
    policy_logits, values = model(states, legal_masks)
    
    assert policy_logits.shape == (batch_size, 25), f"Wrong policy shape: {policy_logits.shape}"
    assert values.shape == (batch_size, 1), f"Wrong value shape: {values.shape}"
    
    # Check that illegal moves are masked
    illegal_positions = ~legal_masks
    assert torch.all(policy_logits[illegal_positions] == -float('inf')), "Illegal moves not properly masked"
    
    print("  Architecture: ✓")

def test_training_step():
    """Test training step."""
    print("Testing training step...")
    
    model = PolicyValueNet()
    trainer = PolicyValueTrainer(model, learning_rate=1e-3)
    
    # Create batch of training data
    batch_size = 8
    states = torch.randn(batch_size, 2, 5, 5).to(device)
    
    # Random legal masks (at least one move must be legal)
    legal_masks = torch.randint(0, 2, (batch_size, 25)).bool().to(device)
    for i in range(batch_size):
        if not legal_masks[i].any():
            legal_masks[i, 0] = True  # Ensure at least one legal move
    
    # Random target policies (normalized over legal moves)
    policies = torch.zeros(batch_size, 25).to(device)
    for i in range(batch_size):
        legal_indices = legal_masks[i].nonzero(as_tuple=True)[0]
        random_policy = torch.rand(len(legal_indices)).to(device)
        policies[i, legal_indices] = random_policy / random_policy.sum()
    
    # Random target values
    values = torch.randn(batch_size, 1).clamp(-1, 1).to(device)
    
    # Training step
    metrics = trainer.train_step(states, policies, values, legal_masks)
    
    assert 'total_loss' in metrics, "Missing total_loss in metrics"
    assert 'policy_loss' in metrics, "Missing policy_loss in metrics"
    assert 'value_loss' in metrics, "Missing value_loss in metrics"
    assert all(isinstance(v, float) for v in metrics.values()), "Metrics should be Python floats"
    
    print("  Training step: ✓")

def test_evaluation():
    """Test evaluation mode."""
    print("Testing evaluation...")
    
    model = PolicyValueNet().to(device)
    trainer = PolicyValueTrainer(model)
    
    batch_size = 4
    states = torch.randn(batch_size, 2, 5, 5).to(device)
    legal_masks = torch.ones(batch_size, 25).bool().to(device)  # All moves legal
    
    policy_probs, values = trainer.evaluate(states, legal_masks)
    
    assert policy_probs.shape == (batch_size, 25), f"Wrong policy shape: {policy_probs.shape}"
    assert values.shape == (batch_size, 1), f"Wrong value shape: {values.shape}"
    
    # Check that probabilities sum to 1
    prob_sums = policy_probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums)), "Probabilities don't sum to 1"
    
    # Check value range
    assert torch.all(values >= -1) and torch.all(values <= 1), "Values outside [-1, 1] range"
    
    print("  Evaluation: ✓")

def benchmark_throughput():
    """Benchmark model throughput."""
    print("Benchmarking throughput...")
    
    model = PolicyValueNet().to(device)
    model.eval()
    
    batch_size = 32
    states = torch.randn(batch_size, 2, 5, 5).to(device)
    legal_masks = torch.ones(batch_size, 25).bool().to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(states, legal_masks)
    
    # Measure throughput
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    num_iterations = 100
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        start_time.record()
    else:
        import time
        start = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(states, legal_masks)
    
    if torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        elapsed_ms = start_time.elapsed_time(end_time)
        throughput = (num_iterations * batch_size) / (elapsed_ms / 1000)
    else:
        end = time.time()
        elapsed_s = end - start
        throughput = (num_iterations * batch_size) / elapsed_s
    
    print(f"  Throughput: {throughput:.1f} inferences/second")

def demonstrate_usage():
    """Demonstrate network usage with Gomoku environment."""
    print("\nDemonstrating network usage:")
    print("="*40)
    
    # Import Gomoku environment
    import sys
    sys.path.append('/Users/taehoon.kim/Desktop/Sources/RL2025/Lectures/Lecture11/experiments')
    try:
        from exp03_gomoku_environment import Gomoku5x5, Player
        
        # Create environment and model
        env = Gomoku5x5()
        model = PolicyValueNet().to(device)
        trainer = PolicyValueTrainer(model)
        
        # Reset environment
        obs, _ = env.reset()
        
        print("Initial board:")
        print(env)
        print()
        
        # Get network prediction
        obs_batch = obs.unsqueeze(0).to(device)  # [1, 2, 5, 5]
        legal_mask = env.legal_actions().unsqueeze(0).to(device)  # [1, 25]
        
        policy_probs, value = trainer.evaluate(obs_batch, legal_mask)
        
        print(f"Network predictions:")
        print(f"  Value estimate: {value.item():.3f}")
        print(f"  Policy entropy: {-torch.sum(policy_probs * torch.log(policy_probs + 1e-8)).item():.3f}")
        
        # Show top 5 move probabilities
        probs = policy_probs.squeeze(0).cpu()
        top_actions = torch.topk(probs, 5)
        
        print(f"  Top 5 moves:")
        for i, (prob, action) in enumerate(zip(top_actions.values, top_actions.indices)):
            row, col = divmod(action.item(), 5)
            print(f"    {i+1}. Position ({row},{col}): {prob.item():.3f}")
        
    except ImportError:
        print("Gomoku environment not available for demonstration")

def main():
    print("="*60)
    print("Experiment 04: Policy-Value Network for Gomoku")
    print("="*60)
    
    # Run tests
    test_network_architecture()
    test_training_step()
    test_evaluation()
    
    print("\nAll tests passed! ✓")
    
    # Benchmark performance
    benchmark_throughput()
    
    # Demonstrate usage
    demonstrate_usage()
    
    print(f"\nPolicy-Value Network ready!")
    print("Network specifications:")
    print(f"  Input shape: [B, 2, 5, 5] (current player stones, opponent stones)")
    print(f"  Policy output: [B, 25] (action logits with illegal move masking)")
    print(f"  Value output: [B, 1] (position evaluation in [-1, +1])")
    print(f"  Mixed precision: {'Enabled' if amp_enabled else 'Disabled'}")
    print(f"  Device: {device}")

if __name__ == "__main__":
    main()
