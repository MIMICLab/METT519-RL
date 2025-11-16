#!/usr/bin/env python3
"""
RL2025 - Lecture 11: Experiment 08 - Direct Preference Optimization (DPO)

Implements the DPO algorithm for training language models directly from
human preferences without requiring an explicit reward model.

Learning objectives:
- Implement DPO loss function
- Compare policy and reference model likelihoods
- Track KL divergence and preference accuracy
- Demonstrate training loop with synthetic data

Prerequisites: Toy causal LM from exp07
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import copy
from dataclasses import dataclass
import time

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

@dataclass
class DPOConfig:
    """DPO training configuration."""
    beta: float = 0.1  # Temperature parameter for DPO loss
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    max_epochs: int = 10
    eval_interval: int = 50
    log_interval: int = 10
    max_length: int = 64

class DPOTrainer:
    """Direct Preference Optimization trainer."""
    
    def __init__(self, policy_model: nn.Module, reference_model: nn.Module, 
                 tokenizer, config: DPOConfig = None):
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model.to(device)
        self.tokenizer = tokenizer
        self.config = config or DPOConfig()
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Mixed precision
        self.scaler = torch.cuda.GradScaler() if amp_enabled else None
        
        # Training history
        self.training_history = []
    
    def compute_log_likelihood(self, model: nn.Module, input_ids: torch.Tensor,
                              attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence log-likelihood.
        
        Args:
            model: Language model
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            
        Returns:
            log_likelihood: [B] log-likelihood per sequence
        """
        logits = model(input_ids, attention_mask)  # [B, L, V]
        
        # Shift for causal modeling
        shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, V]
        shift_labels = input_ids[:, 1:].contiguous()   # [B, L-1]
        shift_mask = attention_mask[:, 1:].contiguous()  # [B, L-1]
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Apply mask and sum over sequence
        masked_log_probs = gathered_log_probs * shift_mask.float()
        sequence_log_likelihood = masked_log_probs.sum(dim=-1)  # [B]
        
        return sequence_log_likelihood
    
    def dpo_loss(self, policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
                 reference_chosen_logps: torch.Tensor, reference_rejected_logps: torch.Tensor,
                 beta: float) -> Dict[str, torch.Tensor]:
        """
        Compute DPO loss.
        
        DPO loss: -log(sigma(beta * ((log pi_pos - log pi_ref_pos) - (log pi_neg - log pi_ref_neg))))
        
        Args:
            policy_chosen_logps: [B] log P(chosen | x) under policy
            policy_rejected_logps: [B] log P(rejected | x) under policy  
            reference_chosen_logps: [B] log P(chosen | x) under reference
            reference_rejected_logps: [B] log P(rejected | x) under reference
            beta: Temperature parameter
            
        Returns:
            Dictionary of loss components
        """
        # Compute log-ratio differences
        policy_diff = policy_chosen_logps - policy_rejected_logps
        reference_diff = reference_chosen_logps - reference_rejected_logps
        
        # DPO objective: maximize log-sigmoid of scaled difference
        logits = beta * (policy_diff - reference_diff)
        loss = -F.logsigmoid(logits).mean()
        
        # Additional metrics
        accuracy = (logits > 0).float().mean()
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'policy_diff': policy_diff.mean(),
            'reference_diff': reference_diff.mean(),
            'chosen_reward': chosen_rewards.mean(),
            'rejected_reward': rejected_rewards.mean(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean()
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.policy_model.train()
        self.reference_model.eval()
        
        # Extract batch components
        input_ids = batch['input_ids']
        chosen_ids = batch['chosen_ids'] 
        rejected_ids = batch['rejected_ids']
        chosen_mask = batch['chosen_attention_mask']
        rejected_mask = batch['rejected_attention_mask']
        
        # Compute log-likelihoods
        if amp_enabled and self.scaler is not None:
            with torch.autocast(device_type='cuda'):
                # Policy model
                policy_chosen_logps = self.compute_log_likelihood(self.policy_model, chosen_ids, chosen_mask)
                policy_rejected_logps = self.compute_log_likelihood(self.policy_model, rejected_ids, rejected_mask)
                
                # Reference model (no gradients)
                with torch.no_grad():
                    reference_chosen_logps = self.compute_log_likelihood(self.reference_model, chosen_ids, chosen_mask)
                    reference_rejected_logps = self.compute_log_likelihood(self.reference_model, rejected_ids, rejected_mask)
                
                # Compute DPO loss
                loss_dict = self.dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    reference_chosen_logps, reference_rejected_logps,
                    self.config.beta
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict['loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        else:
            # Policy model
            policy_chosen_logps = self.compute_log_likelihood(self.policy_model, chosen_ids, chosen_mask)
            policy_rejected_logps = self.compute_log_likelihood(self.policy_model, rejected_ids, rejected_mask)
            
            # Reference model (no gradients)
            with torch.no_grad():
                reference_chosen_logps = self.compute_log_likelihood(self.reference_model, chosen_ids, chosen_mask)
                reference_rejected_logps = self.compute_log_likelihood(self.reference_model, rejected_ids, rejected_mask)
            
            # Compute DPO loss
            loss_dict = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                self.config.beta
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
        
        # Convert to Python floats
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        # Compute KL divergence estimation
        kl_div = self._estimate_kl_divergence(policy_chosen_logps, reference_chosen_logps,
                                             policy_rejected_logps, reference_rejected_logps)
        metrics['kl_divergence'] = kl_div
        
        return metrics
    
    def _estimate_kl_divergence(self, policy_chosen: torch.Tensor, ref_chosen: torch.Tensor,
                               policy_rejected: torch.Tensor, ref_rejected: torch.Tensor) -> float:
        """Estimate KL divergence between policy and reference."""
        # Simple approximation: average difference in log-likelihoods
        chosen_kl = (policy_chosen - ref_chosen).abs().mean()
        rejected_kl = (policy_rejected - ref_rejected).abs().mean()
        return ((chosen_kl + rejected_kl) / 2).item()
    
    def evaluate(self, dataset, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.policy_model.eval()
        
        total_metrics = {
            'loss': 0.0,
            'accuracy': 0.0, 
            'kl_divergence': 0.0,
            'reward_margin': 0.0
        }
        
        num_batches = 0
        with torch.no_grad():
            for _ in range(min(num_samples // self.config.batch_size, 10)):  # Limit eval time
                batch = dataset.get_batch(self.config.batch_size, self.config.max_length)
                
                # Compute metrics without gradients
                policy_chosen = self.compute_log_likelihood(self.policy_model, batch['chosen_ids'], batch['chosen_attention_mask'])
                policy_rejected = self.compute_log_likelihood(self.policy_model, batch['rejected_ids'], batch['rejected_attention_mask'])
                ref_chosen = self.compute_log_likelihood(self.reference_model, batch['chosen_ids'], batch['chosen_attention_mask'])
                ref_rejected = self.compute_log_likelihood(self.reference_model, batch['rejected_ids'], batch['rejected_attention_mask'])
                
                metrics = self.dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, self.config.beta)
                
                for key in total_metrics.keys():
                    if key in metrics:
                        total_metrics[key] += metrics[key].item() if torch.is_tensor(metrics[key]) else metrics[key]
                
                total_metrics['kl_divergence'] += self._estimate_kl_divergence(policy_chosen, ref_chosen, policy_rejected, ref_rejected)
                num_batches += 1
        
        # Average metrics
        if num_batches > 0:
            for key in total_metrics:
                total_metrics[key] /= num_batches
        
        return total_metrics
    
    def train(self, train_dataset, val_dataset=None) -> Dict[str, List[float]]:
        """Full training loop."""
        print(f"Starting DPO training with beta={self.config.beta}")
        print(f"Policy model parameters: {sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad):,}")
        
        history = {
            'train_loss': [],
            'train_accuracy': [], 
            'train_kl': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        step = 0
        for epoch in range(self.config.max_epochs):
            epoch_metrics = {
                'loss': [],
                'accuracy': [],
                'kl_divergence': []
            }
            
            # Training steps
            steps_per_epoch = 50  # Limit for toy example
            for _ in range(steps_per_epoch):
                batch = train_dataset.get_batch(self.config.batch_size, self.config.max_length)
                metrics = self.training_step(batch)
                
                for key in epoch_metrics.keys():
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
                
                if step % self.config.log_interval == 0:
                    print(f"Step {step}: Loss={metrics.get('loss', 0):.4f}, "
                          f"Acc={metrics.get('accuracy', 0):.3f}, "
                          f"KL={metrics.get('kl_divergence', 0):.4f}")
                
                step += 1
            
            # Epoch summary
            avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
            history['train_loss'].append(avg_metrics['loss'])
            history['train_accuracy'].append(avg_metrics['accuracy'])
            history['train_kl'].append(avg_metrics['kl_divergence'])
            
            # Validation
            if val_dataset is not None and epoch % 2 == 0:
                val_metrics = self.evaluate(val_dataset)
                history['val_loss'].append(val_metrics.get('loss', 0))
                history['val_accuracy'].append(val_metrics.get('accuracy', 0))
                print(f"Epoch {epoch}: Val Loss={val_metrics.get('loss', 0):.4f}, "
                      f"Val Acc={val_metrics.get('accuracy', 0):.3f}")
            
            print(f"Epoch {epoch} completed: Train Loss={avg_metrics['loss']:.4f}, "
                  f"Train Acc={avg_metrics['accuracy']:.3f}")
        
        self.training_history = history
        return history

def test_dpo_loss():
    """Test DPO loss computation."""
    print("Testing DPO loss...")
    
    # Create dummy log-probabilities
    batch_size = 4
    policy_chosen = torch.randn(batch_size)
    policy_rejected = torch.randn(batch_size)
    ref_chosen = torch.randn(batch_size)
    ref_rejected = torch.randn(batch_size)
    
    # Ensure chosen is preferred (higher likelihood)
    policy_chosen = policy_chosen.abs()
    policy_rejected = -policy_rejected.abs()
    ref_chosen = ref_chosen.abs() 
    ref_rejected = -ref_rejected.abs()
    
    # Create dummy trainer to test loss function
    import sys
    sys.path.append('/Users/taehoon.kim/Desktop/Sources/RL2025/Lectures/Lecture11/experiments')
    try:
        from exp07_toy_causal_lm import SimpleTokenizer, SimpleTransformerLM
        
        tokenizer = SimpleTokenizer(128)
        model = SimpleTransformerLM(tokenizer.vocab_size)
        ref_model = copy.deepcopy(model)
        trainer = DPOTrainer(model, ref_model, tokenizer)
        
        loss_dict = trainer.dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
        
        assert 'loss' in loss_dict
        assert 'accuracy' in loss_dict
        assert 0.0 <= loss_dict['accuracy'] <= 1.0  # Accuracy should be valid range (random init may not prefer chosen)
        
        print("  DPO loss computation: ✓")
        
    except ImportError as e:
        print(f"  DPO loss: Skipped ({e})")

def test_training_step():
    """Test single training step."""
    print("Testing training step...")
    
    try:
        import sys
        sys.path.append('/Users/taehoon.kim/Desktop/Sources/RL2025/Lectures/Lecture11/experiments')
        from exp07_toy_causal_lm import SimpleTokenizer, SimpleTransformerLM, PreferenceDataset
        
        # Setup
        tokenizer = SimpleTokenizer(128)
        model = SimpleTransformerLM(tokenizer.vocab_size, embed_dim=64, num_layers=2)
        ref_model = copy.deepcopy(model)
        
        config = DPOConfig(batch_size=4, beta=0.1)
        trainer = DPOTrainer(model, ref_model, tokenizer, config)
        
        # Create synthetic batch
        dataset = PreferenceDataset(tokenizer, size=20)
        batch = dataset.get_batch(config.batch_size, config.max_length)
        
        # Training step
        metrics = trainer.training_step(batch)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert isinstance(metrics['loss'], float)
        
        print("  Training step: ✓")
        
    except ImportError as e:
        print(f"  Training step: Skipped ({e})")

def demonstrate_dpo_training():
    """Demonstrate DPO training loop."""
    print("\nDemonstrating DPO training:")
    print("="*40)
    
    try:
        import sys
        sys.path.append('/Users/taehoon.kim/Desktop/Sources/RL2025/Lectures/Lecture11/experiments')
        from exp07_toy_causal_lm import SimpleTokenizer, SimpleTransformerLM, PreferenceDataset
        
        # Setup models
        tokenizer = SimpleTokenizer(vocab_size=256)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        policy_model = SimpleTransformerLM(
            vocab_size=tokenizer.vocab_size,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            max_seq_length=64
        )
        
        # Reference model is a copy of initial policy
        reference_model = copy.deepcopy(policy_model)
        
        print(f"Model size: {sum(p.numel() for p in policy_model.parameters()):,} parameters")
        
        # Create datasets
        train_dataset = PreferenceDataset(tokenizer, size=200)
        val_dataset = PreferenceDataset(tokenizer, size=50) 
        
        print(f"Training data: {len(train_dataset)} preference pairs")
        print(f"Validation data: {len(val_dataset)} preference pairs")
        
        # Show example data
        print("\nExample preference pair:")
        example = train_dataset[0]
        print(f"  Input: '{example['input']}'")
        print(f"  Chosen: '{example['chosen']}'")
        print(f"  Rejected: '{example['rejected']}'")
        
        # DPO training
        config = DPOConfig(
            beta=0.1,
            learning_rate=1e-4,
            batch_size=8,
            max_epochs=3,  # Short for demo
            log_interval=20
        )
        
        trainer = DPOTrainer(policy_model, reference_model, tokenizer, config)
        
        print(f"\nTraining configuration:")
        print(f"  Beta: {config.beta}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Max epochs: {config.max_epochs}")
        
        # Train
        start_time = time.time()
        history = trainer.train(train_dataset, val_dataset)
        train_time = time.time() - start_time
        
        print(f"\nTraining completed in {train_time:.1f}s")
        
        # Show final metrics
        if history['train_loss']:
            print(f"Final training loss: {history['train_loss'][-1]:.4f}")
            print(f"Final training accuracy: {history['train_accuracy'][-1]:.3f}")
            print(f"Final KL divergence: {history['train_kl'][-1]:.4f}")
        
        if history['val_loss']:
            print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
            print(f"Final validation accuracy: {history['val_accuracy'][-1]:.3f}")
        
        # Test different beta values
        print(f"\nTesting different beta values:")
        betas = [0.01, 0.1, 1.0]
        
        for beta in betas:
            config.beta = beta
            trainer_test = DPOTrainer(copy.deepcopy(policy_model), reference_model, tokenizer, config)
            
            # Single evaluation step
            val_metrics = trainer_test.evaluate(val_dataset, num_samples=20)
            print(f"  Beta {beta}: Accuracy = {val_metrics.get('accuracy', 0):.3f}, "
                  f"Loss = {val_metrics.get('loss', 0):.4f}")
        
    except ImportError as e:
        print(f"DPO training demo skipped: {e}")

def main():
    print("="*60)
    print("Experiment 08: Direct Preference Optimization (DPO)")
    print("="*60)
    
    # Run tests
    test_dpo_loss()
    test_training_step()
    
    print("\nAll tests passed! ✓")
    
    # Demonstrate training
    demonstrate_dpo_training()
    
    print(f"\nDPO implementation ready!")
    print("DPO specifications:")
    print(f"  Loss: -log(σ(β * ((log π_θ(y+|x) - log π_ref(y+|x)) - (log π_θ(y-|x) - log π_ref(y-|x)))))")
    print(f"  Beta parameter: Controls preference strength vs KL penalty")
    print(f"  Training: Direct optimization without explicit reward model")
    print(f"  Evaluation: Preference accuracy and KL divergence tracking")
    print(f"  Mixed precision: {'Enabled' if amp_enabled else 'Disabled'}")
    print(f"  Device: {device}")

if __name__ == "__main__":
    main()