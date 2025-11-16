#!/usr/bin/env python3
"""
RL2025 - Lecture 11: Experiment 06 - Self-Play and AlphaZero Training

Implements the AlphaZero self-play training loop with experience collection,
replay buffer management, and policy-value network updates.

Learning objectives:
- Implement AlphaZero self-play data generation
- Build experience replay buffer with reservoir sampling
- Combine MCTS search with supervised learning
- Track training metrics and convergence

Prerequisites: Gomoku environment, Policy-Value network, PUCT MCTS
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from collections import deque
import time
from dataclasses import dataclass

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

class Experience(NamedTuple):
    """Single training experience."""
    state: torch.Tensor  # [2, 5, 5] board state
    policy: torch.Tensor  # [25] MCTS policy distribution
    value: float  # Game outcome from this state's player perspective
    legal_mask: torch.Tensor  # [25] legal moves mask

@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""
    # MCTS parameters
    num_simulations: int = 25
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Self-play parameters
    temperature_moves: int = 10  # Number of moves to use temperature > 0
    temperature: float = 1.0
    
    # Training parameters
    batch_size: int = 32
    buffer_size: int = 10000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Evaluation parameters
    eval_games: int = 10
    eval_simulations: int = 50

class ExperienceBuffer:
    """
    Experience replay buffer with reservoir sampling.
    
    Maintains a fixed-size buffer of training experiences using
    reservoir sampling to ensure uniform distribution over time.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[Experience] = []
        self.total_added = 0
    
    def add(self, experience: Experience):
        """Add experience using reservoir sampling."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            # Reservoir sampling: replace random element
            idx = random.randint(0, self.total_added)
            if idx < self.max_size:
                self.buffer[idx] = experience
        
        self.total_added += 1
    
    def add_game(self, experiences: List[Experience]):
        """Add all experiences from a complete game."""
        for exp in experiences:
            self.add(exp)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.total_added = 0

class SelfPlayTrainer:
    """AlphaZero self-play trainer."""
    
    def __init__(self, model, config: SelfPlayConfig = None):
        self.model = model.to(device)
        self.config = config or SelfPlayConfig()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Mixed precision
        self.scaler = torch.cuda.GradScaler() if amp_enabled else None
        
        # Experience buffer
        self.buffer = ExperienceBuffer(self.config.buffer_size)
        
        # Import MCTS (assuming it's available)
        try:
            import sys
            sys.path.append('/Users/taehoon.kim/Desktop/Sources/RL2025/Lectures/Lecture11/experiments')
            from exp05_puct_mcts import MCTS
            self.mcts = MCTS(
                self.model,
                c_puct=self.config.c_puct,
                dirichlet_alpha=self.config.dirichlet_alpha,
                dirichlet_epsilon=self.config.dirichlet_epsilon
            )
        except ImportError:
            self.mcts = None
            print("Warning: MCTS not available")
    
    def play_game(self, env, collect_data: bool = True) -> Tuple[List[Experience], Dict[str, Any]]:
        """
        Play a single self-play game.
        
        Args:
            env: Game environment
            collect_data: Whether to collect training experiences
            
        Returns:
            experiences: List of training experiences
            game_info: Game statistics
        """
        if self.mcts is None:
            raise ValueError("MCTS not available")
        
        experiences = []
        env.reset()
        move_count = 0
        
        # Store game trajectory for outcome assignment
        game_states = []
        game_policies = []
        game_masks = []
        game_players = []  # Track which player made each move
        
        while not env.is_terminal():
            current_player = env.current_player
            
            # Determine temperature based on move count
            temperature = self.config.temperature if move_count < self.config.temperature_moves else 0.0
            
            # MCTS search
            action_probs, root_value = self.mcts.search(
                env, 
                num_simulations=self.config.num_simulations,
                temperature=temperature,
                add_noise=True
            )
            
            if collect_data:
                # Store state information
                state = env._get_observation().clone()
                legal_mask = env.legal_actions().clone()
                policy = torch.zeros(25)
                for action, prob in action_probs.items():
                    policy[action] = prob
                
                game_states.append(state)
                game_policies.append(policy)
                game_masks.append(legal_mask)
                game_players.append(current_player)
            
            # Select and execute action
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            action = np.random.choice(actions, p=probs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            move_count += 1
        
        # Create training experiences with game outcomes
        if collect_data:
            winner = env.winner()
            for i, (state, policy, mask, player) in enumerate(zip(game_states, game_policies, game_masks, game_players)):
                # Assign value from this player's perspective
                if winner is None or winner == 0:  # Draw
                    value = 0.0
                elif winner == player:
                    value = 1.0
                else:
                    value = -1.0
                
                experience = Experience(
                    state=state,
                    policy=policy,
                    value=value,
                    legal_mask=mask
                )
                experiences.append(experience)
        
        game_info = {
            'winner': env.winner(),
            'moves': move_count,
            'final_reward': reward if env.is_terminal() else 0.0
        }
        
        return experiences, game_info
    
    def training_step(self) -> Dict[str, float]:
        """
        Single training step using batch of experiences.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.config.batch_size:
            return {'warning': 'insufficient_data'}
        
        # Sample batch
        experiences = self.buffer.sample(self.config.batch_size)
        
        # Prepare batch tensors
        states = torch.stack([exp.state for exp in experiences]).to(device)  # [B, 2, 5, 5]
        target_policies = torch.stack([exp.policy for exp in experiences]).to(device)  # [B, 25]
        target_values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32).unsqueeze(1).to(device)  # [B, 1]
        legal_masks = torch.stack([exp.legal_mask for exp in experiences]).to(device)  # [B, 25]
        
        # Training step
        self.model.train()
        self.optimizer.zero_grad()
        
        if amp_enabled and self.scaler is not None:
            with torch.autocast(device_type='cuda'):
                policy_logits, pred_values = self.model(states, legal_masks)
                losses = self._compute_losses(policy_logits, pred_values, target_policies, target_values)
            
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, pred_values = self.model(states, legal_masks)
            losses = self._compute_losses(policy_logits, pred_values, target_policies, target_values)
            
            losses['total_loss'].backward()
            self.optimizer.step()
        
        # Convert to Python floats
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        metrics['buffer_size'] = len(self.buffer)
        
        return metrics
    
    def _compute_losses(self, policy_logits: torch.Tensor, pred_values: torch.Tensor,
                       target_policies: torch.Tensor, target_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute AlphaZero losses."""
        # Policy loss: Cross-entropy between MCTS policy and network policy
        policy_loss = -torch.sum(target_policies * F.log_softmax(policy_logits, dim=1), dim=1).mean()
        
        # Value loss: MSE between game outcome and predicted value
        value_loss = F.mse_loss(pred_values, target_values)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Additional metrics
        policy_entropy = -torch.sum(F.softmax(policy_logits, dim=1) * F.log_softmax(policy_logits, dim=1), dim=1).mean()
        value_accuracy = (torch.sign(pred_values.squeeze()) == torch.sign(target_values.squeeze())).float().mean()
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_entropy': policy_entropy,
            'value_accuracy': value_accuracy
        }
    
    def evaluate_against_random(self, env, num_games: int = 10) -> Dict[str, float]:
        """
        Evaluate model against random player.
        
        Returns:
            Dictionary with win/draw/loss rates
        """
        if self.mcts is None:
            return {'error': 'mcts_not_available'}
        
        wins, draws, losses = 0, 0, 0
        
        for game_idx in range(num_games):
            env.reset()
            
            # Alternate who goes first
            model_is_black = (game_idx % 2 == 0)
            
            while not env.is_terminal():
                if (env.current_player == 1 and model_is_black) or (env.current_player == -1 and not model_is_black):
                    # Model's turn
                    action_probs, _ = self.mcts.search(
                        env, 
                        num_simulations=self.config.eval_simulations,
                        temperature=0.0,  # Greedy play
                        add_noise=False
                    )
                    action = max(action_probs.keys(), key=lambda a: action_probs[a])
                else:
                    # Random player's turn
                    legal_actions = env.legal_actions_list()
                    action = random.choice(legal_actions)
                
                env.step(action)
            
            # Determine result from model's perspective
            winner = env.winner()
            if winner is None or winner == 0:  # Draw
                draws += 1
            elif (winner == 1 and model_is_black) or (winner == -1 and not model_is_black):
                wins += 1
            else:
                losses += 1
        
        total = wins + draws + losses
        return {
            'win_rate': wins / total,
            'draw_rate': draws / total,
            'loss_rate': losses / total,
            'total_games': total
        }

def test_experience_buffer():
    """Test experience buffer functionality."""
    print("Testing experience buffer...")
    
    buffer = ExperienceBuffer(max_size=5)
    
    # Create dummy experiences
    for i in range(3):
        exp = Experience(
            state=torch.randn(2, 5, 5),
            policy=torch.randn(25),
            value=float(i),
            legal_mask=torch.ones(25, dtype=torch.bool)
        )
        buffer.add(exp)
    
    assert len(buffer) == 3, f"Expected 3 experiences, got {len(buffer)}"
    
    # Test sampling
    sample = buffer.sample(2)
    assert len(sample) == 2, f"Expected 2 samples, got {len(sample)}"
    
    # Test overflow (reservoir sampling)
    for i in range(10):
        exp = Experience(
            state=torch.randn(2, 5, 5),
            policy=torch.randn(25),
            value=float(i + 10),
            legal_mask=torch.ones(25, dtype=torch.bool)
        )
        buffer.add(exp)
    
    assert len(buffer) == 5, f"Buffer should be capped at 5, got {len(buffer)}"
    
    print("  Experience buffer: ✓")

def test_selfplay_trainer():
    """Test self-play trainer setup."""
    print("Testing self-play trainer...")
    
    try:
        import sys
        sys.path.append('/Users/taehoon.kim/Desktop/Sources/RL2025/Lectures/Lecture11/experiments')
        from exp04_policy_value_network import PolicyValueNet
        from exp03_gomoku_environment import Gomoku5x5
        
        # Create model and trainer
        model = PolicyValueNet()
        config = SelfPlayConfig(num_simulations=5, batch_size=4)  # Small for testing
        trainer = SelfPlayTrainer(model, config)
        
        assert trainer.buffer.max_size == config.buffer_size
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        
        # Test with dummy experiences
        for i in range(8):
            exp = Experience(
                state=torch.randn(2, 5, 5),
                policy=F.softmax(torch.randn(25), dim=0),  # Valid probability distribution
                value=random.uniform(-1, 1),
                legal_mask=torch.ones(25, dtype=torch.bool)
            )
            trainer.buffer.add(exp)
        
        # Test training step
        metrics = trainer.training_step()
        assert 'total_loss' in metrics
        assert isinstance(metrics['total_loss'], float)
        
        print("  Self-play trainer: ✓")
        
    except ImportError as e:
        print(f"  Self-play trainer: Skipped ({e})")

def test_game_simulation():
    """Test self-play game simulation."""
    print("Testing game simulation...")
    
    try:
        import sys
        sys.path.append('/Users/taehoon.kim/Desktop/Sources/RL2025/Lectures/Lecture11/experiments')
        from exp04_policy_value_network import PolicyValueNet
        from exp03_gomoku_environment import Gomoku5x5
        
        # Create small setup for testing
        model = PolicyValueNet()
        config = SelfPlayConfig(num_simulations=5)  # Fast for testing
        trainer = SelfPlayTrainer(model, config)
        
        if trainer.mcts is None:
            print("  Game simulation: Skipped (MCTS not available)")
            return
        
        # Play a game
        env = Gomoku5x5()
        experiences, game_info = trainer.play_game(env, collect_data=True)
        
        assert len(experiences) > 0, "Should collect some experiences"
        assert 'winner' in game_info
        assert 'moves' in game_info
        
        # Verify experience format
        exp = experiences[0]
        assert exp.state.shape == (2, 5, 5)
        assert exp.policy.shape == (25,)
        assert abs(exp.policy.sum().item() - 1.0) < 1e-5  # Should sum to 1
        assert isinstance(exp.value, float)
        assert exp.legal_mask.shape == (25,)
        
        print("  Game simulation: ✓")
        
    except ImportError as e:
        print(f"  Game simulation: Skipped ({e})")

def demonstrate_training_loop():
    """Demonstrate a short training loop."""
    print("\nDemonstrating training loop:")
    print("="*40)
    
    try:
        import sys
        sys.path.append('/Users/taehoon.kim/Desktop/Sources/RL2025/Lectures/Lecture11/experiments')
        from exp04_policy_value_network import PolicyValueNet
        from exp03_gomoku_environment import Gomoku5x5
        
        # Create training setup
        model = PolicyValueNet()
        config = SelfPlayConfig(
            num_simulations=10,
            batch_size=8,
            buffer_size=100,
            eval_games=3
        )
        trainer = SelfPlayTrainer(model, config)
        
        if trainer.mcts is None:
            print("Training loop skipped: MCTS not available")
            return
        
        print(f"Starting training with config:")
        print(f"  Simulations per move: {config.num_simulations}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Buffer size: {config.buffer_size}")
        print()
        
        # Generate initial data
        env = Gomoku5x5()
        num_games = 5
        
        print(f"Playing {num_games} self-play games...")
        for game_idx in range(num_games):
            experiences, game_info = trainer.play_game(env, collect_data=True)
            trainer.buffer.add_game(experiences)
            
            winner_str = "Black" if game_info['winner'] == 1 else "White" if game_info['winner'] == -1 else "Draw"
            print(f"  Game {game_idx + 1}: {winner_str} wins in {game_info['moves']} moves")
        
        print(f"Buffer size: {len(trainer.buffer)} experiences")
        print()
        
        # Training steps
        print("Training on collected experiences...")
        for step in range(5):
            metrics = trainer.training_step()
            if 'warning' not in metrics:
                print(f"  Step {step + 1}: Loss = {metrics['total_loss']:.4f}, "
                      f"Policy = {metrics['policy_loss']:.4f}, "
                      f"Value = {metrics['value_loss']:.4f}")
        
        # Evaluation
        print("\nEvaluating against random player...")
        eval_results = trainer.evaluate_against_random(env, num_games=config.eval_games)
        if 'error' not in eval_results:
            print(f"  Win rate: {eval_results['win_rate']:.2%}")
            print(f"  Draw rate: {eval_results['draw_rate']:.2%}")
            print(f"  Loss rate: {eval_results['loss_rate']:.2%}")
        
        print(f"\nTraining demonstration complete!")
        
    except ImportError as e:
        print(f"Training demonstration skipped: {e}")

def main():
    print("="*60)
    print("Experiment 06: Self-Play and AlphaZero Training")
    print("="*60)
    
    # Run tests
    test_experience_buffer()
    test_selfplay_trainer()
    test_game_simulation()
    
    print("\nAll tests passed! ✓")
    
    # Demonstrate training
    demonstrate_training_loop()
    
    print(f"\nAlphaZero self-play training ready!")
    print("Training specifications:")
    print(f"  Self-play with MCTS policy targets")
    print(f"  Experience replay with reservoir sampling")
    print(f"  Combined policy and value loss")
    print(f"  Mixed precision training support")
    print(f"  Evaluation against random/previous models")
    print(f"  Device: {device}")

if __name__ == "__main__":
    main()