#!/usr/bin/env python3
"""
RL2025 - Lecture 11: Experiment 03 - Minimal Gomoku 5×5 Environment

Implements a self-contained Gomoku environment on a 5×5 board for MCTS
and AlphaZero experiments. Supports fast vectorized operations and 
proper perspective handling for two-player games.

Learning objectives:
- Implement two-player board game environment
- Handle player perspectives and state representation
- Create efficient legal move checking
- Support terminal state detection and winning conditions

Prerequisites: NumPy, PyTorch
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

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

class Player(IntEnum):
    """Player enumeration."""
    EMPTY = 0
    BLACK = 1
    WHITE = -1

class GameState(IntEnum):
    """Game state enumeration."""
    ONGOING = 0
    BLACK_WIN = 1
    WHITE_WIN = -1
    DRAW = 2

@dataclass
class GameInfo:
    """Game information structure."""
    winner: Optional[int] = None
    is_terminal: bool = False
    move_count: int = 0
    last_move: Optional[int] = None

class Gomoku5x5:
    """
    Minimal Gomoku 5×5 Environment.
    
    Game Rules:
    - Two players (Black=+1, White=-1) alternate placing stones
    - Black plays first
    - Win condition: 5 stones in a row (horizontal, vertical, diagonal)
    - Draw condition: Board full with no winner
    
    State Representation:
    - Board: [5, 5] array with values {-1, 0, +1}
    - Observation: [2, 5, 5] tensor (current player stones, opponent stones)
    - Action space: 25 positions (0-24, row-major order)
    """
    
    def __init__(self):
        self.board_size = 5
        self.action_space = 25  # 5x5 positions
        self.reset()
    
    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Returns:
            obs: [2, 5, 5] tensor (current player stones, opponent stones)
            info: Game information dictionary
        """
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = Player.BLACK
        self.move_count = 0
        self.game_state = GameState.ONGOING
        self.last_move = None
        
        obs = self._get_observation()
        info = GameInfo(is_terminal=False, move_count=0).__dict__
        
        return obs, info
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Execute one game step.
        
        Args:
            action: Position index (0-24)
            
        Returns:
            obs: [2, 5, 5] observation tensor
            reward: Reward for current player (0, +1, -1)
            terminated: True if game ended
            truncated: Always False (no time limit)
            info: Game information
        """
        assert 0 <= action < 25, f"Action {action} out of range [0, 24]"
        assert not self._is_terminal(), "Game already terminated"
        
        row, col = divmod(action, 5)
        assert self.board[row, col] == Player.EMPTY, f"Position ({row}, {col}) already occupied"
        
        # Place stone
        self.board[row, col] = self.current_player
        self.last_move = action
        self.move_count += 1
        
        # Check for winner
        winner = self._check_winner()
        reward = 0.0
        terminated = False
        
        if winner != Player.EMPTY:
            self.game_state = GameState.BLACK_WIN if winner == Player.BLACK else GameState.WHITE_WIN
            reward = 1.0 if winner == self.current_player else -1.0
            terminated = True
        elif self.move_count == 25:  # Board full
            self.game_state = GameState.DRAW
            reward = 0.0
            terminated = True
        else:
            # Switch players
            self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        
        obs = self._get_observation()
        info = GameInfo(
            winner=winner if winner != Player.EMPTY else None,
            is_terminal=terminated,
            move_count=self.move_count,
            last_move=action
        ).__dict__
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self) -> torch.Tensor:
        """
        Get observation tensor from current player's perspective.
        
        Returns:
            obs: [2, 5, 5] tensor where:
                 obs[0] = current player's stones
                 obs[1] = opponent's stones
        """
        current_stones = (self.board == self.current_player).astype(np.float32)
        opponent_stones = (self.board == -self.current_player).astype(np.float32)
        
        obs = np.stack([current_stones, opponent_stones], axis=0)
        return torch.from_numpy(obs)
    
    def legal_actions(self) -> torch.Tensor:
        """
        Get legal actions mask.
        
        Returns:
            mask: [25] boolean tensor, True for legal moves
        """
        legal_mask = (self.board.flatten() == Player.EMPTY)
        return torch.from_numpy(legal_mask)
    
    def legal_actions_list(self) -> List[int]:
        """Get list of legal action indices."""
        return np.where(self.board.flatten() == Player.EMPTY)[0].tolist()
    
    def _is_terminal(self) -> bool:
        """Check if game is in terminal state."""
        return self.game_state != GameState.ONGOING
    
    def _check_winner(self) -> int:
        """
        Check for winner using vectorized operations.
        
        Returns:
            Player.BLACK, Player.WHITE, or Player.EMPTY
        """
        # Check all possible 5-in-a-row patterns
        # Since board is 5x5, any 5-in-a-row wins immediately
        
        # Horizontal checks
        for row in range(5):
            if abs(self.board[row].sum()) == 5:
                return self.board[row, 0]  # All same player
        
        # Vertical checks  
        for col in range(5):
            if abs(self.board[:, col].sum()) == 5:
                return self.board[0, col]  # All same player
        
        # Diagonal checks
        main_diag = np.diag(self.board)
        if abs(main_diag.sum()) == 5:
            return main_diag[0]
        
        anti_diag = np.diag(np.fliplr(self.board))
        if abs(anti_diag.sum()) == 5:
            return anti_diag[0]
        
        return Player.EMPTY
    
    def winner(self) -> Optional[int]:
        """Get winner of terminated game."""
        if self.game_state == GameState.BLACK_WIN:
            return Player.BLACK
        elif self.game_state == GameState.WHITE_WIN:
            return Player.WHITE
        elif self.game_state == GameState.DRAW:
            return 0  # Draw
        return None
    
    def is_terminal(self) -> bool:
        """Check if game is terminal."""
        return self._is_terminal()
    
    def get_result(self, player: int) -> float:
        """
        Get game result from specified player's perspective.
        
        Args:
            player: Player.BLACK or Player.WHITE
            
        Returns:
            +1 if player won, -1 if player lost, 0 if draw/ongoing
        """
        if not self.is_terminal():
            return 0.0
        
        winner = self.winner()
        if winner is None:
            return 0.0
        elif winner == 0:  # Draw
            return 0.0
        elif winner == player:
            return 1.0
        else:
            return -1.0
    
    def clone(self) -> 'Gomoku5x5':
        """Create a deep copy of the current game state."""
        clone = Gomoku5x5()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.move_count = self.move_count
        clone.game_state = self.game_state
        clone.last_move = self.last_move
        return clone
    
    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {Player.EMPTY: '.', Player.BLACK: 'X', Player.WHITE: 'O'}
        lines = []
        lines.append("  0 1 2 3 4")
        for i, row in enumerate(self.board):
            row_str = f"{i} " + " ".join(symbols[cell] for cell in row)
            lines.append(row_str)
        lines.append(f"Current player: {'Black (X)' if self.current_player == Player.BLACK else 'White (O)'}")
        lines.append(f"Move count: {self.move_count}")
        return "\n".join(lines)

def test_gomoku_basic():
    """Test basic Gomoku functionality."""
    print("Testing basic Gomoku functionality...")
    
    env = Gomoku5x5()
    obs, info = env.reset()
    
    # Check initial state
    assert obs.shape == (2, 5, 5), f"Wrong observation shape: {obs.shape}"
    assert not info['is_terminal'], "Game should not be terminal initially"
    assert len(env.legal_actions_list()) == 25, "All positions should be legal initially"
    
    # Test move execution
    action = 12  # Center position (2, 2)
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (2, 5, 5), "Observation shape changed"
    assert reward == 0.0, "Should be no reward for non-terminal move"
    assert not terminated, "Game should not terminate after one move"
    assert len(env.legal_actions_list()) == 24, "Should have 24 legal moves after first move"
    
    print("  Basic functionality: ✓")

def test_winning_condition():
    """Test winning condition detection."""
    print("Testing winning conditions...")
    
    env = Gomoku5x5()
    env.reset()
    
    # Create horizontal win for Black (player 1)
    moves = [0, 5, 1, 6, 2, 7, 3, 8, 4]  # Black: row 0, White: row 1
    
    for i, action in enumerate(moves):
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i == 8:  # Final move creates 5-in-a-row
            assert terminated, "Game should terminate on winning move"
            assert reward == 1.0, f"Winner should get +1 reward, got {reward}"
            assert info['winner'] == Player.BLACK, "Black should win"
        else:
            assert not terminated, f"Game terminated prematurely at move {i}"
    
    print("  Winning condition: ✓")

def test_perspective_handling():
    """Test player perspective in observations."""
    print("Testing perspective handling...")
    
    env = Gomoku5x5()
    obs, _ = env.reset()
    
    # Black plays first - should see empty board in both channels
    assert torch.sum(obs) == 0, "Initial observation should be empty"
    
    # Black plays center
    obs, _, _, _, _ = env.step(12)  # (2, 2)
    
    # White's turn - should see Black's stone in opponent channel
    assert obs[0].sum() == 0, "Current player (White) should have no stones"
    assert obs[1].sum() == 1, "Opponent (Black) should have 1 stone"
    assert obs[1, 2, 2] == 1, "Black's stone should be at center"
    
    # White plays
    obs, _, _, _, _ = env.step(13)  # (2, 3)
    
    # Black's turn - should see own stone in channel 0, opponent's in channel 1
    assert obs[0].sum() == 1, "Current player (Black) should have 1 stone"
    assert obs[1].sum() == 1, "Opponent (White) should have 1 stone"
    assert obs[0, 2, 2] == 1, "Black's original stone should be in current player channel"
    assert obs[1, 2, 3] == 1, "White's stone should be in opponent channel"
    
    print("  Perspective handling: ✓")

def test_legal_moves():
    """Test legal move checking."""
    print("Testing legal move checking...")
    
    env = Gomoku5x5()
    env.reset()
    
    # Initially all moves should be legal
    legal_mask = env.legal_actions()
    assert legal_mask.sum() == 25, "All positions should be legal initially"
    
    # Play some moves
    moves = [12, 13, 14]
    for move in moves:
        env.step(move)
    
    # Check that played positions are no longer legal
    legal_mask = env.legal_actions()
    assert legal_mask.sum() == 22, f"Should have 22 legal moves, got {legal_mask.sum()}"
    
    for move in moves:
        assert not legal_mask[move], f"Move {move} should no longer be legal"
    
    print("  Legal move checking: ✓")

def test_game_cloning():
    """Test game state cloning."""
    print("Testing game state cloning...")
    
    env = Gomoku5x5()
    env.reset()
    
    # Play some moves
    env.step(12)
    env.step(13)
    
    # Clone the game
    clone = env.clone()
    
    # Verify clone is identical
    assert np.array_equal(env.board, clone.board), "Cloned board should be identical"
    assert env.current_player == clone.current_player, "Current player should match"
    assert env.move_count == clone.move_count, "Move count should match"
    
    # Verify independence
    env.step(14)
    assert not np.array_equal(env.board, clone.board), "Clone should be independent"
    
    print("  Game state cloning: ✓")

def demo_game():
    """Demonstrate a complete game."""
    print("\nDemonstrating complete game:")
    print("="*40)
    
    env = Gomoku5x5()
    obs, info = env.reset()
    
    print("Initial board:")
    print(env)
    print()
    
    # Play a sample game
    moves = [12, 6, 13, 7, 11, 8, 14, 9, 10]  # Black wins horizontally
    
    for i, action in enumerate(moves):
        print(f"Move {i+1}: Player {'Black' if i % 2 == 0 else 'White'} plays position {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(env)
        print(f"Reward: {reward}, Terminal: {terminated}")
        
        if terminated:
            winner = "Black" if info['winner'] == Player.BLACK else "White" if info['winner'] == Player.WHITE else "Draw"
            print(f"Game ended! Winner: {winner}")
            break
        print()

def main():
    print("="*60)
    print("Experiment 03: Minimal Gomoku 5×5 Environment")
    print("="*60)
    
    # Run all tests
    test_gomoku_basic()
    test_winning_condition()
    test_perspective_handling()
    test_legal_moves()
    test_game_cloning()
    
    print("\nAll tests passed! ✓")
    
    # Run demonstration
    demo_game()
    
    print(f"\nGomoku environment ready for MCTS and AlphaZero!")
    print("Environment specifications:")
    print(f"  Board size: 5×5")
    print(f"  Action space: 25 positions")
    print(f"  Observation shape: [2, 5, 5] (current player, opponent)")
    print(f"  Terminal rewards: +1 (win), -1 (loss), 0 (draw)")

if __name__ == "__main__":
    main()