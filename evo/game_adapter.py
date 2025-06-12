"""
Game state adapter to bridge existing AbstractGame with EA requirements.
"""

import copy
import pickle
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from games.abstract_game import AbstractGame
    from .features import StateFeatures


class StormboundAdapter:
    """Adapter to bridge existing AbstractGame with EA requirements"""
    
    def __init__(self, game: 'AbstractGame'):
        self.game = game
        self.initial_observation = None
        self._cached_observation = None
    
    def clone_state(self) -> 'StormboundAdapter':
        """Create a deep copy of the current game state"""
        try:
            # Attempt to serialize and deserialize the game state
            # This is a deep copy that preserves all game state
            game_copy = copy.deepcopy(self.game)
            new_adapter = StormboundAdapter(game_copy)
            new_adapter._cached_observation = self._cached_observation
            return new_adapter
        except Exception as e:
            # If deep copy fails, create a new game and try to restore state
            # This is a fallback - in practice we might need game-specific state saving
            from games.stormbound import Game
            new_game = Game(seed=None)
            new_adapter = StormboundAdapter(new_game)
            # TODO: Implement proper state restoration if needed
            return new_adapter
        
    def get_legal_actions(self) -> List[int]:
        """Get legal actions for current player"""
        return self.game.legal_actions()
        
    def apply_action(self, action: int) -> 'StormboundAdapter':
        """Apply action and return new state"""
        new_state = self.clone_state()
        observation, reward, done = new_state.game.step(action)
        new_state._cached_observation = observation
        return new_state
        
    def extract_features(self) -> 'StateFeatures':
        """Extract meaningful features from current game state"""
        from .features import StateFeatures
        
        if self._cached_observation is not None:
            observation = self._cached_observation
        else:
            observation = self.game.env.get_observation()
            self._cached_observation = observation
            
        return StateFeatures(observation, self.game.to_play())
        
    def is_terminal(self) -> bool:
        """Check if game is over"""
        try:
            return self.game.env.have_winner() is not None
        except AttributeError:
            # Fallback if have_winner doesn't exist
            legal_actions = self.game.legal_actions()
            return len(legal_actions) == 0
        
    def get_result(self) -> Tuple[float, float]:
        """Get game result (player1_score, player2_score)"""
        try:
            winner = self.game.env.have_winner()
            if winner == 0:
                return (1.0, 0.0)  # Player 0 wins
            elif winner == 1:
                return (0.0, 1.0)  # Player 1 wins
            else:
                return (0.5, 0.5)  # Draw/ongoing
        except AttributeError:
            # Fallback if have_winner doesn't exist
            return (0.5, 0.5)  # Assume draw if we can't determine winner
    
    def get_current_player(self) -> int:
        """Get the current player index"""
        return self.game.to_play()
    
    def get_observation(self) -> np.ndarray:
        """Get the current game observation"""
        if self._cached_observation is not None:
            return self._cached_observation
        else:
            observation = self.game.env.get_observation()
            self._cached_observation = observation
            return observation 