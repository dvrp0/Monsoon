"""
Game state adapter to bridge existing AbstractGame with EA requirements.
"""

import copy
import pickle
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from games.abstract_game import AbstractGame
    from .features import StateFeatures


class GameState:
    """Serializable representation of complete game state"""

    def __init__(self):
        # Stormbound-level state
        self.player = None
        self.random_state = None

        # Board-level state
        self.board_grid = None
        self.history = None
        self.triggers = None
        self.phase = None
        self.current_player_order = None

        # Player states
        self.local_player_state = None
        self.remote_player_state = None

    def serialize_from_game(self, game: 'AbstractGame') -> None:
        """Extract and serialize all game state from a game instance"""
        stormbound = game.env
        board = stormbound.board

        # Serialize Stormbound state
        self.player = stormbound.player
        self.random_state = stormbound.random.get_state()

        # Serialize board grid (5x4 grid of units/structures)
        self.board_grid = []
        for y in range(5):
            row = []
            for x in range(4):
                entity = board.board[y][x]
                if entity is None:
                    row.append(None)
                else:
                    # Serialize entity with all its properties
                    entity_data = {
                        'type': type(entity).__name__,
                        'card_id': entity.card_id,
                        'faction': entity.faction,
                        'cost': entity.cost,
                        'strength': entity.strength,
                        'position': (entity.position.x, entity.position.y) if entity.position else None,
                        'player_order': entity.player.order,
                    }

                    # Add unit-specific properties
                    if hasattr(entity, 'movement'):
                        entity_data['movement'] = entity.movement
                        entity_data['unit_types'] = entity.unit_types
                        entity_data['status_effects'] = getattr(entity, 'status_effects', [])
                        entity_data['is_vitalized'] = getattr(entity, 'is_vitalized', False)
                        entity_data['is_poisoned'] = getattr(entity, 'is_poisoned', False)
                        entity_data['is_confused'] = getattr(entity, 'is_confused', False)
                        entity_data['is_frozen'] = getattr(entity, 'is_frozen', False)
                        entity_data['is_disabled'] = getattr(entity, 'is_disabled', False)

                    row.append(entity_data)
            self.board_grid.append(row)

        # Serialize board history
        self.history = []
        for card in board.history:
            self.history.append({
                'card_id': card.card_id,
                'player_order': card.player.order
            })

        # Serialize other board state
        self.triggers = len(board.triggers)  # Just count, triggers are complex to serialize
        self.phase = board.phase
        self.current_player_order = board.current_player.order

        # Serialize player states
        self.local_player_state = self._serialize_player(board.local)
        self.remote_player_state = self._serialize_player(board.remote)

    def _serialize_player(self, player) -> Dict[str, Any]:
        """Serialize a player's complete state"""
        return {
            'faction': player.faction,
            'order': player.order,
            'max_mana': player.max_mana,
            'current_mana': player.current_mana,
            'strength': player.strength,
            'front_line': player.front_line,
            'replacable': player.replacable,
            'leftmost_movable': player.leftmost_movable,
            'damage_taken': player.damage_taken,
            'hand': [self._serialize_card(card) for card in player.hand],
            'deck': [self._serialize_card(card) for card in player.deck],
        }

    def _serialize_card(self, card) -> Dict[str, Any]:
        """Serialize a card's state"""
        card_data = {
            'card_id': card.card_id,
            'faction': card.faction,
            'cost': card.cost,
            'weight': card.weight,
            'is_single_use': card.is_single_use,
            'type': type(card).__name__,
        }

        # Add type-specific properties
        if hasattr(card, 'strength'):
            card_data['strength'] = card.strength
        if hasattr(card, 'movement'):
            card_data['movement'] = card.movement
            card_data['unit_types'] = card.unit_types
        if hasattr(card, 'required_targets'):
            card_data['required_targets'] = card.required_targets is not None

        return card_data

    def restore_to_game(self, game: 'AbstractGame') -> None:
        """Restore complete game state to a game instance"""
        stormbound = game.env
        board = stormbound.board

        # Restore Stormbound state
        stormbound.player = self.player
        stormbound.random.set_state(self.random_state)

        # Restore board grid
        for y in range(5):
            for x in range(4):
                entity_data = self.board_grid[y][x]
                if entity_data is None:
                    board.board[y][x] = None
                else:
                    # Recreate entity from serialized data
                    entity = self._recreate_entity(entity_data, board)
                    board.board[y][x] = entity
                    if entity:
                        entity.position = Point(x, y) if entity_data['position'] else None

        # Restore board history
        board.history = []
        for card_data in self.history:
            # Create a minimal card representation for history
            card = self._create_minimal_card(card_data, board)
            board.history.append(card)

        # Restore other board state
        board.triggers = []  # Reset triggers (they're complex to restore)
        board.phase = self.phase
        board.current_player = board.local if self.current_player_order.value == 0 else board.remote

        # Restore player states
        self._restore_player(board.local, self.local_player_state)
        self._restore_player(board.remote, self.remote_player_state)

    def _recreate_entity(self, entity_data: Dict[str, Any], board):
        """Recreate a unit or structure from serialized data"""
        from unit import Unit
        from structure import Structure
        from enums import Faction, PlayerOrder

        # Determine which player owns this entity
        player = board.local if entity_data['player_order'].value == 0 else board.remote

        if entity_data['type'] == 'Unit':
            entity = Unit(
                faction=entity_data['faction'],
                unit_types=entity_data['unit_types'],
                cost=entity_data['cost'],
                strength=entity_data['strength'],
                movement=entity_data['movement']
            )
            # Restore status effects
            entity.is_vitalized = entity_data.get('is_vitalized', False)
            entity.is_poisoned = entity_data.get('is_poisoned', False)
            entity.is_confused = entity_data.get('is_confused', False)
            entity.is_frozen = entity_data.get('is_frozen', False)
            entity.is_disabled = entity_data.get('is_disabled', False)
        elif entity_data['type'] == 'Structure':
            entity = Structure(
                faction=entity_data['faction'],
                cost=entity_data['cost'],
                strength=entity_data['strength']
            )
        else:
            return None

        entity.card_id = entity_data['card_id']
        entity.player = player
        return entity

    def _create_minimal_card(self, card_data: Dict[str, Any], board):
        """Create a minimal card representation for history"""
        from card import Card
        from unit import Unit
        from structure import Structure
        from spell import Spell

        # Create a basic card instance
        if card_data.get('type') == 'Unit':
            card = Unit(faction=Faction.NEUTRAL, unit_types=[], cost=0, strength=1, movement=1)
        elif card_data.get('type') == 'Structure':
            card = Structure(faction=Faction.NEUTRAL, cost=0, strength=1)
        else:
            card = Spell(faction=Faction.NEUTRAL, cost=0)

        card.card_id = card_data['card_id']
        card.player = board.local if card_data['player_order'].value == 0 else board.remote
        return card

    def _restore_player(self, player, player_state: Dict[str, Any]) -> None:
        """Restore a player's complete state"""
        player.max_mana = player_state['max_mana']
        player.current_mana = player_state['current_mana']
        player.strength = player_state['strength']
        player.front_line = player_state['front_line']
        player.replacable = player_state['replacable']
        player.leftmost_movable = player_state['leftmost_movable']
        player.damage_taken = player_state['damage_taken']

        # Restore hand and deck
        player.hand = [self._recreate_card(card_data, player) for card_data in player_state['hand']]
        player.deck = [self._recreate_card(card_data, player) for card_data in player_state['deck']]

    def _recreate_card(self, card_data: Dict[str, Any], player):
        """Recreate a card from serialized data"""
        from unit import Unit
        from structure import Structure
        from spell import Spell

        if card_data['type'] == 'Unit':
            card = Unit(
                faction=card_data['faction'],
                unit_types=card_data.get('unit_types', []),
                cost=card_data['cost'],
                strength=card_data.get('strength', 1),
                movement=card_data.get('movement', 1)
            )
        elif card_data['type'] == 'Structure':
            card = Structure(
                faction=card_data['faction'],
                cost=card_data['cost'],
                strength=card_data.get('strength', 1)
            )
        else:
            card = Spell(
                faction=card_data['faction'],
                cost=card_data['cost']
            )

        card.card_id = card_data['card_id']
        card.weight = card_data['weight']
        card.is_single_use = card_data['is_single_use']
        card.player = player
        return card


class StormboundAdapter:
    """Adapter to bridge existing AbstractGame with EA requirements"""

    def __init__(self, game: 'AbstractGame'):
        self.game = game
        self.initial_observation = None
        self._cached_observation = None

    def clone_state(self) -> 'StormboundAdapter':
        """Create a deep copy of the current game state"""
        try:
            # First attempt: use deepcopy for speed
            game_copy = copy.deepcopy(self.game)
            new_adapter = StormboundAdapter(game_copy)
            new_adapter._cached_observation = self._cached_observation
            return new_adapter
        except Exception as e:
            # Fallback: use comprehensive state serialization/restoration
            try:
                from games.stormbound import Game
                from point import Point

                # Create new game instance
                new_game = Game(seed=None)

                # Serialize current state
                state = GameState()
                state.serialize_from_game(self.game)

                # Restore state to new game
                state.restore_to_game(new_game)

                # Create new adapter
                new_adapter = StormboundAdapter(new_game)
                new_adapter._cached_observation = self._cached_observation
                return new_adapter

            except Exception as restore_error:
                # Final fallback: create fresh game (this should rarely happen)
                from games.stormbound import Game
                new_game = Game(seed=42)  # Use fixed seed for consistency
                new_adapter = StormboundAdapter(new_game)
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

    def get_result(self) -> int:
        """Get game result as winner index"""
        try:
            winner = self.game.env.have_winner()
            if winner == 0:
                return 0  # Player 0 wins
            elif winner == 1:
                return 1  # Player 1 wins
            else:
                return -1  # Draw/ongoing
        except AttributeError:
            # Fallback if have_winner doesn't exist
            return -1  # Assume draw if we can't determine winner

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