"""
Evolutionary Stormbound Game

A variant of the Stormbound game that integrates with DeckEvolutionConfig
to support the three-phase evolutionary training strategy.
"""

import json
import numpy as np
from typing import List, Optional

from games.abstract_game import AbstractGame
from games.stormbound import Stormbound
from utils import DeckEvolutionConfig
from enums import Faction, PlayerOrder
from player import Player
from board import Board
from point import Point


class EvolutionaryStormbound(AbstractGame):
    """
    Stormbound game with evolutionary deck configuration support.

    This class integrates the three-phase evolutionary strategy:
    1. Exploit: Uses pre-built archetype decks
    2. Explore: Gradually introduces randomness
    3. Balance: Maintains steady mix of archetypes and random decks
    """

    def __init__(self, seed: Optional[int] = None, generation: int = 0, 
                 deck_config: Optional[DeckEvolutionConfig] = None):
        """
        Initialize evolutionary Stormbound game.

        Args:
            seed: Random seed for game
            generation: Current generation number (affects deck configuration)
            deck_config: Deck evolution configuration. If None, uses default.
        """
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.generation = generation

        # Set up deck configuration
        if deck_config is None:
            deck_config = self._create_default_deck_config()
        self.deck_config = deck_config

        # Get decks for current generation
        self.player1_deck, self.player2_deck = self.deck_config.get_deck_configuration(generation)

        # Initialize players with evolutionary decks
        self.local = Player(
            self.deck_config.player1_faction, 
            self.player1_deck, 
            PlayerOrder.FIRST, 
            self.random
        )
        self.remote = Player(
            self.deck_config.player2_faction, 
            self.player2_deck, 
            PlayerOrder.SECOND, 
            self.random
        )

        # Initialize board
        self.board = Board(self.local, self.remote, self.random)
        self.player = 1

        # Load action mappings (same as original Stormbound)
        with open("actions.txt", "r") as f:
            self.actions = f.read().splitlines()

        with open("cards.json", "r", encoding="utf-8") as f:
            self.cards = json.load(f)

    def _create_default_deck_config(self) -> DeckEvolutionConfig:
        """Create a default deck configuration if none provided."""
        # Import default archetypes (same as original Stormbound)
        from cards.ua07 import UA07
        from cards.u007 import U007
        from cards.u306 import U306
        from cards.u061 import U061
        from cards.b304 import B304
        from cards.u305 import U305
        from cards.u320 import U320
        from cards.u302 import U302
        from cards.u313 import U313
        from cards.ua02 import UA02
        from cards.ut32 import UT32
        from cards.u316 import U316

        from cards.u001 import U001
        from cards.u053 import U053
        from cards.ue01 import UE01
        from cards.u211 import U211
        from cards.u206 import U206
        from cards.u071 import U071
        from cards.u020 import U020
        from cards.s013 import S013
        from cards.b001 import B001


        # Create default archetype decks
        ironclad_deck = [
            UA07(), U007(), U306(), U061(), B304(), U305(),
            U320(), U302(), U313(), UA02(), UT32(), U316()
        ]

        swarm_deck = [
            UA07(), U007(), U001(), U053(), UE01(), U211(),
            U206(), U071(), U020(), S013(), B001(), U061()
        ]

        return DeckEvolutionConfig(
            player1_archetype=ironclad_deck,
            player2_archetype=swarm_deck,
            exploit_generations=30,
            explore_generations=30,
            max_random_ratio=0.5,
            balance_archetype_ratio=0.7
        )

    def to_play(self):
        """Return current player (0 or 1)."""
        return 0 if self.player == 1 else 1

    def reset(self):
        """Reset the game for a new round."""
        # Get fresh decks for current generation
        self.player1_deck, self.player2_deck = self.deck_config.get_deck_configuration(self.generation)

        # Create new players with fresh decks
        self.local = Player(
            self.deck_config.player1_faction, 
            self.player1_deck, 
            PlayerOrder.FIRST, 
            self.random
        )
        self.remote = Player(
            self.deck_config.player2_faction, 
            self.player2_deck, 
            PlayerOrder.SECOND, 
            self.random
        )

        # Reset board
        self.board = Board(self.local, self.remote, self.random)
        self.player = 1

        return self.get_observation()

    def set_generation(self, generation: int):
        """Update the generation for deck evolution."""
        self.generation = generation
        # Note: Deck configuration will be applied on next reset()

    def get_phase_info(self) -> dict:
        """Get current phase information."""
        return self.deck_config.get_phase_info(self.generation)

    # Delegate other methods to the core Stormbound implementation
    def step(self, action: int):
        """Apply action to the game (same as original Stormbound)."""
        # Use the same step logic as original Stormbound
        stormbound_instance = Stormbound(self.seed)
        stormbound_instance.board = self.board
        stormbound_instance.player = self.player
        stormbound_instance.actions = self.actions
        stormbound_instance.cards = self.cards

        result = stormbound_instance.step(action)

        # Update our state
        self.board = stormbound_instance.board
        self.player = stormbound_instance.player

        return result

    def legal_actions(self):
        """Get legal actions (same as original Stormbound)."""
        stormbound_instance = Stormbound(self.seed)
        stormbound_instance.board = self.board
        stormbound_instance.player = self.player

        return stormbound_instance.legal_actions()

    def get_observation(self):
        """Get game observation (same as original Stormbound)."""
        stormbound_instance = Stormbound(self.seed)
        stormbound_instance.board = self.board
        stormbound_instance.player = self.player

        return stormbound_instance.get_observation()

    def have_winner(self):
        """Check if game has winner (same as original Stormbound)."""
        stormbound_instance = Stormbound(self.seed)
        stormbound_instance.board = self.board

        return stormbound_instance.have_winner()

    def render(self):
        """Display the game state."""
        print(f"=== Evolutionary Stormbound (Generation {self.generation}) ===")
        phase_info = self.get_phase_info()
        print(f"Phase: {phase_info['phase']} | Random Ratio: {phase_info['random_ratio']:.2f}")

        # Show deck info
        print(f"Player 1 ({self.deck_config.player1_faction.name}): {len(self.player1_deck)} cards")
        print(f"Player 2 ({self.deck_config.player2_faction.name}): {len(self.player2_deck)} cards")

        # Show board state
        print(self.board)

    def close(self):
        """Close the game."""
        pass

    def expert_agent(self):
        """Expert agent for evaluation (placeholder)."""
        # Return a random legal action for now
        legal_actions = self.legal_actions()
        if legal_actions:
            return self.random.choice(legal_actions)
        return 155  # Pass action

    def action_to_string(self, action_number: int) -> str:
        """Convert action number to string."""
        if action_number < len(self.actions):
            return self.actions[action_number]
        return f"Action_{action_number}" 