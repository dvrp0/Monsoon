from enums import Faction
from colorama import Back, Fore, Style
import random
from typing import List, Tuple, Optional

def get_faction_color(faction: Faction):
    match faction:
        case Faction.NEUTRAL:
            back = Back.LIGHTBLACK_EX
            fore = Fore.WHITE
        case Faction.WINTER:
            back = Back.BLUE
            fore = Fore.WHITE
        case Faction.SWARM:
            back = Back.YELLOW
            fore = Fore.BLACK
        case Faction.IRONCLAD:
            back = Back.RED
            fore = Fore.WHITE
        case Faction.SHADOWFEN:
            back = Back.GREEN
            fore = Fore.BLACK

    return back, fore

def generate_random_deck(faction: Faction, original: List = None, preserve_ratio: float = 0.0) -> List:
    # Import cards inside the function to avoid circular imports
    try:
        import cards
        import inspect
    except ImportError as e:
        print(f"Error importing cards module: {e}")
        return []

    # Handle preserve_ratio bounds
    preserve_ratio = max(0.0, min(1.0, preserve_ratio))

    # If we have an original deck and want to preserve some cards
    if original and preserve_ratio > 0.0:
        # Calculate how many cards to preserve from original
        cards_to_preserve = min(int(12 * preserve_ratio), len(original), 12)

        # If preserve_ratio is 1.0, return the original deck (up to 12 cards)
        if preserve_ratio == 1.0:
            preserved_cards = original[:12]
            # Pad with random cards if original has fewer than 12 cards
            if len(preserved_cards) < 12:
                cards_needed = 12 - len(preserved_cards)
            else:
                return preserved_cards
        else:
            # Randomly select cards to preserve from original
            preserved_cards = random.sample(original, cards_to_preserve)
            cards_needed = 12 - cards_to_preserve
    else:
        preserved_cards = []
        cards_needed = 12

    # Generate random cards for the remaining slots
    if cards_needed > 0:
        available_cards = []

        # Get all attributes from cards module
        try:
            card_names = [name for name in dir(cards) if not name.startswith('_')]
        except Exception as e:
            print(f"Error accessing cards module: {e}")
            return preserved_cards

        for name in card_names:
            try:
                obj = getattr(cards, name)
                # Check if it's a class and not a test case or other utility
                if (inspect.isclass(obj) and 
                    hasattr(obj, "__init__") and 
                    not name.endswith("Test") and
                    not name.startswith("_")):

                    # Try to create a temporary instance to check its faction
                    try:
                        temp_card = obj()
                        if hasattr(temp_card, "faction"):
                            # Include cards that match the requested faction or are neutral
                            if temp_card.faction == faction or temp_card.faction == Faction.NEUTRAL:
                                available_cards.append(obj)
                    except Exception:
                        # Skip cards that can't be instantiated
                        continue
            except Exception:
                # Skip any problematic attributes
                continue

        # Safety check: if no cards are available for random generation
        if not available_cards:
            print(f"Warning: No cards available for faction {faction.name}")
            return preserved_cards

        # Randomly select cards for the remaining slots
        if cards_needed > len(available_cards):
            # If not enough cards available, allow duplicates
            selected_card_classes = random.choices(available_cards, k=cards_needed)
        else:
            selected_card_classes = random.sample(available_cards, cards_needed)

        # Create card instances for random cards
        random_cards = []
        for card_class in selected_card_classes:
            try:
                random_cards.append(card_class())
            except Exception as e:
                print(f"Warning: Could not create instance of {card_class.__name__}: {e}")
                continue

        # Combine preserved and random cards
        deck = preserved_cards + random_cards
    else:
        deck = preserved_cards

    return deck

class DeckEvolutionConfig:
    """
    Manages deck configuration evolution through three phases:
    1. Exploit: Use pre-built archetypes for rapid progress
    2. Explore: Gradually inject randomized decks 
    3. Balance: Maintain steady mix of pre-built and random
    """

    def __init__(self, 
                 player1_archetype: List,
                 player2_archetype: List,
                 exploit_generations: int = 30,
                 explore_generations: int = 30,
                 max_random_ratio: float = 0.5,
                 balance_archetype_ratio: float = 0.7):
        """
        Initialize deck evolution configuration.

        Args:
            player1_archetype: Pre-built deck for player 1
            player2_archetype: Pre-built deck for player 2
            exploit_generations: Generations in exploit phase (0% random)
            explore_generations: Generations to ramp up to max_random_ratio
            max_random_ratio: Maximum randomness during explore phase
            balance_archetype_ratio: Ratio of archetype decks in balance phase
        """
        self.player1_archetype = player1_archetype
        self.player2_archetype = player2_archetype
        self.exploit_generations = exploit_generations
        self.explore_generations = explore_generations
        self.max_random_ratio = max_random_ratio
        self.balance_archetype_ratio = balance_archetype_ratio

        # Derive factions from archetype decks
        self.player1_faction = player1_archetype[0].faction if player1_archetype else Faction.NEUTRAL
        self.player2_faction = player2_archetype[0].faction if player2_archetype else Faction.NEUTRAL

    def get_deck_configuration(self, generation: int) -> Tuple[List, List]:
        """
        Get deck configuration for the given generation.

        Returns:
            Tuple of (player1_deck, player2_deck)
        """
        if generation < self.exploit_generations:
            # Phase 1: Exploit - use pre-built archetypes
            return self._get_exploit_decks()

        elif generation < self.exploit_generations + self.explore_generations:
            # Phase 2: Explore - gradually increase randomness
            return self._get_explore_decks(generation)

        else:
            # Phase 3: Balance - steady mix
            return self._get_balance_decks()

    def _get_exploit_decks(self) -> Tuple[List, List]:
        """Phase 1: Return the original archetype decks"""
        return (
            [card.copy() for card in self.player1_archetype],
            [card.copy() for card in self.player2_archetype]
        )

    def _get_explore_decks(self, generation: int) -> Tuple[List, List]:
        """Phase 2: Gradually inject randomness"""
        # Calculate how far into explore phase we are
        explore_progress = (generation - self.exploit_generations) / self.explore_generations
        current_random_ratio = explore_progress * self.max_random_ratio

        # Generate decks with increasing randomness
        player1_deck = generate_random_deck(
            self.player1_faction,
            original=self.player1_archetype,
            preserve_ratio=1.0 - current_random_ratio
        )

        player2_deck = generate_random_deck(
            self.player2_faction,
            original=self.player2_archetype,
            preserve_ratio=1.0 - current_random_ratio
        )

        return player1_deck, player2_deck

    def _get_balance_decks(self) -> Tuple[List, List]:
        """Phase 3: Maintain steady archetype/random balance"""
        # Decide whether to use archetype or random for each player
        use_archetype_p1 = random.random() < self.balance_archetype_ratio
        use_archetype_p2 = random.random() < self.balance_archetype_ratio

        if use_archetype_p1:
            player1_deck = [card.copy() for card in self.player1_archetype]
        else:
            player1_deck = generate_random_deck(self.player1_faction)

        if use_archetype_p2:
            player2_deck = [card.copy() for card in self.player2_archetype]
        else:
            player2_deck = generate_random_deck(self.player2_faction)

        return player1_deck, player2_deck

    def get_phase_info(self, generation: int) -> dict:
        """Get information about current phase and parameters"""
        if generation < self.exploit_generations:
            phase = "Exploit"
            random_ratio = 0.0
        elif generation < self.exploit_generations + self.explore_generations:
            phase = "Explore"
            explore_progress = (generation - self.exploit_generations) / self.explore_generations
            random_ratio = explore_progress * self.max_random_ratio
        else:
            phase = "Balance"
            random_ratio = 1.0 - self.balance_archetype_ratio

        return {
            "phase": phase,
            "generation": generation,
            "random_ratio": random_ratio,
            "exploit_complete": generation >= self.exploit_generations,
            "explore_complete": generation >= self.exploit_generations + self.explore_generations
        }