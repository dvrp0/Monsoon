"""
Fitness evaluation system for evolutionary algorithms.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

if TYPE_CHECKING:
    from .weights import WeightVector
    from .config import EvolutionaryConfig
    from .heuristic_agent import HeuristicAgent
    from .game_adapter import StormboundAdapter
    from utils import DeckEvolutionConfig


class FitnessEvaluator:
    """Evaluates fitness of weight vectors through competitive coevolution"""

    def __init__(self, config: 'EvolutionaryConfig', deck_config: 'DeckEvolutionConfig' = None):
        self.config = config
        self.deck_config = deck_config
        self.total_games = 0
        self.total_time = 0.0

        # Hall of Fame to store elite opponents from previous generations
        self.hall_of_fame: List['WeightVector'] = []
        self.hall_of_fame_size = 5  # Keep best 5 from each generation
        self.use_hall_of_fame = True  # Enable hall of fame

    def evaluate_population(self, population: List['WeightVector'], generation: int = 0) -> List[float]:
        """
        Evaluate entire population using competitive coevolution.

        Each individual plays against all others in round-robin tournament.
        Plus additional games against hall of fame opponents if available.
        Fitness = wins + 0.5 * draws
        """
        n_individuals = len(population)

        # Create extended opponent list: current population + hall of fame
        all_opponents = population.copy()
        if self.use_hall_of_fame and len(self.hall_of_fame) > 0:
            all_opponents.extend(self.hall_of_fame)
            print(f"Using {len(self.hall_of_fame)} hall of fame opponents")

        n_total_opponents = len(all_opponents)

        # Calculate pairings: each individual vs all opponents (except itself)
        total_pairings = 0
        pairings = []
        for i in range(n_individuals):
            for j in range(n_total_opponents):
                # Skip self-play within current population
                if j < n_individuals and i == j:
                    continue
                pairings.append((i, j))
                total_pairings += 1

        print(f"Evaluating {n_individuals} individuals...")
        print(f"Total opponents per individual: {n_total_opponents - 1}")
        print(f"Total pairings: {total_pairings}")
        print(f"Games per pairing: {self.config.games_per_pairing}")
        print(f"Total games: {total_pairings * self.config.games_per_pairing}")

        # Show deck evolution info if available
        if self.deck_config:
            phase_info = self.deck_config.get_phase_info(generation)
            print(f"Deck Phase: {phase_info['phase']} | Random Ratio: {phase_info['random_ratio']:.2f}")

        # Initialize fitness scores
        fitness_scores = [0.0] * n_individuals

        start_time = time.time()

        # Evaluate pairings in parallel
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all pairing evaluations
            future_to_pairing = {}
            for i, j in pairings:
                future = executor.submit(
                    self._evaluate_pairing,
                    population[i], all_opponents[j],
                    i, j, generation
                )
                future_to_pairing[future] = (i, j)

            # Collect results
            completed_pairings = 0
            for future in as_completed(future_to_pairing):
                i, j = future_to_pairing[future]
                try:
                    player1_score, player2_score = future.result()
                    fitness_scores[i] += player1_score
                    # Note: we don't update fitness for hall of fame opponents

                    completed_pairings += 1
                    if completed_pairings % 20 == 0:
                        progress = completed_pairings / len(pairings) * 100
                        print(f"Progress: {completed_pairings}/{len(pairings)} pairings ({progress:.1f}%)")

                except Exception as e:
                    print(f"Error evaluating pairing ({i}, {j}): {e}")
                    # Give current population individual average score on error
                    fitness_scores[i] += 0.5 * self.config.games_per_pairing

        evaluation_time = time.time() - start_time
        self.total_time += evaluation_time

        # Normalize fitness scores by number of games played per individual
        n_games_per_individual = (n_total_opponents - 1) * self.config.games_per_pairing
        normalized_fitness = [score / n_games_per_individual for score in fitness_scores]

        print(f"Evaluation completed in {evaluation_time:.2f}s")
        print(f"Games per second: {(len(pairings) * self.config.games_per_pairing) / evaluation_time:.1f}")

        # Update hall of fame with best performers from current population
        self._update_hall_of_fame(population, normalized_fitness)

        return normalized_fitness

    def _evaluate_pairing(self, weights1: 'WeightVector', weights2: 'WeightVector', 
                         idx1: int, idx2: int, generation: int = 0) -> Tuple[float, float]:
        """Evaluate a single pairing of individuals"""
        from .heuristic_agent import HeuristicAgent
        from .game_adapter import StormboundAdapter
        from games.evolutionary_stormbound import EvolutionaryStormbound

        player1_score = 0.0
        player2_score = 0.0

        for game_num in range(self.config.games_per_pairing):
            try:
                # Create evolutionary game instance with generation-aware deck config
                if self.deck_config:
                    game = EvolutionaryStormbound(
                        seed=self.config.seed or game_num, 
                        generation=generation,
                        deck_config=self.deck_config
                    )
                else:
                    # Fallback to regular game
                    from games.stormbound import Game
                    game = Game()

                game.reset()

                # Create game adapter
                adapter = StormboundAdapter(game)

                # Create agents
                agent1 = HeuristicAgent(weights1, player_idx=0)
                agent2 = HeuristicAgent(weights2, player_idx=1)

                # Play the game
                result = self._play_game(adapter, agent1, agent2)

                # Update scores based on result
                if result == 0:  # Player 1 wins
                    player1_score += 1.0
                elif result == 1:  # Player 2 wins
                    player2_score += 1.0
                else:  # Draw
                    player1_score += 0.5
                    player2_score += 0.5

                self.total_games += 1

            except Exception as e:
                print(f"Error in game {game_num} between {idx1} and {idx2}: {e}")
                # Give both players a draw on error
                player1_score += 0.5
                player2_score += 0.5

        return player1_score, player2_score

    def _play_game(self, adapter: 'StormboundAdapter', 
                   agent1: 'HeuristicAgent', agent2: 'HeuristicAgent') -> int:
        """
        Play a single game between two agents.

        Returns:
            0 if agent1 wins, 1 if agent2 wins, -1 if draw/timeout
        """
        max_turns = self.config.max_turns
        turn_count = 0

        # Reset agents for new game
        agent1.reset_for_new_game()
        agent2.reset_for_new_game()

        while not adapter.is_terminal() and turn_count < max_turns:
            try:
                # Get current player
                current_player = adapter.get_current_player()

                # Select agent based on current player
                if current_player == 0:
                    action = agent1.select_action(adapter)
                else:
                    action = agent2.select_action(adapter)

                # Apply action
                adapter = adapter.apply_action(action)
                turn_count += 1

            except Exception as e:
                print(f"Error during game turn {turn_count}: {e}")
                break

        # Determine game result
        if adapter.is_terminal():
            try:
                # Check winner using the game's have_winner method
                winner = adapter.game.env.have_winner()
                if winner == 0:
                    return 0  # Player 0 (agent1) wins
                elif winner == 1:
                    return 1  # Player 1 (agent2) wins
                else:
                    return -1  # No winner yet (shouldn't happen if is_terminal is true)
            except AttributeError:
                # Fallback: if have_winner doesn't exist, treat as draw
                return -1

        # Game didn't terminate naturally - it's a draw/timeout
        return -1

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        avg_time_per_game = self.total_time / max(self.total_games, 1)
        games_per_second = self.total_games / max(self.total_time, 1e-6)

        return {
            "total_games": self.total_games,
            "total_time": self.total_time,
            "avg_time_per_game": avg_time_per_game,
            "games_per_second": games_per_second
        }

    def reset_stats(self):
        """Reset evaluation statistics"""
        self.total_games = 0
        self.total_time = 0.0 

    def _update_hall_of_fame(self, population: List['WeightVector'], fitness: List[float]):
        """Update hall of fame with best performers from current population"""
        # Create pairs of (fitness, individual) and sort by fitness
        fitness_individual_pairs = list(zip(fitness, population))
        fitness_individual_pairs.sort(key=lambda x: x[0], reverse=True)

        # Extract the best individuals (weight vectors)
        best_individuals = [individual for _, individual in fitness_individual_pairs[:self.hall_of_fame_size]]

        # Update hall of fame with copies to avoid reference issues
        self.hall_of_fame = [individual.copy() for individual in best_individuals]

        if len(self.hall_of_fame) > 0:
            print(f"Hall of Fame updated with {len(self.hall_of_fame)} elite opponents (best fitness: {fitness_individual_pairs[0][0]:.4f})") 