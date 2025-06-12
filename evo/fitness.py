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


class FitnessEvaluator:
    """Evaluates fitness of weight vectors through competitive coevolution"""
    
    def __init__(self, config: 'EvolutionaryConfig'):
        self.config = config
        self.total_games = 0
        self.total_time = 0.0
        
    def evaluate_population(self, population: List['WeightVector']) -> List[float]:
        """
        Evaluate entire population using competitive coevolution.
        
        Each individual plays against all others in round-robin tournament.
        Fitness = wins + 0.5 * draws
        """
        n_individuals = len(population)
        n_pairings = n_individuals * (n_individuals - 1) // 2
        
        print(f"Evaluating {n_individuals} individuals...")
        print(f"Total pairings: {n_pairings}")
        print(f"Games per pairing: {self.config.games_per_pairing}")
        print(f"Total games: {n_pairings * self.config.games_per_pairing}")
        
        # Initialize fitness scores
        fitness_scores = [0.0] * n_individuals
        
        # Create all pairings for round-robin tournament
        pairings = []
        for i in range(n_individuals):
            for j in range(i + 1, n_individuals):
                pairings.append((i, j))
        
        start_time = time.time()
        
        # Evaluate pairings in parallel
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit all pairing evaluations
            future_to_pairing = {}
            for i, j in pairings:
                future = executor.submit(
                    self._evaluate_pairing,
                    population[i], population[j],
                    i, j
                )
                future_to_pairing[future] = (i, j)
            
            # Collect results
            completed_pairings = 0
            for future in as_completed(future_to_pairing):
                i, j = future_to_pairing[future]
                try:
                    player1_score, player2_score = future.result()
                    fitness_scores[i] += player1_score
                    fitness_scores[j] += player2_score
                    
                    completed_pairings += 1
                    if completed_pairings % 10 == 0:
                        progress = completed_pairings / len(pairings) * 100
                        print(f"Progress: {completed_pairings}/{len(pairings)} pairings ({progress:.1f}%)")
                        
                except Exception as e:
                    print(f"Error evaluating pairing ({i}, {j}): {e}")
                    # Give both players average score on error
                    fitness_scores[i] += 0.5 * self.config.games_per_pairing
                    fitness_scores[j] += 0.5 * self.config.games_per_pairing
        
        evaluation_time = time.time() - start_time
        self.total_time += evaluation_time
        
        # Normalize fitness scores by number of games played
        n_games_per_individual = (n_individuals - 1) * self.config.games_per_pairing
        normalized_fitness = [score / n_games_per_individual for score in fitness_scores]
        
        print(f"Evaluation completed in {evaluation_time:.2f}s")
        print(f"Games per second: {(len(pairings) * self.config.games_per_pairing) / evaluation_time:.1f}")
        
        return normalized_fitness
    
    def _evaluate_pairing(self, weights1: 'WeightVector', weights2: 'WeightVector', 
                         idx1: int, idx2: int) -> Tuple[float, float]:
        """Evaluate a single pairing of individuals"""
        from .heuristic_agent import HeuristicAgent
        from .game_adapter import StormboundAdapter
        from games.stormbound import Game
        
        player1_score = 0.0
        player2_score = 0.0
        
        for game_num in range(self.config.games_per_pairing):
            try:
                # Create game instance
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
        max_turns = self.config.max_game_length
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
            result = adapter.get_result()
            if result is not None:
                return result
        
        # Game didn't terminate naturally - it's a draw
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