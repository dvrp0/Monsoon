#!/usr/bin/env python3
"""
Watch a trained evolutionary agent play against the built-in expert agent.

Usage:
    python play_vs_expert.py [--checkpoint path/to/checkpoint.pkl] [--games 10] [--verbose]
"""

import argparse
import pickle
import sys
import os
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from games.stormbound import Game
from evo.heuristic_agent import HeuristicAgent
from evo.game_adapter import StormboundAdapter
from evo.population import Population
from evo.config import EvolutionaryConfig


def load_trained_agent(checkpoint_path: str) -> HeuristicAgent:
    """Load the best trained agent from a checkpoint file."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load the population data
    try:
        with open(checkpoint_path, 'rb') as f:
            population_data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Extract the population and find the best individual
    individuals = population_data["individuals"]
    fitness_scores = population_data["fitness_scores"]
    generation = population_data["generation"]

    print(f"Loaded population from generation {generation}")
    print(f"Population size: {len(individuals)}")

    if not fitness_scores:
        raise ValueError("No fitness scores found in checkpoint")

    # Find the best individual
    best_idx = fitness_scores.index(max(fitness_scores))
    best_weights = individuals[best_idx]
    best_fitness = fitness_scores[best_idx]

    print(f"Best agent fitness: {best_fitness:.6f}")
    print(f"Weight vector size: {best_weights.size}")

    # Create the agent with the best weights
    agent = HeuristicAgent(best_weights, player_idx=1)  # Agent will be player 1

    return agent


class ExpertAgent:
    """Wrapper for the built-in expert agent"""
    
    def __init__(self, player_idx: int):
        self.player_idx = player_idx
        self.action_count = 0
        self.game_count = 0
    
    def select_action(self, adapter: StormboundAdapter) -> int:
        """Select action using the built-in expert strategy"""
        try:
            # Use the game's built-in expert_agent method
            action = adapter.game.env.expert_action()
            self.action_count += 1
            return action
        except Exception as e:
            print(f"Warning: Expert agent failed: {e}")
            # Fallback to random legal action
            legal_actions = adapter.get_legal_actions()
            if legal_actions:
                import random
                return random.choice(legal_actions)
            else:
                return 155  # PASS action
    
    def reset_for_new_game(self):
        """Reset agent state for a new game"""
        self.action_count = 0
        self.game_count += 1


def play_game(evolutionary_agent: HeuristicAgent, expert_agent: ExpertAgent, 
              game_num: int, verbose: bool = True) -> dict:
    """
    Play a game between evolutionary agent and expert agent.
    
    Returns:
        dict with keys: 'winner', 'turns', 'agent_health', 'expert_health'
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"GAME {game_num}")
        print("="*60)
        print("Evolutionary Agent (Player 1) - IRONCLAD faction")
        print("Expert Agent (Player 2) - SWARM faction")

    # Create game
    game = Game()
    observation = game.reset()

    # Create game adapter
    adapter = StormboundAdapter(game)

    # Reset agents for new game
    evolutionary_agent.reset_for_new_game()
    expert_agent.reset_for_new_game()

    turn_count = 0
    max_turns = 300  # Prevent infinite games

    while turn_count < max_turns:
        current_player = game.to_play()

        if verbose:
            print(f"\n--- Turn {turn_count + 1} ---")
            game.render()

        # Check if game is over
        if game.env.have_winner():
            # Determine winner based on player strengths
            if game.env.board.local.strength <= 0:
                winner = 1  # Remote player (player 1) wins
            elif game.env.board.remote.strength <= 0:
                winner = 0  # Local player (player 0) wins
            else:
                winner = -1  # Draw/unknown

            # Get final health values
            agent_health = game.env.board.local.strength  # Evolutionary agent (Player 1/local)
            expert_health = game.env.board.remote.strength  # Expert agent (Player 2/remote)

            if verbose:
                if winner == 0:
                    winner_name = "IRONCLAD (Evolutionary Agent)"
                elif winner == 1:
                    winner_name = "SWARM (Expert Agent)"
                else:
                    winner_name = "UNKNOWN"

                print(f"\n🎉 GAME OVER! {winner_name} wins!")

            return {
                'winner': winner,
                'turns': turn_count + 1,
                'agent_health': agent_health,
                'expert_health': expert_health
            }

        # Check for legal actions
        legal_actions = game.legal_actions()
        if not legal_actions:
            if verbose:
                print("No legal actions available. Game ends in draw.")
            return {
                'winner': -1,
                'turns': turn_count + 1,
                'agent_health': game.env.board.local.strength,
                'expert_health': game.env.board.remote.strength
            }

        # Get action from current player
        try:
            # Update adapter with current game state
            adapter = StormboundAdapter(game)

            if current_player == 0:  # Evolutionary agent's turn
                if verbose:
                    print(f"\n>> Evolutionary Agent's turn (Player {current_player + 1})")
                    print("Evolutionary agent is thinking...")
                
                action = evolutionary_agent.select_action(adapter)
                
                if verbose:
                    print(f"Evolutionary agent selected action: {action}")
                    if action < len(game.env.actions):
                        action_description = game.env.actions[action]
                        print(f"Action: {action_description}")

            else:  # Expert agent's turn
                if verbose:
                    print(f"\n>> Expert Agent's turn (Player {current_player + 1})")
                    print("Expert agent is thinking...")

                action = expert_agent.select_action(adapter)
                
                if verbose:
                    print(f"Expert agent selected action: {action}")
                    if action < len(game.env.actions):
                        action_description = game.env.actions[action]
                        print(f"Action: {action_description}")

        except Exception as e:
            if verbose:
                print(f"Error getting action: {e}")
            # Fallback to random legal action
            action = legal_actions[0] if legal_actions else 155

        # Apply the action
        try:
            observation, reward, done = game.step(action)
            turn_count += 1

            if done:
                break

        except Exception as e:
            if verbose:
                print(f"Error applying action {action}: {e}")
            # Return error result instead of breaking without return
            return {
                'winner': -1,
                'turns': turn_count + 1,
                'agent_health': game.env.board.local.strength,
                'expert_health': game.env.board.remote.strength
            }

    # This should handle the case where loop exits normally
    if turn_count >= max_turns:
        if verbose:
            print(f"\nGame ended after {max_turns} turns (draw due to turn limit)")
        return {
            'winner': -1,
            'turns': max_turns,
            'agent_health': game.env.board.local.strength,
            'expert_health': game.env.board.remote.strength
        }
    
    # Final fallback - check if game actually ended
    if game.env.have_winner():
        # Determine winner based on player strengths
        if game.env.board.local.strength <= 0:
            winner = 1  # Remote player wins
        elif game.env.board.remote.strength <= 0:
            winner = 0  # Local player wins
        else:
            winner = -1  # Draw/unknown
        
        return {
            'winner': winner,
            'turns': turn_count,
            'agent_health': game.env.board.local.strength,
            'expert_health': game.env.board.remote.strength
        }
    else:
        # Game ended without clear winner
        return {
            'winner': -1,
            'turns': turn_count,
            'agent_health': game.env.board.local.strength,
            'expert_health': game.env.board.remote.strength
        }


def run_tournament(evolutionary_agent: HeuristicAgent, expert_agent: ExpertAgent, 
                   num_games: int, verbose: bool = True) -> dict:
    """Run a tournament between the agents"""
    
    print(f"\n🏟️  Starting tournament: {num_games} games")
    print(f"Evolutionary Agent vs Expert Agent")
    
    results = {
        'evolutionary_wins': 0,
        'expert_wins': 0,
        'draws': 0,
        'total_games': num_games,
        'game_details': []
    }
    
    for game_num in range(1, num_games + 1):
        game_result = play_game(evolutionary_agent, expert_agent, game_num, verbose=verbose)
        
        # Store detailed game information
        results['game_details'].append(game_result)
        
        # Determine result string
        if game_result['winner'] == 0:
            result_str = "Evolutionary Agent wins"
            results['evolutionary_wins'] += 1
        elif game_result['winner'] == 1:
            result_str = "Expert Agent wins"
            results['expert_wins'] += 1
        else:
            result_str = "Draw"
            results['draws'] += 1
        
        # Print game result in the requested format
        print(f"Game {game_num}: {result_str} ({game_result['turns']} turns, EA: {game_result['agent_health']} / Expert: {game_result['expert_health']})")
    
    # Print tournament summary
    print(f"\n{'='*60}")
    print("🏆 TOURNAMENT RESULTS")
    print("="*60)
    print(f"Total games played: {results['total_games']}")
    print(f"Evolutionary Agent wins: {results['evolutionary_wins']} ({results['evolutionary_wins']/num_games*100:.1f}%)")
    print(f"Expert Agent wins: {results['expert_wins']} ({results['expert_wins']/num_games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    
    if results['evolutionary_wins'] > results['expert_wins']:
        print("🎊 Evolutionary Agent wins the tournament!")
    elif results['expert_wins'] > results['evolutionary_wins']:
        print("🤖 Expert Agent wins the tournament!")
    else:
        print("🤝 Tournament ends in a tie!")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Watch evolutionary agent vs expert agent matches")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="results/evolutionary/interrupted_checkpoint.pkl",
        help="Path to the checkpoint file (default: results/evolutionary/interrupted_checkpoint.pkl)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games to play (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed game progress (default: False)"
    )

    args = parser.parse_args()

    try:
        # Load the trained evolutionary agent
        evolutionary_agent = load_trained_agent(args.checkpoint)
        print("Evolutionary agent loaded successfully!")

        # Create expert agent
        expert_agent = ExpertAgent(player_idx=1)
        print("Expert agent created successfully!")

        # Run tournament
        results = run_tournament(evolutionary_agent, expert_agent, args.games, args.verbose)

        # Save results if desired
        print(f"\nTournament completed! 🎮")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the checkpoint file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 