#!/usr/bin/env python3
"""
Play against a trained evolutionary agent.

Usage:
    python play_vs_agent.py [--checkpoint path/to/checkpoint.pkl]
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


def play_game(agent: HeuristicAgent, human_starts: bool = True):
    """Play a game between human and agent."""

    print("\n" + "="*60)
    print("STARTING NEW GAME")
    print("="*60)

    if human_starts:
        print("You are Player 1 (FIRST) - IRONCLAD faction")
        print("Agent is Player 2 (SECOND) - SWARM faction")
    else:
        print("Agent is Player 1 (FIRST) - IRONCLAD faction") 
        print("You are Player 2 (SECOND) - SWARM faction")

    print("\nHow to play:")
    print("- Type card_index x y to play a unit/structure (e.g., '0 1 3')")
    print("- Type card_index x y to cast a spell at target (e.g., '2 0 2')")
    print("- Type 'replace X' to replace card at index X")
    print("- Type 'end' to end your turn")
    print("- Type 'quit' to exit the game")

    # Create game
    game = Game()
    observation = game.reset()

    # Create game adapter for the agent
    adapter = StormboundAdapter(game)

    # Reset agent for new game
    agent.reset_for_new_game()

    turn_count = 0
    max_turns = 300  # Prevent infinite games

    while turn_count < max_turns:
        current_player = game.to_play()

        # Render the game state
        print(f"\n--- Turn {turn_count + 1} ---")
        game.render()

        # Check if game is over
        if game.env.have_winner():
            # Determine winner based on player strengths
            # In the game, players lose when their strength <= 0
            if game.env.board.local.strength <= 0:
                winner = 1  # Remote player (player 1) wins
            elif game.env.board.remote.strength <= 0:
                winner = 0  # Local player (player 0) wins
            else:
                # This shouldn't happen if have_winner() returned True
                winner = -1  # Draw/unknown

            if winner == 0:
                winner_name = "IRONCLAD (Player 1)"
            elif winner == 1:
                winner_name = "SWARM (Player 2)"
            else:
                winner_name = "UNKNOWN"

            print(f"\n🎉 GAME OVER! {winner_name} wins!")

            if human_starts:
                if winner == 0:  # Player 1 wins
                    print("Congratulations! You won! 🎊")
                else:
                    print("The agent won this time. Better luck next time! 🤖")
            else:
                if winner == 0:  # Player 1 wins (agent)
                    print("The agent won this time. Better luck next time! 🤖")
                else:
                    print("Congratulations! You won! 🎊")
            break

        # Check for legal actions
        legal_actions = game.legal_actions()
        if not legal_actions:
            print("No legal actions available. Game ends in draw.")
            break

        # Determine whose turn it is
        is_human_turn = (current_player == 0 and human_starts) or (current_player == 1 and not human_starts)

        if is_human_turn:
            # Human turn
            print(f"\n>> Your turn (Player {current_player + 1})")

            # Get human action
            try:
                action = game.human_to_action()
                if action == -1:  # Special code for quit
                    print("Thanks for playing!")
                    return
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                return
            except Exception as e:
                print(f"Error getting human action: {e}")
                continue
        else:
            # Agent turn
            print(f"\n>> Agent's turn (Player {current_player + 1})")
            print("Agent is thinking...")

            try:
                # Update adapter with current game state
                adapter = StormboundAdapter(game)

                # Get agent action
                action = agent.select_action(adapter)
                print(f"Agent selected action: {action}")

                # Show what the agent is doing
                if action < len(game.env.actions):
                    action_description = game.env.actions[action]
                    print(f"Agent action: {action_description}")

            except Exception as e:
                print(f"Error getting agent action: {e}")
                # Fallback to random legal action
                action = legal_actions[0] if legal_actions else 155

        # Apply the action
        try:
            observation, reward, done = game.step(action)
            turn_count += 1

            if done:
                break

        except Exception as e:
            print(f"Error applying action {action}: {e}")
            break

    if turn_count >= max_turns:
        print(f"\nGame ended after {max_turns} turns (draw due to turn limit)")


def modify_human_to_action():
    """Modify the human_to_action method to handle quit command."""
    import games.stormbound

    original_human_to_action = games.stormbound.Game.human_to_action

    def new_human_to_action(self):
        """Enhanced human_to_action with quit support."""
        print(f"Current player: {self.env.board.local.order}")
        print(f"Max mana: {self.env.board.local.max_mana}, Current mana: {self.env.board.local.current_mana}")
        print(f"Hand:")
        for i, card in enumerate(self.env.board.local.hand):
            for entry in self.env.cards:
                if entry["id"] == card.card_id:
                    print(f"{i}: {entry['name']} {card.card_id} {card.cost} {card.strength if hasattr(card, 'strength') else ''} "
                        f"{card.movement if hasattr(card, 'movement') else ''}")

        while True:
            action = input("> ").strip().lower()

            if action == "quit" or action == "exit":
                return -1  # Special code for quit

            # Use the original logic for other actions
            from games.stormbound import Action, ActionType
            from point import Point
            from spell import Spell

            action_representation = Action(ActionType.PASS)

            if action == "end":
                print("Turn ended")
                break
            elif action.startswith("replace"):
                if not self.env.board.local.replacable:
                    print("Already replaced")
                    continue

                try:
                    target = int(action.split()[1])
                    if target < 0 or target >= len(self.env.board.local.hand):
                        print("Invalid card index")
                        continue
                    self.env.board.local.cycle(self.env.board.local.hand[target])
                    action_representation = Action(ActionType.REPLACE, target)
                    print(f"Replaced card {target}")
                    break
                except (IndexError, ValueError):
                    print("Usage: replace <card_index>")
                    continue
            else:
                try:
                    inputs = [int(x) for x in action.split()]
                    if len(inputs) < 1:
                        print("Please specify at least a card index")
                        continue

                    card_idx = inputs[0]
                    if card_idx < 0 or card_idx >= len(self.env.board.local.hand):
                        print("Invalid card index")
                        continue

                    card = self.env.board.local.hand[card_idx]

                    if card.cost > self.env.board.local.current_mana:
                        print("Not enough mana")
                        continue

                    if len(inputs) >= 3:
                        point = Point(inputs[1], inputs[2])
                        if not isinstance(card, Spell) and inputs[2] < self.env.board.local.front_line:
                            print("Can only place behind front line")
                            continue
                        action_representation = Action(ActionType.USE if isinstance(card, Spell) else ActionType.PLACE, card_idx, point)
                    else:
                        # Card with no target required
                        action_representation = Action(ActionType.USE if isinstance(card, Spell) else ActionType.PLACE, card_idx)

                    break

                except (ValueError, IndexError):
                    print("Invalid input. Use: <card_index> [x y] or 'replace <card_index>' or 'end'")
                    continue

        return action_representation.to_int()

    # Replace the method
    games.stormbound.Game.human_to_action = new_human_to_action


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Play against a trained evolutionary agent")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="results/evolutionary/interrupted_checkpoint.pkl",
        help="Path to the checkpoint file (default: results/evolutionary/interrupted_checkpoint.pkl)"
    )
    parser.add_argument(
        "--agent-starts",
        action="store_true",
        help="Let the agent go first (default: human goes first)"
    )

    args = parser.parse_args()

    # Modify the human_to_action method to support quitting
    modify_human_to_action()

    try:
        # Load the trained agent
        agent = load_trained_agent(args.checkpoint)

        print("Trained agent loaded successfully!")
        print("\nReady to play! 🎮")

        while True:
            # Play a game
            human_starts = not args.agent_starts
            play_game(agent, human_starts)

            # Ask if they want to play again
            while True:
                play_again = input("\nWould you like to play again? (y/n): ").strip().lower()
                if play_again in ['y', 'yes']:
                    break
                elif play_again in ['n', 'no']:
                    print("Thanks for playing! 👋")
                    return
                else:
                    print("Please enter 'y' for yes or 'n' for no.")

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