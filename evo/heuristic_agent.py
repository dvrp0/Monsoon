"""
Heuristic agent implementation with weight-based action scoring.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .weights import WeightVector
    from .game_adapter import StormboundAdapter
    from .features import StateFeatures


class HeuristicAgent:
    """Heuristic agent that uses weighted feature evaluation for action selection"""

    def __init__(self, weights: 'WeightVector', player_idx: int):
        self.weights = weights
        self.player_idx = player_idx
        self.action_count = 0
        self.game_count = 0

    def score_action(self, state: 'StormboundAdapter', action: int) -> float:
        """
        Score an action based on the weighted feature delta.

        Implements the PRD formula:
        Δ(a,S) = Δ_state(enemy) – Δ_state(agent) – Δ_resource
        """
        try:
            # Apply action to get resulting state
            next_state = state.apply_action(action)

            # Extract features before and after
            current_features = state.extract_features()
            next_features = next_state.extract_features()

            # Compute weighted feature deltas as per PRD formula
            agent_delta = self._compute_feature_delta(current_features, next_features, for_agent=True)
            enemy_delta = self._compute_feature_delta(current_features, next_features, for_agent=False)
            resource_delta = self._compute_resource_delta(current_features, next_features)

            # Final score: enemy improvement - agent cost - resource cost
            score = enemy_delta - agent_delta - resource_delta

            return score

        except Exception as e:
            # If scoring fails, return a default score
            print(f"Warning: Action scoring failed for action {action}: {e}")
            return 0.0

    def select_action(self, state: 'StormboundAdapter') -> int:
        """Select the best action using heuristic evaluation"""
        try:
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                return 155  # PASS action based on existing action space

            # Score all legal actions
            scores = []
            for action in legal_actions:
                score = self.score_action(state, action)
                scores.append(score)

            # Select action with highest score
            best_idx = np.argmax(scores)
            best_action = legal_actions[best_idx]

            self.action_count += 1
            return best_action

        except Exception as e:
            print(f"Warning: Action selection failed: {e}")
            # Fallback to random legal action
            legal_actions = state.get_legal_actions()
            if legal_actions:
                return np.random.choice(legal_actions)
            else:
                return 155  # PASS

    def _compute_feature_delta(self, before: 'StateFeatures', after: 'StateFeatures', for_agent: bool) -> float:
        """Compute weighted sum of feature changes"""
        try:
            # Get feature vectors
            before_features = before.get_feature_vector()
            after_features = after.get_feature_vector()

            # Compute feature delta
            feature_delta = after_features - before_features

            # Flip perspective for opponent evaluation
            if not for_agent:
                feature_delta = -feature_delta

            # Apply weights to compute score
            score = self.weights.dot_product(feature_delta)

            return score

        except Exception as e:
            print(f"Warning: Feature delta computation failed: {e}")
            return 0.0

    def _compute_resource_delta(self, before: 'StateFeatures', after: 'StateFeatures') -> float:
        """Compute resource usage penalty"""
        try:
            # Calculate mana efficiency change (higher efficiency is generally good)
            efficiency_change = after.mana_efficiency - before.mana_efficiency

            # Small penalty for dropping efficiency too much (wasting mana)
            # But allow for strategic mana usage
            if efficiency_change < -0.3:  # Significant efficiency drop
                resource_penalty = abs(efficiency_change) * 0.2
            else:
                resource_penalty = 0.0

            return resource_penalty

        except Exception as e:
            print(f"Warning: Resource delta computation failed: {e}")
            return 0.0

    def reset_for_new_game(self):
        """Reset agent state for a new game"""
        self.action_count = 0
        self.game_count += 1

    def get_weights(self) -> 'WeightVector':
        """Get the agent's weight vector"""
        return self.weights

    def set_weights(self, weights: 'WeightVector'):
        """Set the agent's weight vector"""
        self.weights = weights

    def get_player_idx(self) -> int:
        """Get the player index this agent represents"""
        return self.player_idx

    def get_stats(self) -> dict:
        """Get agent statistics"""
        return {
            'player_idx': self.player_idx,
            'action_count': self.action_count,
            'game_count': self.game_count,
            'weight_vector_size': self.weights.size
        }

    def __str__(self) -> str:
        """String representation of the agent"""
        return f"HeuristicAgent(player={self.player_idx}, weights_size={self.weights.size})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__() 