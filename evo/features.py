"""
Feature extraction system for heuristic evaluation from game observations.
"""

import numpy as np
from typing import Tuple


class StateFeatures:
    """Extract meaningful features from game observation for heuristic evaluation"""

    def __init__(self, observation: np.ndarray, current_player: int):
        self.observation = observation  # 27×5×4 observation array
        self.current_player = current_player

        # Extract structured features from observation
        self._parse_observation()

        # Resource features - Updated for Stormbound mechanics
        self.mana_efficiency = self._calculate_mana_efficiency()  # How well mana is being used
        self.health_advantage = self.player_health - self.opponent_health

        # Board presence features
        self.board_control = self._calculate_board_control()
        self.front_line_advantage = self._calculate_front_line_advantage()

        # Unit/structure features
        self.total_strength = self._calculate_total_strength()
        self.unit_count = self._count_units()
        self.structure_count = self._count_structures()

        # Tactical features
        self.threatened_base = self._calculate_base_threat()
        self.protection_value = self._calculate_protection()
        self.hand_quality = self._calculate_hand_quality()

    def _parse_observation(self):
        """Parse the observation array to extract game state information"""
        # Based on stormbound.py observation format (27×5×4):
        # 0: local_unit_ids
        # 1: local_unit_strengths  
        # 2: local_unit_movements
        # 3: local_unit_statuses
        # 4: local_structure_ids
        # 5: local_structure_strengths
        # 6: local_hand
        # 7-12: local_deck (6 layers)
        # 13: local_mana
        # 14: local_base (health)
        # 15: local_faction
        # 16: remote_unit_ids
        # 17: remote_unit_strengths
        # 18: remote_unit_movements
        # 19: remote_unit_statuses
        # 20: remote_structure_ids
        # 21: remote_structure_strengths
        # 22: remote_mana
        # 23: remote_base (health)
        # 24: remote_faction
        # 25: current_player
        # 26: history

        # Extract basic information
        self.player_mana = self._extract_player_mana()
        self.opponent_mana = self._extract_opponent_mana()
        self.player_health = self._extract_player_health()
        self.opponent_health = self._extract_opponent_health()
        self.board_state = self._extract_board_state()

    def _extract_player_mana(self) -> float:
        """Extract current player's mana from observation"""
        try:
            # Layer 13 contains local player's mana
            mana = self.observation[13, 0, 0]
            return float(mana) if mana != -1 else 0.0
        except:
            return 0.0

    def _extract_opponent_mana(self) -> float:
        """Extract opponent's mana from observation"""
        try:
            # Layer 22 contains remote player's mana
            mana = self.observation[22, 0, 0]
            return float(mana) if mana != -1 else 0.0
        except:
            return 0.0

    def _extract_player_health(self) -> float:
        """Extract current player's health from observation"""
        try:
            # Layer 14 contains local player's base strength (health)
            health = self.observation[14, 0, 0]
            return float(health) if health != -1 else 20.0
        except:
            return 20.0  # Default starting health

    def _extract_opponent_health(self) -> float:
        """Extract opponent's health from observation"""
        try:
            # Layer 23 contains remote player's base strength (health)  
            health = self.observation[23, 0, 0]
            return float(health) if health != -1 else 20.0
        except:
            return 20.0  # Default starting health

    def _extract_board_state(self) -> np.ndarray:
        """Extract board state information"""
        try:
            # Extract unit and structure information from the board
            # Local units: layers 0-3, structures: layers 4-5
            # Remote units: layers 16-19, structures: layers 20-21
            local_units = self.observation[0:4, :, :]  # IDs, strengths, movements, statuses
            local_structures = self.observation[4:6, :, :]  # IDs, strengths
            remote_units = self.observation[16:20, :, :]  # IDs, strengths, movements, statuses
            remote_structures = self.observation[20:22, :, :]  # IDs, strengths

            # Combine into a unified board representation
            board_state = np.concatenate([
                local_units, local_structures, 
                remote_units, remote_structures
            ], axis=0)

            return board_state
        except:
            return np.zeros((10, 5, 4))  # Default empty board state

    def _calculate_board_control(self) -> float:
        """Calculate board control metric based on unit positions and strengths"""
        try:
            # Get unit strength information from board state
            local_unit_strengths = self.observation[1, :, :]  # Layer 1: local unit strengths
            remote_unit_strengths = self.observation[17, :, :]  # Layer 17: remote unit strengths
            local_structure_strengths = self.observation[5, :, :]  # Layer 5: local structure strengths
            remote_structure_strengths = self.observation[21, :, :]  # Layer 21: remote structure strengths

            # Calculate total strength on board for each player
            player_strength = np.sum(local_unit_strengths[local_unit_strengths != -1]) + \
                            np.sum(local_structure_strengths[local_structure_strengths != -1])
            opponent_strength = np.sum(remote_unit_strengths[remote_unit_strengths != -1]) + \
                              np.sum(remote_structure_strengths[remote_structure_strengths != -1])

            total_strength = player_strength + opponent_strength
            if total_strength == 0:
                return 0.0

            return (player_strength - opponent_strength) / total_strength
        except:
            return 0.0

    def _calculate_front_line_advantage(self) -> float:
        """Calculate front line position advantage"""
        try:
            # Check unit positions to determine front line advantage
            local_unit_ids = self.observation[0, :, :]  # Layer 0: local unit IDs
            remote_unit_ids = self.observation[16, :, :]  # Layer 16: remote unit IDs

            # Find furthest advanced positions (lowest row numbers for player, highest for opponent)
            player_positions = []
            opponent_positions = []

            for row in range(5):
                for col in range(4):
                    if local_unit_ids[row, col] != -1:
                        player_positions.append(row)
                    if remote_unit_ids[row, col] != -1:
                        opponent_positions.append(row)

            if not player_positions and not opponent_positions:
                return 0.0

            # Lower row numbers mean closer to opponent's base
            player_advance = min(player_positions) if player_positions else 4
            opponent_advance = max(opponent_positions) if opponent_positions else 0

            # Normalize the advantage (-1 to 1)
            return (opponent_advance - player_advance) / 4.0
        except:
            return 0.0

    def _calculate_total_strength(self) -> float:
        """Calculate total strength advantage"""
        try:
            # Sum all unit and structure strengths
            local_unit_strengths = self.observation[1, :, :]
            local_structure_strengths = self.observation[5, :, :]
            remote_unit_strengths = self.observation[17, :, :]
            remote_structure_strengths = self.observation[21, :, :]

            player_total = np.sum(local_unit_strengths[local_unit_strengths != -1]) + \
                          np.sum(local_structure_strengths[local_structure_strengths != -1])
            opponent_total = np.sum(remote_unit_strengths[remote_unit_strengths != -1]) + \
                           np.sum(remote_structure_strengths[remote_structure_strengths != -1])

            return float(player_total - opponent_total)
        except:
            return 0.0

    def _count_units(self) -> float:
        """Count number of units on board"""
        try:
            local_unit_ids = self.observation[0, :, :]
            remote_unit_ids = self.observation[16, :, :]

            player_units = np.count_nonzero(local_unit_ids != -1)
            opponent_units = np.count_nonzero(remote_unit_ids != -1)

            return float(player_units - opponent_units)
        except:
            return 0.0

    def _count_structures(self) -> float:
        """Count number of structures on board"""
        try:
            local_structure_ids = self.observation[4, :, :]
            remote_structure_ids = self.observation[20, :, :]

            player_structures = np.count_nonzero(local_structure_ids != -1)
            opponent_structures = np.count_nonzero(remote_structure_ids != -1)

            return float(player_structures - opponent_structures)
        except:
            return 0.0

    def _calculate_base_threat(self) -> float:
        """Calculate threat to player's base"""
        try:
            # Calculate threat based on opponent units close to player's base
            remote_unit_strengths = self.observation[17, :, :]

            threat_level = 0.0

            # Rows closer to base (higher row indices) pose greater threat
            for row in range(5):
                for col in range(4):
                    if remote_unit_strengths[row, col] != -1:
                        # Weight by distance to base (row 4 is closest to player's base)
                        distance_weight = (row + 1) / 5.0
                        threat_level += remote_unit_strengths[row, col] * distance_weight

            return threat_level
        except:
            return 0.0

    def _calculate_protection(self) -> float:
        """Calculate protection value for player's base"""
        try:
            # Calculate protection based on player units close to own base
            local_unit_strengths = self.observation[1, :, :]
            local_structure_strengths = self.observation[5, :, :]

            protection = 0.0

            # Units/structures closer to own base provide more protection
            for row in range(5):
                for col in range(4):
                    distance_weight = (5 - row) / 5.0  # Higher weight for back rows

                    if local_unit_strengths[row, col] != -1:
                        protection += local_unit_strengths[row, col] * distance_weight
                    if local_structure_strengths[row, col] != -1:
                        protection += local_structure_strengths[row, col] * distance_weight

            return protection
        except:
            return 0.0

    def _calculate_mana_efficiency(self) -> float:
        """Calculate how efficiently player is using available mana"""
        try:
            current_mana = self.player_mana

            # Mana efficiency = how much mana has been spent this turn
            # We estimate max mana based on typical game progression
            # In Stormbound, max mana increases each turn up to about 10
            estimated_max_mana = min(10, max(3, current_mana + 2))

            # Higher efficiency when more mana has been used
            mana_usage_ratio = 1.0 - (current_mana / estimated_max_mana)

            return np.clip(mana_usage_ratio, 0.0, 1.0)
        except:
            return 0.5

    def _calculate_hand_quality(self) -> float:
        """Calculate quality/playability of current hand"""
        try:
            # Extract hand info from observation (layer 6)
            hand_data = self.observation[6, :4, :]  # First 4 rows contain hand cards

            # Calculate hand metrics
            playable_cards = 0
            total_value = 0.0
            valid_cards = 0

            for i in range(4):  # Up to 4 cards in hand
                card_id = hand_data[i, 0]
                if card_id != -1 and card_id != 32767:  # Valid card (not empty or sentinel)
                    cost = hand_data[i, 1] 
                    strength = hand_data[i, 2] if hand_data[i, 2] != -1 else 0

                    valid_cards += 1

                    # Simple value heuristic: strength per mana cost
                    if cost > 0:
                        card_value = strength / cost
                        total_value += card_value

                        # Count as playable if cost <= current mana
                        if cost <= self.player_mana:
                            playable_cards += 1

            if valid_cards == 0:
                return 0.0

            # Combine playability and average card value
            playability = playable_cards / valid_cards if valid_cards > 0 else 0
            avg_value = total_value / valid_cards if valid_cards > 0 else 0

            # Normalize and combine metrics
            normalized_value = np.clip(avg_value / 3.0, 0.0, 1.0)  # Assume 3.0 is good value
            hand_quality = (playability + normalized_value) / 2.0

            return hand_quality
        except:
            return 0.5

    def get_feature_vector(self) -> np.ndarray:
        """Get all features as a single vector for weight-based evaluation"""
        features = np.array([
            self.mana_efficiency,
            self.health_advantage,
            self.board_control,
            self.front_line_advantage,
            self.total_strength,
            self.unit_count,
            self.structure_count,
            self.threatened_base,
            self.protection_value,
            self.hand_quality
        ])

        return features

    @staticmethod
    def get_feature_count() -> int:
        """Get the number of features in the feature vector"""
        return 10  # Number of features in get_feature_vector()

    @staticmethod
    def get_feature_names() -> list:
        """Get names of all features"""
        return [
            "mana_efficiency",
            "health_advantage", 
            "board_control",
            "front_line_advantage",
            "total_strength",
            "unit_count",
            "structure_count",
            "threatened_base",
            "protection_value",
            "hand_quality"
        ] 