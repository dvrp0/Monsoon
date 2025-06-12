"""
Feature extraction system for heuristic evaluation from game observations.
"""

import numpy as np
from typing import Tuple


class StateFeatures:
    """Extract meaningful features from game observation for heuristic evaluation"""
    
    def __init__(self, observation: np.ndarray, current_player: int):
        self.observation = observation  # 35×5×4 observation array
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
        # The observation format is 35×5×4 as specified in MuZeroConfig
        # We need to decode this format to extract meaningful game state
        
        # For now, we'll implement basic feature extraction
        # TODO: This needs to be updated based on the actual observation format
        
        # Extract basic information (placeholder implementation)
        self.player_mana = self._extract_player_mana()
        self.opponent_mana = self._extract_opponent_mana()
        self.player_health = self._extract_player_health()
        self.opponent_health = self._extract_opponent_health()
        self.board_state = self._extract_board_state()
        
    def _extract_player_mana(self) -> float:
        """Extract current player's mana from observation"""
        # Placeholder implementation - needs actual observation format
        # Assuming mana information is encoded in specific channels/positions
        try:
            # This is a placeholder - actual implementation depends on observation format
            return float(np.mean(self.observation[0, :, 0]))
        except:
            return 0.0
    
    def _extract_opponent_mana(self) -> float:
        """Extract opponent's mana from observation"""
        try:
            # Placeholder implementation
            return float(np.mean(self.observation[1, :, 0]))
        except:
            return 0.0
    
    def _extract_player_health(self) -> float:
        """Extract current player's health from observation"""
        try:
            # Placeholder implementation
            return float(np.mean(self.observation[0, :, 1]))
        except:
            return 20.0  # Default starting health
    
    def _extract_opponent_health(self) -> float:
        """Extract opponent's health from observation"""
        try:
            # Placeholder implementation
            return float(np.mean(self.observation[1, :, 1]))
        except:
            return 20.0  # Default starting health
    
    def _extract_board_state(self) -> np.ndarray:
        """Extract board state information"""
        try:
            # Extract board-related information from observation
            # This should contain unit positions, strengths, etc.
            return self.observation[2:, :, :]
        except:
            return np.zeros((33, 5, 4))  # Default empty board
        
    def _calculate_board_control(self) -> float:
        """Calculate board control metric based on unit positions"""
        try:
            # Calculate total strength and positioning advantage
            player_units = self.board_state[:, :, 0]  # Player units
            opponent_units = self.board_state[:, :, 1]  # Opponent units
            
            player_control = np.sum(player_units)
            opponent_control = np.sum(opponent_units)
            
            total_control = player_control + opponent_control
            if total_control == 0:
                return 0.0
            
            return (player_control - opponent_control) / total_control
        except:
            return 0.0
        
    def _calculate_front_line_advantage(self) -> float:
        """Calculate front line position advantage"""
        try:
            # Find the furthest advanced positions for each player
            player_positions = np.where(self.board_state[:, :, 0] > 0)
            opponent_positions = np.where(self.board_state[:, :, 1] > 0)
            
            if len(player_positions[0]) == 0 and len(opponent_positions[0]) == 0:
                return 0.0
            
            player_front = np.max(player_positions[0]) if len(player_positions[0]) > 0 else 0
            opponent_front = np.max(opponent_positions[0]) if len(opponent_positions[0]) > 0 else 0
            
            return float(player_front - opponent_front) / 35.0  # Normalize by board size
        except:
            return 0.0
        
    def _calculate_total_strength(self) -> float:
        """Calculate total strength advantage"""
        try:
            player_strength = np.sum(self.board_state[:, :, 2])  # Assuming strength in channel 2
            opponent_strength = np.sum(self.board_state[:, :, 3])
            
            return player_strength - opponent_strength
        except:
            return 0.0
        
    def _count_units(self) -> float:
        """Count number of units on board"""
        try:
            player_units = np.count_nonzero(self.board_state[:, :, 0])
            opponent_units = np.count_nonzero(self.board_state[:, :, 1])
            
            return float(player_units - opponent_units)
        except:
            return 0.0
        
    def _count_structures(self) -> float:
        """Count number of structures on board"""
        try:
            # Structures might be encoded differently - this is a placeholder
            # Need to understand the actual observation format
            return 0.0
        except:
            return 0.0
        
    def _calculate_base_threat(self) -> float:
        """Calculate threat to player's base"""
        try:
            # Calculate how many opponent units are threatening the base
            # This depends on the observation format and game rules
            threat_level = 0.0
            
            # Look for opponent units in threatening positions
            for row in range(min(5, self.board_state.shape[0])):  # Front rows are more threatening
                threat_level += np.sum(self.board_state[row, :, 1]) * (5 - row) / 5.0
            
            return threat_level
        except:
            return 0.0
        
    def _calculate_protection(self) -> float:
        """Calculate protection value for player's base"""
        try:
            # Calculate defensive positioning
            protection = 0.0
            
            # Look for player units in defensive positions
            for row in range(min(5, self.board_state.shape[0])):
                protection += np.sum(self.board_state[row, :, 0]) * (5 - row) / 5.0
            
            return protection
        except:
            return 0.0
        
    def _calculate_mana_efficiency(self) -> float:
        """Calculate how efficiently player is using available mana"""
        try:
            # Extract mana info from observation (row 17 for player 1, row 30 for player 2)
            if self.current_player == 0:
                current_mana = self.observation[17, 0, 0]  # Player 1's current mana
            else:
                current_mana = self.observation[30, 0, 0]  # Player 2's current mana
            
            # Mana efficiency = how much mana has been spent this turn
            # Higher values indicate more aggressive/active play
            # This assumes mana resets to max each turn, so low mana = high usage
            max_possible_mana = 10  # Reasonable upper bound for late game
            mana_usage_ratio = 1.0 - (current_mana / max_possible_mana)
            
            return np.clip(mana_usage_ratio, 0.0, 1.0)
        except:
            return 0.5
    
    def _calculate_hand_quality(self) -> float:
        """Calculate quality/playability of current hand"""
        try:
            # Extract hand info from observation (row 10 for player)
            # Hand format: [card_id, cost, strength, movement] for each card
            hand_data = self.observation[10, :, :]  # 5x4 array with hand info
            
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
                        
                        # Count as playable if cost <= reasonable mana threshold
                        if cost <= 8:  # Most cards should be playable by mid-game
                            playable_cards += 1
            
            if valid_cards == 0:
                return 0.0
                
            # Combine playability and average card value
            playability = playable_cards / valid_cards
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