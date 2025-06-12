"""
Weight vector management with self-adaptive mutation for evolutionary strategies.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EvolutionaryConfig


class WeightVector:
    """Weight vector with self-adaptive mutation strengths for evolutionary optimization"""
    
    def __init__(self, size: int):
        self.weights = np.random.uniform(0, 1, size)
        self.sigmas = np.full(size, 0.1)  # Self-adaptive mutation strengths
        self.size = size
    
    def mutate(self, tau: float, tau_prime: float, min_sigma: float):
        """
        Self-adaptive ES mutation as per PRD formula:
        σᵢ' = max(σᵢ · exp(τ'·N(0,1) + τ·Nᵢ(0,1)), ε)
        wᵢ' = clamp(wᵢ + N(0,σᵢ'), 0,1)
        """
        # Generate random noise
        global_noise = np.random.normal(0, 1)
        individual_noise = np.random.normal(0, 1, len(self.sigmas))
        
        # Update mutation strengths (sigmas) first
        self.sigmas = np.maximum(
            self.sigmas * np.exp(tau_prime * global_noise + tau * individual_noise),
            min_sigma
        )
        
        # Mutate weights using updated sigmas
        self.weights = np.clip(
            self.weights + np.random.normal(0, self.sigmas),
            0, 1
        )
    
    def copy(self) -> 'WeightVector':
        """Create a deep copy of this weight vector"""
        new_vector = WeightVector(len(self.weights))
        new_vector.weights = self.weights.copy()
        new_vector.sigmas = self.sigmas.copy()
        new_vector.size = self.size
        return new_vector
    
    def distance_to(self, other: 'WeightVector') -> float:
        """Calculate Euclidean distance to another weight vector"""
        if self.size != other.size:
            raise ValueError("Cannot compute distance between vectors of different sizes")
        return np.linalg.norm(self.weights - other.weights)
    
    def dot_product(self, features: np.ndarray) -> float:
        """Compute weighted sum of features using this weight vector"""
        if len(features) != self.size:
            raise ValueError(f"Feature vector size {len(features)} doesn't match weight vector size {self.size}")
        return np.dot(self.weights, features)
    
    def get_weights(self) -> np.ndarray:
        """Get the weight values as numpy array"""
        return self.weights.copy()
    
    def get_sigmas(self) -> np.ndarray:
        """Get the mutation strengths as numpy array"""
        return self.sigmas.copy()
    
    def set_weights(self, weights: np.ndarray):
        """Set the weight values from numpy array"""
        if len(weights) != self.size:
            raise ValueError(f"Weight array size {len(weights)} doesn't match expected size {self.size}")
        self.weights = np.clip(weights, 0, 1)
    
    def set_sigmas(self, sigmas: np.ndarray):
        """Set the mutation strengths from numpy array"""
        if len(sigmas) != self.size:
            raise ValueError(f"Sigma array size {len(sigmas)} doesn't match expected size {self.size}")
        self.sigmas = np.maximum(sigmas, 1e-10)  # Ensure positive sigmas
    
    def normalize_weights(self):
        """Normalize weights to sum to 1"""
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights = self.weights / weight_sum
    
    def __str__(self) -> str:
        """String representation of weight vector"""
        return f"WeightVector(size={self.size}, weights={self.weights}, sigmas={self.sigmas})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()
    
    @classmethod
    def from_arrays(cls, weights: np.ndarray, sigmas: np.ndarray = None) -> 'WeightVector':
        """Create WeightVector from existing arrays"""
        vector = cls(len(weights))
        vector.set_weights(weights)
        if sigmas is not None:
            vector.set_sigmas(sigmas)
        return vector
    
    @classmethod
    def zeros(cls, size: int) -> 'WeightVector':
        """Create a zero-initialized weight vector"""
        vector = cls(size)
        vector.weights = np.zeros(size)
        return vector
    
    @classmethod
    def ones(cls, size: int) -> 'WeightVector':
        """Create a ones-initialized weight vector"""
        vector = cls(size)
        vector.weights = np.ones(size)
        return vector
    
    @classmethod
    def random_uniform(cls, size: int, low: float = 0.0, high: float = 1.0) -> 'WeightVector':
        """Create a uniformly random weight vector"""
        vector = cls(size)
        vector.weights = np.random.uniform(low, high, size)
        return vector 