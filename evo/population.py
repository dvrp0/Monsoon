"""
Population management for evolutionary algorithms.
"""

import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
import time

if TYPE_CHECKING:
    from .weights import WeightVector
    from .config import EvolutionaryConfig


class Population:
    """Manages a population of weight vectors for evolutionary optimization"""
    
    def __init__(self, config: 'EvolutionaryConfig'):
        self.config = config
        self.individuals: List['WeightVector'] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def initialize_population(self, feature_count: int):
        """Initialize population with random weight vectors"""
        from .weights import WeightVector
        
        self.individuals = []
        for i in range(self.config.mu):
            # Create weight vector with correct size
            individual = WeightVector(feature_count)
            
            # Initialize with random weights between -1 and 1
            weights = np.random.uniform(-1.0, 1.0, feature_count)
            individual.set_weights(np.clip(weights, 0, 1))  # Ensure weights are in [0,1]
            
            # Initialize mutation strengths based on config
            sigmas = np.full(feature_count, self.config.initial_sigma)
            individual.set_sigmas(sigmas)
            
            self.individuals.append(individual)
        
        # Initialize fitness scores
        self.fitness_scores = [0.0] * self.config.mu
        
        print(f"Initialized population with {len(self.individuals)} individuals")
        print(f"Each individual has {feature_count} features")
    
    def get_parents(self) -> List['WeightVector']:
        """Get the parent population"""
        return self.individuals[:self.config.mu]
    
    def generate_offspring(self) -> List['WeightVector']:
        """Generate offspring through mutation"""
        offspring = []
        
        for i in range(self.config.lambda_):
            # Select random parent
            parent_idx = np.random.randint(0, self.config.mu)
            parent = self.individuals[parent_idx]
            
            # Create offspring through mutation
            child = parent.copy()
            child.mutate(self.config.tau, self.config.tau_prime, self.config.min_sigma)
            offspring.append(child)
        
        return offspring
    
    def select_from_combined(self, all_individuals: List['WeightVector'], fitness_scores: List[float]):
        """Select best μ individuals from combined parents + offspring"""
        if len(all_individuals) != len(fitness_scores):
            raise ValueError(f"Individuals ({len(all_individuals)}) must match fitness scores ({len(fitness_scores)})")
        
        # Sort by fitness (descending - higher is better)
        sorted_pairs = sorted(
            zip(fitness_scores, all_individuals),
            key=lambda x: x[0],
            reverse=True
        )
        
        # Select top μ individuals as the new population
        self.fitness_scores = [pair[0] for pair in sorted_pairs[:self.config.mu]]
        self.individuals = [pair[1] for pair in sorted_pairs[:self.config.mu]]
        
        # Increment generation
        self.generation += 1
        
        print(f"Generation {self.generation}: "
              f"Best fitness = {self.fitness_scores[0]:.4f}, "
              f"Avg fitness = {np.mean(self.fitness_scores):.4f}")

    def selection(self, fitness_scores: List[float]):
        """Perform (μ + λ) selection to update population"""
        
        # For first generation or when we only have μ individuals
        if self.generation == 0 or len(fitness_scores) == self.config.mu:
            # Just update fitness scores for current population
            if len(fitness_scores) != len(self.individuals):
                raise ValueError(f"Fitness scores ({len(fitness_scores)}) must match individuals ({len(self.individuals)})")
            
            self.fitness_scores = fitness_scores
        else:
            # (μ + λ) selection: we have μ parents + λ offspring
            expected_size = self.config.mu + self.config.lambda_
            if len(fitness_scores) != expected_size:
                raise ValueError(f"Expected {expected_size} fitness scores for μ+λ selection, got {len(fitness_scores)}")
            
            # Generate offspring for selection
            offspring = self.generate_offspring()
            all_individuals = self.individuals + offspring
            
            if len(all_individuals) != len(fitness_scores):
                raise ValueError(f"Individuals ({len(all_individuals)}) must match fitness scores ({len(fitness_scores)})")
            
            # Sort by fitness (descending - higher is better)
            sorted_pairs = sorted(
                zip(fitness_scores, all_individuals),
                key=lambda x: x[0],
                reverse=True
            )
            
            # Select top μ individuals as the new population
            self.fitness_scores = [pair[0] for pair in sorted_pairs[:self.config.mu]]
            self.individuals = [pair[1] for pair in sorted_pairs[:self.config.mu]]
        
        # Increment generation
        self.generation += 1
        
        print(f"Generation {self.generation}: "
              f"Best fitness = {self.fitness_scores[0]:.4f}, "
              f"Avg fitness = {np.mean(self.fitness_scores):.4f}")
    
    def get_best_individual(self) -> Tuple['WeightVector', float]:
        """Get the best individual and its fitness"""
        if not self.fitness_scores:
            raise ValueError("No fitness scores available")
        
        best_idx = np.argmax(self.fitness_scores)
        return self.individuals[best_idx], self.fitness_scores[best_idx]
    
    def get_population_stats(self) -> dict:
        """Get population statistics"""
        if not self.fitness_scores:
            return {"error": "No fitness scores available"}
        
        fitness_array = np.array(self.fitness_scores)
        
        # Calculate diversity metrics
        all_weights = np.array([ind.get_weights() for ind in self.individuals])
        diversity = np.mean(np.std(all_weights, axis=0))
        
        # Calculate average mutation strength
        all_sigmas = np.array([ind.get_sigmas() for ind in self.individuals])
        avg_sigma = np.mean(all_sigmas)
        
        return {
            "generation": self.generation,
            "population_size": len(self.individuals),
            "best_fitness": float(fitness_array.max()),
            "worst_fitness": float(fitness_array.min()),
            "mean_fitness": float(fitness_array.mean()),
            "std_fitness": float(fitness_array.std()),
            "diversity": float(diversity),
            "avg_mutation_strength": float(avg_sigma)
        }
    
    def should_terminate(self) -> bool:
        """Check if evolution should terminate"""
        # Terminate if max generations reached
        if self.generation >= self.config.generations:
            return True
        
        # Only check convergence after a few generations and if we have fitness scores
        if self.generation > 2 and len(self.fitness_scores) > 0:
            fitness_std = np.std(self.fitness_scores)
            fitness_mean = np.mean(self.fitness_scores)
            
            # Only consider converged if std is very small AND mean is not zero
            # This prevents early termination when all fitness scores are 0
            if fitness_std < 1e-6 and abs(fitness_mean) > 1e-6:
                print(f"Population converged at generation {self.generation}")
                return True
        
        return False
    
    def save_population(self, filepath: str):
        """Save current population to file"""
        import pickle
        
        population_data = {
            "generation": self.generation,
            "individuals": self.individuals,
            "fitness_scores": self.fitness_scores,
            "config": self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(population_data, f)
        
        print(f"Population saved to {filepath}")
    
    def load_population(self, filepath: str):
        """Load population from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            population_data = pickle.load(f)
        
        self.generation = population_data["generation"]
        self.individuals = population_data["individuals"]
        self.fitness_scores = population_data["fitness_scores"]
        self.config = population_data["config"]
        
        print(f"Population loaded from {filepath}")
        print(f"Resumed at generation {self.generation}")
    
    def __len__(self) -> int:
        """Return population size"""
        return len(self.individuals)
    
    def __str__(self) -> str:
        """String representation of population"""
        if self.fitness_scores:
            best_fitness = max(self.fitness_scores)
            avg_fitness = np.mean(self.fitness_scores)
            return (f"Population(gen={self.generation}, size={len(self.individuals)}, "
                   f"best={best_fitness:.4f}, avg={avg_fitness:.4f})")
        else:
            return f"Population(gen={self.generation}, size={len(self.individuals)}, uneval)" 