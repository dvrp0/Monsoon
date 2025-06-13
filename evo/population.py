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
        if config.seed is not None:
            np.random.seed(config.seed)

    def initialize_population(self, feature_count: int):
        """Initialize population with random weight vectors"""
        from .weights import WeightVector

        self.individuals = []
        for i in range(self.config.mu):
            # Create weight vector with correct size
            individual = WeightVector(feature_count)

            # Create more diverse initial weights using different strategies
            if i < self.config.mu // 3:
                # Group 1: Conservative weights (0.2-0.8 range)
                weights = np.random.uniform(0.2, 0.8, feature_count)
            elif i < 2 * self.config.mu // 3:
                # Group 2: Extreme weights (0.0-1.0 range with bias toward extremes)
                weights = np.random.choice([0.0, 1.0], feature_count, p=[0.3, 0.7])
                weights = weights + np.random.normal(0, 0.1, feature_count)
                weights = np.clip(weights, 0, 1)
            else:
                # Group 3: Random uniform weights (original approach)
                weights = np.random.uniform(0.0, 1.0, feature_count)

            individual.set_weights(weights)

            # Initialize mutation strengths with more diversity
            base_sigma = self.config.initial_sigma
            sigma_variation = np.random.uniform(0.5, 2.0)  # 50%-200% of base sigma
            sigmas = np.full(feature_count, base_sigma * sigma_variation)

            # Add some per-feature variation in sigma
            sigma_noise = np.random.uniform(0.8, 1.2, feature_count)
            sigmas = sigmas * sigma_noise

            individual.set_sigmas(sigmas)
            self.individuals.append(individual)

        # Initialize fitness scores
        self.fitness_scores = [0.0] * self.config.mu

        print(f"Initialized population with {len(self.individuals)} individuals")
        print(f"Each individual has {feature_count} features")
        print(f"Population diversity groups: Conservative({self.config.mu//3}), Extreme({self.config.mu//3}), Random({self.config.mu - 2*(self.config.mu//3)})")

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

        # Enhanced logging with fitness score details
        fitness_std = np.std(self.fitness_scores)
        fitness_min = min(self.fitness_scores)
        fitness_max = max(self.fitness_scores)

        print(f"Generation {self.generation}: "
              f"Best fitness = {self.fitness_scores[0]:.6f}, "
              f"Avg fitness = {np.mean(self.fitness_scores):.6f}, "
              f"Std = {fitness_std:.6f}, Range = [{fitness_min:.6f}, {fitness_max:.6f}]")

        # Calculate and display mutation strength stats
        all_sigmas = np.array([ind.get_sigmas() for ind in self.individuals])
        avg_sigma = np.mean(all_sigmas)
        min_sigma = np.min(all_sigmas)
        max_sigma = np.max(all_sigmas)

        print(f"  Mutation strengths - Avg: {avg_sigma:.6f}, Range: [{min_sigma:.6f}, {max_sigma:.6f}]")

        # Reset mutation strengths if they've become too small
        if avg_sigma < self.config.min_sigma * 10:  # If avg sigma is close to minimum
            print(f"  WARNING: Mutation strengths very low (avg={avg_sigma:.2e})")
            print(f"  RESETTING mutation strengths to {self.config.initial_sigma}")
            for individual in self.individuals:
                # Reset sigmas to initial values with some randomness
                new_sigmas = np.random.uniform(
                    self.config.initial_sigma * 0.5, 
                    self.config.initial_sigma * 1.5, 
                    len(individual.get_sigmas())
                )
                individual.set_sigmas(new_sigmas)

        # Debug: print all fitness scores if they're suspiciously similar
        if fitness_std < 1e-3:
            print(f"  WARNING: Very low fitness variance (std={fitness_std:.2e})")
            print(f"  All fitness scores: {[f'{score:.6f}' for score in self.fitness_scores]}")

            # If ALL individuals have identical fitness, inject diversity
            if fitness_std == 0.0 and len(set(self.fitness_scores)) == 1:
                print(f"  CRITICAL: All individuals have identical fitness ({self.fitness_scores[0]:.6f})")
                print(f"  INJECTING DIVERSITY: Randomly mutating 50% of population")

                # Randomly select half the population for diversity injection
                num_to_mutate = max(1, len(self.individuals) // 2)
                indices_to_mutate = np.random.choice(len(self.individuals), num_to_mutate, replace=False)

                for idx in indices_to_mutate:
                    # Apply strong random mutation to break convergence
                    individual = self.individuals[idx]

                    # Temporarily boost mutation strength for diversity injection
                    original_sigmas = individual.get_sigmas().copy()
                    boosted_sigmas = original_sigmas * 5.0  # 5x stronger mutation
                    individual.set_sigmas(boosted_sigmas)

                    # Apply multiple mutations
                    for _ in range(3):  # Multiple mutation rounds
                        individual.mutate(self.config.tau * 2, self.config.tau_prime * 2, self.config.min_sigma)

                    # Restore original sigma levels (but keep them reasonably high)
                    individual.set_sigmas(np.maximum(original_sigmas, self.config.initial_sigma * 0.5))

                print(f"  Diversity injection completed on {num_to_mutate} individuals")

        # Check for early termination conditions
        if self.generation > 2:  # Only after a few generations
            if fitness_std < 1e-6:
                print(f"  NOTICE: Fitness std ({fitness_std:.2e}) is very low - may trigger early termination")

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

        # Only check convergence after a reasonable number of generations and if we have fitness scores
        min_generations_before_convergence = max(10, self.config.generations // 10)
        if self.generation > min_generations_before_convergence and len(self.fitness_scores) > 1:
            fitness_std = np.std(self.fitness_scores)
            fitness_mean = np.mean(self.fitness_scores)

            # Much more conservative convergence criteria:
            # 1. Standard deviation must be extremely small (1e-10 instead of 1e-6)
            # 2. Mean fitness must be significantly above zero
            # 3. All fitness scores must be identical (additional safety check)
            all_identical = len(set(np.round(self.fitness_scores, 10))) == 1

            if (fitness_std < 1e-10 and 
                abs(fitness_mean) > 1e-3 and 
                all_identical and
                self.generation > self.config.generations // 2):  # Only allow convergence after 50% of generations
                print(f"Population converged at generation {self.generation}")
                print(f"Fitness std: {fitness_std:.2e}, Mean: {fitness_mean:.6f}")
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