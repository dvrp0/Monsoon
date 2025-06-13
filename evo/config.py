"""
Configuration management for evolutionary algorithms.
"""

import dataclasses
import json
from typing import Optional, Dict, Any


@dataclasses.dataclass
class EvolutionaryConfig:
    """EA-specific configuration extending existing MuZero config pattern"""

    # Population parameters
    mu: int = 10                    # Parent population size
    lambda_: int = 10               # Offspring size  
    generations: int = 100          # Total generations

    # Evaluation parameters
    games_per_pairing: int = 20     # Games between each agent pair
    deck_configs: int = 3           # Number of deck configurations

    # Mutation parameters
    tau: float = 0.1                # Global mutation strength
    tau_prime: float = 0.01         # Individual mutation strength
    min_sigma: float = 1e-5         # Minimum mutation step size
    initial_sigma: float = 0.1      # Initial mutation strength

    # Adaptive mutation parameters
    sigma_reset_threshold: float = 1e-4  # Threshold for resetting sigmas
    sigma_boost_factor: float = 2.0      # Factor to boost sigmas when stuck
    fitness_stagnation_gens: int = 10    # Generations to detect stagnation

    # Simulation parameters
    max_turns: int = 100            # Maximum turns per game
    num_workers: int = 128            # Parallel evaluation workers
    timeout_seconds: int = 30       # Timeout per game

    # Logging & checkpoints
    checkpoint_interval: int = 10   # Save checkpoint every N generations
    save_best_n: int = 5           # Number of best agents to save
    log_level: str = "INFO"         # Logging level
    save_logs: bool = True          # Whether to save logs
    save_generation_details: bool = True  # Save detailed generation stats
    track_weight_evolution: bool = True   # Track weight evolution
    results_dir: str = "results/evolutionary2"  # Results directory

    # Convergence criteria
    min_generations: int = 50                  # Minimum generations before early stopping
    fitness_plateau_threshold: float = 0.001  # Fitness plateau detection threshold
    plateau_generations: int = 25             # Generations to confirm plateau

    # Random seed for reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.mu <= 0:
            raise ValueError("Parent population size (mu) must be positive")
        if self.lambda_ <= 0:
            raise ValueError("Offspring size (lambda_) must be positive")
        if self.generations <= 0:
            raise ValueError("Generations must be positive")
        if self.games_per_pairing <= 0:
            raise ValueError("Games per pairing must be positive")
        if self.tau <= 0 or self.tau_prime <= 0:
            raise ValueError("Mutation parameters (tau, tau_prime) must be positive")
        if self.min_sigma <= 0:
            raise ValueError("Minimum sigma must be positive")
        if self.num_workers <= 0:
            raise ValueError("Number of workers must be positive")

    @classmethod
    def from_json(cls, json_path: str) -> 'EvolutionaryConfig':
        """Create configuration from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract nested configuration values
        config_dict = {}

        if 'population' in data:
            config_dict.update({
                'mu': data['population']['mu'],
                'lambda_': data['population']['lambda_']
            })

        if 'evolution' in data:
            config_dict.update({
                'generations': data['evolution']['generations']
            })

        if 'evaluation' in data:
            config_dict.update({
                'games_per_pairing': data['evaluation']['games_per_pairing'],
                'deck_configs': data['evaluation']['deck_configs']
            })

        if 'mutation' in data:
            config_dict.update({
                'tau': data['mutation']['tau'],
                'tau_prime': data['mutation']['tau_prime'], 
                'min_sigma': data['mutation']['min_sigma'],
                'initial_sigma': data['mutation']['initial_sigma'],
                'sigma_reset_threshold': data['mutation']['sigma_reset_threshold'],
                'sigma_boost_factor': data['mutation']['sigma_boost_factor'],
                'fitness_stagnation_gens': data['mutation']['fitness_stagnation_gens']
            })

        if 'simulation' in data:
            config_dict.update({
                'max_turns': data['simulation']['max_turns'],
                'num_workers': data['simulation']['num_workers'],
                'timeout_seconds': data['simulation']['timeout_seconds']
            })

        if 'checkpointing' in data:
            config_dict.update({
                'checkpoint_interval': data['checkpointing']['checkpoint_interval'],
                'save_best_n': data['checkpointing']['save_best_n'],
                'results_dir': data['checkpointing']['results_dir']
            })

        if 'logging' in data:
            config_dict.update({
                'log_level': data['logging']['log_level'],
                'save_generation_details': data['logging']['save_generation_details'],
                'track_weight_evolution': data['logging']['track_weight_evolution']
            })

        if 'convergence_criteria' in data:
            config_dict.update({
                'min_generations': data['convergence_criteria']['min_generations'],
                'fitness_plateau_threshold': data['convergence_criteria']['fitness_plateau_threshold'],
                'plateau_generations': data['convergence_criteria']['plateau_generations']
            })

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'EvolutionaryConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return dataclasses.asdict(self) 