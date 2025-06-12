#!/usr/bin/env python3
"""
Main training script for evolutionary algorithm.

Usage:
    python train_evolutionary.py [--config config.json] [--resume checkpoint.pkl]
"""

import argparse
import os
import json
import sys
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evo.config import EvolutionaryConfig
from evo.evolution import EvolutionEngine


def create_default_config() -> EvolutionaryConfig:
    """Create default configuration for evolutionary algorithm"""
    return EvolutionaryConfig(
        # Population parameters
        mu=10,                      # Parent population size
        lambda_=10,                 # Offspring population size
        generations=100,            # Maximum number of generations
        
        # Evaluation parameters
        games_per_pairing=12,       # Games played between each pair
        deck_configs=3,             # Number of deck configurations
        
        # Mutation parameters
        tau=0.1,                    # Global learning rate
        tau_prime=0.05,             # Individual learning rate
        min_sigma=1e-5,             # Minimum mutation step size
        
        # Simulation parameters
        max_turns=200,              # Maximum turns per game
        num_workers=4,              # Number of parallel workers
        
        # Logging and saving
        checkpoint_interval=10,     # Save checkpoint every N generations
        log_level="INFO",           # Logging level
        seed=42                     # Random seed for reproducibility
    )


def load_config_from_file(config_file: str) -> EvolutionaryConfig:
    """Load configuration from JSON file"""
    print(f"Loading configuration from {config_file}")
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    return EvolutionaryConfig.from_dict(config_dict)


def save_config_to_file(config: EvolutionaryConfig, config_file: str):
    """Save configuration to JSON file"""
    print(f"Saving configuration to {config_file}")
    
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Stormbound AI using evolutionary algorithms")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (JSON format)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint file to resume training from"
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save default configuration to specified file and exit"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override number of workers"
    )
    parser.add_argument(
        "--generations",
        type=int,
        help="Override maximum number of generations"
    )
    parser.add_argument(
        "--population",
        type=int,
        help="Override population size (both mu and lambda)"
    )
    
    args = parser.parse_args()
    
    # Handle save-config option
    if args.save_config:
        config = create_default_config()
        save_config_to_file(config, args.save_config)
        print(f"Default configuration saved to {args.save_config}")
        return
    
    # Load or create configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        print("No configuration file specified, using default configuration")
        config = create_default_config()
    
    # Apply command line overrides
    if args.workers:
        config.num_workers = args.workers
        print(f"Override: Using {args.workers} workers")
    
    if args.generations:
        config.generations = args.generations
        print(f"Override: Using {args.generations} generations")
    
    if args.population:
        config.mu = args.population
        config.lambda_ = args.population
        print(f"Override: Using population size {args.population}")
    
    # Create engine
    engine = EvolutionEngine(config)
    
    try:
        # Resume from checkpoint or initialize fresh
        if args.resume:
            engine.load_checkpoint(args.resume)
        else:
            engine.initialize()
        
        # Save the actual configuration used
        config_file = os.path.join(config.results_dir, "config.json")
        save_config_to_file(config, config_file)
        
        # Run training
        print("\n" + "="*60)
        print("Starting Evolutionary Algorithm Training")
        print("="*60)
        
        results = engine.run()
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print(f"Best fitness: {results['best_fitness']:.6f}")
        print(f"Total generations: {results['final_generation']}")
        print(f"Total time: {results['total_time']:.2f}s ({results['total_time']/3600:.2f} hours)")
        print(f"Results saved to: {config.results_dir}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        
        # Save current state if possible
        if engine.population is not None:
            try:
                checkpoint_file = os.path.join(config.results_dir, "interrupted_checkpoint.pkl")
                engine.population.save_population(checkpoint_file)
                print(f"Current state saved to {checkpoint_file}")
                print(f"Resume training with: python train_evolutionary.py --resume {checkpoint_file}")
            except Exception as e:
                print(f"Failed to save interrupted state: {e}")
        
        sys.exit(1)
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_training():
    """Quick test function for development"""
    print("Running quick test training...")
    
    # Create minimal test configuration
    config = EvolutionaryConfig(
        mu=4,                       # Small population for testing
        lambda_=4,
        generations=3,              # Only 3 generations
        games_per_pairing=2,        # Fewer games per pairing
        num_workers=2,              # Fewer workers
        max_turns=50,               # Shorter games
        results_dir="results/test",
        checkpoint_interval=1
    )
    
    engine = EvolutionEngine(config)
    engine.initialize()
    
    results = engine.run()
    
    print(f"Test completed! Best fitness: {results['best_fitness']:.4f}")


if __name__ == "__main__":
    # Uncomment the line below for quick testing
    # test_training()
    
    # Run main training
    main() 