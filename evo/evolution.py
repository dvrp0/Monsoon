"""
Main evolutionary algorithm engine for training Stormbound AI agents.
"""

import os
import time
import numpy as np
from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .config import EvolutionaryConfig
    from .population import Population
    from .fitness import FitnessEvaluator
    from .features import StateFeatures


class EvolutionEngine:
    """Main engine for evolutionary algorithm training"""
    
    def __init__(self, config: 'EvolutionaryConfig'):
        self.config = config
        self.population: Optional['Population'] = None
        self.fitness_evaluator: Optional['FitnessEvaluator'] = None
        self.start_time: Optional[float] = None
        self.results_dir = config.results_dir
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Evolution Engine initialized")
        print(f"Results directory: {self.results_dir}")
    
    def initialize(self):
        """Initialize the evolutionary algorithm components"""
        from .population import Population
        from .fitness import FitnessEvaluator
        from .features import StateFeatures
        
        print("Initializing evolutionary algorithm...")
        
        # Initialize population
        self.population = Population(self.config)
        
        # Get feature count from StateFeatures
        feature_count = StateFeatures.get_feature_count()
        print(f"Feature vector size: {feature_count}")
        
        # Initialize population with random individuals
        self.population.initialize_population(feature_count)
        
        # Initialize fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(self.config)
        
        print("Initialization complete!")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete evolutionary algorithm"""
        if self.population is None or self.fitness_evaluator is None:
            raise ValueError("Engine not initialized. Call initialize() first.")
        
        print("Starting evolutionary algorithm training...")
        print(f"Population size (μ): {self.config.mu}")
        print(f"Offspring size (λ): {self.config.lambda_}")
        print(f"Max generations: {self.config.generations}")
        print(f"Games per pairing: {self.config.games_per_pairing}")
        
        self.start_time = time.time()
        
        # Main evolution loop
        while not self.population.should_terminate():
            generation_start = time.time()
            
            # Always evaluate the current population first
            current_individuals = self.population.get_parents()
            
            # For generations after the first, also generate and evaluate offspring
            if self.population.generation > 0:
                offspring = self.population.generate_offspring()
                all_individuals = current_individuals + offspring
            else:
                all_individuals = current_individuals
            
            # Evaluate fitness of all individuals
            fitness_scores = self.fitness_evaluator.evaluate_population(all_individuals)
            
            # Perform selection
            if self.population.generation == 0:
                # First generation: just assign fitness to current population
                self.population.fitness_scores = fitness_scores
                self.population.generation += 1
            else:
                # (μ + λ) selection: select best individuals from parents + offspring
                self.population.select_from_combined(all_individuals, fitness_scores)
            
            # Log generation results
            generation_time = time.time() - generation_start
            self._log_generation(generation_time)
            
            # Save checkpoint every few generations
            if self.population.generation % self.config.checkpoint_frequency == 0:
                self._save_checkpoint()
        
        # Training completed
        total_time = time.time() - self.start_time
        results = self._finalize_training(total_time)
        
        return results
    
    def _log_generation(self, generation_time: float):
        """Log generation statistics"""
        stats = self.population.get_population_stats()
        eval_stats = self.fitness_evaluator.get_stats()
        
        print(f"\n=== Generation {stats['generation']} ===")
        print(f"Time: {generation_time:.2f}s")
        print(f"Best fitness: {stats['best_fitness']:.4f}")
        print(f"Mean fitness: {stats['mean_fitness']:.4f}")
        print(f"Fitness std: {stats['std_fitness']:.4f}")
        print(f"Population diversity: {stats['diversity']:.4f}")
        print(f"Avg mutation strength: {stats['avg_mutation_strength']:.4f}")
        print(f"Games/sec: {eval_stats['games_per_second']:.1f}")
        
        # Save generation log
        if self.config.save_logs:
            self._save_generation_log(stats, eval_stats, generation_time)
    
    def _save_generation_log(self, stats: Dict, eval_stats: Dict, generation_time: float):
        """Save generation statistics to CSV"""
        log_file = os.path.join(self.results_dir, "training_log.csv")
        
        # Create header if file doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("generation,time,best_fitness,mean_fitness,std_fitness,diversity,avg_sigma,games_per_sec\n")
        
        # Append generation data
        with open(log_file, 'a') as f:
            f.write(f"{stats['generation']},{generation_time:.2f},{stats['best_fitness']:.6f},"
                   f"{stats['mean_fitness']:.6f},{stats['std_fitness']:.6f},{stats['diversity']:.6f},"
                   f"{stats['avg_mutation_strength']:.6f},{eval_stats['games_per_second']:.1f}\n")
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(self.results_dir, f"checkpoint_gen{self.population.generation}_{timestamp}.pkl")
        
        self.population.save_population(checkpoint_file)
    
    def _finalize_training(self, total_time: float) -> Dict[str, Any]:
        """Finalize training and save results"""
        print(f"\n=== Training Complete ===")
        print(f"Total time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
        print(f"Total generations: {self.population.generation}")
        
        # Get final statistics
        final_stats = self.population.get_population_stats()
        eval_stats = self.fitness_evaluator.get_stats()
        
        # Get best individual
        best_weights, best_fitness = self.population.get_best_individual()
        
        print(f"Best fitness achieved: {best_fitness:.4f}")
        print(f"Total games played: {eval_stats['total_games']}")
        print(f"Average games/sec: {eval_stats['games_per_second']:.1f}")
        
        # Save final results
        results = {
            "config": self.config,
            "total_time": total_time,
            "final_generation": self.population.generation,
            "best_fitness": best_fitness,
            "best_weights": best_weights,
            "final_stats": final_stats,
            "eval_stats": eval_stats
        }
        
        # Save best weights
        self._save_best_weights(best_weights)
        
        # Save final population
        final_pop_file = os.path.join(self.results_dir, "final_population.pkl")
        self.population.save_population(final_pop_file)
        
        # Save results summary
        self._save_results_summary(results)
        
        return results
    
    def _save_best_weights(self, best_weights):
        """Save the best weight vector"""
        import pickle
        
        weights_file = os.path.join(self.results_dir, "best_weights.pkl")
        with open(weights_file, 'wb') as f:
            pickle.dump(best_weights, f)
        
        # Also save as numpy array for easier loading
        weights_npy = os.path.join(self.results_dir, "best_weights.npy")
        np.save(weights_npy, best_weights.get_weights())
        
        print(f"Best weights saved to {weights_file} and {weights_npy}")
    
    def _save_results_summary(self, results: Dict[str, Any]):
        """Save a human-readable results summary"""
        summary_file = os.path.join(self.results_dir, "results_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("=== Evolutionary Algorithm Training Results ===\n\n")
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {results['total_time']:.2f}s ({results['total_time']/3600:.2f} hours)\n")
            f.write(f"Generations completed: {results['final_generation']}\n")
            f.write(f"Best fitness achieved: {results['best_fitness']:.6f}\n\n")
            
            f.write("=== Configuration ===\n")
            f.write(f"Population size (μ): {self.config.mu}\n")
            f.write(f"Offspring size (λ): {self.config.lambda_}\n")
            f.write(f"Max generations: {self.config.generations}\n")
            f.write(f"Games per pairing: {self.config.games_per_pairing}\n")
            f.write(f"Workers: {self.config.n_workers}\n")
            f.write(f"Max game length: {self.config.max_game_length}\n\n")
            
            f.write("=== Final Statistics ===\n")
            stats = results['final_stats']
            f.write(f"Mean fitness: {stats['mean_fitness']:.6f}\n")
            f.write(f"Fitness std: {stats['std_fitness']:.6f}\n")
            f.write(f"Population diversity: {stats['diversity']:.6f}\n")
            f.write(f"Avg mutation strength: {stats['avg_mutation_strength']:.6f}\n\n")
            
            f.write("=== Evaluation Statistics ===\n")
            eval_stats = results['eval_stats']
            f.write(f"Total games played: {eval_stats['total_games']}\n")
            f.write(f"Average time per game: {eval_stats['avg_time_per_game']:.4f}s\n")
            f.write(f"Average games per second: {eval_stats['games_per_second']:.1f}\n")
        
        print(f"Results summary saved to {summary_file}")
    
    def load_checkpoint(self, checkpoint_file: str):
        """Load training from a checkpoint"""
        from .population import Population
        from .fitness import FitnessEvaluator
        
        print(f"Loading checkpoint from {checkpoint_file}")
        
        # Initialize components
        self.population = Population(self.config)
        self.population.load_population(checkpoint_file)
        
        self.fitness_evaluator = FitnessEvaluator(self.config)
        
        print("Checkpoint loaded successfully!")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status"""
        if self.population is None:
            return {"status": "not_initialized"}
        
        status = {
            "status": "running" if not self.population.should_terminate() else "completed",
            "generation": self.population.generation,
            "max_generations": self.config.generations,
            "population_size": len(self.population),
        }
        
        if self.population.fitness_scores:
            status.update({
                "best_fitness": max(self.population.fitness_scores),
                "mean_fitness": np.mean(self.population.fitness_scores),
            })
        
        if self.start_time is not None:
            status["elapsed_time"] = time.time() - self.start_time
        
        return status 