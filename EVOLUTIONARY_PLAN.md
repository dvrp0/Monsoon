# Evolutionary Algorithm Agent Implementation Plan

## Executive Summary

This document outlines the implementation plan for integrating an Evolutionary Algorithm (EA) agent into the existing Monsoon Stormbound AI project. Based on analysis of the current codebase and the provided PRD, this plan adapts the general evolutionary methodology to work with the existing `AbstractGame` interface and `Stormbound` implementation.

## Current Project Analysis

### Existing Architecture Strengths
- **Mature Game Simulator**: Complete Stormbound implementation with 110+ cards
- **Clean Game Abstraction**: `AbstractGame` interface provides standardized game API
- **Comprehensive Game State**: Rich observation space (35×5×4) with detailed game features
- **Well-Defined Action Space**: 156-action space with legal action masking
- **Existing Game Implementation**: `Stormbound` class implements full game logic with proper action handling

### Current Limitations for EA Implementation
- **No Heuristic Agent Framework**: Currently only has neural network (MuZero) and basic expert agents
- **No Weight-Based Scoring**: No parameterized evaluation functions
- **No Evolutionary Infrastructure**: Missing population management, mutation, selection operators
- **Limited Configuration Management**: No systematic hyperparameter handling for EA

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Configuration Management System
**File**: `evo/config.py`
```python
# EA-specific configuration extending existing MuZero config pattern
class EvolutionaryConfig:
    # Population parameters
    mu = 10              # Parent population size
    lambda_ = 10         # Offspring size  
    generations = 100    # Total generations
    
    # Evaluation parameters
    games_per_pairing = 20   # Games between each agent pair
    deck_configs = 3         # Number of deck configurations
    
    # Mutation parameters
    tau = 0.1               # Global mutation strength
    tau_prime = 0.01        # Individual mutation strength
    min_sigma = 1e-5        # Minimum mutation step size
    
    # Simulation parameters
    max_turns = 100         # Maximum turns per game
    num_workers = 4         # Parallel evaluation workers
    
    # Logging & checkpoints
    checkpoint_interval = 10
    log_level = "INFO"
```

#### 1.2 Game State Adapter
**File**: `evo/game_adapter.py`
```python
# Adapter to bridge existing AbstractGame with EA requirements
class StormboundAdapter:
    def __init__(self, game: AbstractGame):
        self.game = game
        self.initial_observation = None
    
    def clone_state(self) -> 'StormboundAdapter':
        """Create a deep copy of the current game state"""
        # Serialize current state and create new game instance
        new_game = Game(seed=None)  # Will need to save/restore state properly
        new_adapter = StormboundAdapter(new_game)
        # Implementation to copy game state
        return new_adapter
        
    def get_legal_actions(self) -> List[int]:
        """Get legal actions for current player"""
        return self.game.legal_actions()
        
    def apply_action(self, action: int) -> 'StormboundAdapter':
        """Apply action and return new state"""
        new_state = self.clone_state()
        observation, reward, done = new_state.game.step(action)
        return new_state
        
    def extract_features(self) -> 'StateFeatures':
        """Extract meaningful features from current game state"""
        observation = self.game.env.get_observation()
        return StateFeatures(observation, self.game.to_play())
        
    def is_terminal(self) -> bool:
        """Check if game is over"""
        return self.game.env.have_winner() is not None
        
    def get_result(self) -> Tuple[float, float]:
        """Get game result (player1_score, player2_score)"""
        winner = self.game.env.have_winner()
        if winner == 0:
            return (1.0, 0.0)  # Player 0 wins
        elif winner == 1:
            return (0.0, 1.0)  # Player 1 wins
        else:
            return (0.5, 0.5)  # Draw/ongoing
```

#### 1.3 Feature Extraction System
**File**: `evo/features.py`
```python
# Extract meaningful features from game observation for heuristic evaluation
class StateFeatures:
    def __init__(self, observation: np.ndarray, current_player: int):
        self.observation = observation  # 35×5×4 observation array
        self.current_player = current_player
        
        # Extract structured features from observation
        self._parse_observation()
        
        # Resource features
        self.mana_advantage = self.player_mana - self.opponent_mana
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
        self.tempo_advantage = self._calculate_tempo()
    
    def _parse_observation(self):
        """Parse the observation array to extract game state information"""
        # Implementation to decode the 35×5×4 observation format
        # Extract player resources, board state, hand information, etc.
        self.player_mana = self._extract_player_mana()
        self.opponent_mana = self._extract_opponent_mana()
        self.player_health = self._extract_player_health()
        self.opponent_health = self._extract_opponent_health()
        self.board_state = self._extract_board_state()
        
    def _calculate_board_control(self) -> float:
        """Calculate board control metric based on unit positions"""
        # Implementation using self.board_state
        
    def _calculate_front_line_advantage(self) -> float:
        """Calculate front line position advantage"""
        # Implementation based on observation format
        
    # Additional feature calculation methods...
```

### Phase 2: Heuristic Agent Framework (Week 2-3)

#### 2.1 Weight Vector Management
**File**: `evo/weights.py`
```python
class WeightVector:
    def __init__(self, size: int):
        self.weights = np.random.uniform(0, 1, size)
        self.sigmas = np.full(size, 0.1)  # Self-adaptive mutation strengths
    
    def mutate(self, tau: float, tau_prime: float, min_sigma: float):
        # Self-adaptive ES mutation as per PRD formula
        global_noise = np.random.normal(0, 1)
        individual_noise = np.random.normal(0, 1, len(self.sigmas))
        
        self.sigmas = np.maximum(
            self.sigmas * np.exp(tau_prime * global_noise + tau * individual_noise),
            min_sigma
        )
        
        self.weights = np.clip(
            self.weights + np.random.normal(0, self.sigmas),
            0, 1
        )
    
    def copy(self) -> 'WeightVector':
        new_vector = WeightVector(len(self.weights))
        new_vector.weights = self.weights.copy()
        new_vector.sigmas = self.sigmas.copy()
        return new_vector
```

#### 2.2 Heuristic Scoring Function
**File**: `evo/heuristic_agent.py`
```python
class HeuristicAgent:
    def __init__(self, weights: WeightVector, player_idx: int):
        self.weights = weights
        self.player_idx = player_idx
    
    def score_action(self, state: StormboundAdapter, action: int) -> float:
        # Apply action to get resulting state
        next_state = state.apply_action(action)
        
        # Extract features before and after
        current_features = state.extract_features()
        next_features = next_state.extract_features()
        
        # Compute weighted feature deltas as per PRD formula:
        # Δ(a,S) = Δ_state(enemy) – Δ_state(agent) – Δ_resource
        agent_delta = self._compute_feature_delta(current_features, next_features, for_agent=True)
        enemy_delta = self._compute_feature_delta(current_features, next_features, for_agent=False)
        resource_delta = self._compute_resource_delta(current_features, next_features)
        
        return enemy_delta - agent_delta - resource_delta
    
    def select_action(self, state: StormboundAdapter) -> int:
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return 155  # PASS action based on existing action space
            
        scores = [self.score_action(state, action) for action in legal_actions]
        best_idx = np.argmax(scores)
        return legal_actions[best_idx]
    
    def _compute_feature_delta(self, before: StateFeatures, after: StateFeatures, for_agent: bool) -> float:
        """Compute weighted sum of feature changes"""
        # Implementation using self.weights to score feature deltas
```

#### 2.3 Integration with Existing Game Interface
**File**: `evo/stormbound_ea_game.py`
```python
class StormboundEAGame(AbstractGame):
    """EA-compatible wrapper for Stormbound"""
    
    def __init__(self, agent1_weights: WeightVector, agent2_weights: WeightVector, seed=None):
        self.game = Game(seed)
        self.agents = [
            HeuristicAgent(agent1_weights, 0),
            HeuristicAgent(agent2_weights, 1)
        ]
        self.adapter = StormboundAdapter(self.game)
    
    def play_full_game(self) -> Tuple[float, float]:
        """Play complete game and return results for both agents"""
        observation = self.game.reset()
        done = False
        
        while not done:
            current_player = self.game.to_play()
            current_agent = self.agents[current_player]
            
            # Get action from heuristic agent
            action = current_agent.select_action(self.adapter)
            
            # Apply action
            observation, reward, done = self.game.step(action)
            
            # Update adapter state
            self.adapter = StormboundAdapter(self.game)
        
        # Return results based on game outcome
        return self.adapter.get_result()
    
    def step(self, action):
        return self.game.step(action)
    
    def reset(self):
        return self.game.reset()
    
    def legal_actions(self):
        return self.game.legal_actions()
    
    def to_play(self):
        return self.game.to_play()
    
    def render(self):
        return self.game.render()
```

### Phase 3: Evolutionary Algorithm Core (Week 3-4)

#### 3.1 Population Management
**File**: `evo/population.py`
```python
class Individual:
    def __init__(self, weights: WeightVector):
        self.weights = weights
        self.fitness = 0.0
        self.generation = 0
        self.id = uuid.uuid4()
    
    def mutate(self, config: EvolutionaryConfig):
        child = Individual(self.weights.copy())
        child.weights.mutate(config.tau, config.tau_prime, config.min_sigma)
        child.generation = self.generation + 1
        return child

class Population:
    def __init__(self, config: EvolutionaryConfig):
        # Determine weight vector size based on feature count
        feature_count = self._calculate_feature_count()
        
        self.individuals = [
            Individual(WeightVector(feature_count)) 
            for _ in range(config.mu)
        ]
        self.generation = 0
    
    def generate_offspring(self, config: EvolutionaryConfig) -> List[Individual]:
        offspring = []
        for _ in range(config.lambda_):
            parent = random.choice(self.individuals)
            child = parent.mutate(config)
            offspring.append(child)
        return offspring
    
    def select_survivors(self, offspring: List[Individual], config: EvolutionaryConfig):
        # (μ + λ) selection: combine parents and offspring, keep best μ
        combined = self.individuals + offspring
        combined.sort(key=lambda x: x.fitness, reverse=True)
        self.individuals = combined[:config.mu]
        self.generation += 1
```

#### 3.2 Fitness Evaluation
**File**: `evo/fitness_evaluator.py`
```python
class FitnessEvaluator:
    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.deck_configurations = self._load_deck_configs()
    
    def evaluate_population(self, individuals: List[Individual]) -> None:
        """Evaluate fitness via competitive coevolution"""
        # Reset fitness
        for individual in individuals:
            individual.fitness = 0.0
        
        # Round-robin tournament across all pairs and deck configurations
        for i, agent1 in enumerate(individuals):
            for j, agent2 in enumerate(individuals):
                if i != j:
                    wins = self._play_matches(agent1, agent2)
                    agent1.fitness += wins
    
    def _play_matches(self, agent1: Individual, agent2: Individual) -> float:
        """Play g games across D deck configurations"""
        total_wins = 0.0
        
        for deck_config in self.deck_configurations:
            for game_num in range(self.config.games_per_pairing):
                seed = hash((agent1.id, agent2.id, deck_config.id, game_num))
                
                game = StormboundEAGame(
                    agent1.weights, agent2.weights, seed=seed
                )
                
                result1, result2 = game.play_full_game()
                total_wins += result1
        
        return total_wins

    def _load_deck_configs(self) -> List['DeckConfiguration']:
        """Load deck configurations using existing card system"""
        return [
            DeckConfiguration(
                deck1_cards=["B304", "UA07", "U007", "U061", "U053", "U106", 
                           "U302", "U305", "U306", "U320", "UD31", "UE04"],
                deck2_cards=["S012", "UA07", "U007", "U211", "U061", "U206", 
                           "U053", "U001", "U216", "S013", "U071", "UA04"],
                id="ironclad_vs_shadowfen"
            )
        ]
```

#### 3.3 Parallel Simulation Orchestrator
**File**: `evo/simulation_orchestrator.py`
```python
class SimulationOrchestrator:
    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.num_workers
        )
    
    def evaluate_population_parallel(self, individuals: List[Individual]):
        """Parallel fitness evaluation using thread pool"""
        match_tasks = []
        
        # Create all match tasks
        for i, agent1 in enumerate(individuals):
            for j, agent2 in enumerate(individuals):
                if i != j:
                    task = self.executor.submit(self._evaluate_pair, agent1, agent2)
                    match_tasks.append((task, agent1))
        
        # Collect results
        for task, agent1 in match_tasks:
            wins = task.result()
            agent1.fitness += wins
    
    def _evaluate_pair(self, agent1: Individual, agent2: Individual) -> float:
        """Evaluate a single agent pair"""
        # Use existing FitnessEvaluator logic
        evaluator = FitnessEvaluator(self.config)
        return evaluator._play_matches(agent1, agent2)
```

### Phase 4: Training Pipeline (Week 4-5)

#### 4.1 Evolutionary Driver
**File**: `evo/evolutionary_driver.py`
```python
class EvolutionaryDriver:
    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.population = Population(config)
        self.evaluator = FitnessEvaluator(config)
        self.orchestrator = SimulationOrchestrator(config)
        self.logger = EvolutionaryLogger(config)
    
    def run_evolution(self):
        """Main evolutionary loop"""
        for generation in range(self.config.generations):
            print(f"Generation {generation + 1}/{self.config.generations}")
            
            # Generate offspring
            offspring = self.population.generate_offspring(self.config)
            
            # Evaluate fitness (parents + offspring)
            all_individuals = self.population.individuals + offspring
            self.orchestrator.evaluate_population_parallel(all_individuals)
            
            # Selection
            self.population.select_survivors(offspring, self.config)
            
            # Logging and checkpointing
            self.logger.log_generation(self.population, generation)
            
            if generation % self.config.checkpoint_interval == 0:
                self.save_checkpoint(generation)
        
        return self.population.individuals[0]  # Best individual
    
    def save_checkpoint(self, generation: int):
        """Save population state for resuming training"""
        checkpoint = {
            'generation': generation,
            'population': [
                {
                    'weights': ind.weights.weights.tolist(),
                    'sigmas': ind.weights.sigmas.tolist(),
                    'fitness': ind.fitness,
                    'id': str(ind.id)
                }
                for ind in self.population.individuals
            ],
            'config': self.config.__dict__
        }
        
        checkpoint_path = f"results/evolutionary/checkpoints/gen_{generation}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
```

#### 4.2 Deck Configuration Management
**File**: `evo/deck_manager.py`
```python
class DeckConfiguration:
    def __init__(self, deck1_cards: List[str], deck2_cards: List[str], id: str):
        self.deck1_cards = deck1_cards
        self.deck2_cards = deck2_cards
        self.id = id

class DeckManager:
    @staticmethod
    def load_default_configurations() -> List[DeckConfiguration]:
        """Load predefined competitive deck configurations"""
        return [
            DeckConfiguration(
                deck1_cards=["B304", "UA07", "U007", "U061", "U053", "U106", 
                           "U302", "U305", "U306", "U320", "UD31", "UE04"],
                deck2_cards=["S012", "UA07", "U007", "U211", "U061", "U206", 
                           "U053", "U001", "U216", "S013", "U071", "UA04"],
                id="ironclad_vs_shadowfen"
            ),
            # Add more balanced deck configurations
        ]
    
    @staticmethod
    def create_balanced_decks() -> List[DeckConfiguration]:
        """Create balanced deck matchups for fair evaluation"""
        # Implementation to create diverse, balanced deck configurations
```

### Phase 5: Logging, Monitoring & Analysis (Week 5-6)

#### 5.1 Comprehensive Logging System
**File**: `evo/logger.py`
```python
class EvolutionaryLogger:
    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
    
    def log_generation(self, population: Population, generation: int):
        """Log comprehensive generation statistics"""
        fitness_stats = self._compute_fitness_statistics(population)
        weight_stats = self._compute_weight_statistics(population)
        
        metrics = {
            'generation': generation,
            'fitness_mean': fitness_stats['mean'],
            'fitness_std': fitness_stats['std'],
            'fitness_max': fitness_stats['max'],
            'fitness_min': fitness_stats['min'],
            'weight_diversity': weight_stats['diversity'],
            'convergence_metric': weight_stats['convergence']
        }
        
        self._log_to_file(metrics)
        self._log_to_console(metrics)
        
    def save_best_agents(self, population: Population, top_n: int = 5):
        """Export top N weight vectors for deployment"""
        sorted_pop = sorted(population.individuals, key=lambda x: x.fitness, reverse=True)
        
        for i, individual in enumerate(sorted_pop[:top_n]):
            self._save_weight_vector(individual.weights, f"best_agent_{i+1}")
    
    def _log_to_file(self, metrics: dict):
        """Log metrics to JSON file"""
        log_path = f"results/evolutionary/logs/{self.experiment_id}.jsonl"
        with open(log_path, 'a') as f:
            json.dump(metrics, f)
            f.write('\n')
    
    def _log_to_console(self, metrics: dict):
        """Log key metrics to console"""
        print(f"Gen {metrics['generation']}: "
              f"Max={metrics['fitness_max']:.3f}, "
              f"Mean={metrics['fitness_mean']:.3f}, "
              f"Diversity={metrics['weight_diversity']:.3f}")
```

#### 5.2 Analysis and Visualization Tools
**File**: `evo/analysis.py`
```python
class EvolutionaryAnalyzer:
    def analyze_convergence(self, log_file: str):
        """Analyze convergence patterns and plateaus"""
        
    def cluster_solutions(self, population: Population):
        """Use Ward's method to identify solution clusters"""
        
    def visualize_weight_evolution(self, log_file: str):
        """Create weight histogram evolution over generations"""
        
    def performance_analysis(self, log_file: str):
        """Analyze games/second, evaluate scalability"""
```

### Phase 6: Integration & Testing (Week 6-7)

#### 6.1 Unit Testing Framework
**File**: `tests/test_evolutionary.py`
```python
class TestEvolutionaryComponents:
    def test_weight_mutation(self):
        """Test self-adaptive mutation mechanics"""
        
    def test_heuristic_scoring(self):
        """Test scoring function consistency"""
        
    def test_game_adapter(self):
        """Test state cloning and action application"""
        
    def test_population_operations(self):
        """Test selection, mutation, fitness assignment"""
    
    def test_game_simulation(self):
        """Test complete game simulation with heuristic agents"""
```

#### 6.2 Integration Testing
```python
class TestEvolutionaryIntegration:
    def test_small_scale_evolution(self):
        """Run μ=3, λ=3, G=5 evolution and verify convergence"""
        
    def test_deterministic_reproduction(self):
        """Verify same seeds produce identical results"""
        
    def test_parallel_evaluation(self):
        """Test parallel fitness evaluation correctness"""
```

#### 6.3 Main Training Script
**File**: `train_evolutionary.py`
```python
#!/usr/bin/env python3
"""
Main script to train evolutionary agents
"""
from evo.config import EvolutionaryConfig
from evo.evolutionary_driver import EvolutionaryDriver

def main():
    # Load configuration
    config = EvolutionaryConfig()
    
    # Initialize and run evolution
    driver = EvolutionaryDriver(config)
    best_agent = driver.run_evolution()
    
    print(f"Evolution complete! Best agent fitness: {best_agent.fitness}")
    
    # Save final results
    driver.logger.save_best_agents(driver.population)

if __name__ == "__main__":
    main()
```

## Implementation Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | Week 1-2 | Configuration system, game adapter, feature extraction |
| Phase 2 | Week 2-3 | Heuristic agent framework, weight management, scoring function |
| Phase 3 | Week 3-4 | Population management, fitness evaluation, parallel orchestration |
| Phase 4 | Week 4-5 | Evolutionary driver, deck management, training pipeline |
| Phase 5 | Week 5-6 | Logging system, analysis tools, visualization |
| Phase 6 | Week 6-7 | Integration testing, performance optimization |

## Project Structure

```
Monsoon/
├── evo/                           # New EA module
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── game_adapter.py           # Game state adapter for AbstractGame
│   ├── features.py               # Feature extraction
│   ├── weights.py                # Weight vector management
│   ├── heuristic_agent.py        # Heuristic agent implementation
│   ├── stormbound_ea_game.py     # EA game wrapper
│   ├── population.py             # Population management
│   ├── fitness_evaluator.py      # Fitness evaluation
│   ├── simulation_orchestrator.py # Parallel execution
│   ├── evolutionary_driver.py    # Main evolution loop
│   ├── deck_manager.py           # Deck configuration
│   ├── logger.py                 # Logging and monitoring
│   └── analysis.py               # Post-training analysis
├── tests/
│   ├── test_evolutionary.py      # EA unit tests
│   └── test_integration.py       # Integration tests
├── configs/
│   ├── ea_default.yaml           # Default EA configuration
│   └── decks/                    # Deck configurations
├── train_evolutionary.py         # Main training script
└── results/                      # Training results and logs
    └── evolutionary/
        ├── checkpoints/
        ├── logs/
        └── best_agents/
```

## Risk Mitigation

### Technical Risks
1. **Performance Bottlenecks**: Implement profiling and optimize game simulation speed
2. **Memory Usage**: Monitor population size vs. available memory  
3. **Convergence Issues**: Implement adaptive parameter tuning

### Integration Risks
1. **AbstractGame Compatibility**: Ensure proper implementation of AbstractGame interface
2. **State Serialization**: Handle game state cloning/copying correctly
3. **Action Space Mapping**: Maintain consistency with existing 156-action space

## Success Metrics

### Functional Metrics
- [ ] Successfully evolve agents that beat random baseline
- [ ] Demonstrate improvement over hand-coded expert agent
- [ ] Achieve stable convergence within 100 generations
- [ ] Support parallel evaluation on multi-core systems

### Performance Metrics
- [ ] Minimum 100 games/second simulation speed
- [ ] Memory usage < 4GB for default population size
- [ ] Linear scaling with number of CPU cores

### Code Quality Metrics
- [ ] >90% test coverage for EA components
- [ ] Complete documentation for all public APIs
- [ ] Successful integration testing with existing codebase

## Future Extensions

1. **Multi-Objective Evolution**: Optimize for multiple objectives (win rate, game length, etc.)
2. **Hierarchical Strategies**: Evolve separate weights for different game phases
3. **Transfer Learning**: Apply evolved weights to different card sets
4. **Human-AI Collaboration**: Use evolved agents to suggest moves to human players

This implementation plan provides a robust foundation for integrating evolutionary algorithms into the Monsoon project while leveraging the existing AbstractGame abstraction and maintaining compatibility with the established architecture. 