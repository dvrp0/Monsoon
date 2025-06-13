# Deck Evolution System

## Overview

This document describes the Deck Evolution System, which implements a three-phase evolutionary strategy for training AI agents on card games. The system gradually transitions from exploiting known strong archetypes to exploring novel strategies, finally balancing between the two approaches.

## Three-Phase Strategy

### Phase 1: Exploit (Pure Archetypes)
- **Purpose**: Establish strong "core" weight patterns through known successful strategies
- **Duration**: Configurable (default: 30 generations)
- **Behavior**: Uses only pre-built archetype decks
- **Random Ratio**: 0% - completely deterministic
- **Benefits**: Rapid initial progress, stable baseline performance

### Phase 2: Explore (Gradual Randomization)
- **Purpose**: Inject novel feature combinations to prevent local optima
- **Duration**: Configurable (default: 30 generations)
- **Behavior**: Gradually increases randomness from 0% to max_random_ratio
- **Random Ratio**: Linear ramp (e.g., 0% → 50%)
- **Benefits**: Discovers new strategies while maintaining some archetype guidance

### Phase 3: Balance (Steady Mix)
- **Purpose**: Maintain meta strengths while rewarding generalist strategies
- **Duration**: Remainder of training
- **Behavior**: Steady mix of archetype and random decks
- **Random Ratio**: Constant (e.g., 30% random when balance_archetype_ratio=0.7)
- **Benefits**: Robust agents that handle both known and novel situations

## Implementation

### Core Classes

#### `DeckEvolutionConfig`
Main configuration class that manages the three-phase strategy.

```python
from utils import DeckEvolutionConfig

config = DeckEvolutionConfig(
    player1_archetype=ironclad_deck,    # Pre-built archetype for player 1
    player2_archetype=swarm_deck,       # Pre-built archetype for player 2
    exploit_generations=30,             # Generations in exploit phase
    explore_generations=30,             # Generations to ramp up randomness
    max_random_ratio=0.5,              # Maximum randomness in explore phase
    balance_archetype_ratio=0.7        # Archetype ratio in balance phase
)
```

#### `EvolutionaryStormbound`
Game wrapper that integrates deck evolution with the Stormbound game engine.

```python
from games.evolutionary_stormbound import EvolutionaryStormbound

game = EvolutionaryStormbound(
    seed=42,
    generation=10,              # Current generation affects deck config
    deck_config=config
)
```

### Integration with Evolution Engine

The system integrates with the existing evolutionary algorithm:

```python
from evo.evolution import EvolutionEngine
from evo.config import EvolutionaryConfig

# Create configurations
evolution_config = EvolutionaryConfig(...)
deck_config = DeckEvolutionConfig(...)

# Create and run evolution engine
engine = EvolutionEngine(evolution_config, deck_config)
engine.initialize()
results = engine.run()
```

## Key Features

### Generation-Aware Deck Configuration
- Decks are generated based on current generation number
- Automatic phase transition based on configured thresholds
- Preserves deterministic behavior for reproducible experiments

### Gradual Randomization
- Uses the improved `generate_random_deck()` function with `preserve_ratio`
- Smooth transition from pure archetypes to mixed strategies
- Configurable randomness levels and transition rates

### Flexible Archetype Support
- Supports any faction combination
- Automatic faction detection from archetype decks
- Easy to swap or modify archetype strategies

### Phase Information Tracking
- `get_phase_info()` provides detailed phase status
- Useful for logging and monitoring training progress
- Helps with debugging and experiment analysis

## Configuration Parameters

### Phase Duration
- `exploit_generations`: How long to use pure archetypes
- `explore_generations`: How long to gradually increase randomness
- Balance phase continues for remaining generations

### Randomness Control
- `max_random_ratio`: Peak randomness during explore phase (0.0-1.0)
- `balance_archetype_ratio`: Fraction of archetype decks in balance phase (0.0-1.0)

### Archetype Decks
- `player1_archetype`: List of cards for player 1's archetype
- `player2_archetype`: List of cards for player 2's archetype
- Must be valid card instances with proper faction assignments

## Usage Examples

### Basic Usage
```python
# Create deck configuration
deck_config = DeckEvolutionConfig(
    player1_archetype=my_ironclad_deck,
    player2_archetype=my_swarm_deck
)

# Get decks for specific generation
p1_deck, p2_deck = deck_config.get_deck_configuration(generation=15)

# Check phase information
phase_info = deck_config.get_phase_info(generation=15)
print(f"Phase: {phase_info['phase']}, Random Ratio: {phase_info['random_ratio']}")
```

### Integration with Training
```python
def train_with_deck_evolution():
    # Set up configurations
    config = EvolutionaryConfig(generations=100, ...)
    deck_config = DeckEvolutionConfig(...)

    # Create and run evolution engine
    engine = EvolutionEngine(config, deck_config)
    engine.initialize()

    # Training automatically uses evolving decks
    results = engine.run()
    return results
```

### Custom Phase Configuration
```python
# Short phases for quick experimentation
deck_config = DeckEvolutionConfig(
    player1_archetype=deck1,
    player2_archetype=deck2,
    exploit_generations=10,     # Short exploit phase
    explore_generations=15,     # Medium explore phase
    max_random_ratio=0.8,       # High randomness
    balance_archetype_ratio=0.6 # More random in balance
)
```

## Testing

Run the test suite to verify the system works correctly:

```bash
python test_deck_evolution.py
```

This tests:
- Phase progression logic
- Deck generation across all phases
- Phase boundary transitions
- Configuration parameter handling

## Benefits of This Approach

1. **Faster Initial Learning**: Start with known good strategies
2. **Exploration of Novel Strategies**: Gradually introduce randomness
3. **Robust Final Agents**: Balance between known and unknown situations
4. **Configurable Trade-offs**: Adjust phase durations and randomness levels
5. **Reproducible Results**: Deterministic behavior based on generation number
6. **Easy Integration**: Works with existing evolutionary algorithm infrastructure

## Implementation Notes

### Card Copying
- Uses `card.copy()` method to create independent card instances
- Prevents interference between different deck configurations
- Maintains proper player assignments

### Random Deck Generation
- Leverages the enhanced `generate_random_deck()` function
- Supports preserve_ratio for gradual randomization
- Handles faction constraints and card availability

### Error Handling
- Graceful degradation when card modules aren't available
- Fallback to original game implementation when deck_config is None
- Clear error messages for configuration issues

## Future Extensions

### Potential Enhancements
1. **Adaptive Phase Transitions**: Adjust phase duration based on fitness progress
2. **Multi-Archetype Support**: Cycle through multiple archetype sets
3. **Opponent-Specific Strategies**: Different evolution strategies per player
4. **Meta-Learning**: Learn which archetypes work best in different phases
5. **Dynamic Randomness**: Adjust randomness based on population diversity

### Configuration Extensions
- Support for more than two players
- Per-generation custom deck configurations
- Integration with external deck databases
- Automatic archetype discovery from successful agents

This system provides a robust foundation for evolutionary training with structured deck progression, balancing the exploitation of known strategies with exploration of novel approaches. 