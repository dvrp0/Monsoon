"""
Evolutionary Algorithm module for Monsoon Stormbound AI.

This module implements a competitive coevolution approach for training
Stormbound AI agents using weighted heuristic evaluation.
"""

__version__ = "1.0.0"
__author__ = "Monsoon Stormbound AI"

# Core components
from .config import EvolutionaryConfig
from .weights import WeightVector
from .features import StateFeatures
from .game_adapter import StormboundAdapter
from .heuristic_agent import HeuristicAgent

# Evolution components
from .population import Population
from .fitness import FitnessEvaluator
from .evolution import EvolutionEngine

__all__ = [
    # Core components
    "EvolutionaryConfig",
    "WeightVector", 
    "StateFeatures",
    "StormboundAdapter",
    "HeuristicAgent",
    
    # Evolution components
    "Population",
    "FitnessEvaluator",
    "EvolutionEngine",
] 