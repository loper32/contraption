"""
Convergence Package for Workforce Management Analytics
Implements dual-loop convergence algorithms and binary search solvers.
"""

from .engine import (
    ConvergenceEngine,
    ConvergenceConfig,
    ConvergenceResult,
    ConvergenceIteration,
    RelationshipPredictor,
    LoopAResults,
    LoopBResults,
    create_convergence_config,
    create_relationship_predictor_from_models
)

from .binary_search import (
    BinarySearchSolver,
    BinarySearchResult,
    create_binary_search_solver,
    solve_for_target_occupancy,
    solve_for_target_service_level,
    solve_for_required_fte
)

__all__ = [
    # Core convergence classes
    'ConvergenceEngine',
    'ConvergenceConfig',
    'ConvergenceResult',
    'ConvergenceIteration',
    'RelationshipPredictor',
    'LoopAResults',
    'LoopBResults',

    # Binary search classes
    'BinarySearchSolver',
    'BinarySearchResult',

    # Factory functions
    'create_convergence_config',
    'create_relationship_predictor_from_models',
    'create_binary_search_solver',

    # Utility functions
    'solve_for_target_occupancy',
    'solve_for_target_service_level',
    'solve_for_required_fte'
]