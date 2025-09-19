"""
Binary Search Solver for Inverse Relationship Calculations
Implements binary search algorithms for finding inverse relationships in WFM calculations.

Based on WFM v15 binary search algorithms for inverse relationship calculations.
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BinarySearchResult:
    """Result from binary search operation"""
    value: float
    iterations: int
    converged: bool
    error: float
    target_output: float


class BinarySearchSolver:
    """Binary search solver for inverse relationship calculations"""

    def __init__(self, tolerance: float = 0.01, max_iterations: int = 50):
        """
        Initialize binary search solver

        Args:
            tolerance: Convergence tolerance (default 1%)
            max_iterations: Maximum iterations before giving up (default 50)
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def find_inverse(self,
                    predictor: Callable[[float], float],
                    target_output: float,
                    min_input: float = 0.0,
                    max_input: float = 100.0) -> BinarySearchResult:
        """
        Binary search for inverse relationship calculations

        Given a predictor function f(x) and a target output y,
        finds the input x such that f(x) ≈ y within tolerance.

        Args:
            predictor: Function that maps input to output
            target_output: Desired output value
            min_input: Minimum bound for input search
            max_input: Maximum bound for input search

        Returns:
            BinarySearchResult with found value and convergence info
        """
        iterations = 0
        current_min = min_input
        current_max = max_input

        # Initial bounds checking
        try:
            min_output = predictor(current_min)
            max_output = predictor(current_max)

            # Ensure outputs are scalars, not arrays
            min_output = float(np.asarray(min_output).item()) if hasattr(min_output, '__len__') else float(min_output)
            max_output = float(np.asarray(max_output).item()) if hasattr(max_output, '__len__') else float(max_output)

        except (ValueError, OverflowError, ZeroDivisionError) as e:
            logger.warning(f"Error evaluating predictor at bounds: {e}")
            return BinarySearchResult(
                value=(current_min + current_max) / 2,
                iterations=0,
                converged=False,
                error=float('inf'),
                target_output=target_output
            )

        # Check if target is within bounds
        if not (min(min_output, max_output) <= target_output <= max(min_output, max_output)):
            logger.warning(f"Target {target_output} may be outside bounds "
                          f"[{min_output}, {max_output}]")

        while iterations < self.max_iterations and (current_max - current_min) > self.tolerance:
            mid_input = (current_min + current_max) / 2

            try:
                predicted_output = predictor(mid_input)

                # Ensure output is scalar, not array
                predicted_output = float(np.asarray(predicted_output).item()) if hasattr(predicted_output, '__len__') else float(predicted_output)

                # Check for numerical issues (inf, nan)
                import math
                if not math.isfinite(predicted_output):
                    raise ValueError(f"Non-finite output: {predicted_output}")

            except (ValueError, OverflowError, ZeroDivisionError):
                # Handle mathematical errors by backing off
                if iterations == 0:
                    # If first iteration fails, return middle value
                    return BinarySearchResult(
                        value=mid_input,
                        iterations=iterations + 1,
                        converged=False,
                        error=float('inf'),
                        target_output=target_output
                    )
                # For extreme functions, try a more conservative approach
                # Narrow the search space towards the safer side
                if mid_input > (current_min + current_max) / 2:
                    current_max = mid_input
                else:
                    current_min = mid_input
                iterations += 1
                continue

            error = abs(predicted_output - target_output)

            # For extreme functions with very steep curves, use relative error as well
            relative_error = error / max(abs(target_output), 1e-6)

            # Enhanced convergence check for extreme functions
            if error < self.tolerance or (relative_error < 0.01 and error < 1.0):
                return BinarySearchResult(
                    value=mid_input,
                    iterations=iterations + 1,
                    converged=True,
                    error=error,
                    target_output=target_output
                )

            # Determine which half to search
            # Assume predictor is monotonic (either increasing or decreasing)
            if min_output <= max_output:  # Increasing function
                if predicted_output < target_output:
                    current_min = mid_input
                else:
                    current_max = mid_input
            else:  # Decreasing function
                if predicted_output > target_output:
                    current_min = mid_input
                else:
                    current_max = mid_input

            iterations += 1

        # Return best approximation
        final_value = (current_min + current_max) / 2
        try:
            final_output = predictor(final_value)

            # Ensure output is scalar, not array
            final_output = float(np.asarray(final_output).item()) if hasattr(final_output, '__len__') else float(final_output)

            final_error = abs(final_output - target_output)
        except (ValueError, OverflowError, ZeroDivisionError):
            final_error = float('inf')

        return BinarySearchResult(
            value=final_value,
            iterations=iterations,
            converged=iterations < self.max_iterations,
            error=final_error,
            target_output=target_output
        )

    def find_service_level_for_occupancy(self,
                                       sl_to_occupancy_predictor: Callable[[float], float],
                                       target_occupancy: float) -> BinarySearchResult:
        """
        Find service level that produces target occupancy

        Args:
            sl_to_occupancy_predictor: Function that maps SL% to occupancy%
            target_occupancy: Target occupancy (0.0 to 1.0)

        Returns:
            BinarySearchResult with service level (0.0 to 1.0)
        """
        # Convert target to percentage for predictor
        target_occupancy_pct = target_occupancy * 100

        # Wrapper function to handle percentage conversion
        def predictor(sl_pct: float) -> float:
            return sl_to_occupancy_predictor(sl_pct)

        result = self.find_inverse(
            predictor=predictor,
            target_output=target_occupancy_pct,
            min_input=10.0,   # 10% service level minimum
            max_input=99.0    # 99% service level maximum
        )

        # Convert result back to decimal
        result.value = result.value / 100
        result.target_output = target_occupancy

        return result

    def find_occupancy_for_service_level(self,
                                       occupancy_to_sl_predictor: Callable[[float], float],
                                       target_service_level: float) -> BinarySearchResult:
        """
        Find occupancy that produces target service level

        Args:
            occupancy_to_sl_predictor: Function that maps occupancy% to SL%
            target_service_level: Target service level (0.0 to 1.0)

        Returns:
            BinarySearchResult with occupancy (0.0 to 1.0)
        """
        # Convert target to percentage for predictor
        target_sl_pct = target_service_level * 100

        # Wrapper function to handle percentage conversion
        def predictor(occ_pct: float) -> float:
            return occupancy_to_sl_predictor(occ_pct)

        result = self.find_inverse(
            predictor=predictor,
            target_output=target_sl_pct,
            min_input=30.0,   # 30% occupancy minimum
            max_input=99.0    # 99% occupancy maximum
        )

        # Convert result back to decimal
        result.value = result.value / 100
        result.target_output = target_service_level

        return result

    def find_fte_for_service_level(self,
                                 base_calls: float,
                                 base_aht: float,
                                 target_service_level: float,
                                 occupancy_to_sl_predictor: Callable[[float], float],
                                 hours_per_period: float = 40.0,
                                 shrinkage: float = 0.15) -> BinarySearchResult:
        """
        Find FTE that produces target service level

        Args:
            base_calls: Expected call volume
            base_aht: Expected average handle time (seconds)
            target_service_level: Target service level (0.0 to 1.0)
            occupancy_to_sl_predictor: Function that maps occupancy% to SL%
            hours_per_period: Hours per FTE per period (default 40 for weekly)
            shrinkage: Shrinkage factor (default 15%)

        Returns:
            BinarySearchResult with required FTE
        """
        # Calculate workload hours
        workload_hours = (base_calls * base_aht) / 3600

        # Convert target to percentage for predictor
        target_sl_pct = target_service_level * 100

        def fte_to_service_level(fte: float) -> float:
            """Calculate service level from FTE"""
            if fte <= 0:
                return 0.0

            # Calculate net supply hours
            total_supply_hours = fte * hours_per_period
            net_supply_hours = total_supply_hours * (1 - shrinkage)

            # Calculate occupancy (as decimal 0-1)
            occupancy = min(0.99, workload_hours / net_supply_hours)

            # Predict service level (convert occupancy to percentage for predictor)
            return occupancy_to_sl_predictor(occupancy * 100)

        result = self.find_inverse(
            predictor=fte_to_service_level,
            target_output=target_sl_pct,
            min_input=0.1,    # Minimum 0.1 FTE
            max_input=100.0   # Maximum 100 FTE
        )

        result.target_output = target_service_level
        return result

    def update_config(self, tolerance: Optional[float] = None,
                     max_iterations: Optional[int] = None):
        """Update solver configuration"""
        if tolerance is not None:
            self.tolerance = tolerance
        if max_iterations is not None:
            self.max_iterations = max_iterations


# Factory functions for easy integration
def create_binary_search_solver(tolerance: float = 0.01,
                               max_iterations: int = 50) -> BinarySearchSolver:
    """Create binary search solver with specified parameters"""
    return BinarySearchSolver(tolerance=tolerance, max_iterations=max_iterations)


def solve_for_target_occupancy(sl_to_occupancy_predictor: Callable[[float], float],
                              target_occupancy: float,
                              tolerance: float = 0.01) -> BinarySearchResult:
    """
    Utility function to find service level for target occupancy

    Args:
        sl_to_occupancy_predictor: Function mapping SL% to occupancy%
        target_occupancy: Target occupancy (0.0 to 1.0)
        tolerance: Convergence tolerance

    Returns:
        BinarySearchResult with service level
    """
    solver = BinarySearchSolver(tolerance=tolerance)
    return solver.find_service_level_for_occupancy(sl_to_occupancy_predictor, target_occupancy)


def solve_for_target_service_level(occupancy_to_sl_predictor: Callable[[float], float],
                                 target_service_level: float,
                                 tolerance: float = 0.01) -> BinarySearchResult:
    """
    Utility function to find occupancy for target service level

    Args:
        occupancy_to_sl_predictor: Function mapping occupancy% to SL%
        target_service_level: Target service level (0.0 to 1.0)
        tolerance: Convergence tolerance

    Returns:
        BinarySearchResult with occupancy
    """
    solver = BinarySearchSolver(tolerance=tolerance)
    return solver.find_occupancy_for_service_level(occupancy_to_sl_predictor, target_service_level)


def solve_for_required_fte(base_calls: float,
                          base_aht: float,
                          target_service_level: float,
                          occupancy_to_sl_predictor: Callable[[float], float],
                          hours_per_period: float = 40.0,
                          shrinkage: float = 0.15,
                          tolerance: float = 0.01) -> BinarySearchResult:
    """
    Utility function to find FTE for target service level

    Args:
        base_calls: Expected call volume
        base_aht: Expected average handle time (seconds)
        target_service_level: Target service level (0.0 to 1.0)
        occupancy_to_sl_predictor: Function mapping occupancy% to SL%
        hours_per_period: Hours per FTE per period
        shrinkage: Shrinkage factor
        tolerance: Convergence tolerance

    Returns:
        BinarySearchResult with required FTE
    """
    solver = BinarySearchSolver(tolerance=tolerance)
    return solver.find_fte_for_service_level(
        base_calls=base_calls,
        base_aht=base_aht,
        target_service_level=target_service_level,
        occupancy_to_sl_predictor=occupancy_to_sl_predictor,
        hours_per_period=hours_per_period,
        shrinkage=shrinkage
    )