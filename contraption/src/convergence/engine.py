"""
Convergence Engine for Workforce Management Analytics
Implements dual-loop convergence algorithms for service level predictions.

Based on WFM v15 convergence algorithms with dual circular calculations:
- Loop A: Call Volume Feedback (abandons → retry calls → workload increase)
- Loop B: Occupancy/AHT Feedback (stress → AHT increase → occupancy change)

Migrated from WFM v14.1 convergence algorithms with enhanced Python implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConvergenceConfig:
    """Configuration for convergence algorithm parameters"""
    max_iterations: int = 10
    tolerance: float = 0.01  # 1% convergence tolerance
    retry_rate: float = 0.3  # 30% of abandons retry
    shrinkage: float = 0.15  # 15% shrinkage factor
    damping_factor: float = 0.7  # Default damping to prevent over-correction
    step_size: float = 1.0

    # Early stopping configuration
    early_stopping_enabled: bool = True

    # Convergence mode configuration
    use_percentage_based_changes: bool = True  # Conservative mode by default
    max_percentage_change: float = 0.30  # Maximum ±30% change in percentage mode
    stability_window: int = 3
    stability_threshold: float = 0.001

    # Business constraints
    min_service_level: float = 0.0
    max_service_level: float = 1.0
    min_occupancy: float = 0.0
    max_occupancy: float = 0.99
    min_fte: float = 0.1
    max_fte: float = 1000.0

    # Debug settings
    debug_enabled: bool = False
    log_iterations: bool = False


@dataclass
class LoopAResults:
    """Results from Loop A: Call Volume Feedback"""
    abandon_rate: float
    retry_calls: float
    total_calls: float
    workload_hours: float
    occupancy: float
    service_level: float


@dataclass
class LoopBResults:
    """Results from Loop B: Occupancy/AHT Feedback"""
    adjusted_aht: float
    final_workload_hours: float
    final_occupancy: float
    final_service_level: float


@dataclass
class ConvergenceIteration:
    """Data for a single convergence iteration"""
    iteration: int
    calls: float
    aht: float
    occupancy: float
    service_level: float
    abandon_rate: float
    convergence_error: float
    loop_a_results: LoopAResults
    loop_b_results: LoopBResults


@dataclass
class ConvergenceResult:
    """Final convergence calculation result"""
    converged: bool
    iterations: int
    final_error: float
    execution_time: float

    # Input parameters
    base_calls: float
    base_aht: float
    planned_fte: float

    # Final results
    final_calls: float
    final_service_level: float
    final_occupancy: float
    final_aht: float
    final_abandon_rate: float

    # Quality metrics
    stability: str
    confidence: float

    # Detailed history
    iteration_history: List[ConvergenceIteration] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RelationshipPredictor:
    """Interface for relationship prediction functions"""

    def __init__(self,
                 predict_sl_from_occupancy: Callable[[float], float],
                 predict_abandon_from_sl: Callable[[float], float],
                 predict_aht_from_occupancy: Callable[[float], float]):
        self.predict_sl_from_occupancy = predict_sl_from_occupancy
        self.predict_abandon_from_sl = predict_abandon_from_sl
        self.predict_aht_from_occupancy = predict_aht_from_occupancy


class ConvergenceEngine:
    """Main convergence engine implementing dual-loop feedback algorithms"""

    def __init__(self, config: Optional[ConvergenceConfig] = None):
        self.config = config or ConvergenceConfig()
        self.iteration_history: List[ConvergenceIteration] = []
        self.start_time: float = 0.0

    def iterate_convergence(self,
                          base_calls: float,
                          base_aht: float,
                          planned_fte: float,
                          period: str,
                          relationship_predictor: RelationshipPredictor) -> ConvergenceResult:
        """
        Primary convergence algorithm implementing dual-loop feedback

        Args:
            base_calls: Initial call volume for the period
            base_aht: Initial average handle time in seconds
            planned_fte: Planned staffing level (FTE)
            period: Planning period ('weekly', 'daily', 'monthly')
            relationship_predictor: Functions for predicting relationships

        Returns:
            ConvergenceResult with final calculations and iteration history
        """
        self.start_time = time.time()
        self.iteration_history = []

        # Validate inputs
        self._validate_inputs(base_calls, base_aht, planned_fte)

        # Calculate hours per period
        hours_per_period = self._calculate_hours_per_period(period)

        # SUPPLY/DEMAND MODEL IMPLEMENTATION
        workload_hours = (base_calls * base_aht) / 3600  # Convert to hours
        total_supply_hours = planned_fte * hours_per_period
        net_supply_hours = total_supply_hours * (1 - self.config.shrinkage)

        # Base occupancy calculation: demand ÷ net supply
        initial_occupancy = min(self.config.max_occupancy, workload_hours / net_supply_hours)

        # Calculate initial metrics
        initial_sl = relationship_predictor.predict_sl_from_occupancy(initial_occupancy)
        initial_abandon = relationship_predictor.predict_abandon_from_sl(initial_sl)

        if self.config.debug_enabled:
            logger.info(f"Initial conditions: Calls={base_calls}, AHT={base_aht}s, "
                       f"Occ={initial_occupancy:.1%}, SL={initial_sl:.1%}")

        # Initialize convergence tracking
        current_calls = base_calls
        current_aht = base_aht
        current_sl = initial_sl
        previous_sl = 0.0
        final_occupancy = initial_occupancy

        # Set baseline values for percentage change calculations
        relationship_predictor._baseline_aht = base_aht
        relationship_predictor._baseline_occupancy = initial_occupancy

        if self.config.debug_enabled:
            logger.info(f"Starting convergence with: SL={current_sl:.1%}, Occ={initial_occupancy:.1%}, "
                       f"Calls={current_calls:.0f}, AHT={current_aht:.0f}s")

        # Enhanced oscillation detection tracking
        sl_history = []

        # DUAL-LOOP CONVERGENCE ALGORITHM
        for iteration in range(self.config.max_iterations):
            if self.config.log_iterations:
                logger.info(f"=== ITERATION {iteration + 1} ===")

            # Early convergence check: if we're already close to a reasonable SL (85-95%), be more conservative
            if iteration == 0 and current_sl > 0.85:
                conservative_tolerance = self.config.tolerance * 2  # Double tolerance for good starting conditions
                if self.config.debug_enabled:
                    logger.info(f"Good initial SL ({current_sl:.1%}), using conservative tolerance: {conservative_tolerance:.1%}")
            else:
                conservative_tolerance = self.config.tolerance

            # **LOOP A: CALL VOLUME FEEDBACK**
            loop_a_results = self._execute_loop_a(
                current_sl,
                base_calls,
                current_aht,
                net_supply_hours,
                relationship_predictor
            )

            current_calls = loop_a_results.total_calls

            # **LOOP B: OCCUPANCY/AHT FEEDBACK**
            loop_b_results = self._execute_loop_b(
                loop_a_results.occupancy,
                current_calls,
                net_supply_hours,
                relationship_predictor
            )

            # Update current values for next iteration with bounds checking
            current_aht = loop_b_results.adjusted_aht
            current_sl = loop_b_results.final_service_level

            # Apply realistic bounds to prevent extreme swings
            # Don't allow occupancy to change more than 20% per iteration
            max_occ_change = 0.20
            if abs(loop_b_results.final_occupancy - final_occupancy) > max_occ_change:
                if loop_b_results.final_occupancy > final_occupancy:
                    final_occupancy = min(loop_b_results.final_occupancy, final_occupancy + max_occ_change)
                else:
                    final_occupancy = max(loop_b_results.final_occupancy, final_occupancy - max_occ_change)
                logger.debug(f"Bounded occupancy change to {final_occupancy:.1%} (was {loop_b_results.final_occupancy:.1%})")
            else:
                final_occupancy = loop_b_results.final_occupancy

            # Apply damping if configured
            if self.config.damping_factor < 1.0:
                damped_change = (current_sl - previous_sl) * self.config.damping_factor
                current_sl = previous_sl + damped_change

            # Store iteration data
            iteration_data = ConvergenceIteration(
                iteration=iteration + 1,
                calls=current_calls,
                aht=current_aht,
                occupancy=final_occupancy,
                service_level=current_sl,
                abandon_rate=loop_a_results.abandon_rate,
                convergence_error=abs(current_sl - previous_sl),
                loop_a_results=loop_a_results,
                loop_b_results=loop_b_results
            )

            self.iteration_history.append(iteration_data)

            # **CONVERGENCE CHECKING**
            sl_change = abs(current_sl - previous_sl)

            if self.config.log_iterations:
                logger.info(f"  Convergence: SL change = {sl_change:.3%} "
                           f"(tolerance: {self.config.tolerance:.1%})")

            # Track oscillation
            sl_history.append(current_sl)
            if len(sl_history) > 6:  # Keep last 6 values
                sl_history.pop(0)

            # Check for convergence
            if sl_change < conservative_tolerance:
                if self.config.debug_enabled:
                    logger.info(f"Converged after {iteration + 1} iterations "
                               f"with {sl_change:.3%} tolerance")
                break

            # Check for oscillation
            if self._detect_oscillation(sl_history) and len(sl_history) >= 4:
                if self.config.debug_enabled:
                    logger.info(f"Oscillation detected - convergence achieved "
                               f"within acceptable range after {iteration + 1} iterations")
                break

            # Early stopping check
            if (self.config.early_stopping_enabled and
                self._should_early_stop(sl_history)):
                if self.config.debug_enabled:
                    logger.info("Early stopping - stability achieved")
                break

            previous_sl = current_sl

        # Validate results before returning
        if abs(current_sl - initial_sl) > 0.5:  # More than 50% change
            logger.warning(f"Large service level change detected: {initial_sl:.1%} → {current_sl:.1%}")

        if current_aht < 30 or current_aht > 3600:  # Less than 30s or more than 1 hour
            logger.warning(f"Extreme AHT value detected: {current_aht:.0f}s")

        # Generate final results
        return self._generate_convergence_result(
            iteration + 1,
            current_calls,
            current_aht,
            final_occupancy,
            current_sl,
            base_calls,
            base_aht,
            initial_occupancy,
            initial_sl
        )

    def _execute_loop_a(self,
                       current_sl: float,
                       base_calls: float,
                       current_aht: float,
                       net_supply_hours: float,
                       relationship_predictor: RelationshipPredictor) -> LoopAResults:
        """Execute Loop A: Call Volume Feedback"""

        if self.config.log_iterations:
            logger.info("Loop A - Call Volume Feedback:")

        # A1: Current SL → Abandon Rate
        abandon_rate = relationship_predictor.predict_abandon_from_sl(current_sl)
        if self.config.log_iterations:
            logger.info(f"  A1: {current_sl:.1%} SL → {abandon_rate:.2%} abandon")

        # A2: Calculate retry calls from abandons
        retry_calls = base_calls * abandon_rate * self.config.retry_rate
        if self.config.log_iterations:
            logger.info(f"  A2: {base_calls} * {abandon_rate:.2%} * {self.config.retry_rate} "
                       f"= {retry_calls:.0f} retry calls")

        # A3: Update total call volume
        total_calls = base_calls + retry_calls
        if self.config.log_iterations:
            logger.info(f"  A3: Total calls = {base_calls} + {retry_calls:.0f} = {total_calls:.0f}")

        # A4: Recalculate workload and occupancy with new call volume
        workload_hours = (total_calls * current_aht) / 3600
        occupancy = min(self.config.max_occupancy, workload_hours / net_supply_hours)
        if self.config.log_iterations:
            logger.info(f"  A4: New workload = {workload_hours:.1f}h, "
                       f"occupancy = {occupancy:.1%}")

        # A5: Calculate SL after call volume increase
        service_level = relationship_predictor.predict_sl_from_occupancy(occupancy)
        if self.config.log_iterations:
            logger.info(f"  A5: SL after call increase = {service_level:.1%}")

        return LoopAResults(
            abandon_rate=abandon_rate,
            retry_calls=retry_calls,
            total_calls=total_calls,
            workload_hours=workload_hours,
            occupancy=occupancy,
            service_level=service_level
        )

    def _execute_loop_b(self,
                       current_occupancy: float,
                       current_calls: float,
                       net_supply_hours: float,
                       relationship_predictor: RelationshipPredictor) -> LoopBResults:
        """Execute Loop B: Occupancy/AHT Feedback"""

        if self.config.log_iterations:
            logger.info("Loop B - Occupancy/AHT Feedback:")

        # B1: Current occupancy → Adjusted AHT (configurable approach)
        predicted_aht = relationship_predictor.predict_aht_from_occupancy(current_occupancy)

        if self.config.use_percentage_based_changes:
            # CONSERVATIVE MODE: Use percentage-based changes
            baseline_aht = getattr(relationship_predictor, '_baseline_aht', None)
            baseline_occupancy = getattr(relationship_predictor, '_baseline_occupancy', None)

            if baseline_aht is None or baseline_occupancy is None:
                logger.warning("No baseline AHT/occupancy set - using direct prediction as fallback")
                adjusted_aht = predicted_aht
            else:
                # Calculate what the model predicts for baseline occupancy
                baseline_predicted_aht = relationship_predictor.predict_aht_from_occupancy(baseline_occupancy)

                # Calculate percentage change suggested by the model
                if baseline_predicted_aht > 0:
                    aht_change_ratio = predicted_aht / baseline_predicted_aht
                    # Apply this percentage change to the actual starting AHT
                    adjusted_aht = baseline_aht * aht_change_ratio

                    # Apply configurable bounds
                    max_change_ratio = 1 + self.config.max_percentage_change
                    min_change_ratio = 1 - self.config.max_percentage_change
                    bounded_ratio = max(min_change_ratio, min(max_change_ratio, aht_change_ratio))
                    adjusted_aht = baseline_aht * bounded_ratio

                    if self.config.log_iterations:
                        mode_info = f"CONSERVATIVE: {aht_change_ratio:.1%} ratio → {adjusted_aht:.0f}s"
                        logger.info(f"  B1: {current_occupancy:.1%} occ → {mode_info}")
                else:
                    logger.warning("Invalid baseline predicted AHT - using direct prediction")
                    adjusted_aht = predicted_aht
        else:
            # AGGRESSIVE MODE: Use direct model predictions
            adjusted_aht = predicted_aht
            if self.config.log_iterations:
                logger.info(f"  B1: {current_occupancy:.1%} occ → AGGRESSIVE: {adjusted_aht:.0f}s AHT")

        if self.config.log_iterations:
            logger.info(f"  B1: {current_occupancy:.1%} occ → {adjusted_aht:.0f}s AHT")

        # B2: Recalculate workload with adjusted AHT
        final_workload_hours = (current_calls * adjusted_aht) / 3600
        final_occupancy = min(self.config.max_occupancy,
                             final_workload_hours / net_supply_hours)
        if self.config.log_iterations:
            logger.info(f"  B2: Final workload = {final_workload_hours:.1f}h, "
                       f"occupancy = {final_occupancy:.1%}")

        # B3: Calculate final SL for this iteration
        final_service_level = relationship_predictor.predict_sl_from_occupancy(final_occupancy)
        if self.config.log_iterations:
            logger.info(f"  B3: Final SL = {final_service_level:.1%}")

        return LoopBResults(
            adjusted_aht=adjusted_aht,
            final_workload_hours=final_workload_hours,
            final_occupancy=final_occupancy,
            final_service_level=final_service_level
        )

    def _detect_oscillation(self, sl_history: List[float]) -> bool:
        """Detect oscillation in convergence values"""
        if len(sl_history) < 4:
            return False

        # Calculate range and average change
        sl_range = max(sl_history) - min(sl_history)
        recent_changes = [abs(sl_history[i] - sl_history[i-1])
                         for i in range(1, len(sl_history))]
        avg_change = np.mean(recent_changes) if recent_changes else 0

        # Detect oscillation: small range and small average change
        is_oscillating = (sl_range < self.config.tolerance * 2 and
                         avg_change < self.config.tolerance * 0.5)

        return is_oscillating

    def _should_early_stop(self, sl_history: List[float]) -> bool:
        """Check if early stopping criteria are met"""
        if (not self.config.early_stopping_enabled or
            len(sl_history) < self.config.stability_window):
            return False

        recent = sl_history[-self.config.stability_window:]
        changes = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        max_change = max(changes) if changes else float('inf')

        return max_change < self.config.stability_threshold

    def _calculate_hours_per_period(self, period: str) -> float:
        """Calculate hours per period based on time granularity"""
        period_lower = period.lower()

        if 'daily' in period_lower:
            return 8.0  # 8 hours per day
        elif 'monthly' in period_lower:
            return 160.0  # Approximate monthly hours (4 weeks)
        else:
            return 40.0  # Default to weekly (40 hours)

    def _validate_inputs(self, base_calls: float, base_aht: float, planned_fte: float):
        """Validate convergence inputs"""
        if base_calls <= 0:
            raise ValueError("Base calls must be positive")
        if base_aht <= 0:
            raise ValueError("Base AHT must be positive")
        if planned_fte <= 0:
            raise ValueError("Planned FTE must be positive")
        if self.config.tolerance <= 0:
            raise ValueError("Tolerance must be positive")

    def _generate_convergence_result(self,
                                   iterations: int,
                                   final_calls: float,
                                   final_aht: float,
                                   final_occupancy: float,
                                   final_sl: float,
                                   base_calls: float,
                                   base_aht: float,
                                   initial_occupancy: float,
                                   initial_sl: float) -> ConvergenceResult:
        """Generate final convergence result"""

        execution_time = time.time() - self.start_time
        converged = iterations < self.config.max_iterations
        final_error = (self.iteration_history[-1].convergence_error
                      if self.iteration_history else 0.0)

        # Calculate confidence score
        confidence = max(0.0, min(1.0, 1 - (final_error / self.config.tolerance)))

        # Determine stability
        if final_error < self.config.tolerance * 0.1:
            stability = "excellent"
        elif final_error < self.config.tolerance * 0.5:
            stability = "good"
        elif final_error < self.config.tolerance * 2:
            stability = "fair"
        else:
            stability = "poor"

        # Generate warnings
        warnings = []
        if not converged:
            warnings.append("Maximum iterations reached without convergence")
        if final_occupancy > 0.95:
            warnings.append("Occupancy exceeds 95% - may be unrealistic")
        if final_sl < 0.5:
            warnings.append("Service level below 50% - check staffing adequacy")

        final_abandon_rate = (self.iteration_history[-1].abandon_rate
                             if self.iteration_history else 0.0)

        if self.config.debug_enabled:
            logger.info("=== CONVERGENCE COMPLETE ===")
            logger.info(f"📊 FINAL COMPARISON:")
            logger.info(f"   BASE  → Calls: {base_calls}, AHT: {base_aht}s, "
                       f"Occ: {initial_occupancy:.1%}, SL: {initial_sl:.1%}")
            logger.info(f"   FINAL → Calls: {final_calls:.0f}, AHT: {final_aht:.0f}s, "
                       f"Occ: {final_occupancy:.1%}, SL: {final_sl:.1%}")
            logger.info(f"Iterations: {iterations}/{self.config.max_iterations}, "
                       f"Status: {stability}")

        return ConvergenceResult(
            converged=converged,
            iterations=iterations,
            final_error=final_error,
            execution_time=execution_time,
            base_calls=base_calls,
            base_aht=base_aht,
            planned_fte=0.0,  # Will be set by caller
            final_calls=final_calls,
            final_service_level=final_sl,
            final_occupancy=final_occupancy,
            final_aht=final_aht,
            final_abandon_rate=final_abandon_rate,
            stability=stability,
            confidence=confidence,
            iteration_history=self.iteration_history.copy(),
            warnings=warnings
        )

    def get_iteration_history(self) -> List[ConvergenceIteration]:
        """Get convergence iteration history"""
        return self.iteration_history.copy()

    def update_config(self, new_config: Dict[str, Any]):
        """Update convergence configuration"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


def create_convergence_config(calculation_type: str = 'fte') -> ConvergenceConfig:
    """Create default convergence configuration for different calculation types"""

    if calculation_type == 'fte':
        return ConvergenceConfig(
            max_iterations=10,
            tolerance=0.01,
            debug_enabled=False
        )
    elif calculation_type == 'service_level':
        return ConvergenceConfig(
            max_iterations=50,
            tolerance=0.01,
            debug_enabled=False
        )
    else:
        return ConvergenceConfig(
            max_iterations=20,
            tolerance=0.01,
            debug_enabled=False
        )


def create_relationship_predictor_from_models(models: Dict[str, Any]) -> RelationshipPredictor:
    """Create relationship predictor from trained curve fitting models"""

    # Enhanced debugging for relationship predictor creation
    logger.info(f"Creating relationship predictor from {len(models)} models")

    # Create standardized relationship lookup by converting display names to metric pairs
    relationship_map = {}

    # Log detailed model information and build relationship map
    valid_models = 0
    for display_name, model_info in models.items():
        if model_info:
            x_metric = model_info.get('x_metric', 'NOT_FOUND')
            y_metric = model_info.get('y_metric', 'NOT_FOUND')
            model_type = model_info.get('model_type', 'NOT_FOUND')
            logger.info(f"FOUND {display_name}: {x_metric} → {y_metric} ({model_type})")

            # Normalize field names to match expected keys
            x_normalized = x_metric
            y_normalized = y_metric

            # Handle field name variations
            if y_metric == 'abandonment':
                y_normalized = 'abandonment_rate'
            elif y_metric == 'aht':
                y_normalized = 'average_handle_time'

            # Create standardized lookup key for WFM v14.1 style relationships
            metric_key = f"{x_normalized}_to_{y_normalized}"
            relationship_map[metric_key] = model_info

            logger.info(f"  Mapped to key: {metric_key}")

            valid_models += 1
        else:
            logger.warning(f"Empty model info for {display_name}")

    logger.info(f"Total valid models: {valid_models}/{len(models)}")
    logger.info(f"Relationship map keys: {list(relationship_map.keys())}")

    # WFM v14.1 style validation - check for critical relationships
    critical_relationships = ['service_level_to_occupancy', 'occupancy_to_service_level']
    available_critical = [key for key in critical_relationships if key in relationship_map]

    logger.info(f"🔍 CONVERGENCE DIAGNOSTICS:")
    logger.info(f"  Critical relationships needed: {critical_relationships}")
    logger.info(f"  Available critical: {available_critical}")
    logger.info(f"  Full relationship map keys: {list(relationship_map.keys())}")

    # Check each critical relationship individually for debugging
    for crit_rel in critical_relationships:
        exists = crit_rel in relationship_map
        status = "✅ FOUND" if exists else "❌ MISSING"
        logger.info(f"  '{crit_rel}': {status}")

    # Detailed relationship analysis
    if available_critical:
        logger.info(f"✅ Critical relationships available: {available_critical}")
        for rel_key in available_critical:
            rel_info = relationship_map[rel_key]
            model_type = rel_info.get('model_type', 'unknown')
            params = rel_info.get('parameters', rel_info.get('params', []))
            logger.info(f"  {rel_key}: {model_type} model with {len(params)} parameters")
    else:
        available_relationships = list(relationship_map.keys())
        logger.warning(f"⚠️ No direct critical relationships found. Available: {available_relationships}")
        logger.warning("   Will attempt inverse calculations and use empirical fallbacks if needed.")

    # Enhanced validation - check if we have at least one SL-Occupancy relationship in either direction
    has_any_sl_occ_relationship = any(
        'service_level' in key and 'occupancy' in key
        for key in relationship_map.keys()
    )

    if has_any_sl_occ_relationship:
        logger.info("✅ Found at least one Service Level ↔ Occupancy relationship")
    else:
        logger.warning("⚠️ No Service Level ↔ Occupancy relationships found - will use empirical fallbacks")

    def predict_sl_from_occupancy(occupancy: float) -> float:
        """Predict service level from occupancy using trained model"""

        # WFM v14.1 style direct lookup - try direct relationship first
        direct_key = "occupancy_to_service_level"
        if direct_key in relationship_map:
            model_info = relationship_map[direct_key]
            logger.info(f"Using direct Occupancy→SL relationship")
            try:
                # Apply model - occupancy is 0-1, model expects percentage
                result = _apply_curve_model(occupancy * 100, model_info)
                result_decimal = result / 100  # Convert back to decimal
                logger.info(f"Direct Occ→SL prediction: {occupancy:.1%} → {result_decimal:.1%}")
                return max(0.01, min(0.99, result_decimal))
            except Exception as e:
                logger.warning(f"Error in direct occupancy→SL prediction: {e}")

        # Try inverse calculation using SL→Occupancy relationship
        inverse_key = "service_level_to_occupancy"
        if inverse_key in relationship_map:
            model_info = relationship_map[inverse_key]
            logger.info(f"Using SL→Occ relationship for inverse calculation, target occupancy={occupancy:.1%}")
            try:
                # Use binary search to find the service level that produces the target occupancy
                from .binary_search import BinarySearchSolver

                def sl_to_occ_predictor(sl_percent):
                    try:
                        result = _apply_curve_model(sl_percent, model_info)
                        # Ensure result is a scalar, not array
                        if hasattr(result, '__len__'):
                            result = float(np.asarray(result).item())
                        else:
                            result = float(result)

                        # Validate result is finite using math.isfinite
                        import math
                        if not math.isfinite(result):
                            logger.warning(f"Non-finite result for SL {sl_percent}%: {result}")
                            return 50.0  # Fallback to 50% occupancy

                        logger.debug(f"SL {sl_percent}% → Occ {result}%")
                        return result
                    except Exception as e:
                        logger.warning(f"Error in SL predictor for {sl_percent}%: {e}")
                        return 50.0  # Fallback to 50% occupancy

                # Test the predictor at boundaries for debugging
                try:
                    test_1 = sl_to_occ_predictor(1.0)
                    test_50 = sl_to_occ_predictor(50.0)
                    test_99 = sl_to_occ_predictor(99.0)
                    logger.info(f"Predictor test: SL 1% → Occ {test_1:.1f}%, SL 50% → Occ {test_50:.1f}%, SL 99% → Occ {test_99:.1f}%")
                except Exception as e:
                    logger.warning(f"Error testing predictor boundaries: {e}")

                target_occ_percent = occupancy * 100
                logger.info(f"Target occupancy: {target_occ_percent:.1f}%")

                # For extreme power functions like y = -0.255 × x^10.866 + 0.913,
                # we need a more sophisticated approach than standard binary search

                # First, test the function behavior to understand its range
                test_points = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 95.0]
                valid_points = []

                for test_sl in test_points:
                    try:
                        test_occ = sl_to_occ_predictor(test_sl)
                        if math.isfinite(test_occ) and 0 <= test_occ <= 100:
                            valid_points.append((test_sl, test_occ))
                            logger.debug(f"Valid point: SL {test_sl}% → Occ {test_occ:.1f}%")
                    except Exception as e:
                        logger.debug(f"Invalid point SL {test_sl}%: {e}")

                if len(valid_points) < 2:
                    logger.warning(f"Insufficient valid points for inverse calculation: {len(valid_points)}")
                    # Fallback to empirical WFM relationship
                    logger.info("Using empirical WFM relationship for occupancy→SL prediction")
                    if occupancy <= 0.70:
                        estimated_sl = 0.95  # 70% occupancy can typically achieve 95% SL
                    elif occupancy <= 0.80:
                        estimated_sl = 0.85  # 80% occupancy typically achieves 85% SL
                    elif occupancy <= 0.90:
                        estimated_sl = 0.70  # 90% occupancy typically achieves 70% SL
                    else:
                        estimated_sl = 0.50  # >90% occupancy typically achieves 50% SL

                    logger.info(f"EMPIRICAL FALLBACK: {occupancy:.1%} occupancy → {estimated_sl:.1%} service level")
                    return max(0.01, min(0.99, estimated_sl))

                # Sort points by occupancy to find bounds
                valid_points.sort(key=lambda x: x[1])  # Sort by occupancy

                # Find the bracketing points for our target
                target_occ_percent = occupancy * 100

                # Check if target is within achievable range
                min_achievable_occ = valid_points[0][1]
                max_achievable_occ = valid_points[-1][1]

                logger.info(f"Achievable occupancy range: {min_achievable_occ:.1f}% to {max_achievable_occ:.1f}%")
                logger.info(f"Target occupancy: {target_occ_percent:.1f}%")

                # Check if the achievable range is realistic for WFM (occupancy should be 20-99%)
                if max_achievable_occ < 20 or min_achievable_occ > 99 or (max_achievable_occ - min_achievable_occ) < 5:
                    logger.warning("Mathematical model produces unrealistic occupancy range, using empirical fallback")
                    if occupancy <= 0.70:
                        estimated_sl = 0.95
                    elif occupancy <= 0.80:
                        estimated_sl = 0.85
                    elif occupancy <= 0.90:
                        estimated_sl = 0.70
                    else:
                        estimated_sl = 0.50

                    logger.info(f"EMPIRICAL FALLBACK: {occupancy:.1%} occupancy → {estimated_sl:.1%} service level")
                    return max(0.01, min(0.99, estimated_sl))

                if target_occ_percent < min_achievable_occ:
                    # Target below minimum achievable - use the SL that gives minimum occupancy
                    result_sl = valid_points[0][0]
                    logger.info(f"Target below minimum achievable, using SL {result_sl:.1f}%")
                elif target_occ_percent > max_achievable_occ:
                    # Target above maximum achievable - use the SL that gives maximum occupancy
                    result_sl = valid_points[-1][0]
                    logger.info(f"Target above maximum achievable, using SL {result_sl:.1f}%")
                else:
                    # Target is within range - use binary search with tight bounds
                    bracket_lower = None
                    bracket_upper = None

                    for i in range(len(valid_points) - 1):
                        occ1, occ2 = valid_points[i][1], valid_points[i + 1][1]
                        if min(occ1, occ2) <= target_occ_percent <= max(occ1, occ2):
                            bracket_lower = valid_points[i][0]
                            bracket_upper = valid_points[i + 1][0]
                            break

                    if bracket_lower is None or bracket_upper is None:
                        # Fallback to full range
                        bracket_lower = valid_points[0][0]
                        bracket_upper = valid_points[-1][0]

                    logger.info(f"Binary search bounds: SL {bracket_lower:.1f}% to {bracket_upper:.1f}%")

                    # Use binary search with optimized settings for power functions
                    solver = BinarySearchSolver(tolerance=0.1, max_iterations=100)  # Relaxed tolerance, more iterations

                    result = solver.find_inverse(
                        predictor=sl_to_occ_predictor,
                        target_output=target_occ_percent,
                        min_input=bracket_lower,
                        max_input=bracket_upper
                    )

                    if result.converged:
                        result_sl = result.value
                        logger.info(f"Binary search converged: SL {result_sl:.1f}% for target occupancy {target_occ_percent:.1f}%")
                    else:
                        # Use interpolation as fallback
                        result_sl = (bracket_lower + bracket_upper) / 2
                        logger.warning(f"Binary search failed, using midpoint: SL {result_sl:.1f}%")

                # Convert result to decimal and return
                result_decimal = result_sl / 100  # Convert from percentage to decimal
                logger.info(f"SUCCESS: Inverse SL→occupancy prediction: target {occupancy:.1%} → found SL {result_decimal:.1%}")
                return max(0.01, min(0.99, result_decimal))

            except Exception as e:
                logger.error(f"ERROR in inverse SL→occupancy prediction: {e}", exc_info=True)
                # Fallback: Use empirical estimation based on typical WFM relationships
                logger.info("Using empirical fallback for occupancy→SL prediction")
                # Typical contact center relationship: higher occupancy = lower achievable service level
                if occupancy <= 0.70:
                    estimated_sl = 0.95  # 70% occupancy can typically achieve 95% SL
                elif occupancy <= 0.80:
                    estimated_sl = 0.85  # 80% occupancy typically achieves 85% SL
                elif occupancy <= 0.90:
                    estimated_sl = 0.70  # 90% occupancy typically achieves 70% SL
                else:
                    estimated_sl = 0.50  # >90% occupancy typically achieves 50% SL

                logger.info(f"EMPIRICAL FALLBACK: {occupancy:.1%} occupancy → {estimated_sl:.1%} service level")
                return max(0.01, min(0.99, estimated_sl))

        # Final fallback if no relationships found at all
        # This prevents the ValueError and provides a conservative estimate
        logger.warning("No Service Level ↔ Occupancy relationship found, using empirical estimation")
        available_relationships = list(relationship_map.keys())
        logger.warning(f"Available relationships: {available_relationships}")

        # Use empirical WFM relationship for final fallback
        if occupancy <= 0.70:
            estimated_sl = 0.95
        elif occupancy <= 0.80:
            estimated_sl = 0.85
        elif occupancy <= 0.90:
            estimated_sl = 0.70
        else:
            estimated_sl = 0.50

        logger.info(f"FINAL EMPIRICAL FALLBACK: {occupancy:.1%} occupancy → {estimated_sl:.1%} service level")
        return max(0.01, min(0.99, estimated_sl))

    def predict_abandon_from_sl(service_level: float) -> float:
        """Predict abandon rate from service level using trained model"""
        # WFM v14.1 style direct lookup - handle both field name variations
        abandon_key = "service_level_to_abandonment_rate"
        abandon_key_alt = "service_level_to_abandonment"
        if abandon_key in relationship_map:
            model_info = relationship_map[abandon_key]
            logger.info(f"Found SL→Abandon relationship: {abandon_key}")
            logger.debug(f"Model info keys: {list(model_info.keys())}")
            logger.debug(f"Model type: {model_info.get('model_type', 'UNKNOWN')}")
            logger.debug(f"Parameters: {model_info.get('parameters', 'MISSING')}")

            try:
                # Apply model - service_level is already 0-1, model expects percentage
                sl_percentage = service_level * 100
                logger.debug(f"Applying model to SL: {sl_percentage:.1f}%")
                result = _apply_curve_model(sl_percentage, model_info)
                result_decimal = result / 100  # Convert from percentage to decimal

                # Check for invalid results (negative, NaN, infinite)
                import math
                if result_decimal < 0:
                    logger.warning(f"Model produced negative abandon rate {result_decimal:.3f} for SL {service_level:.1%}. Using empirical fallback.")
                    raise ValueError(f"Negative abandon rate: {result_decimal}")
                elif not math.isfinite(result_decimal):
                    logger.warning(f"Model produced non-finite abandon rate {result_decimal} for SL {service_level:.1%}. Using empirical fallback.")
                    raise ValueError(f"Non-finite abandon rate: {result_decimal}")
                else:
                    logger.info(f"SUCCESS: SL→Abandon prediction: {service_level:.1%} → {result_decimal:.2%}")
                    return min(0.5, result_decimal)  # Bound to maximum 50% abandon
            except Exception as e:
                # Alert user to calculation problem rather than silently guessing
                logger.error(f"ALERT: SL→Abandon model calculation failed for {service_level:.1%} service level")
                logger.error(f"Model type: {model_info.get('model_type', 'UNKNOWN')}")
                logger.error(f"Error: {e}")

                # Use realistic empirical fallback based on historical data context
                logger.warning(f"Using empirical abandon rate fallback due to model calculation failure")

                # Use historical data from model if available to calibrate fallback
                baseline_abandon = 0.04  # Default 4% abandon rate
                if abandon_key in relationship_map:
                    model_info = relationship_map[abandon_key]
                    if 'data_summary' in model_info and 'y_mean' in model_info['data_summary']:
                        baseline_abandon = model_info['data_summary']['y_mean'] / 100  # Convert from percentage

                # Apply realistic inverse relationship with service level
                # Higher service level = lower abandon rate, with more natural decimal values
                if service_level >= 0.95:
                    return baseline_abandon * 0.52   # ~52% of baseline at 95%+ SL (creates decimals like 2.1%)
                elif service_level >= 0.85:
                    return baseline_abandon * 0.73  # ~73% of baseline at 85-95% SL (creates decimals like 2.9%)
                elif service_level >= 0.75:
                    return baseline_abandon * 1.02   # ~Baseline abandon at 75-85% SL (creates decimals like 4.1%)
                elif service_level >= 0.65:
                    return baseline_abandon * 1.23  # ~123% of baseline at 65-75% SL (creates decimals like 4.9%)
                elif service_level >= 0.50:
                    return baseline_abandon * 1.47   # ~147% of baseline at 50-65% SL (creates decimals like 5.9%)
                else:
                    return baseline_abandon * 1.98   # ~198% of baseline below 50% SL (creates decimals like 7.9%)
        elif abandon_key_alt in relationship_map:
            model_info = relationship_map[abandon_key_alt]
            logger.debug(f"Using SL→Abandon relationship (alternative field name): {model_info}")
            try:
                # Apply model - service_level is already 0-1, model expects percentage
                result = _apply_curve_model(service_level * 100, model_info)
                result_decimal = result / 100  # Convert from percentage to decimal

                # Check for invalid results (negative, NaN, infinite)
                import math
                if result_decimal < 0:
                    logger.warning(f"Alternative model produced negative abandon rate {result_decimal:.3f} for SL {service_level:.1%}. Using empirical fallback.")
                    raise ValueError(f"Negative abandon rate: {result_decimal}")
                elif not math.isfinite(result_decimal):
                    logger.warning(f"Alternative model produced non-finite abandon rate {result_decimal} for SL {service_level:.1%}. Using empirical fallback.")
                    raise ValueError(f"Non-finite abandon rate: {result_decimal}")
                else:
                    logger.debug(f"SL→Abandon prediction (alt): {service_level} → {result_decimal}")
                    return min(0.5, result_decimal)  # Bound to maximum 50% abandon
            except Exception as e:
                # Alert user to calculation problem rather than silently guessing
                logger.error(f"ALERT: SL→Abandon alternative model calculation failed for {service_level:.1%} service level")
                logger.error(f"Error: {e}")

                # Use realistic empirical fallback based on historical data context
                logger.warning(f"Using empirical abandon rate fallback due to alternative model calculation failure")

                # Use historical data from model if available to calibrate fallback
                baseline_abandon = 0.04  # Default 4% abandon rate
                if abandon_key_alt in relationship_map:
                    model_info = relationship_map[abandon_key_alt]
                    if 'data_summary' in model_info and 'y_mean' in model_info['data_summary']:
                        baseline_abandon = model_info['data_summary']['y_mean'] / 100  # Convert from percentage

                # Apply realistic inverse relationship with service level
                # Higher service level = lower abandon rate, with more natural decimal values
                if service_level >= 0.95:
                    return baseline_abandon * 0.52   # ~52% of baseline at 95%+ SL (creates decimals like 2.1%)
                elif service_level >= 0.85:
                    return baseline_abandon * 0.73  # ~73% of baseline at 85-95% SL (creates decimals like 2.9%)
                elif service_level >= 0.75:
                    return baseline_abandon * 1.02   # ~Baseline abandon at 75-85% SL (creates decimals like 4.1%)
                elif service_level >= 0.65:
                    return baseline_abandon * 1.23  # ~123% of baseline at 65-75% SL (creates decimals like 4.9%)
                elif service_level >= 0.50:
                    return baseline_abandon * 1.47   # ~147% of baseline at 50-65% SL (creates decimals like 5.9%)
                else:
                    return baseline_abandon * 1.98   # ~198% of baseline below 50% SL (creates decimals like 7.9%)

        # This should not happen if models are properly trained
        available_relationships = list(relationship_map.keys())
        raise ValueError(f"No Service Level → Abandonment relationship found. Available relationships: {available_relationships}. Need 'service_level_to_abandonment_rate' for convergence analysis.")

    def predict_aht_from_occupancy(occupancy: float) -> float:
        """Predict AHT from occupancy using trained model"""
        # WFM v14.1 style direct lookup - handle both field name variations
        aht_key = "occupancy_to_average_handle_time"
        aht_key_alt = "occupancy_to_aht"
        if aht_key in relationship_map:
            model_info = relationship_map[aht_key]
            logger.debug(f"Using Occupancy→AHT relationship")
            try:
                result = _apply_curve_model(occupancy * 100, model_info)

                # Check for invalid results
                import math
                if result <= 0:
                    logger.warning(f"Occupancy→AHT model produced invalid result {result:.1f}s for occupancy {occupancy:.1%}. Using empirical relationship.")
                    raise ValueError(f"Invalid AHT result: {result}")
                elif not math.isfinite(result):
                    logger.warning(f"Occupancy→AHT model produced non-finite result {result} for occupancy {occupancy:.1%}. Using empirical relationship.")
                    raise ValueError(f"Non-finite AHT result: {result}")
                else:
                    # Log raw result before bounds to understand the issue
                    if result > 1800:
                        logger.warning(f"ALERT: Occupancy→AHT model produced extreme result {result:.0f}s (>30min) for occupancy {occupancy:.1%}. Model may be unrealistic - using empirical relationship.")
                        raise ValueError(f"Extreme AHT result: {result:.0f}s")

                    bounded_result = max(30, min(1800, result))  # Bound to realistic values (30s to 30min)
                    logger.info(f"SUCCESS: Occupancy→AHT prediction: {occupancy:.1%} → {bounded_result:.0f}s (raw: {result:.0f}s)")
                    return bounded_result
            except Exception as e:
                # Alert user to calculation problem
                logger.error(f"ALERT: Occupancy→AHT model calculation failed for {occupancy:.1%} occupancy")
                logger.error(f"Error: {e}")
                # Fall through to empirical relationship
        elif aht_key_alt in relationship_map:
            model_info = relationship_map[aht_key_alt]
            logger.debug(f"Using Occupancy→AHT relationship (alternative field name)")
            try:
                result = _apply_curve_model(occupancy * 100, model_info)

                # Check for invalid results
                import math
                if result <= 0:
                    logger.warning(f"Alternative Occupancy→AHT model produced invalid result {result:.1f}s for occupancy {occupancy:.1%}. Using empirical relationship.")
                    raise ValueError(f"Invalid AHT result: {result}")
                elif not math.isfinite(result):
                    logger.warning(f"Alternative Occupancy→AHT model produced non-finite result {result} for occupancy {occupancy:.1%}. Using empirical relationship.")
                    raise ValueError(f"Non-finite AHT result: {result}")
                else:
                    # Log raw result before bounds to understand the issue
                    if result > 1800:
                        logger.warning(f"ALERT: Alternative Occupancy→AHT model produced extreme result {result:.0f}s (>30min) for occupancy {occupancy:.1%}. Model may be unrealistic - using empirical relationship.")
                        raise ValueError(f"Extreme AHT result: {result:.0f}s")

                    bounded_result = max(30, min(1800, result))  # Bound to realistic values (30s to 30min)
                    logger.info(f"SUCCESS: Alternative Occupancy→AHT prediction: {occupancy:.1%} → {bounded_result:.0f}s (raw: {result:.0f}s)")
                    return bounded_result
            except Exception as e:
                # Alert user to calculation problem
                logger.error(f"ALERT: Alternative Occupancy→AHT model calculation failed for {occupancy:.1%} occupancy")
                logger.error(f"Error: {e}")
                # Fall through to empirical relationship

        # Empirical fallback relationship based on workforce management principles
        logger.warning(f"Using empirical Occupancy→AHT relationship due to model failure or absence")

        # Use actual baseline AHT from the relationship map if available
        baseline_aht = 300  # Default fallback
        if aht_key in relationship_map:
            model_info = relationship_map[aht_key]
            if 'data_summary' in model_info and 'y_mean' in model_info['data_summary']:
                baseline_aht = model_info['data_summary']['y_mean']
        elif aht_key_alt in relationship_map:
            model_info = relationship_map[aht_key_alt]
            if 'data_summary' in model_info and 'y_mean' in model_info['data_summary']:
                baseline_aht = model_info['data_summary']['y_mean']

        # Apply workforce management stress relationship
        # Higher occupancy typically increases AHT due to agent stress/fatigue
        if occupancy <= 0.50:
            stress_factor = 1.0  # No stress at 50% or below
        elif occupancy <= 0.70:
            stress_factor = 1.0 + (occupancy - 0.50) * 0.3  # Linear increase to 6% at 70%
        elif occupancy <= 0.85:
            stress_factor = 1.06 + (occupancy - 0.70) * 0.4  # Linear increase to 12% at 85%
        else:
            stress_factor = 1.12 + (occupancy - 0.85) * 0.6  # Linear increase to 21% at 95%

        result_aht = baseline_aht * stress_factor
        logger.info(f"EMPIRICAL: {occupancy:.1%} occupancy → {stress_factor:.1%} stress factor → {result_aht:.0f}s AHT")
        return result_aht

    return RelationshipPredictor(
        predict_sl_from_occupancy=predict_sl_from_occupancy,
        predict_abandon_from_sl=predict_abandon_from_sl,
        predict_aht_from_occupancy=predict_aht_from_occupancy
    )


def _get_historical_abandon_rate_fallback(model_info: Dict[str, Any]) -> float:
    """Extract a reasonable abandon rate from historical training data"""
    try:
        # Try to get some representative abandon rate from the model or data
        # If we have training data ranges, use a median value
        if 'data_summary' in model_info:
            data_summary = model_info['data_summary']
            if 'y_mean' in data_summary:
                return min(0.5, data_summary['y_mean'] / 100)  # Convert from percentage
            elif 'y_median' in data_summary:
                return min(0.5, data_summary['y_median'] / 100)  # Convert from percentage

        # If no data summary, try to evaluate the model at a typical service level (85%)
        try:
            typical_result = _apply_curve_model(85.0, model_info)  # 85% service level
            return min(0.5, typical_result / 100)  # Convert from percentage
        except:
            pass

        # Last resort: use a conservative 2% abandon rate
        logger.error("FALLBACK: No valid abandon rate data found - using conservative 2% abandon rate")
        logger.error(f"Model info available: {list(model_info.keys()) if model_info else 'None'}")
        return 0.02

    except Exception as e:
        logger.warning(f"Error in historical abandon rate fallback: {e}")
        return 0.02  # Conservative fallback


def _apply_curve_model(x: float, model_info: Dict[str, Any]) -> float:
    """Apply curve fitting model to predict y value from x"""
    import math

    # Robust input conversion - handle arrays, scalars, and any numeric type
    try:
        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            # It's an array-like object
            x_list = list(x)
            if len(x_list) == 1:
                x_val = float(x_list[0])
            else:
                raise ValueError(f"Input x must be scalar, got array with {len(x_list)} elements")
        else:
            # It's a scalar
            x_val = float(x)
    except (TypeError, ValueError) as e:
        logger.error(f"Cannot convert input to scalar: {x}, type: {type(x)}, error: {e}")
        return 0.0

    model_type = model_info.get('model_type', 'linear')

    # Robust parameter extraction
    params = model_info.get('parameters', model_info.get('params', []))

    # Check if parameters exist using explicit None check and length check
    params_exist = params is not None
    if params_exist and hasattr(params, '__len__'):
        params_exist = len(params) > 0

    if not params_exist:
        logger.warning(f"No parameters found in model_info: {model_info}")
        return x_val  # Fallback to identity

    # Convert parameters to list to avoid any array boolean operations
    try:
        params_list = list(params)
    except (TypeError, ValueError):
        logger.error(f"Cannot convert parameters to list: {params}")
        return x_val

    try:
        if model_type == 'linear':
            if len(params_list) < 2:
                a, b = 1.0, 0.0
            else:
                a, b = float(params_list[0]), float(params_list[1])
            result = a * x_val + b

        elif model_type == 'polynomial':
            result = 0.0
            num_params = len(params_list)
            for i in range(num_params):
                coeff = float(params_list[i])
                power = num_params - 1 - i
                result += coeff * (x_val ** power)

        elif model_type == 'exponential':
            if len(params_list) < 3:
                a, b, c = 1.0, 0.0, 0.0
            else:
                a, b, c = float(params_list[0]), float(params_list[1]), float(params_list[2])

            # Clamp input to prevent overflow
            x_clamped = max(-50.0, min(50.0, x_val))
            try:
                result = a * math.exp(b * x_clamped) + c
            except OverflowError:
                result = c  # Fallback to constant

        elif model_type == 'power':
            if len(params_list) < 3:
                a, b, c = 1.0, 1.0, 0.0
            else:
                a, b, c = float(params_list[0]), float(params_list[1]), float(params_list[2])

            # Handle negative or zero input for power functions
            if x_val <= 0:
                result = c  # Return just the constant term
            else:
                # Clamp values to prevent overflow
                b_clamped = max(-50.0, min(50.0, b))
                x_clamped = max(0.001, min(100.0, x_val))

                try:
                    power_result = x_clamped ** b_clamped
                    # Check if result is finite using math.isfinite
                    if math.isfinite(power_result):
                        result = a * power_result + c
                    else:
                        result = c  # Fallback to constant
                except (OverflowError, ValueError, ZeroDivisionError):
                    result = c  # Fallback to constant

        elif model_type == 'logarithmic':
            if len(params_list) < 3:
                a, b, c = 1.0, 0.0, 0.0
            else:
                a, b, c = float(params_list[0]), float(params_list[1]), float(params_list[2])

            try:
                log_input = max(0.001, x_val + b)
                result = a * math.log(log_input) + c
            except (ValueError, OverflowError):
                result = c  # Fallback to constant

        else:
            # Default to linear for unknown types
            if len(params_list) >= 2:
                a, b = float(params_list[0]), float(params_list[1])
            else:
                a, b = 1.0, 0.0
            result = a * x_val + b

        # Final result validation using math.isfinite
        if not math.isfinite(result):
            logger.warning(f"Non-finite result from {model_type} model: {result}")
            return x_val  # Fallback to identity

        return float(result)

    except Exception as e:
        logger.error(f"Error applying {model_type} model: {e}")
        return x_val