"""
Historical metrics relationship analysis for workforce management data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MetricsRelationshipAnalyzer:
    """
    Analyzes relationships between different workforce management metrics
    to identify patterns and correlations for predictive modeling.
    """

    def __init__(self):
        """Initialize the relationship analyzer."""
        self.correlation_results = {}
        self.relationship_strength = {}
        self.scaler = StandardScaler()

    def calculate_correlations(
        self,
        df: pd.DataFrame,
        target_metrics: Optional[List[str]] = None,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between metrics.

        Args:
            df: Input DataFrame with metrics
            target_metrics: Specific metrics to analyze (None for all numeric)
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix DataFrame
        """
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_metrics:
            numeric_cols = [col for col in numeric_cols if col in target_metrics]

        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation analysis")

        # Calculate correlation matrix
        if method == "pearson":
            correlation_matrix = df[numeric_cols].corr(method='pearson')
        elif method == "spearman":
            correlation_matrix = df[numeric_cols].corr(method='spearman')
        elif method == "kendall":
            correlation_matrix = df[numeric_cols].corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Store results
        self.correlation_results[method] = correlation_matrix

        logger.info(f"Calculated {method} correlations for {len(numeric_cols)} metrics")

        return correlation_matrix

    def find_strong_relationships(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.7,
        exclude_self: bool = True
    ) -> List[Dict]:
        """
        Identify strong correlations between metrics.

        Args:
            correlation_matrix: Correlation matrix
            threshold: Minimum correlation strength (absolute value)
            exclude_self: Whether to exclude self-correlations

        Returns:
            List of strong relationship dictionaries
        """
        strong_relationships = []

        for i, metric1 in enumerate(correlation_matrix.columns):
            for j, metric2 in enumerate(correlation_matrix.columns):
                if exclude_self and i == j:
                    continue

                if i <= j:  # Avoid duplicates
                    continue

                correlation = correlation_matrix.loc[metric1, metric2]

                if abs(correlation) >= threshold:
                    relationship = {
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': correlation,
                        'strength': self._categorize_strength(abs(correlation)),
                        'direction': 'positive' if correlation > 0 else 'negative'
                    }
                    strong_relationships.append(relationship)

        # Sort by correlation strength
        strong_relationships.sort(key=lambda x: abs(x['correlation']), reverse=True)

        logger.info(f"Found {len(strong_relationships)} strong relationships above {threshold}")

        return strong_relationships

    def _categorize_strength(self, correlation: float) -> str:
        """Categorize correlation strength."""
        if correlation >= 0.9:
            return "very_strong"
        elif correlation >= 0.7:
            return "strong"
        elif correlation >= 0.5:
            return "moderate"
        elif correlation >= 0.3:
            return "weak"
        else:
            return "very_weak"

    def analyze_lag_correlations(
        self,
        df: pd.DataFrame,
        metric1: str,
        metric2: str,
        max_lag: int = 7
    ) -> Dict:
        """
        Analyze time-lagged correlations between two metrics.

        Args:
            df: Input DataFrame with datetime index
            metric1: First metric name
            metric2: Second metric name
            max_lag: Maximum lag periods to test

        Returns:
            Dictionary with lag correlation results
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for lag analysis")

        if metric1 not in df.columns or metric2 not in df.columns:
            raise ValueError(f"Metrics {metric1} or {metric2} not found in DataFrame")

        lag_results = {'lags': [], 'correlations': []}

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                correlation = df[metric1].corr(df[metric2])
            elif lag > 0:
                # metric1 leads metric2
                shifted_metric2 = df[metric2].shift(lag)
                correlation = df[metric1].corr(shifted_metric2)
            else:
                # metric2 leads metric1
                shifted_metric1 = df[metric1].shift(abs(lag))
                correlation = shifted_metric1.corr(df[metric2])

            lag_results['lags'].append(lag)
            lag_results['correlations'].append(correlation)

        # Find optimal lag
        max_corr_idx = np.argmax(np.abs(lag_results['correlations']))
        optimal_lag = lag_results['lags'][max_corr_idx]
        optimal_correlation = lag_results['correlations'][max_corr_idx]

        lag_results.update({
            'optimal_lag': optimal_lag,
            'optimal_correlation': optimal_correlation,
            'metric1': metric1,
            'metric2': metric2
        })

        logger.info(f"Optimal lag between {metric1} and {metric2}: {optimal_lag} periods "
                   f"(correlation: {optimal_correlation:.3f})")

        return lag_results

    def detect_seasonal_patterns(
        self,
        df: pd.DataFrame,
        metric: str,
        seasonality_periods: Optional[List[int]] = None
    ) -> Dict:
        """
        Detect seasonal patterns in a time series metric.

        Args:
            df: Input DataFrame with datetime index
            metric: Metric to analyze
            seasonality_periods: Periods to test (default: daily, weekly, monthly)

        Returns:
            Dictionary with seasonality analysis results
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for seasonality analysis")

        if metric not in df.columns:
            raise ValueError(f"Metric {metric} not found in DataFrame")

        if seasonality_periods is None:
            seasonality_periods = [7, 30, 365]  # Weekly, monthly, yearly

        seasonality_results = {
            'metric': metric,
            'patterns': {},
            'strongest_pattern': None
        }

        series = df[metric].dropna()

        for period in seasonality_periods:
            if len(series) < period * 2:
                continue

            # Calculate autocorrelation at the seasonal lag
            autocorr = series.autocorr(lag=period)

            # Perform seasonal decomposition if enough data
            if len(series) >= period * 3:
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    decomposition = seasonal_decompose(
                        series.resample('D').mean().fillna(method='ffill'),
                        model='additive',
                        period=period
                    )
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(series)
                except:
                    seasonal_strength = abs(autocorr)
            else:
                seasonal_strength = abs(autocorr)

            seasonality_results['patterns'][period] = {
                'autocorrelation': autocorr,
                'seasonal_strength': seasonal_strength,
                'period_name': self._get_period_name(period)
            }

        # Find strongest seasonal pattern
        if seasonality_results['patterns']:
            strongest = max(
                seasonality_results['patterns'].items(),
                key=lambda x: x[1]['seasonal_strength']
            )
            seasonality_results['strongest_pattern'] = {
                'period': strongest[0],
                'strength': strongest[1]['seasonal_strength'],
                'name': strongest[1]['period_name']
            }

        logger.info(f"Analyzed seasonality for {metric}")

        return seasonality_results

    def _get_period_name(self, period: int) -> str:
        """Get human-readable name for seasonality period."""
        if period == 7:
            return "weekly"
        elif period == 30:
            return "monthly"
        elif period == 365:
            return "yearly"
        else:
            return f"{period}_day"

    def analyze_metric_dependencies(
        self,
        df: pd.DataFrame,
        target_metric: str,
        predictor_metrics: Optional[List[str]] = None,
        method: str = "linear"
    ) -> Dict:
        """
        Analyze dependencies between a target metric and potential predictors.

        Args:
            df: Input DataFrame
            target_metric: Target metric to predict
            predictor_metrics: List of predictor metrics (None for all others)
            method: Analysis method ('linear', 'rank', 'mutual_info')

        Returns:
            Dictionary with dependency analysis results
        """
        if target_metric not in df.columns:
            raise ValueError(f"Target metric {target_metric} not found in DataFrame")

        # Get predictor metrics
        if predictor_metrics is None:
            predictor_metrics = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col != target_metric
            ]

        dependencies = {}

        for predictor in predictor_metrics:
            if predictor not in df.columns:
                continue

            # Remove rows with missing values
            clean_data = df[[target_metric, predictor]].dropna()

            if len(clean_data) < 10:
                continue

            target_values = clean_data[target_metric]
            predictor_values = clean_data[predictor]

            if method == "linear":
                # Linear correlation and R-squared
                correlation, p_value = pearsonr(predictor_values, target_values)
                r_squared = r2_score(target_values, predictor_values)

                dependencies[predictor] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'r_squared': r_squared,
                    'significance': 'significant' if p_value < 0.05 else 'not_significant',
                    'sample_size': len(clean_data)
                }

            elif method == "rank":
                # Spearman rank correlation
                correlation, p_value = spearmanr(predictor_values, target_values)

                dependencies[predictor] = {
                    'rank_correlation': correlation,
                    'p_value': p_value,
                    'significance': 'significant' if p_value < 0.05 else 'not_significant',
                    'sample_size': len(clean_data)
                }

            elif method == "mutual_info":
                # Mutual information
                from sklearn.feature_selection import mutual_info_regression

                mi_score = mutual_info_regression(
                    predictor_values.values.reshape(-1, 1),
                    target_values.values
                )[0]

                dependencies[predictor] = {
                    'mutual_info': mi_score,
                    'sample_size': len(clean_data)
                }

        # Sort by strength of relationship
        if method == "linear":
            sorted_deps = sorted(
                dependencies.items(),
                key=lambda x: abs(x[1]['correlation']),
                reverse=True
            )
        elif method == "rank":
            sorted_deps = sorted(
                dependencies.items(),
                key=lambda x: abs(x[1]['rank_correlation']),
                reverse=True
            )
        else:  # mutual_info
            sorted_deps = sorted(
                dependencies.items(),
                key=lambda x: x[1]['mutual_info'],
                reverse=True
            )

        result = {
            'target_metric': target_metric,
            'method': method,
            'dependencies': dict(sorted_deps),
            'top_predictors': [dep[0] for dep in sorted_deps[:5]]
        }

        logger.info(f"Analyzed dependencies for {target_metric} using {method} method")

        return result

    def generate_relationship_report(
        self,
        df: pd.DataFrame,
        target_metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate comprehensive relationship analysis report.

        Args:
            df: Input DataFrame
            target_metrics: Metrics to focus on (None for all)

        Returns:
            Comprehensive relationship report
        """
        report = {
            'data_summary': {
                'total_rows': len(df),
                'total_metrics': len(df.select_dtypes(include=[np.number]).columns),
                'date_range': {
                    'start': df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
                    'end': df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None,
                }
            },
            'correlations': {},
            'strong_relationships': [],
            'dependencies': {},
            'seasonality': {}
        }

        # Calculate correlations
        try:
            pearson_corr = self.calculate_correlations(df, target_metrics, method="pearson")
            spearman_corr = self.calculate_correlations(df, target_metrics, method="spearman")

            report['correlations'] = {
                'pearson': pearson_corr.to_dict(),
                'spearman': spearman_corr.to_dict()
            }

            # Find strong relationships
            report['strong_relationships'] = self.find_strong_relationships(pearson_corr)

        except Exception as e:
            logger.warning(f"Error calculating correlations: {str(e)}")

        # Analyze dependencies for key metrics
        if target_metrics is None:
            target_metrics = df.select_dtypes(include=[np.number]).columns.tolist()[:5]

        for metric in target_metrics:
            if metric in df.columns:
                try:
                    dependencies = self.analyze_metric_dependencies(df, metric)
                    report['dependencies'][metric] = dependencies
                except Exception as e:
                    logger.warning(f"Error analyzing dependencies for {metric}: {str(e)}")

        # Analyze seasonality for time series data
        if isinstance(df.index, pd.DatetimeIndex):
            for metric in target_metrics:
                if metric in df.columns:
                    try:
                        seasonality = self.detect_seasonal_patterns(df, metric)
                        report['seasonality'][metric] = seasonality
                    except Exception as e:
                        logger.warning(f"Error analyzing seasonality for {metric}: {str(e)}")

        logger.info("Generated comprehensive relationship analysis report")

        return report