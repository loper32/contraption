"""
Data processing utilities for workforce management metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class WFMDataProcessor:
    """
    Processes and transforms workforce management data for analysis.
    Handles common WFM calculations and data cleaning tasks.
    """

    def __init__(self):
        """Initialize the WFM data processor."""
        self.metric_mappings = {
            # Standard WFM metric name mappings
            'fte': ['fte', 'FTE', 'fulltime_equivalent', 'headcount', 'staff'],
            'volume': ['volume', 'Volume', 'calls', 'Calls', 'contacts', 'transactions'],
            'aht': ['aht', 'AHT', 'average_handle_time', 'handle_time', 'talk_time'],
            'service_level': ['service_level', 'ServiceLevel', 'SL', 'sl', 'service_lvl'],
            'occupancy': ['occupancy', 'Occupancy', 'occ', 'utilization'],
            'shrinkage': ['shrinkage', 'Shrinkage', 'shrink', 'attrition'],
            'forecast': ['forecast', 'Forecast', 'predicted', 'expected']
        }

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to common WFM terminology.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized column names
        """
        df_copy = df.copy()

        # Create reverse mapping for lookups
        reverse_mapping = {}
        for standard_name, variations in self.metric_mappings.items():
            for variation in variations:
                reverse_mapping[variation.lower()] = standard_name

        # Rename columns
        renamed_columns = {}
        for col in df_copy.columns:
            standard_name = reverse_mapping.get(col.lower())
            if standard_name and standard_name != col:
                renamed_columns[col] = standard_name

        if renamed_columns:
            df_copy = df_copy.rename(columns=renamed_columns)
            logger.info(f"Standardized column names: {renamed_columns}")

        return df_copy

    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common derived WFM metrics.

        Args:
            df: Input DataFrame with base metrics

        Returns:
            DataFrame with additional calculated metrics
        """
        df_copy = df.copy()

        # Calculate FTE utilization if FTE and volume exist
        if 'fte' in df_copy.columns and 'volume' in df_copy.columns:
            df_copy['fte_per_volume'] = df_copy['fte'] / df_copy['volume'].replace(0, np.nan)

        # Calculate productivity (volume per FTE)
        if 'volume' in df_copy.columns and 'fte' in df_copy.columns:
            df_copy['productivity'] = df_copy['volume'] / df_copy['fte'].replace(0, np.nan)

        # Calculate capacity (theoretical max calls)
        if 'fte' in df_copy.columns and 'aht' in df_copy.columns:
            # Assuming 8-hour workday = 480 minutes
            available_minutes = 480
            if 'occupancy' in df_copy.columns:
                df_copy['theoretical_capacity'] = (
                    df_copy['fte'] * available_minutes * df_copy['occupancy'] / 100
                ) / df_copy['aht'].replace(0, np.nan)
            else:
                df_copy['theoretical_capacity'] = (
                    df_copy['fte'] * available_minutes * 0.85  # Assume 85% occupancy
                ) / df_copy['aht'].replace(0, np.nan)

        # Calculate efficiency (actual vs theoretical)
        if 'volume' in df_copy.columns and 'theoretical_capacity' in df_copy.columns:
            df_copy['efficiency'] = (
                df_copy['volume'] / df_copy['theoretical_capacity'].replace(0, np.nan)
            )

        logger.info(f"Calculated derived metrics: {[col for col in df_copy.columns if col not in df.columns]}")

        return df_copy

    def clean_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove or cap outliers in specified columns.

        Args:
            df: Input DataFrame
            columns: Columns to clean (None for all numeric columns)
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers handled
        """
        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        outliers_removed = 0

        for col in columns:
            if col not in df_copy.columns:
                continue

            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
                outliers_removed += outlier_mask.sum()
                df_copy.loc[outlier_mask, col] = np.nan

            elif method == 'zscore':
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                outlier_mask = z_scores > threshold
                outliers_removed += outlier_mask.sum()
                df_copy.loc[outlier_mask, col] = np.nan

        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outliers using {method} method")

        return df_copy

    def interpolate_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'linear',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Interpolate missing values in time series data.

        Args:
            df: Input DataFrame (should have datetime index)
            method: Interpolation method ('linear', 'time', 'spline')
            columns: Columns to interpolate (None for all numeric)

        Returns:
            DataFrame with interpolated values
        """
        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df_copy.columns:
                missing_before = df_copy[col].isnull().sum()

                if method == 'spline':
                    df_copy[col] = df_copy[col].interpolate(method='spline', order=2)
                else:
                    df_copy[col] = df_copy[col].interpolate(method=method)

                missing_after = df_copy[col].isnull().sum()
                filled_values = missing_before - missing_after

                if filled_values > 0:
                    logger.info(f"Interpolated {filled_values} missing values in {col}")

        return df_copy

    def aggregate_to_period(
        self,
        df: pd.DataFrame,
        period: str = 'D',
        aggregation_methods: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate data to specified time period.

        Args:
            df: Input DataFrame with datetime index
            period: Aggregation period ('H', 'D', 'W', 'M')
            aggregation_methods: Dictionary mapping columns to aggregation methods

        Returns:
            Aggregated DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for aggregation")

        # Default aggregation methods for common WFM metrics
        default_methods = {
            'fte': 'mean',
            'volume': 'sum',
            'aht': 'mean',
            'service_level': 'mean',
            'occupancy': 'mean',
            'productivity': 'mean',
            'efficiency': 'mean'
        }

        if aggregation_methods is None:
            aggregation_methods = {}

        # Apply aggregation
        agg_dict = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in aggregation_methods:
                agg_dict[col] = aggregation_methods[col]
            else:
                # Use default method if available, otherwise mean
                for metric, method in default_methods.items():
                    if metric in col.lower():
                        agg_dict[col] = method
                        break
                else:
                    agg_dict[col] = 'mean'

        aggregated_df = df.resample(period).agg(agg_dict)

        logger.info(f"Aggregated data to {period} periods: {len(df)} → {len(aggregated_df)} rows")

        return aggregated_df

    def calculate_rolling_metrics(
        self,
        df: pd.DataFrame,
        window: int = 7,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling window statistics for trend analysis.

        Args:
            df: Input DataFrame
            window: Rolling window size
            metrics: Columns to calculate rolling metrics for

        Returns:
            DataFrame with additional rolling metric columns
        """
        df_copy = df.copy()

        if metrics is None:
            metrics = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        for metric in metrics:
            if metric in df_copy.columns:
                # Rolling mean
                df_copy[f'{metric}_rolling_mean'] = df_copy[metric].rolling(window=window).mean()

                # Rolling standard deviation
                df_copy[f'{metric}_rolling_std'] = df_copy[metric].rolling(window=window).std()

                # Rolling trend (simple linear regression slope)
                df_copy[f'{metric}_trend'] = (
                    df_copy[metric].rolling(window=window)
                    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan)
                )

        logger.info(f"Calculated rolling metrics with window size {window}")

        return df_copy

    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality assessment.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_quality': {}
        }

        # Per-column quality metrics
        for col in df.columns:
            col_quality = {
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_values': df[col].nunique(),
                'data_type': str(df[col].dtype)
            }

            if df[col].dtype in ['int64', 'float64']:
                col_quality.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'zeros_percentage': ((df[col] == 0).sum() / len(df)) * 100
                })

            quality_report['column_quality'][col] = col_quality

        return quality_report