"""
WFM Data Preprocessor - Handles formatting differences between WFM Excel files
Standardizes column names, datetime formats, and prepares data for merging
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class WFMPreprocessor:
    """
    Preprocesses WFM Excel files to handle formatting differences and standardize data.
    Uses column header names for robust data extraction and transformation.
    """

    def __init__(self):
        """Initialize the preprocessor with column mapping configurations."""

        # Column mappings for different file types
        self.column_mappings = {
            'calls': {
                'date_col': ['Date', 'date', 'DATE'],
                'interval_col': ['Interval', 'interval', 'INTERVAL', 'Time', 'time'],
                'metrics': {
                    'service_level': ['%SL', '%sl', 'Service Level', 'SL%', 'sl%'],
                    'abandonment_rate': ['%Aban', '%aban', 'Abandon%', 'abandon%', 'Abandonment Rate'],
                    'average_speed_answer': ['ASA', 'asa', 'Average Speed Answer', 'Avg Speed Answer'],
                    'average_handle_time': ['AHT', 'aht', 'Average Handle Time', 'Avg Handle Time'],
                    'offered': ['Offered', 'offered', 'OFFERED', 'Calls Offered'],
                    'answered': ['Answered', 'answered', 'ANSWERED', 'Calls Answered'],
                    'abandoned': ['Abandoned', 'abandoned', 'ABANDONED', 'Calls Abandoned']
                }
            },
            'staff': {
                'date_col': ['Daily', 'daily', 'Date', 'date', 'DATE'],
                'interval_col': ['Interval', 'interval', 'INTERVAL', 'Time', 'time'],
                'metrics': {
                    'service_level': ['Service Level', 'service level', 'SL', 'sl'],
                    'adj_aps': ['Adj. APS', 'adj aps', 'Adjusted APS', 'APS Adjusted'],
                    'adj_lock': ['Adj. Lock', 'adj lock', 'Adjusted Lock', 'Lock Adjusted'],
                    'line_adherence': ['Line Adherence', 'line adherence', 'Adherence', 'adherence'],
                    'billable_hours': ['Billable Hrs', 'billable hrs', 'Billable Hours', 'Hours Billable'],
                    'aps_lock': ['APS/Lock', 'aps/lock', 'APS Lock', 'aps lock'],
                    'aps': ['APS', 'aps']
                }
            },
            'occupancy': {
                'date_col': ['Daily', 'daily', 'Date', 'date', 'DATE'],
                'interval_col': ['Interval', 'interval', 'INTERVAL', 'Time', 'time'],
                'metrics': {
                    'occupancy': ['Occupancy', 'occupancy', 'OCCUPANCY', 'Occ', 'occ']
                }
            }
        }

    def find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """
        Find a column in the DataFrame using a list of possible names.

        Args:
            df: DataFrame to search
            possible_names: List of possible column names to look for

        Returns:
            Actual column name if found, None otherwise
        """
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    def identify_file_type(self, df: pd.DataFrame) -> str:
        """
        Identify the type of WFM file based on its columns.

        Args:
            df: DataFrame to analyze

        Returns:
            File type: 'calls', 'staff', or 'occupancy'
        """
        cols_lower = [col.lower() for col in df.columns]

        # Check for occupancy file indicators first (most specific)
        if any(indicator in cols_lower for indicator in ['occupancy', 'occ']):
            return 'occupancy'

        # Check for staff file indicators (more specific than calls)
        elif any(indicator in cols_lower for indicator in ['adj. aps', 'billable hrs', 'line adherence', 'aps/lock']):
            return 'staff'

        # Check for calls file indicators (least specific, as service level can appear in staff files too)
        elif any(indicator in cols_lower for indicator in ['%sl', '%aban', 'abandon', 'asa', 'aht', 'offered', 'answered']):
            return 'calls'

        else:
            logger.warning("Could not identify file type, defaulting to 'calls'")
            return 'calls'

    def create_datetime_column(self, df: pd.DataFrame, date_col: str, interval_col: str) -> pd.DataFrame:
        """
        Create a unified datetime column from separate date and interval columns.

        Args:
            df: DataFrame with separate date and interval columns
            date_col: Name of the date column
            interval_col: Name of the interval column

        Returns:
            DataFrame with new 'datetime' column
        """
        df = df.copy()

        # Filter out 'Total' and 'Subtotal' rows BEFORE datetime conversion
        # Check both date and interval columns for summary rows
        date_mask = ~df[date_col].astype(str).str.contains('Total|Subtotal|total|subtotal|Applied filters', case=False, na=False)
        interval_mask = ~df[interval_col].astype(str).str.contains('Total|Subtotal|total|subtotal', case=False, na=False)
        # Combine both masks - filter out rows where EITHER column contains summary text
        clean_mask = date_mask & interval_mask
        df_intervals = df[clean_mask].copy()

        logger.info(f"Filtered out {len(df) - len(df_intervals)} summary/total rows")

        # Convert date column to datetime AFTER filtering
        if df_intervals[date_col].dtype == 'object':
            df_intervals[date_col] = pd.to_datetime(df_intervals[date_col], errors='coerce')

        if len(df_intervals) == 0:
            logger.warning("No interval data found (only Total rows)")
            return df

        # Create datetime column for interval rows
        datetime_values = []

        for idx, row in df_intervals.iterrows():
            try:
                base_date = row[date_col]
                interval_val = row[interval_col]

                if pd.isna(base_date) or pd.isna(interval_val):
                    datetime_values.append(pd.NaT)
                    continue

                # Handle different interval formats
                if isinstance(interval_val, time):
                    # interval_val is already a time object
                    hour = interval_val.hour
                    minute = interval_val.minute
                elif isinstance(interval_val, str):
                    # Parse time string like "09:30:00" or "09:30"
                    try:
                        time_parts = interval_val.split(':')
                        hour = int(time_parts[0])
                        minute = int(time_parts[1])
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse interval: {interval_val}")
                        datetime_values.append(pd.NaT)
                        continue
                else:
                    logger.warning(f"Unexpected interval format: {type(interval_val)} - {interval_val}")
                    datetime_values.append(pd.NaT)
                    continue

                # Combine date and time
                combined_datetime = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                datetime_values.append(combined_datetime)

            except Exception as e:
                logger.warning(f"Error processing datetime for row {idx}: {e}")
                datetime_values.append(pd.NaT)

        # Update the DataFrame with datetime values
        df_intervals['datetime'] = datetime_values

        # Filter out any remaining NaT values to prevent index issues
        valid_datetime_mask = pd.notna(df_intervals['datetime'])
        nat_count = (~valid_datetime_mask).sum()
        if nat_count > 0:
            logger.warning(f"Removing {nat_count} rows with invalid datetime values")
            df_intervals = df_intervals[valid_datetime_mask].copy()

        # Check for duplicate timestamps and warn if found
        if len(df_intervals) > 0:
            unique_timestamps = df_intervals['datetime'].nunique()
            total_rows = len(df_intervals)
            if unique_timestamps < total_rows:
                duplicate_count = total_rows - unique_timestamps
                logger.warning(f"Found {duplicate_count} duplicate timestamps. Keeping last occurrence of each.")
                # Keep the last occurrence of each timestamp (most recent data)
                df_intervals = df_intervals.drop_duplicates(subset=['datetime'], keep='last')

        # Return only the interval rows with valid, unique datetime
        return df_intervals

    def filter_call_volume_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows with zero or NaN call volumes.

        Args:
            df: DataFrame to filter

        Returns:
            DataFrame with call volume filtering applied
        """
        df = df.copy()
        original_rows = len(df)

        # Look for call volume columns (both original and standardized names)
        call_volume_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['offered', 'calls', 'volume', 'answered']):
                call_volume_cols.append(col)

        if not call_volume_cols:
            logger.info("No call volume columns found, skipping call volume filtering")
            return df

        # Apply filtering based on primary call volume indicator (usually 'offered' or first found)
        primary_vol_col = call_volume_cols[0]
        logger.info(f"Filtering based on '{primary_vol_col}' column")

        # Filter out zero and NaN values
        before_filter = len(df)

        # Remove rows where call volume is 0 or NaN
        df = df[(df[primary_vol_col] > 0) & (df[primary_vol_col].notna())]

        after_filter = len(df)
        removed_count = before_filter - after_filter

        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with zero/NaN call volume ({before_filter} → {after_filter})")
        else:
            logger.info("No rows removed by call volume filtering")

        return df

    def standardize_column_names(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """
        Standardize column names based on file type mapping.

        Args:
            df: DataFrame to standardize
            file_type: Type of file ('calls', 'staff', 'occupancy')

        Returns:
            DataFrame with standardized column names
        """
        df = df.copy()

        if file_type not in self.column_mappings:
            logger.warning(f"Unknown file type: {file_type}")
            return df

        # Create rename mapping
        rename_mapping = {}

        # Map metrics columns
        for standard_name, possible_names in self.column_mappings[file_type]['metrics'].items():
            found_col = self.find_column(df, possible_names)
            if found_col:
                rename_mapping[found_col] = standard_name

        # Apply renaming
        if rename_mapping:
            df = df.rename(columns=rename_mapping)
            logger.info(f"Renamed columns: {rename_mapping}")

        return df

    def preprocess_file(self, file_path: str, file_type: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess a single WFM Excel file.

        Args:
            file_path: Path to the Excel file
            file_type: Optional file type override ('calls', 'staff', 'occupancy')

        Returns:
            Preprocessed DataFrame with standardized format
        """
        try:
            # Load the Excel file
            df = pd.read_excel(file_path, sheet_name=0)  # Read first sheet
            logger.info(f"Loaded {file_path}: {df.shape}")

            # Auto-detect file type if not provided
            if file_type is None:
                file_type = self.identify_file_type(df)
                logger.info(f"Identified file type: {file_type}")

            # Find date and interval columns
            mappings = self.column_mappings[file_type]
            date_col = self.find_column(df, mappings['date_col'])
            interval_col = self.find_column(df, mappings['interval_col'])

            if not date_col:
                raise ValueError(f"Could not find date column in {file_path}")
            if not interval_col:
                raise ValueError(f"Could not find interval column in {file_path}")

            logger.info(f"Found columns - Date: '{date_col}', Interval: '{interval_col}'")

            # Create datetime column and filter to interval data only
            df = self.create_datetime_column(df, date_col, interval_col)

            # Standardize column names
            df = self.standardize_column_names(df, file_type)

            # Filter out zero/NaN call volume data (applies to all file types)
            df = self.filter_call_volume_data(df)

            # Add file type metadata
            df['source_file_type'] = file_type

            # Remove original date/interval columns if datetime was created successfully
            if 'datetime' in df.columns:
                cols_to_drop = [date_col, interval_col]
                df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

            logger.info(f"Preprocessed {file_path}: {df.shape} final shape")
            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

            return df

        except Exception as e:
            logger.error(f"Error preprocessing {file_path}: {e}")
            raise

    def merge_preprocessed_files(self, dataframes: List[pd.DataFrame],
                                merge_strategy: str = "outer") -> pd.DataFrame:
        """
        Merge multiple preprocessed DataFrames on datetime.

        Args:
            dataframes: List of preprocessed DataFrames
            merge_strategy: How to merge ('outer', 'inner', 'left', 'right')

        Returns:
            Merged DataFrame
        """
        if not dataframes:
            raise ValueError("No DataFrames provided for merging")

        if len(dataframes) == 1:
            return dataframes[0].set_index('datetime')

        # Start with first DataFrame
        merged_df = dataframes[0].set_index('datetime')

        # Merge additional DataFrames
        for i, df in enumerate(dataframes[1:], 1):
            df_indexed = df.set_index('datetime')

            # Add suffix to avoid column conflicts
            file_type = df['source_file_type'].iloc[0] if 'source_file_type' in df.columns else f'file_{i}'

            merged_df = merged_df.merge(
                df_indexed,
                left_index=True,
                right_index=True,
                how=merge_strategy,
                suffixes=('', f'_{file_type}')
            )

        logger.info(f"Merged {len(dataframes)} files: {merged_df.shape}")
        return merged_df

    def get_available_metrics(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get available metrics by category from the processed DataFrame.

        Args:
            df: Processed DataFrame

        Returns:
            Dictionary of metric categories and their available columns
        """
        all_metrics = {
            'service_level': ['service_level'],
            'occupancy': ['occupancy'],
            'aht': ['average_handle_time'],
            'abandonment': ['abandonment_rate', 'abandoned'],
            'volume': ['offered', 'answered'],
            'staffing': ['adj_aps', 'aps', 'billable_hours', 'line_adherence']
        }

        available = {}
        for category, metric_names in all_metrics.items():
            found_metrics = [col for col in metric_names if col in df.columns]
            if found_metrics:
                available[category] = found_metrics

        return available