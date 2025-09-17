"""
Excel data loader and timestamp merger for workforce management data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelDataLoader:
    """
    Handles loading and merging Excel files based on timestamp information.
    Designed for workforce management metrics with time-based data.
    """

    def __init__(self, timestamp_column: str = "timestamp"):
        """
        Initialize the Excel data loader.

        Args:
            timestamp_column: Name of the column containing timestamp data
        """
        self.timestamp_column = timestamp_column
        self.loaded_files: Dict[str, pd.DataFrame] = {}

    def load_excel_file(
        self,
        file_path: Union[str, Path],
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a single Excel file and standardize timestamp column.

        Args:
            file_path: Path to the Excel file or file-like object
            sheet_name: Specific sheet to load (None for first sheet)
            **kwargs: Additional arguments passed to pd.read_excel()

        Returns:
            DataFrame with standardized timestamp column

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If timestamp column is missing or invalid
        """
        # Handle file-like objects (e.g., from Streamlit upload)
        if hasattr(file_path, 'read'):
            filename = getattr(file_path, 'name', 'uploaded_file')
            try:
                # Ensure we're reading a single sheet as DataFrame, not dict
                if sheet_name is None:
                    sheet_name = 0  # Read first sheet by default

                df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

                # Check if result is a dictionary (multiple sheets)
                if isinstance(df, dict):
                    # Take the first sheet
                    df = list(df.values())[0]
                    logger.info(f"File {filename} contained multiple sheets, using first sheet")

                # Ensure we have a DataFrame
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(f"Could not load {filename} as DataFrame, got {type(df)}")

                # If the DataFrame only has one row, it might be using the first row as headers incorrectly
                if len(df) == 1 and len(df.columns) > 10:
                    logger.warning(f"Only 1 row detected in {filename}, trying without header")
                    file_path.seek(0)  # Reset file pointer
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, **kwargs)
                    if isinstance(df, dict):
                        df = list(df.values())[0]
                    # Set first row as header if it looks like headers
                    if df.iloc[0].dtype == object:
                        df.columns = df.iloc[0]
                        df = df.drop(0).reset_index(drop=True)

                logger.info(f"Loaded Excel file: {filename} ({len(df)} rows, {len(df.columns)} columns)")

            except Exception as e:
                logger.error(f"Error reading Excel file {filename}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        else:
            # Handle file paths
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Excel file not found: {file_path}")

            filename = file_path.name

            try:
                # Load Excel file
                df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
                logger.info(f"Loaded Excel file: {file_path} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Error loading Excel file {file_path}: {str(e)}")
                raise

        # Ensure we have a DataFrame before timestamp processing
        if isinstance(df, dict):
            # Take the first sheet
            df = list(df.values())[0]
            logger.info(f"File {filename} was a dict, extracted first sheet")

        # Standardize timestamp column
        df = self._standardize_timestamps(df, filename)

        # Store loaded file
        self.loaded_files[filename] = df

        return df

    def _standardize_timestamps(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Standardize timestamp column format and handle common variations.

        Args:
            df: Input DataFrame
            filename: Name of source file (for error reporting)

        Returns:
            DataFrame with standardized timestamp column
        """
        # First check if df is actually a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame but got {type(df)} for {filename}")

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError(f"DataFrame is empty for {filename}")

        # Look for timestamp column variations
        timestamp_candidates = [
            self.timestamp_column,
            'date', 'Date', 'DATE',
            'time', 'Time', 'TIME',
            'datetime', 'DateTime', 'DATETIME',
            'period', 'Period', 'PERIOD',
            'week', 'Week', 'WEEK',
            'month', 'Month', 'MONTH',
            'day', 'Day', 'DAY',
            'interval', 'Interval', 'INTERVAL'
        ]

        # Also check if first column might be dates
        if df.index.name and any(term in str(df.index.name).lower() for term in ['date', 'time', 'period', 'week']):
            df = df.reset_index()
            timestamp_col = df.columns[0]
        else:
            timestamp_col = None
            for col in timestamp_candidates:
                if col in df.columns:
                    timestamp_col = col
                    break

            # If still not found, check if first column contains date-like values
            if timestamp_col is None and len(df.columns) > 0:
                first_col = df.columns[0]
                try:
                    # Try to parse first column as dates
                    pd.to_datetime(df[first_col].iloc[:5])  # Test first 5 values
                    timestamp_col = first_col
                    logger.info(f"Using first column '{first_col}' as timestamp column")
                except:
                    pass

        if timestamp_col is None:
            available_cols = list(df.columns)
            # If no timestamp column found, create one with sequential dates
            logger.warning(f"No timestamp column found in {filename}. Creating sequential dates.")
            df[self.timestamp_column] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            return df

        # Filter out "Total" rows before timestamp processing
        if 'Total' in df[timestamp_col].astype(str).values:
            logger.info(f"Filtering out 'Total' rows from {filename}")
            total_mask = df[timestamp_col].astype(str) != 'Total'
            df = df[total_mask].copy()
            logger.info(f"After filtering: {len(df)} rows remaining")

        # Convert to datetime
        try:
            df[self.timestamp_column] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            logger.warning(f"Error converting timestamps in {filename}: {str(e)}")
            # Try common date formats
            for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    df[self.timestamp_column] = pd.to_datetime(df[timestamp_col], format=date_format)
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not parse timestamps in {filename}")

        # Rename original column if different
        if timestamp_col != self.timestamp_column and timestamp_col in df.columns:
            df = df.drop(columns=[timestamp_col])

        # Sort by timestamp
        df = df.sort_values(self.timestamp_column).reset_index(drop=True)

        logger.info(f"Standardized timestamps for {filename} "
                   f"({df[self.timestamp_column].min()} to {df[self.timestamp_column].max()})")

        return df

    def merge_files_by_timestamp(
        self,
        dataframes: List[pd.DataFrame],
        merge_strategy: str = "outer",
        tolerance: Optional[pd.Timedelta] = None
    ) -> pd.DataFrame:
        """
        Merge multiple DataFrames based on timestamp column.

        Args:
            dataframes: List of DataFrames to merge
            merge_strategy: How to merge ('outer', 'inner', 'left', 'right')
            tolerance: Time tolerance for approximate matching

        Returns:
            Merged DataFrame with timestamp as index
        """
        if not dataframes:
            raise ValueError("No DataFrames provided for merging")

        if len(dataframes) == 1:
            return dataframes[0].set_index(self.timestamp_column)

        # Start with first DataFrame
        merged_df = dataframes[0].set_index(self.timestamp_column)

        # Merge additional DataFrames
        for i, df in enumerate(dataframes[1:], 1):
            df_indexed = df.set_index(self.timestamp_column)

            if tolerance is not None:
                # Use merge_asof for approximate timestamp matching
                merged_df = pd.merge_asof(
                    merged_df.reset_index().sort_values(self.timestamp_column),
                    df_indexed.reset_index().sort_values(self.timestamp_column),
                    on=self.timestamp_column,
                    tolerance=tolerance,
                    direction='nearest'
                ).set_index(self.timestamp_column)
            else:
                # Exact timestamp matching
                merged_df = merged_df.merge(
                    df_indexed,
                    left_index=True,
                    right_index=True,
                    how=merge_strategy,
                    suffixes=('', f'_file_{i}')
                )

        logger.info(f"Merged {len(dataframes)} files into DataFrame with {len(merged_df)} rows")

        return merged_df

    def load_and_merge_files(
        self,
        file_paths: List[Union[str, Path]],
        merge_strategy: str = "outer",
        tolerance: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load multiple Excel files and merge them by timestamp.

        Args:
            file_paths: List of Excel file paths
            merge_strategy: How to merge files
            tolerance: Time tolerance string (e.g., '1H', '30min')

        Returns:
            Merged DataFrame
        """
        # Load all files
        dataframes = []
        for file_path in file_paths:
            df = self.load_excel_file(file_path)
            dataframes.append(df)

        # Convert tolerance to Timedelta
        time_tolerance = pd.Timedelta(tolerance) if tolerance else None

        # Merge files
        merged_df = self.merge_files_by_timestamp(
            dataframes,
            merge_strategy=merge_strategy,
            tolerance=time_tolerance
        )

        return merged_df

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the loaded data.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min() if self.timestamp_column in df.index.names else None,
                'end': df.index.max() if self.timestamp_column in df.index.names else None,
            },
            'columns': list(df.columns),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'missing_data': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }

        return summary

    def validate_wfm_data(self, df: pd.DataFrame) -> List[str]:
        """
        Validate workforce management data for common issues.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        # Check for common WFM columns
        expected_columns = [
            'fte', 'FTE', 'headcount', 'Headcount',
            'volume', 'Volume', 'calls', 'Calls',
            'aht', 'AHT', 'service_level', 'ServiceLevel',
            'occupancy', 'Occupancy'
        ]

        found_columns = [col for col in expected_columns if col in df.columns]
        if not found_columns:
            warnings.append("No common WFM columns detected (FTE, Volume, AHT, etc.)")

        # Check for negative values in key metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                warnings.append(f"Negative values found in {col}")

        # Check for missing timestamps
        if df.index.isnull().any():
            warnings.append("Missing timestamp values detected")

        # Check for duplicate timestamps
        if df.index.duplicated().any():
            warnings.append("Duplicate timestamp values detected")

        return warnings