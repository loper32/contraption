#!/usr/bin/env python3
"""
Examine real WFM data files to understand formatting differences
"""

import pandas as pd
import numpy as np
import os

# Real WFM data files from WFM-Data directory
wfm_files = [
    '../WFM-Data/Calls.xlsx',
    '../WFM-Data/Occ.xlsx',
    '../WFM-Data/Staff.xlsx'
]

def analyze_excel_file(filepath):
    """Analyze an Excel file's structure and content"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    try:
        # Check all sheets
        excel_file = pd.ExcelFile(filepath)
        print(f"📊 Sheets available: {excel_file.sheet_names}")

        for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
            print(f"\n📋 SHEET: {sheet_name}")
            print("-" * 40)

            # Read sheet
            df = pd.read_excel(filepath, sheet_name=sheet_name)

            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Look for date/time columns
            date_cols = []
            for col in df.columns:
                col_str = str(col).lower()
                if any(term in col_str for term in ['date', 'time', 'interval', 'period', 'timestamp', 'hour', 'minute']):
                    date_cols.append(col)

            print(f"Potential date/time columns: {date_cols}")

            # Sample first few rows
            print(f"\nFirst 3 rows:")
            print(df.head(3))

            # Analyze date columns in detail
            for col in date_cols[:2]:  # Limit to avoid spam
                print(f"\nDetailed analysis of '{col}':")
                print(f"  Data type: {df[col].dtype}")
                print(f"  Sample values: {df[col].head(5).tolist()}")
                print(f"  Unique values count: {df[col].nunique()}")

                # Try to detect interval pattern
                if df[col].dtype == 'object':
                    print(f"  String patterns in first 10 values:")
                    for i, val in enumerate(df[col].head(10)):
                        print(f"    {i+1}: '{val}'")

            # Look for key WFM metrics
            wfm_metrics = []
            for col in df.columns:
                col_str = str(col).lower()
                if any(term in col_str for term in [
                    'volume', 'calls', 'aht', 'handle', 'time',
                    'occupancy', 'occ', 'service', 'level', 'sl',
                    'abandon', 'aban', 'fte', 'staff', 'agent',
                    'shrinkage', 'utilization', 'adherence'
                ]):
                    wfm_metrics.append(col)

            print(f"\nWFM metrics found: {wfm_metrics}")

            # If we have more than 3 rows, check for patterns
            if len(df) > 3:
                print(f"\nData pattern analysis:")
                print(f"  Total rows: {len(df)}")

                # Check if data looks like intervals within a day
                if date_cols:
                    first_date_col = date_cols[0]
                    try:
                        # Try to parse as datetime
                        if df[first_date_col].dtype == 'object':
                            parsed_dates = pd.to_datetime(df[first_date_col], errors='coerce')
                            if not parsed_dates.isna().all():
                                df_copy = df.copy()
                                df_copy[first_date_col] = parsed_dates
                                time_diffs = df_copy[first_date_col].diff().dropna()
                                if len(time_diffs) > 0:
                                    most_common_diff = time_diffs.mode()
                                    if len(most_common_diff) > 0:
                                        print(f"  Most common time difference: {most_common_diff.iloc[0]}")
                                    print(f"  Sample time differences: {time_diffs.head(5).tolist()}")
                    except Exception as e:
                        print(f"  Could not analyze time patterns: {e}")

            print("\n" + "="*40)

    except Exception as e:
        print(f"❌ Error analyzing {filepath}: {e}")
        import traceback
        traceback.print_exc()

# Analyze all WFM files
for filepath in wfm_files:
    analyze_excel_file(filepath)