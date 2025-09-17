#!/usr/bin/env python3
"""
Examine Excel files to understand data formats and column structures
"""

import pandas as pd
import numpy as np

# Load and examine each Excel file
files_to_examine = ['Calls.xlsx', 'Staff.xlsx', 'Occ.xlsx']

for file in files_to_examine:
    print(f"\n{'='*50}")
    print(f"EXAMINING: {file}")
    print(f"{'='*50}")

    try:
        # Read the Excel file to see all sheets
        excel_file = pd.ExcelFile(file)
        print(f"Sheets available: {excel_file.sheet_names}")

        # Read the first sheet
        df = pd.read_excel(file, sheet_name=0)

        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Column dtypes:\n{df.dtypes}")

        print(f"\nFirst 5 rows:")
        print(df.head())

        print(f"\nLast 5 rows:")
        print(df.tail())

        # Check for timestamp/date columns
        date_like_cols = []
        for col in df.columns:
            if any(term in str(col).lower() for term in ['date', 'time', 'interval', 'period', 'timestamp']):
                date_like_cols.append(col)

        print(f"\nPotential date/time columns: {date_like_cols}")

        # Sample unique values from potential date columns
        for col in date_like_cols[:2]:  # Limit to first 2 to avoid spam
            print(f"\nSample values from '{col}':")
            print(df[col].head(10).tolist())

        # Check if data looks like 30-minute intervals
        if len(df) > 1:
            print(f"\nData frequency analysis:")
            if date_like_cols:
                first_date_col = date_like_cols[0]
                try:
                    df[first_date_col] = pd.to_datetime(df[first_date_col])
                    time_diff = df[first_date_col].diff().dropna()
                    print(f"Time differences (first 5): {time_diff.head().tolist()}")
                    print(f"Most common time difference: {time_diff.mode().iloc[0] if len(time_diff.mode()) > 0 else 'N/A'}")
                except:
                    print("Could not analyze time differences")

    except Exception as e:
        print(f"Error reading {file}: {e}")
        import traceback
        traceback.print_exc()