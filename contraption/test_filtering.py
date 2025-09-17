#!/usr/bin/env python3
"""
Test the updated filtering logic
"""

import sys
sys.path.insert(0, 'src')

from data.wfm_preprocessor import WFMPreprocessor
import pandas as pd

def test_filtering():
    """Test filtering of totals, zero calls, and NaN calls"""

    print("🔄 Testing Updated Filtering Logic")
    print("=" * 50)

    preprocessor = WFMPreprocessor()

    # Test with Calls file (has call volume data)
    print(f"\n📁 Testing Calls.xlsx with new filtering:")
    print("-" * 40)

    # Load raw first to see original counts
    raw_df = pd.read_excel('../WFM-Data/Calls.xlsx', sheet_name=0)
    print(f"Original rows: {len(raw_df)}")

    # Count different types of data
    total_rows = raw_df[raw_df['Interval'].astype(str).str.contains('Total|total', na=False)]
    print(f"Total/Subtotal rows: {len(total_rows)}")

    zero_offered = len(raw_df[raw_df['Offered'] == 0])
    nan_offered = len(raw_df[raw_df['Offered'].isna()])
    print(f"Zero offered calls: {zero_offered}")
    print(f"NaN offered calls: {nan_offered}")

    # Now process with updated preprocessor
    processed_df = preprocessor.preprocess_file('../WFM-Data/Calls.xlsx')
    print(f"\nAfter new filtering: {len(processed_df)} rows")
    print(f"Total rows removed: {len(raw_df) - len(processed_df)}")

    # Check remaining data
    if 'offered' in processed_df.columns:
        remaining_zero = len(processed_df[processed_df['offered'] == 0])
        remaining_nan = len(processed_df[processed_df['offered'].isna()])
        print(f"Remaining zero calls: {remaining_zero}")
        print(f"Remaining NaN calls: {remaining_nan}")

    # Test with Staff file (no call volume - should only filter totals)
    print(f"\n📁 Testing Staff.xlsx (no call volume):")
    print("-" * 40)

    raw_staff = pd.read_excel('../WFM-Data/Staff.xlsx', sheet_name=0)
    print(f"Original rows: {len(raw_staff)}")

    processed_staff = preprocessor.preprocess_file('../WFM-Data/Staff.xlsx')
    print(f"After filtering: {len(processed_staff)} rows")
    print(f"Should only remove Total rows: {len(raw_staff) - len(processed_staff)}")

    # Test with Occ file (no call volume - should only filter totals)
    print(f"\n📁 Testing Occ.xlsx (no call volume):")
    print("-" * 40)

    raw_occ = pd.read_excel('../WFM-Data/Occ.xlsx', sheet_name=0)
    print(f"Original rows: {len(raw_occ)}")

    processed_occ = preprocessor.preprocess_file('../WFM-Data/Occ.xlsx')
    print(f"After filtering: {len(processed_occ)} rows")
    print(f"Should only remove Total rows: {len(raw_occ) - len(processed_occ)}")

if __name__ == "__main__":
    test_filtering()