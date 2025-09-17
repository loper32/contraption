#!/usr/bin/env python3
"""
Check timestamp alignment across merged files
"""

import sys
sys.path.insert(0, 'src')

from data.wfm_preprocessor import WFMPreprocessor
import pandas as pd

def check_timestamp_alignment():
    """Check if timestamps align properly across all merged files"""

    print("🕐 Checking Timestamp Alignment Across Files")
    print("=" * 60)

    preprocessor = WFMPreprocessor()

    # Process each file individually to see their timestamp ranges
    files = {
        'Calls': '../WFM-Data/Calls.xlsx',
        'Staff': '../WFM-Data/Staff.xlsx',
        'Occ': '../WFM-Data/Occ.xlsx'
    }

    processed_files = {}

    for name, filepath in files.items():
        print(f"\n📁 {name}.xlsx individual processing:")
        print("-" * 30)

        df = preprocessor.preprocess_file(filepath)
        processed_files[name] = df

        print(f"Rows after filtering: {len(df)}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Unique timestamps: {df['datetime'].nunique()}")

        # Show first few timestamps
        print(f"First 5 timestamps:")
        print(df['datetime'].head().tolist())

    # Now merge and check alignment
    print(f"\n🔗 Merging and checking alignment:")
    print("=" * 40)

    dataframes = list(processed_files.values())
    merged_df = preprocessor.merge_preprocessed_files(dataframes, merge_strategy="outer")

    print(f"Merged dataset: {merged_df.shape}")
    print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")

    # Check for data availability by source
    print(f"\n📊 Data coverage by source:")
    print("-" * 30)

    # Count non-null values for each source type
    calls_data = merged_df[~merged_df['offered'].isna()] if 'offered' in merged_df.columns else pd.DataFrame()
    staff_data = merged_df[~merged_df['service_level_staff'].isna()] if 'service_level_staff' in merged_df.columns else pd.DataFrame()
    occ_data = merged_df[~merged_df['occupancy'].isna()] if 'occupancy' in merged_df.columns else pd.DataFrame()

    print(f"Intervals with Calls data: {len(calls_data)}")
    print(f"Intervals with Staff data: {len(staff_data)}")
    print(f"Intervals with Occupancy data: {len(occ_data)}")

    # Check for common timestamps
    if len(calls_data) > 0 and len(occ_data) > 0:
        common_timestamps = set(calls_data.index) & set(occ_data.index)
        print(f"Common timestamps (Calls & Occ): {len(common_timestamps)}")

    if len(calls_data) > 0 and len(staff_data) > 0:
        common_timestamps = set(calls_data.index) & set(staff_data.index)
        print(f"Common timestamps (Calls & Staff): {len(common_timestamps)}")

    # Show sample of merged data to verify alignment
    print(f"\n🔍 Sample merged data (first 10 rows):")
    print("-" * 50)

    # Select key columns for alignment check
    key_cols = []
    if 'offered' in merged_df.columns:
        key_cols.append('offered')
    if 'service_level' in merged_df.columns:
        key_cols.append('service_level')
    if 'occupancy' in merged_df.columns:
        key_cols.append('occupancy')
    if 'service_level_staff' in merged_df.columns:
        key_cols.append('service_level_staff')

    if key_cols:
        sample_data = merged_df[key_cols].head(10)
        print(sample_data)

    # Check for potential misalignment issues
    print(f"\n⚠️  Potential alignment issues:")
    print("-" * 30)

    # Check if we have timestamps with only partial data
    partial_data_count = 0
    for idx, row in merged_df.iterrows():
        non_null_sources = 0
        if 'offered' in merged_df.columns and pd.notna(row['offered']):
            non_null_sources += 1
        if 'occupancy' in merged_df.columns and pd.notna(row['occupancy']):
            non_null_sources += 1
        if 'service_level_staff' in merged_df.columns and pd.notna(row['service_level_staff']):
            non_null_sources += 1

        if non_null_sources == 1:  # Only one source has data
            partial_data_count += 1

    print(f"Timestamps with only partial data: {partial_data_count}")
    print(f"Complete data coverage: {len(merged_df) - partial_data_count} timestamps")

if __name__ == "__main__":
    check_timestamp_alignment()