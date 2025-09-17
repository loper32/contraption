#!/usr/bin/env python3
"""
Check what data is being filtered out during processing
"""

import sys
import pandas as pd
sys.path.insert(0, 'src')

from data.wfm_preprocessor import WFMPreprocessor

def analyze_filtering():
    """Analyze what data gets filtered during preprocessing"""

    print("🔍 Analyzing Data Filtering")
    print("=" * 50)

    # Load raw files first to see original data
    files = {
        'Calls': '../WFM-Data/Calls.xlsx',
        'Staff': '../WFM-Data/Staff.xlsx',
        'Occ': '../WFM-Data/Occ.xlsx'
    }

    for name, file_path in files.items():
        print(f"\n📁 {name}.xlsx Analysis:")
        print("-" * 30)

        # Load raw data
        raw_df = pd.read_excel(file_path, sheet_name=0)
        print(f"Original rows: {len(raw_df)}")

        # Check for Total rows
        if 'Interval' in raw_df.columns:
            total_rows = raw_df[raw_df['Interval'].astype(str) == 'Total']
            interval_rows = raw_df[raw_df['Interval'].astype(str) != 'Total']
            print(f"'Total' rows: {len(total_rows)}")
            print(f"Interval rows: {len(interval_rows)}")

            # Show sample Total row data
            if len(total_rows) > 0:
                print(f"Sample Total row data:")
                print(total_rows.iloc[0])

        # Check for zero call volume (if applicable)
        if name == 'Calls':
            print(f"\n📊 Call Volume Analysis:")
            if 'Offered' in raw_df.columns:
                zero_offered = raw_df[raw_df['Offered'] == 0]
                nan_offered = raw_df[raw_df['Offered'].isna()]
                print(f"Rows with 0 offered calls: {len(zero_offered)}")
                print(f"Rows with NaN offered calls: {len(nan_offered)}")

            if 'Answered' in raw_df.columns:
                zero_answered = raw_df[raw_df['Answered'] == 0]
                nan_answered = raw_df[raw_df['Answered'].isna()]
                print(f"Rows with 0 answered calls: {len(zero_answered)}")
                print(f"Rows with NaN answered calls: {len(nan_answered)}")

        # Now process with preprocessor
        print(f"\n🔄 After Preprocessing:")
        preprocessor = WFMPreprocessor()
        processed_df = preprocessor.preprocess_file(file_path)
        print(f"Processed rows: {len(processed_df)}")
        print(f"Rows removed: {len(raw_df) - len(processed_df)}")

        # Check what happened to call volume data
        if name == 'Calls' and 'offered' in processed_df.columns:
            zero_processed = processed_df[processed_df['offered'] == 0]
            nan_processed = processed_df[processed_df['offered'].isna()]
            print(f"Zero offered calls after processing: {len(zero_processed)}")
            print(f"NaN offered calls after processing: {len(nan_processed)}")

if __name__ == "__main__":
    analyze_filtering()