#!/usr/bin/env python3
"""
Test the WFM Preprocessor with real WFM data files
"""

import sys
import os
sys.path.insert(0, 'src')

from data.wfm_preprocessor import WFMPreprocessor
import pandas as pd

def test_preprocessor():
    """Test the WFM preprocessor with real files"""

    # Initialize preprocessor
    preprocessor = WFMPreprocessor()

    # Real WFM files
    files = {
        'calls': '../WFM-Data/Calls.xlsx',
        'staff': '../WFM-Data/Staff.xlsx',
        'occupancy': '../WFM-Data/Occ.xlsx'
    }

    print("🔄 Testing WFM Preprocessor with Real Data")
    print("=" * 60)

    processed_dfs = []

    for file_type, file_path in files.items():
        print(f"\n📁 Processing {file_type.upper()} file: {file_path}")
        print("-" * 40)

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        try:
            # Preprocess the file
            df = preprocessor.preprocess_file(file_path, file_type=file_type)

            print(f"✅ Successfully processed: {df.shape}")
            print(f"📊 Columns: {list(df.columns)}")
            print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")

            # Show available metrics
            available_metrics = preprocessor.get_available_metrics(df)
            print(f"📈 Available metrics: {available_metrics}")

            # Sample data
            print(f"\n🔍 Sample data (first 3 rows):")
            print(df.head(3))

            processed_dfs.append(df)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🔗 Testing File Merging")
    print("=" * 40)

    if len(processed_dfs) > 1:
        try:
            merged_df = preprocessor.merge_preprocessed_files(processed_dfs)
            print(f"✅ Successfully merged files: {merged_df.shape}")
            print(f"📊 Merged columns: {list(merged_df.columns)}")
            print(f"📅 Date range: {merged_df.index.min()} to {merged_df.index.max()}")

            # Show sample of merged data
            print(f"\n🔍 Merged data sample:")
            print(merged_df.head(3))

            # Check for your target analyses
            print(f"\n🎯 Target Analysis Availability:")

            # Service Level + Occupancy
            sl_cols = [col for col in merged_df.columns if 'service_level' in col.lower()]
            occ_cols = [col for col in merged_df.columns if 'occupancy' in col.lower()]
            print(f"  • Service Level + Occupancy: SL={sl_cols}, Occ={occ_cols}")

            # Occupancy + AHT
            aht_cols = [col for col in merged_df.columns if 'handle_time' in col.lower() or 'aht' in col.lower()]
            print(f"  • Occupancy + AHT: Occ={occ_cols}, AHT={aht_cols}")

            # Service Level + Abandonment
            aban_cols = [col for col in merged_df.columns if 'abandon' in col.lower()]
            print(f"  • Service Level + Abandonment: SL={sl_cols}, Aban={aban_cols}")

            return merged_df

        except Exception as e:
            print(f"❌ Error merging files: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Need at least 2 files to test merging (got {len(processed_dfs)})")

    print(f"\n🎉 Preprocessor testing complete!")

if __name__ == "__main__":
    test_preprocessor()