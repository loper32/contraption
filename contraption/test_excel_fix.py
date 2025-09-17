"""
Test the Excel loader fix
"""

import sys
import os
sys.path.insert(0, 'src')

from data.excel_loader import ExcelDataLoader

# Test with our created files
loader = ExcelDataLoader()

try:
    print("Testing Calls.xlsx...")
    calls_df = loader.load_excel_file('Calls.xlsx')
    print(f"✓ Loaded Calls.xlsx: {len(calls_df)} rows, {len(calls_df.columns)} columns")
    print(f"Columns: {list(calls_df.columns)}")
    print(calls_df.head(3))
    print()

    print("Testing Staff.xlsx...")
    staff_df = loader.load_excel_file('Staff.xlsx')
    print(f"✓ Loaded Staff.xlsx: {len(staff_df)} rows, {len(staff_df.columns)} columns")
    print(f"Columns: {list(staff_df.columns)}")
    print(staff_df.head(3))
    print()

    print("Testing Occ.xlsx...")
    occ_df = loader.load_excel_file('Occ.xlsx')
    print(f"✓ Loaded Occ.xlsx: {len(occ_df)} rows, {len(occ_df.columns)} columns")
    print(f"Columns: {list(occ_df.columns)}")
    print(occ_df.head(3))
    print()

    print("Testing file merging...")
    merged_df = loader.merge_files_by_timestamp([calls_df, staff_df, occ_df])
    print(f"✓ Merged files: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    print(f"Merged columns: {list(merged_df.columns)}")
    print(merged_df.head(3))

    print("\n🎉 All tests passed!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()