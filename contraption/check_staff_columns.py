#!/usr/bin/env python3
"""
Quick check of Staff.xlsx column names
"""

import pandas as pd

# Check Staff.xlsx column names
staff_df = pd.read_excel('../WFM-Data/Staff.xlsx', sheet_name=0)
print("Staff.xlsx columns:")
print(list(staff_df.columns))
print(f"\nFirst few rows:")
print(staff_df.head(3))
print(f"\nData types:")
print(staff_df.dtypes)