"""
Create simple test Excel files for upload testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create simple test data
dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

# File 1: Calls data
calls_data = {
    'Date': dates,
    'Call_Volume': np.random.randint(800, 1200, 30),
    'AHT_Seconds': np.random.randint(250, 350, 30)
}
calls_df = pd.DataFrame(calls_data)
calls_df.to_excel('Calls.xlsx', index=False)

# File 2: Staff data
staff_data = {
    'Date': dates,
    'FTE': np.random.randint(15, 25, 30),
    'Shrinkage_Pct': np.random.uniform(12, 18, 30)
}
staff_df = pd.DataFrame(staff_data)
staff_df.to_excel('Staff.xlsx', index=False)

# File 3: Occupancy data
occ_data = {
    'Date': dates,
    'Occupancy_Pct': np.random.uniform(70, 85, 30),
    'Service_Level_Pct': np.random.uniform(75, 95, 30)
}
occ_df = pd.DataFrame(occ_data)
occ_df.to_excel('Occ.xlsx', index=False)

print("Created test Excel files:")
print("- Calls.xlsx")
print("- Staff.xlsx")
print("- Occ.xlsx")

print("\nCalls data preview:")
print(calls_df.head())
print("\nStaff data preview:")
print(staff_df.head())
print("\nOcc data preview:")
print(occ_df.head())