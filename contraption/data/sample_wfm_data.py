"""
Script to generate sample WFM data for testing the Contraption application
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 6, 30)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate base patterns
n_days = len(dates)

# Generate workforce metrics with realistic patterns
data = {
    'timestamp': dates,

    # Volume metrics (with weekly seasonality and trend)
    'volume': (
        1000 +
        np.random.normal(0, 50, n_days) +  # Random noise
        100 * np.sin(2 * np.pi * np.arange(n_days) / 7) +  # Weekly pattern
        np.arange(n_days) * 2  # Growth trend
    ).clip(min=0),

    # Average Handle Time (in seconds, relatively stable with slight variations)
    'aht': (
        300 +
        np.random.normal(0, 20, n_days) +
        10 * np.sin(2 * np.pi * np.arange(n_days) / 30)  # Monthly pattern
    ).clip(min=200, max=400),

    # Service Level (percentage, inversely related to volume)
    'service_level': np.clip(95 - (np.random.normal(0, 5, n_days)), 70, 100),

    # Occupancy (percentage, related to volume)
    'occupancy': np.clip(75 + np.random.normal(0, 5, n_days), 60, 90),

    # Shrinkage (percentage, relatively stable)
    'shrinkage': np.clip(15 + np.random.normal(0, 2, n_days), 10, 25),
}

# Calculate FTE based on workforce management formula
# FTE = (Volume * AHT) / (Available Time * Occupancy * (1 - Shrinkage))
available_time = 8 * 60 * 60  # 8 hours in seconds

data['fte'] = (
    (data['volume'] * data['aht']) /
    (available_time * (data['occupancy']/100) * (1 - data['shrinkage']/100))
) + np.random.normal(0, 2, n_days)

data['fte'] = np.round(data['fte']).clip(min=10)

# Add some derived metrics
data['productivity'] = data['volume'] / data['fte']
data['efficiency'] = (data['service_level'] / 100) * (data['occupancy'] / 100) * 100

# Create DataFrame
df = pd.DataFrame(data)

# Round numeric columns for cleaner data
numeric_columns = ['volume', 'aht', 'service_level', 'occupancy', 'shrinkage', 'fte', 'productivity', 'efficiency']
for col in numeric_columns:
    if col in ['service_level', 'occupancy', 'shrinkage', 'efficiency']:
        df[col] = df[col].round(1)
    elif col == 'aht':
        df[col] = df[col].round(0)
    else:
        df[col] = df[col].round(0)

# Save to Excel files (split into multiple files for testing merge functionality)
# File 1: Core metrics
core_metrics = df[['timestamp', 'volume', 'aht', 'service_level']].copy()
core_metrics.to_excel('data/wfm_core_metrics.xlsx', index=False)

# File 2: Staffing metrics
staffing_metrics = df[['timestamp', 'fte', 'occupancy', 'shrinkage']].copy()
staffing_metrics.to_excel('data/wfm_staffing_metrics.xlsx', index=False)

# File 3: Performance metrics
performance_metrics = df[['timestamp', 'productivity', 'efficiency']].copy()
performance_metrics.to_excel('data/wfm_performance_metrics.xlsx', index=False)

# Also save complete dataset
df.to_excel('data/wfm_complete_data.xlsx', index=False)

print("Sample WFM data files created successfully!")
print(f"Generated {len(df)} days of data from {start_date.date()} to {end_date.date()}")
print("\nFiles created:")
print("- data/wfm_core_metrics.xlsx")
print("- data/wfm_staffing_metrics.xlsx")
print("- data/wfm_performance_metrics.xlsx")
print("- data/wfm_complete_data.xlsx")
print("\nSample data preview:")
print(df.head(10))