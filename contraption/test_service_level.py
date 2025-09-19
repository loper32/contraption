"""
Quick test script to simulate forecast data for Service Level Prediction testing
"""

import streamlit as st
import pandas as pd
import numpy as np

# Simulate forecast data being in session state
if 'forecast_data' not in st.session_state:
    # Create sample forecast data
    forecast_data = pd.DataFrame({
        'Period': [f'Week {i+1}' for i in range(4)],
        'Calls': [429207, 438126, 484588, 539239],
        'AHT': [690, 689, 683, 679]
    })
    st.session_state['forecast_data'] = forecast_data
    st.session_state['hours_per_day'] = 8
    st.session_state['days_per_week'] = 5
    st.session_state['shrinkage_rate'] = 30

    print("✅ Forecast data loaded into session state")
    print(forecast_data)
else:
    print("Forecast data already in session state")

# Navigate to Service Level Prediction
print("\nNavigate to Service Level Prediction section in the app")
print("The page should now work with the forecast data available")