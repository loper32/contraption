"""Service Level Prediction Module"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def show_service_level_prediction():
    """Service level prediction interface"""
    st.header("🎯 Service Level Prediction")
    st.write("Predict expected service levels based on your actual staffing plans and forecast workload.")

    # Check prerequisites
    if 'forecast_data' not in st.session_state:
        st.warning("⚠️ Please complete the Forecasting section first to generate workload predictions.")

        # Offer to use sample data for testing
        if st.button("🔧 Load Sample Forecast Data for Testing"):
            # Create sample forecast data based on your example
            st.session_state['forecast_data'] = pd.DataFrame({
                'Period': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                'Calls': [429207, 438126, 484588, 539239],
                'AHT': [690, 689, 683, 679]
            })
            st.session_state['hours_per_day'] = 8
            st.session_state['days_per_week'] = 5
            st.session_state['shrinkage_rate'] = 30
            st.success("✅ Sample forecast data loaded! Please refresh to continue.")
            st.rerun()
        return

    # Try to find fitted models - but don't require them
    occupancy_to_sl_relationship = None

    # Check both possible session state locations
    fitted_models = st.session_state.get('fitted_models', st.session_state.get('wfm_relationship_models', {}))

    if fitted_models:
        # Find the Occupancy → Service Level relationship (inverse of what we used before)
        for rel_name, model in fitted_models.items():
            if "occupancy" in rel_name.lower() and "service" in rel_name.lower():
                # We need to check which direction this relationship goes
                if model.get('x_metric') == 'service_level' and model.get('y_metric') == 'occupancy':
                    # This is SL→Occ, we need the inverse
                    pass
                elif model.get('x_metric') == 'occupancy' and model.get('y_metric') == 'service_level':
                    # This is Occ→SL, perfect!
                    occupancy_to_sl_relationship = model
                    break

    if occupancy_to_sl_relationship is None:
        st.info("ℹ️ No trained models found. Using industry-standard Erlang approximations for Service Level predictions.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 FTE Input")

        # Persist input method selection
        if 'fte_input_method' not in st.session_state:
            st.session_state.fte_input_method = "Use Sample Data"

        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Use Sample Data", "Copy/Paste", "Upload File"],
            index=["Use Sample Data", "Copy/Paste", "Upload File"].index(st.session_state.fte_input_method),
            help="Choose how to provide your actual FTE staffing data"
        )

        # Update session state
        st.session_state.fte_input_method = input_method

        fte_data = None

        if input_method == "Use Sample Data":
            # Get forecast data to create sample
            forecast_df = st.session_state.forecast_data
            num_periods = len(forecast_df)

            # Generate sample FTE data based on forecast
            sample_fte = []
            for idx, row in forecast_df.iterrows():
                period = row.get('Period', row.get('Date', f"Week {idx+1}"))
                # Use a reasonable default FTE (could be from previous calculation)
                if 'last_fte_calculation' in st.session_state:
                    base_fte = st.session_state.last_fte_calculation.get(idx, 100)
                else:
                    base_fte = 100 + (idx * 5)  # Simple increasing pattern
                sample_fte.append(f"{period}\t{base_fte}")

            sample_text = "\n".join(sample_fte)
            st.text_area(
                "Sample FTE data (Period, FTE):",
                value=sample_text,
                height=200,
                help="Default sample data for testing",
                disabled=True
            )

            # Parse sample data
            fte_data = []
            for line in sample_text.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) == 2:
                    fte_data.append({
                        'Period': parts[0].strip(),
                        'FTE': int(parts[1].strip())
                    })

        elif input_method == "Copy/Paste":
            paste_format = st.selectbox(
                "Data format:",
                ["Tab-separated", "Comma-separated"],
                help="Choose the format of your pasted data"
            )

            delimiter = '\t' if paste_format == "Tab-separated" else ','

            # Use session state to preserve input
            if 'fte_paste_input' not in st.session_state:
                st.session_state.fte_paste_input = ""

            paste_input = st.text_area(
                "Paste your FTE data (Period, FTE):",
                value=st.session_state.fte_paste_input,
                height=200,
                placeholder=f"Week 1{delimiter}150\nWeek 2{delimiter}155\nWeek 3{delimiter}160",
                help=f"Paste data with periods and FTE values separated by {delimiter}. Can include headers.",
                key="fte_paste_textarea"
            )

            # Update session state when text changes
            st.session_state.fte_paste_input = paste_input

            if st.button("📊 Process Pasted Data", type="primary", disabled=not paste_input):
                if paste_input:
                    try:
                        fte_data = []
                        lines = paste_input.strip().split('\n')

                        # Check if first line is a header
                        skip_first = False
                        if lines and len(lines[0].split(delimiter)) >= 2:
                            try:
                                # Try to parse the second column as a number
                                float(lines[0].split(delimiter)[1].strip())
                            except ValueError:
                                # If it fails, it's likely a header
                                skip_first = True
                                st.info("📋 Detected header row, skipping first line")

                        for i, line in enumerate(lines):
                            if i == 0 and skip_first:
                                continue
                            if not line.strip():  # Skip empty lines
                                continue

                            # Handle both tab and comma, regardless of selection
                            parts = line.split(delimiter) if delimiter in line else line.split('\t' if delimiter == ',' else ',')

                            if len(parts) >= 2:
                                try:
                                    period = parts[0].strip()
                                    # Handle FTE values that might have commas (e.g., "1,234")
                                    fte_str = parts[1].strip().replace(',', '')
                                    fte_value = int(float(fte_str))
                                    fte_data.append({
                                        'Period': period,
                                        'FTE': fte_value
                                    })
                                except ValueError as ve:
                                    # Only warn if it's not a known header word
                                    if not any(header in parts[1].lower() for header in ['fte', 'staff', 'agents', 'headcount']):
                                        st.warning(f"⚠️ Skipping invalid line: {line[:50]}...")
                                    continue
                        if fte_data:
                            st.success(f"✓ Loaded {len(fte_data)} periods")
                            # Store in session state to persist
                            st.session_state['fte_input_data'] = fte_data
                        else:
                            st.error("No valid data found. Please check your format.")
                    except Exception as e:
                        st.error(f"Error parsing data: {str(e)}")
                        fte_data = None

            # Retrieve from session state if it exists
            if 'fte_input_data' in st.session_state and input_method == "Copy/Paste":
                fte_data = st.session_state['fte_input_data']

        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload FTE data file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload a file with Period and FTE columns"
            )

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    # Find FTE column
                    fte_col = None
                    period_col = None
                    for col in df.columns:
                        if 'fte' in col.lower() or 'staff' in col.lower():
                            fte_col = col
                        if 'period' in col.lower() or 'week' in col.lower() or 'date' in col.lower():
                            period_col = col

                    if fte_col and period_col:
                        fte_data = []
                        for _, row in df.iterrows():
                            fte_data.append({
                                'Period': str(row[period_col]),
                                'FTE': int(row[fte_col])
                            })
                        st.success(f"✓ Loaded {len(fte_data)} periods from {uploaded_file.name}")
                    else:
                        st.error("Could not find Period and FTE columns in file")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

    with col2:
        st.subheader("📊 Forecast Workload")

        # Display forecast data that will be used
        forecast_df = st.session_state.forecast_data

        # Show summary of forecast
        st.metric("Total Periods", len(forecast_df))

        total_calls = forecast_df['Calls'].sum() if 'Calls' in forecast_df.columns else 0
        avg_aht = forecast_df['AHT'].mean() if 'AHT' in forecast_df.columns else 300

        st.metric("Total Call Volume", f"{total_calls:,.0f}")
        st.metric("Average AHT", f"{avg_aht:.0f} seconds")

        # Show forecast details
        with st.expander("View Forecast Details"):
            st.dataframe(forecast_df, use_container_width=True)

    # Calculate Service Level Predictions
    if fte_data and len(fte_data) > 0:
        st.markdown("---")
        st.subheader("🔮 Service Level Predictions")

        if len(fte_data) != len(forecast_df):
            st.warning(f"⚠️ FTE data has {len(fte_data)} periods but forecast has {len(forecast_df)} periods. Will use minimum of both.")

        # Get parameters from session state or use defaults
        shrinkage_rate = st.session_state.get('shrinkage_rate', 30)
        hours_per_day = st.session_state.get('hours_per_day', 24)
        days_per_week = st.session_state.get('days_per_week', 7)
        hours_per_period = hours_per_day * days_per_week

        # Calculate service levels
        results = []
        min_periods = min(len(fte_data), len(forecast_df))

        for idx in range(min_periods):
            fte_info = fte_data[idx]
            forecast_row = forecast_df.iloc[idx]

            actual_fte = fte_info['FTE']
            period_label = fte_info['Period']
            calls = forecast_row.get('Calls', 0)
            aht = forecast_row.get('AHT', 300)

            # Calculate occupancy from FTE and workload
            # Occupancy = (Calls × AHT) / (FTE × Hours × 3600 × (1 - Shrinkage))
            total_work_hours = (calls * aht) / 3600
            available_hours = actual_fte * hours_per_period * (1 - shrinkage_rate/100)

            if available_hours > 0:
                occupancy = (total_work_hours / available_hours) * 100
                occupancy = min(98, max(20, occupancy))  # Bound between 20% and 98%
            else:
                occupancy = 0

            # Predict service level from occupancy
            predicted_sl = None

            if occupancy_to_sl_relationship:
                # Use direct Occ→SL relationship
                model_type = occupancy_to_sl_relationship.get('model_type')
                popt = occupancy_to_sl_relationship.get('parameters')

                # Apply the model (occupancy is in percentage form here)
                occ_decimal = occupancy / 100

                try:
                    if model_type == 'linear' and len(popt) >= 2:
                        predicted_sl = popt[0] * occ_decimal + popt[1]
                    elif model_type == 'polynomial' and len(popt) >= 3:
                        predicted_sl = popt[0] * occ_decimal**2 + popt[1] * occ_decimal + popt[2]
                    elif model_type == 'exponential' and len(popt) >= 3:
                        predicted_sl = popt[0] * np.exp(popt[1] * occ_decimal) + popt[2]
                    elif model_type == 'power' and len(popt) >= 3:
                        def power_func(x, a, b, c):
                            return a * np.power(np.abs(x + 0.001), b) + c
                        predicted_sl = power_func(occ_decimal, *popt)
                    elif model_type == 'logarithmic' and len(popt) >= 2:
                        predicted_sl = popt[0] * np.log(occ_decimal + 0.001) + popt[1]

                    # Convert to percentage and bound
                    if predicted_sl is not None:
                        predicted_sl = predicted_sl * 100
                        predicted_sl = min(100, max(0, predicted_sl))
                except Exception as e:
                    st.error(f"Error predicting SL for period {period_label}: {str(e)}")
            else:
                # Use inverse of SL→Occ relationship (simplified Erlang approximation)
                # This is a rough approximation when we don't have the direct relationship
                if occupancy < 50:
                    predicted_sl = 95 + (50 - occupancy) * 0.1
                elif occupancy < 85:
                    predicted_sl = 80 + (85 - occupancy) * 0.5
                else:
                    predicted_sl = max(20, 80 - (occupancy - 85) * 2)

                predicted_sl = min(100, max(0, predicted_sl))

            results.append({
                'Period': period_label,
                'Actual FTE': actual_fte,
                'Call Volume': f"{calls:,}",
                'AHT (sec)': f"{aht:.0f}",
                'Occupancy %': f"{occupancy:.1f}",
                'Predicted SL %': f"{predicted_sl:.1f}" if predicted_sl is not None else "N/A"
            })

        # Display results table
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Create visualization
        if any(r['Predicted SL %'] != "N/A" for r in results):
            fig = go.Figure()

            # Extract data for plotting
            periods = [r['Period'] for r in results]
            sl_values = []
            occ_values = []

            for r in results:
                if r['Predicted SL %'] != "N/A":
                    sl_values.append(float(r['Predicted SL %'].replace('%', '')))
                    occ_values.append(float(r['Occupancy %'].replace('%', '')))
                else:
                    sl_values.append(None)
                    occ_values.append(float(r['Occupancy %'].replace('%', '')))

            # Add service level line
            fig.add_trace(go.Scatter(
                x=periods,
                y=sl_values,
                mode='lines+markers',
                name='Predicted Service Level',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ))

            # Add occupancy line on secondary y-axis
            fig.add_trace(go.Scatter(
                x=periods,
                y=occ_values,
                mode='lines+markers',
                name='Occupancy',
                line=dict(color='blue', width=2, dash='dash'),
                marker=dict(size=6),
                yaxis='y2'
            ))

            # Add target lines
            fig.add_hline(y=80, line_dash="dash", line_color="red",
                         annotation_text="SL Target: 80%", annotation_position="left")

            fig.update_layout(
                title='Service Level Predictions Based on Actual FTE',
                xaxis_title='Period',
                yaxis_title='Service Level (%)',
                yaxis2=dict(
                    title='Occupancy (%)',
                    overlaying='y',
                    side='right'
                ),
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_sl = np.mean([float(r['Predicted SL %'].replace('%', ''))
                             for r in results if r['Predicted SL %'] != "N/A"])
            st.metric("Average Service Level", f"{avg_sl:.1f}%")

        with col2:
            avg_occ = np.mean([float(r['Occupancy %'].replace('%', '')) for r in results])
            st.metric("Average Occupancy", f"{avg_occ:.1f}%")

        with col3:
            total_fte = sum([r['Actual FTE'] for r in results])
            st.metric("Total FTE", f"{total_fte:,}")

        # Insights
        st.markdown("---")
        st.subheader("💡 Insights")

        if avg_sl < 70:
            st.error("⚠️ **Understaffed**: Average service level is below 70%. Consider adding more FTE.")
        elif avg_sl < 80:
            st.warning("📊 **Below Target**: Service level is below the typical 80% target.")
        else:
            st.success("✅ **On Target**: Service level meets or exceeds the 80% target.")

        if avg_occ > 90:
            st.warning("🔥 **High Occupancy**: Staff utilization is very high, which may lead to burnout.")
        elif avg_occ < 60:
            st.info("💼 **Low Occupancy**: There may be excess capacity. Consider cross-training or additional tasks.")