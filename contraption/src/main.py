"""
Contraption - Workforce Management Analytics Platform
Main application entry point using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Import service level prediction module
from src.service_level_prediction import show_service_level_prediction

# Import convergence modules
from src.convergence import (
    ConvergenceEngine,
    create_convergence_config,
    create_relationship_predictor_from_models,
    BinarySearchSolver
)

# Configure page
st.set_page_config(
    page_title="Contraption - WFM Analytics",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    st.title("� Contraption")
    st.subheader("Workforce Management Analytics Platform")

    # Sidebar navigation with status indicators
    with st.sidebar:
        st.header("📍 Navigation")

        # Add status indicators for what's loaded
        if 'merged_data' in st.session_state and st.session_state.merged_data is not None:
            st.success("✅ Data Loaded")
        if 'wfm_relationship_models' in st.session_state and st.session_state.wfm_relationship_models:
            st.success("✅ Models Trained")
        if 'forecast_data' in st.session_state and st.session_state.forecast_data is not None:
            st.success("✅ Forecast Ready")

        st.markdown("---")

        # Use radio buttons for better navigation
        page = st.radio(
            "Select Section:",
            [
                "🏠 Home",
                "📁 Data Upload & Processing",
                "📊 Historical Analysis",
                "📈 Forecasting",
                "🎯 Service Level Prediction",
                "🔄 Convergence Analysis",
                "⚙️ Settings"
            ],
            index=1,  # Default to Data Upload & Processing
            label_visibility="collapsed"
        )

        # Clean up the page name for processing
        page = page.split(" ", 1)[1] if " " in page else page

    # Main content area
    if page == "Home":
        show_home()
    elif page == "Data Upload & Processing":
        show_data_upload()
    elif page == "Historical Analysis":
        show_historical_analysis()
    elif page == "Forecasting":
        show_forecasting()
    elif page == "Service Level Prediction":
        show_service_level_prediction()
    elif page == "Convergence Analysis":
        show_convergence_analysis()
    elif page == "Settings":
        show_settings()

def show_home():
    """Display home page with overview"""
    st.markdown("""
    ## Welcome to Contraption

    A comprehensive workforce management analytics platform that helps you:

    ### =� **Data Integration**
    - Load and merge Excel files based on timestamps
    - Process historical workforce metrics

    ### =� **Predictive Analytics**
    - Establish relationships between workforce metrics
    - Use curve fitting for FTE prediction models

    ### <� **Forecasting & Planning**
    - Apply learned models to new forecast data
    - Calculate FTE requirements with configurable assumptions

    ### = **Service Level Prediction**
    - Predict service levels based on actual staffing plans
    - Analyze staffing scenarios and their outcomes

    ### 🔄 **Convergence Analysis**
    - Advanced dual-loop convergence algorithms
    - Account for abandon/retry cycles and occupancy stress effects

    ### =� **Visualization & Reporting**
    - Interactive plots and dashboards
    - Exportable analysis and recommendations

    ---
    **Get started by uploading your historical data in the 'Data Upload & Processing' section.**
    """)

def show_data_upload():
    """Data upload and processing interface"""
    st.header("📁 Data Upload & Processing")
    st.write("Upload Excel files containing historical workforce metrics.")

    # Import data processing modules
    try:
        from data.excel_loader import ExcelDataLoader
        from data.processor import WFMDataProcessor
        from data.wfm_preprocessor import WFMPreprocessor
    except ImportError:
        st.error("Data processing modules not found. Please ensure the project is properly set up.")
        return

    # Initialize processors
    if 'excel_loader' not in st.session_state:
        st.session_state.excel_loader = ExcelDataLoader()
    if 'wfm_processor' not in st.session_state:
        st.session_state.wfm_processor = WFMDataProcessor()
    if 'wfm_preprocessor' not in st.session_state:
        st.session_state.wfm_preprocessor = WFMPreprocessor()

    # Check if data is already loaded
    if 'merged_data' in st.session_state and st.session_state.merged_data is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ Data already loaded: {st.session_state.merged_data.shape[0]:,} rows, {st.session_state.merged_data.shape[1]} columns")
        with col2:
            if st.button("🔄 Upload New Data"):
                # Clear existing data
                if 'merged_data' in st.session_state:
                    del st.session_state.merged_data
                if 'processed_data' in st.session_state:
                    del st.session_state.processed_data
                if 'uploaded_files_info' in st.session_state:
                    del st.session_state.uploaded_files_info
                st.rerun()

        # Show loaded file information
        if 'uploaded_files_info' in st.session_state:
            with st.expander("📂 Loaded Files", expanded=False):
                for info in st.session_state.uploaded_files_info:
                    st.write(f"• **{info['name']}**: {info['rows']:,} rows")

        # Display preview of loaded data
        st.subheader("📊 Data Preview")
        st.dataframe(st.session_state.merged_data.head(100), use_container_width=True)

        # Show data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Date Range",
                     f"{st.session_state.merged_data.index.min().strftime('%Y-%m-%d')} to {st.session_state.merged_data.index.max().strftime('%Y-%m-%d')}")
        with col2:
            st.metric("Total Records", f"{len(st.session_state.merged_data):,}")
        with col3:
            st.metric("Columns", len(st.session_state.merged_data.columns))
        return  # Exit early since data is already loaded

    # File upload section (only shown if no data loaded)
    st.subheader("Upload Excel Files")
    uploaded_files = st.file_uploader(
        "Choose Excel files",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload one or more Excel files containing workforce management data with timestamp columns."
    )

    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")

        # Processing info
        st.info("📊 Files will be automatically processed and merged based on 30-minute interval timestamps.")

        # Process files button
        if st.button("Process Files", type="primary"):
            try:
                with st.spinner("Processing files..."):
                    # Save uploaded files temporarily and process
                    dataframes = []
                    file_info = []

                    # Debug info
                    st.write(f"Processing {len(uploaded_files)} files...")

                    for uploaded_file in uploaded_files:
                        try:
                            # Reset file pointer to beginning
                            uploaded_file.seek(0)

                            # Use WFM Preprocessor for proper handling of complex WFM files
                            import tempfile
                            import os

                            # Save uploaded file temporarily for preprocessing
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_file_path = tmp_file.name

                            try:
                                # Process using WFM Preprocessor
                                df = st.session_state.wfm_preprocessor.preprocess_file(tmp_file_path)

                                # Debug: show what was loaded
                                st.write(f"✅ Processed {uploaded_file.name}: {len(df)} rows, {len(df.columns)} columns")
                                st.write(f"📊 Columns: {list(df.columns)}")
                                if 'datetime' in df.columns:
                                    st.write(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")

                                dataframes.append(df)
                                file_info.append({
                                    'name': uploaded_file.name,
                                    'rows': len(df),
                                    'columns': list(df.columns)
                                })
                            finally:
                                # Clean up temporary file
                                os.unlink(tmp_file_path)

                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                            # Show more details for debugging
                            import traceback
                            st.code(traceback.format_exc())

                    # Merge files if multiple
                    if len(dataframes) > 1:
                        merged_df = st.session_state.wfm_preprocessor.merge_preprocessed_files(
                            dataframes,
                            merge_strategy="outer"  # Use outer join to preserve all data
                        )
                        st.write(f"✅ Successfully merged {len(dataframes)} files: {merged_df.shape}")
                    else:
                        merged_df = dataframes[0].set_index('datetime')
                        st.write(f"✅ Single file processed: {merged_df.shape}")

                    # Store processed data with both names for compatibility
                    st.session_state.processed_data = merged_df
                    st.session_state.merged_data = merged_df  # Also store as merged_data for consistency
                    st.session_state.uploaded_files_info = file_info  # Store file info for later reference

                st.success("Files processed successfully!")

                # Show file information
                st.subheader("File Information")
                for info in file_info:
                    with st.expander(f"📄 {info['name']} ({info['rows']} rows)"):
                        st.write("**Columns:**", ", ".join(info['columns']))

            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

    # Show processed data preview
    if 'processed_data' in st.session_state:
        st.subheader("Data Preview")

        # Summary statistics
        summary = st.session_state.excel_loader.get_data_summary(st.session_state.processed_data)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", summary['total_rows'])
        with col2:
            st.metric("Columns", len(summary['columns']))
        with col3:
            st.metric("Memory (MB)", f"{summary['memory_usage_mb']:.1f}")

        # Data quality check
        validation_warnings = st.session_state.excel_loader.validate_wfm_data(
            st.session_state.processed_data
        )

        if validation_warnings:
            st.warning("Data Quality Issues Detected:")
            for warning in validation_warnings:
                st.write(f"⚠️ {warning}")

        # Show data preview
        st.write("**Data Preview (first 10 rows):**")
        st.dataframe(st.session_state.processed_data.head(10))

        # Data processing options
        st.subheader("Data Processing Options")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Calculate Derived Metrics"):
                with st.spinner("Calculating derived metrics..."):
                    st.session_state.processed_data = st.session_state.wfm_processor.calculate_derived_metrics(
                        st.session_state.processed_data
                    )
                st.success("Derived metrics calculated!")
                st.rerun()

        with col2:
            if st.button("Clean Outliers"):
                with st.spinner("Cleaning outliers..."):
                    st.session_state.processed_data = st.session_state.wfm_processor.clean_outliers(
                        st.session_state.processed_data
                    )
                st.success("Outliers cleaned!")
                st.rerun()

def show_historical_analysis():
    """Historical analysis interface"""
    st.header("📊 Historical Analysis")
    st.write("Analyze relationships between workforce metrics.")

    # Check if data is available (support both names)
    if 'merged_data' not in st.session_state and 'processed_data' not in st.session_state:
        st.warning("Please upload and process data first in the 'Data Upload & Processing' section.")
        return

    # Use whichever is available
    data = st.session_state.get('merged_data', st.session_state.get('processed_data'))

    # Import analysis modules
    try:
        from analysis.relationships import MetricsRelationshipAnalyzer
        from visualization.correlation_plots import CorrelationVisualizer
    except ImportError:
        st.error("Analysis modules not found. Please ensure the project is properly set up.")
        return

    # Initialize analyzers
    if 'relationship_analyzer' not in st.session_state:
        st.session_state.relationship_analyzer = MetricsRelationshipAnalyzer()
    if 'correlation_visualizer' not in st.session_state:
        st.session_state.correlation_visualizer = CorrelationVisualizer()

    df = st.session_state.processed_data

    # Service Level Target Setting
    st.subheader("🎯 Service Level Target")

    col1, col2 = st.columns([1, 2])

    with col1:
        sl_target = st.number_input(
            "Service Level Target (%)",
            min_value=70.0,
            max_value=99.9,
            value=80.0,
            step=0.1,
            help="Target service level for FTE calculations"
        ) / 100.0  # Convert to decimal

        # Store in session state for use in Forecasting
        st.session_state.service_level_target = sl_target

    with col2:
        st.info(f"Target: {sl_target*100:.1f}% - This will be marked on the curves and used for intersection calculations")

    # WFM Relationship Curves Section
    st.subheader("📈 WFM Relationship Curves")
    st.write("Key workforce management relationships weighted by call volume")

    # Import model modules
    try:
        from models.curve_fitting import FTEPredictionModel
        from scipy.optimize import curve_fit
    except ImportError:
        st.error("Curve fitting model not found. Please ensure the project is properly set up.")
        return

    # Initialize curve fitting model
    if 'curve_model' not in st.session_state:
        st.session_state.curve_model = FTEPredictionModel()

    # Define the three key WFM relationships (ordered so SL→Occ is first for cross-referencing)
    key_relationships = [
        {
            "name": "Service Level → Occupancy",
            "x_metric": "service_level",
            "y_metric": "occupancy",
            "curve_type": "auto"
        },
        {
            "name": "Occupancy → AHT",
            "x_metric": "occupancy",
            "y_metric": "average_handle_time",
            "curve_type": "exponential"
        },
        {
            "name": "Service Level → Abandonment",
            "x_metric": "service_level",
            "y_metric": "abandonment_rate",
            "curve_type": "power"
        }
    ]

    # Check if we have the necessary columns
    required_cols = ['service_level', 'occupancy', 'average_handle_time', 'abandonment_rate']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.warning(f"Missing required columns: {missing_cols}. Upload complete WFM data to see all relationships.")
        return

    # Get call volume for weighting (use 'offered' as primary, 'answered' as backup)
    volume_col = 'offered' if 'offered' in df.columns else 'answered' if 'answered' in df.columns else None

    if st.button("Generate All WFM Relationship Curves", type="primary"):
        with st.spinner("Fitting curves weighted by call volume..."):

            # Create three columns for side-by-side plots
            cols = st.columns(3)

            fitted_models = {}

            for idx, rel in enumerate(key_relationships):
                with cols[idx]:
                    try:
                        # Handle column name variations with fallback logic
                        x_col = rel['x_metric']
                        y_col = rel['y_metric']

                        # Check if columns exist, try alternatives if not
                        if x_col not in df.columns:
                            st.error(f"Column '{x_col}' not found in DataFrame for {rel['name']}")
                            continue

                        if y_col not in df.columns:
                            # Try common variations for y_metric
                            alternatives = {
                                'abandonment_rate': ['abandonment', 'abandon_rate', '%aban', 'abandon%'],
                                'average_handle_time': ['aht', 'handle_time', 'avg_handle_time']
                            }
                            found_alternative = None
                            if y_col in alternatives:
                                for alt in alternatives[y_col]:
                                    if alt in df.columns:
                                        found_alternative = alt
                                        st.warning(f"Using '{alt}' instead of '{y_col}' for {rel['name']}")
                                        break

                            if found_alternative:
                                y_col = found_alternative
                            else:
                                st.error(f"Column '{y_col}' not found in DataFrame for {rel['name']}. Available columns: {list(df.columns)}")
                                continue

                        # Get clean data with volume weighting
                        if volume_col:
                            data_subset = df[[x_col, y_col, volume_col]].dropna()
                            weights = data_subset[volume_col].values
                            # Normalize weights to prevent numerical issues
                            weights = weights / weights.mean()
                        else:
                            data_subset = df[[x_col, y_col]].dropna()
                            weights = None

                        if len(data_subset) < 10:
                            st.error(f"Not enough data for {rel['name']}")
                            continue

                        x_data = data_subset[x_col].values
                        y_data = data_subset[y_col].values

                        # Fit weighted curve using scipy directly for better control
                        if weights is not None and rel['curve_type'] == 'auto':
                            # Try multiple models and pick the best one for weighted fitting
                            best_model = None
                            best_r_squared = -1

                            models_to_try = ['polynomial', 'exponential', 'power', 'logarithmic', 'linear']

                            for test_model_type in models_to_try:
                                try:
                                    if test_model_type == 'polynomial':
                                        def poly_func(x, a, b, c):
                                            return a * x**2 + b * x + c
                                        p0 = [1, 1, np.mean(y_data)]
                                        test_popt, _ = curve_fit(poly_func, x_data, y_data, p0=p0,
                                                               sigma=1/np.sqrt(weights), absolute_sigma=True, maxfev=5000)
                                        test_y_pred = poly_func(x_data, *test_popt)
                                        test_x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                        test_y_smooth = poly_func(test_x_smooth, *test_popt)
                                    elif test_model_type == 'exponential':
                                        def exp_func(x, a, b, c):
                                            return a * np.exp(b * x) + c
                                        p0 = [1, 0.1, np.mean(y_data)]
                                        test_popt, _ = curve_fit(exp_func, x_data, y_data, p0=p0,
                                                               sigma=1/np.sqrt(weights), absolute_sigma=True, maxfev=5000)
                                        test_y_pred = exp_func(x_data, *test_popt)
                                        test_x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                        test_y_smooth = exp_func(test_x_smooth, *test_popt)
                                    elif test_model_type == 'power':
                                        def power_func(x, a, b, c):
                                            return a * np.power(np.abs(x + 0.001), b) + c
                                        p0 = [1, 1, np.mean(y_data)]
                                        test_popt, _ = curve_fit(power_func, x_data, y_data, p0=p0,
                                                               sigma=1/np.sqrt(weights), absolute_sigma=True, maxfev=5000)
                                        test_y_pred = power_func(x_data, *test_popt)
                                        test_x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                        test_y_smooth = power_func(test_x_smooth, *test_popt)
                                    elif test_model_type == 'logarithmic':
                                        def log_func(x, a, b):
                                            return a * np.log(np.abs(x + 0.001)) + b
                                        p0 = [1, np.mean(y_data)]
                                        test_popt, _ = curve_fit(log_func, x_data, y_data, p0=p0,
                                                               sigma=1/np.sqrt(weights), absolute_sigma=True, maxfev=5000)
                                        test_y_pred = log_func(x_data, *test_popt)
                                        test_x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                        test_y_smooth = log_func(test_x_smooth, *test_popt)
                                    else:  # linear
                                        def linear_func(x, a, b):
                                            return a * x + b
                                        test_popt, _ = curve_fit(linear_func, x_data, y_data,
                                                               sigma=1/np.sqrt(weights), absolute_sigma=True)
                                        test_y_pred = linear_func(x_data, *test_popt)
                                        test_x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                        test_y_smooth = linear_func(test_x_smooth, *test_popt)

                                    # Calculate weighted R²
                                    ss_res = np.sum(weights * (y_data - test_y_pred) ** 2)
                                    ss_tot = np.sum(weights * (y_data - np.average(y_data, weights=weights)) ** 2)
                                    test_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                                    if test_r_squared > best_r_squared:
                                        best_r_squared = test_r_squared
                                        best_model = {
                                            'type': test_model_type,
                                            'popt': test_popt,
                                            'y_pred': test_y_pred,
                                            'x_smooth': test_x_smooth,
                                            'y_smooth': test_y_smooth,
                                            'r_squared': test_r_squared
                                        }
                                except:
                                    continue

                            if best_model:
                                popt = best_model['popt']
                                y_pred = best_model['y_pred']
                                x_smooth = best_model['x_smooth']
                                y_smooth = best_model['y_smooth']
                                r_squared = best_model['r_squared']
                                model_type = best_model['type']
                            else:
                                # Fallback to linear
                                def linear_func(x, a, b):
                                    return a * x + b
                                popt, _ = curve_fit(linear_func, x_data, y_data,
                                                 sigma=1/np.sqrt(weights), absolute_sigma=True)
                                x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                y_smooth = linear_func(x_smooth, *popt)
                                y_pred = linear_func(x_data, *popt)
                                model_type = 'linear'
                                ss_res = np.sum(weights * (y_data - y_pred) ** 2)
                                ss_tot = np.sum(weights * (y_data - np.average(y_data, weights=weights)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                        elif weights is not None and rel['curve_type'] != 'auto':
                            # Custom weighted fitting
                            if rel['curve_type'] == 'power':
                                # Power model: y = a * x^b + c
                                def power_func(x, a, b, c):
                                    return a * np.power(np.abs(x + 0.001), b) + c  # Add small value to avoid zero

                                # Initial guess
                                p0 = [1, 1, np.mean(y_data)]

                                # Weighted curve fit
                                popt, _ = curve_fit(power_func, x_data, y_data, p0=p0,
                                                  sigma=1/np.sqrt(weights), absolute_sigma=True,
                                                  maxfev=5000)

                                # Generate smooth curve
                                x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                y_smooth = power_func(x_smooth, *popt)

                                # Calculate R²
                                y_pred = power_func(x_data, *popt)
                                ss_res = np.sum(weights * (y_data - y_pred) ** 2)
                                ss_tot = np.sum(weights * (y_data - np.average(y_data, weights=weights)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                                model_type = 'power'

                            elif rel['curve_type'] == 'exponential':
                                # Exponential model: y = a * exp(b * x) + c
                                def exp_func(x, a, b, c):
                                    return a * np.exp(b * x) + c

                                # Initial guess
                                p0 = [1, 0.1, np.mean(y_data)]

                                # Weighted curve fit
                                try:
                                    popt, _ = curve_fit(exp_func, x_data, y_data, p0=p0,
                                                      sigma=1/np.sqrt(weights), absolute_sigma=True,
                                                      maxfev=5000)
                                    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                    y_smooth = exp_func(x_smooth, *popt)
                                    y_pred = exp_func(x_data, *popt)
                                    model_type = 'exponential'
                                except:
                                    # Fallback to linear if exponential fails
                                    def linear_func(x, a, b):
                                        return a * x + b
                                    popt, _ = curve_fit(linear_func, x_data, y_data,
                                                      sigma=1/np.sqrt(weights), absolute_sigma=True)
                                    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                    y_smooth = linear_func(x_smooth, *popt)
                                    y_pred = linear_func(x_data, *popt)
                                    model_type = 'linear'

                                # Calculate weighted R²
                                ss_res = np.sum(weights * (y_data - y_pred) ** 2)
                                ss_tot = np.sum(weights * (y_data - np.average(y_data, weights=weights)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                            else:
                                # Linear fallback
                                def linear_func(x, a, b):
                                    return a * x + b
                                popt, _ = curve_fit(linear_func, x_data, y_data,
                                                  sigma=1/np.sqrt(weights), absolute_sigma=True)
                                x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                                y_smooth = linear_func(x_smooth, *popt)
                                y_pred = linear_func(x_data, *popt)
                                model_type = 'linear'

                                ss_res = np.sum(weights * (y_data - y_pred) ** 2)
                                ss_tot = np.sum(weights * (y_data - np.average(y_data, weights=weights)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        else:
                            # Use standard fitting without weights
                            model_result = st.session_state.curve_model.fit_model(
                                x_data, y_data, model_type=rel['curve_type']
                            )
                            popt = model_result['parameters']
                            r_squared = model_result['r_squared']
                            x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                            y_smooth = st.session_state.curve_model.predict(model_result, x_smooth)
                            model_type = model_result['model_type']

                        # Create plot
                        fig = go.Figure()

                        # Add scatter points with size based on volume
                        if weights is not None:
                            # Scale marker sizes based on weights
                            marker_sizes = 3 + (weights / weights.max()) * 12
                        else:
                            marker_sizes = 6

                        fig.add_trace(go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='markers',
                            name='Data',
                            marker=dict(
                                color='lightblue',
                                size=marker_sizes,
                                opacity=0.6,
                                line=dict(width=0.5, color='blue')
                            ),
                            hovertemplate=f'{rel["x_metric"]}: %{{x:.3f}}<br>{rel["y_metric"]}: %{{y:.3f}}<extra></extra>'
                        ))

                        # Add fitted curve
                        fig.add_trace(go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            mode='lines',
                            name=f'{model_type.title()} Fit',
                            line=dict(color='red', width=2.5),
                            hovertemplate=f'Predicted: %{{y:.3f}}<extra></extra>'
                        ))

                        # Add Service Level Target intersection marker
                        intersection_y = None
                        intersection_annotation = ""

                        if rel['x_metric'] == 'service_level':
                            # Calculate intersection point for SL target
                            if model_type == 'linear' and len(popt) >= 2:
                                a, b = popt
                                intersection_y = a * sl_target + b
                            elif model_type == 'polynomial' and len(popt) >= 3:
                                # Polynomial: y = a*x^2 + b*x + c
                                a, b, c = popt
                                intersection_y = a * sl_target**2 + b * sl_target + c
                            elif model_type == 'logarithmic' and len(popt) >= 2:
                                a, b = popt
                                intersection_y = a * np.log(sl_target + 0.001) + b
                            elif model_type == 'power':
                                def power_func(x, a, b, c):
                                    return a * np.power(np.abs(x + 0.001), b) + c
                                intersection_y = power_func(sl_target, *popt)
                            elif model_type == 'exponential':
                                if len(popt) == 3:
                                    a, b, c = popt
                                    intersection_y = a * np.exp(b * sl_target) + c
                                else:
                                    a, b = popt
                                    intersection_y = a * np.exp(b * sl_target)

                            if intersection_y is not None:
                                # Add vertical line at SL target
                                fig.add_vline(
                                    x=sl_target,
                                    line_dash="dash",
                                    line_color="gold",
                                    line_width=2,
                                    annotation_text=f"SL Target: {sl_target*100:.1f}%"
                                )

                                # Add intersection marker
                                fig.add_trace(go.Scatter(
                                    x=[sl_target],
                                    y=[intersection_y],
                                    mode='markers',
                                    name='SL Target Intersection',
                                    marker=dict(
                                        color='gold',
                                        size=12,
                                        symbol='star',
                                        line=dict(width=2, color='orange')
                                    ),
                                    hovertemplate=f'SL: {sl_target*100:.1f}%<br>{rel["y_metric"].replace("_", " ").title()}: %{{y:.3f}}<extra></extra>'
                                ))

                                intersection_annotation = f"At {sl_target*100:.1f}% SL → {intersection_y:.3f}"

                        # Special handling for Occupancy → AHT plot: show occupancy from SL→Occ intersection
                        elif rel['x_metric'] == 'occupancy' and rel['y_metric'] == 'average_handle_time':
                            # Look for the SL→Occupancy intersection value if it was calculated
                            sl_occ_intersection = None
                            for other_rel_name, other_model in fitted_models.items():
                                if (other_rel_name == "Service Level → Occupancy" and
                                    'sl_target_intersection' in other_model):
                                    sl_occ_intersection = other_model['sl_target_intersection']['y']
                                    break

                            if sl_occ_intersection is not None:
                                # Calculate AHT at the occupancy point from SL→Occ intersection
                                if rel['curve_type'] == 'exponential':
                                    if len(popt) == 3:
                                        a, b, c = popt
                                        aht_at_sl_occ = a * np.exp(b * sl_occ_intersection) + c
                                    else:
                                        a, b = popt
                                        aht_at_sl_occ = a * np.exp(b * sl_occ_intersection)
                                elif rel['curve_type'] == 'power':
                                    def power_func(x, a, b, c):
                                        return a * np.power(np.abs(x + 0.001), b) + c
                                    aht_at_sl_occ = power_func(sl_occ_intersection, *popt)

                                # Add vertical line at the occupancy from SL→Occ
                                fig.add_vline(
                                    x=sl_occ_intersection,
                                    line_dash="dash",
                                    line_color="green",
                                    line_width=2,
                                    annotation_text=f"Occ from SL Target: {sl_occ_intersection:.1%}"
                                )

                                # Add intersection marker
                                fig.add_trace(go.Scatter(
                                    x=[sl_occ_intersection],
                                    y=[aht_at_sl_occ],
                                    mode='markers',
                                    name='SL-derived Occupancy Point',
                                    marker=dict(
                                        color='green',
                                        size=12,
                                        symbol='diamond',
                                        line=dict(width=2, color='darkgreen')
                                    ),
                                    hovertemplate=f'Occ: {sl_occ_intersection:.1%}<br>AHT: %{{y:.3f}}<extra></extra>'
                                ))

                                intersection_annotation = f"At {sl_occ_intersection:.1%} Occ → {aht_at_sl_occ:.3f} AHT"

                        # Update layout
                        title_text = f'<b>{rel["name"]}</b><br><sub>R² = {r_squared:.3f}'
                        if intersection_annotation:
                            title_text += f'<br>{intersection_annotation}'
                        title_text += '</sub>'

                        fig.update_layout(
                            title=title_text,
                            xaxis_title=rel['x_metric'].replace('_', ' ').title(),
                            yaxis_title=rel['y_metric'].replace('_', ' ').title(),
                            template='plotly_white',
                            showlegend=False,
                            height=400,
                            margin=dict(t=80, b=60, l=60, r=20)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Store the model with intersection data
                        model_data = {
                            'parameters': popt,
                            'r_squared': r_squared,
                            'model_type': model_type,
                            'x_metric': rel['x_metric'],  # Always use standardized names
                            'y_metric': rel['y_metric'],  # Always use standardized names
                            'weighted': weights is not None,
                            'actual_x_column': x_col,    # Track actual column used
                            'actual_y_column': y_col     # Track actual column used
                        }

                        # Store intersection value if calculated
                        if intersection_y is not None:
                            model_data['sl_target_intersection'] = {
                                'x': sl_target,
                                'y': intersection_y,
                                'annotation': intersection_annotation
                            }

                        fitted_models[rel['name']] = model_data

                    except Exception as e:
                        st.error(f"Error fitting {rel['name']}: {str(e)}")

            # Store all models in session state
            if fitted_models:
                if 'wfm_relationship_models' not in st.session_state:
                    st.session_state.wfm_relationship_models = {}
                st.session_state.wfm_relationship_models.update(fitted_models)

                # Also store in the format expected by forecasting section
                fitted_relationships = []
                for name, model in fitted_models.items():
                    # Use the actual metric names from the model data (not parsed from name)
                    fitted_relationships.append({
                        'name': name,
                        'x_metric': model['x_metric'],  # Use actual field names from model
                        'y_metric': model['y_metric'],  # Use actual field names from model
                        'model_type': model['model_type'],
                        'parameters': model['parameters'],
                        'r_squared': model['r_squared'],
                        'quality': 'Good' if model['r_squared'] > 0.7 else 'Fair' if model['r_squared'] > 0.5 else 'Poor'
                    })

                st.session_state.fitted_relationships = fitted_relationships

                # Display summary below the plots
                st.success("✅ All WFM relationship curves fitted and saved for FTE calculations!")

                # Show model equations
                st.subheader("📐 Model Equations")
                eq_cols = st.columns(3)

                for idx, (name, model) in enumerate(fitted_models.items()):
                    with eq_cols[idx]:
                        st.write(f"**{name}**")

                        if model['model_type'] == 'power':
                            a, b, c = model['parameters']
                            st.latex(f"y = {a:.3f} \\cdot x^{{{b:.3f}}} + {c:.3f}")
                        elif model['model_type'] == 'exponential':
                            if len(model['parameters']) == 3:
                                a, b, c = model['parameters']
                                st.latex(f"y = {a:.3f} \\cdot e^{{{b:.3f}x}} + {c:.3f}")
                            else:
                                a, b = model['parameters']
                                st.latex(f"y = {a:.3f} \\cdot e^{{{b:.3f}x}}")
                        else:  # linear
                            if len(model['parameters']) == 2:
                                a, b = model['parameters']
                                st.latex(f"y = {a:.3f}x + {b:.3f}")
                            else:
                                a, b, c = model['parameters']
                                st.latex(f"y = {a:.3f}x + {b:.3f}")

                        st.caption(f"R² = {model['r_squared']:.3f}")
                        if model['weighted']:
                            st.caption("📊 Volume-weighted fit")


def show_forecasting():
    """Forecasting interface - FTE prediction using curve fitting relationships"""
    st.header("🔮 Forecasting")
    st.write("Generate FTE predictions for new forecast data using established relationships.")

    # Check if we have data and fitted relationships
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process historical data first in the 'Data Upload & Processing' section.")
        return

    if 'fitted_relationships' not in st.session_state or not st.session_state.fitted_relationships:
        st.warning("⚠️ Please train curve fitting models first in the 'Historical Analysis' section.")
        return

    # Display available relationships
    st.subheader("📊 Available Relationships")
    relationships_summary = []
    for rel in st.session_state.fitted_relationships:
        relationships_summary.append({
            'Relationship': f"{rel['x_metric']} → {rel['y_metric']}",
            'Model Type': rel['model_type'],
            'R²': f"{rel['r_squared']:.3f}",
            'Quality': rel.get('quality', 'Good')
        })

    if relationships_summary:
        st.dataframe(pd.DataFrame(relationships_summary), use_container_width=True)

    st.subheader("📈 Forecast Data Input")

    # Persist input method selection
    if 'forecast_input_method' not in st.session_state:
        st.session_state.forecast_input_method = "Use Sample Data (for testing)"

    # Input method selection
    input_method = st.radio(
        "Choose forecast data input method:",
        ["Use Sample Data (for testing)", "Manual Entry", "Upload Excel File"],
        index=["Use Sample Data (for testing)", "Manual Entry", "Upload Excel File"].index(st.session_state.forecast_input_method),
        help="Select how you want to provide forecast call volume and AHT data"
    )

    # Update session state
    st.session_state.forecast_input_method = input_method

    forecast_data = None

    if input_method == "Use Sample Data (for testing)":
        st.info("Using realistic sample forecast data based on industry patterns")

        # Create sample data with 20 weeks of realistic forecast data
        sample_data = {
            'Period': [f'Week {i+1}' for i in range(20)],
            'Date': ['10/12/25', '10/19/25', '10/26/25', '11/02/25', '11/09/25', '11/16/25', '11/23/25', '11/30/25',
                    '12/07/25', '12/14/25', '12/21/25', '12/28/25', '01/04/26', '01/11/26', '01/18/26', '01/25/26',
                    '02/01/26', '02/08/26', '02/15/26', '02/22/26'],
            'Calls': [429207, 438126, 484588, 539239, 533150, 549506, 427620, 743309, 471900, 456828, 359031, 638468,
                     718116, 583131, 495801, 573089, 507601, 502630, 473334, 483598],
            'AHT': [690, 689, 683, 679, 684, 685, 675, 678, 683, 682, 667, 693, 670, 680, 679, 677, 678, 677, 679, 677]
        }
        forecast_df = pd.DataFrame(sample_data)

        st.write("**Sample Forecast Data:**")
        st.dataframe(forecast_df, use_container_width=True)
        forecast_data = forecast_df

    elif input_method == "Manual Entry":
        st.write("Enter forecast data in tab or space-separated format:")
        st.code("Date    Calls    AHT\n10/12/25    429207    690\n10/19/25    438126    689", language="text")

        # Use session state to preserve input
        if 'forecast_text_input' not in st.session_state:
            st.session_state.forecast_text_input = ""

        forecast_text = st.text_area(
            "Forecast Data",
            value=st.session_state.forecast_text_input,
            placeholder="10/12/25\t429207\t690\n10/19/25\t438126\t689\n10/26/25\t484588\t683",
            height=150,
            help="Enter one row per time period with Date, Calls, and AHT separated by tabs or spaces",
            key="forecast_textarea"
        )

        # Update session state
        st.session_state.forecast_text_input = forecast_text

        if st.button("📊 Process Forecast Data", type="primary", disabled=not forecast_text.strip()):
            if forecast_text.strip():
                try:
                    lines = [line.strip() for line in forecast_text.strip().split('\n') if line.strip()]
                    parsed_data = []

                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 3:
                            date = parts[0]
                            calls = int(parts[1].replace(',', ''))
                            aht = float(parts[2])
                            parsed_data.append({'Date': date, 'Calls': calls, 'AHT': aht})

                    if parsed_data:
                        forecast_df = pd.DataFrame(parsed_data)
                        st.write("**Parsed Forecast Data:**")
                        st.dataframe(forecast_df, use_container_width=True)
                        forecast_data = forecast_df
                        # Store in session state
                        st.session_state['forecast_data_manual'] = forecast_df
                        st.success(f"✅ Processed {len(parsed_data)} forecast periods")
                    else:
                        st.error("Could not parse the forecast data. Please check the format.")

                except Exception as e:
                    st.error(f"Error parsing forecast data: {str(e)}")

        # Check if we have processed data from session state
        if 'forecast_data_manual' in st.session_state and input_method == "Manual Entry":
            forecast_data = st.session_state['forecast_data_manual']

    elif input_method == "Upload Excel File":
        uploaded_file = st.file_uploader(
            "Upload Forecast Excel File",
            type=['xlsx', 'xls'],
            help="Upload an Excel file containing forecast data with Date, Calls, and AHT columns"
        )

        if uploaded_file:
            try:
                forecast_df = pd.read_excel(uploaded_file)
                st.write("**Uploaded Forecast Data:**")
                st.dataframe(forecast_df.head(), use_container_width=True)

                # Try to identify relevant columns
                call_cols = [col for col in forecast_df.columns if any(term in col.lower()
                           for term in ['call', 'volume', 'offer'])]
                aht_cols = [col for col in forecast_df.columns if 'aht' in col.lower()]

                if call_cols and aht_cols:
                    st.success(f"✅ Detected call column: {call_cols[0]}, AHT column: {aht_cols[0]}")
                    forecast_data = forecast_df
                else:
                    st.warning("⚠️ Could not automatically detect Calls and AHT columns. Please ensure your Excel file has clearly labeled columns.")

            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")

    # FTE Calculation Parameters
    if forecast_data is not None:
        # Store forecast data in session state for use in Service Level Prediction
        st.session_state['forecast_data'] = forecast_data

        st.subheader("⚙️ FTE Calculation Parameters")

        col1, col2 = st.columns(2)

        with col1:
            # Work days per week
            days_per_week = st.number_input("Days per week", min_value=1, max_value=7, value=5)
            # Hardcode hours per day to 8
            hours_per_day = 8

            # Store in session state
            st.session_state['hours_per_day'] = hours_per_day
            st.session_state['days_per_week'] = days_per_week

        with col2:
            # Get target service level from Historical Analysis if available
            if 'service_level_target' in st.session_state:
                target_service_level_pct = st.session_state.service_level_target * 100
                st.info(f"Using Target Service Level from Historical Analysis: {target_service_level_pct:.1f}%")
                target_service_level = st.session_state.service_level_target  # Keep as decimal for curve fitting
            else:
                target_service_level_pct = st.slider("Target Service Level %", min_value=70, max_value=99, value=90, step=1)
                target_service_level = target_service_level_pct / 100  # Convert to decimal for curve fitting

        # Monte Carlo Simulation Parameters
        st.subheader("🎲 Simulation Parameters")

        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:
            num_simulations = st.slider("Simulations", min_value=100, max_value=5000, value=1000, step=100,
                                       help="Number of Monte Carlo simulations to run")
            calls_variance = st.slider("Calls Variance ±%", min_value=0, max_value=20, value=5, step=1,
                                      help="Random variance applied to call volume in simulations")
            aht_variance = st.slider("AHT Variance ±%", min_value=0, max_value=20, value=5, step=1,
                                   help="Random variance applied to AHT in simulations")

        with sim_col2:
            occupancy_variance = st.slider("Occupancy Variance ±%", min_value=0, max_value=20, value=5, step=1,
                                         help="Random variance applied to predicted occupancy in simulations")
            shrinkage_rate = st.slider("Shrinkage Rate", min_value=0, max_value=50, value=30, step=1,
                                      help="Base shrinkage rate for Monte Carlo simulations")

            # Store parameters in session state
            st.session_state['shrinkage_rate'] = shrinkage_rate

        # Calculate FTE Requirements
        if st.button("🚀 Simulate FTE Requirements", type="primary"):
            with st.spinner("Calculating FTE requirements using curve fitting relationships..."):

                # Look for Service Level → Occupancy relationship (preferred)
                service_level_relationship = None
                for rel in st.session_state.fitted_relationships:
                    if rel['x_metric'] in ['service_level'] and rel['y_metric'] in ['occupancy']:
                        service_level_relationship = rel
                        st.info("✅ Using Service Level → Occupancy relationship for predictions")
                        break

                if service_level_relationship is None:
                    st.error("❌ **No Service Level→Occupancy relationship found!** Please ensure you have trained a curve fitting model that relates Service Level to Occupancy in the Historical Analysis section.")
                    st.info("💡 **Tip**: The system requires a Service Level→Occupancy relationship to predict optimal staffing levels from your target service level.")
                    return

                # Get the available time in seconds
                available_seconds_per_day = hours_per_day * 3600
                available_seconds_per_period = available_seconds_per_day * days_per_week

                results = []
                fte_values = []  # For plotting
                period_labels = []  # For plotting

                # Predict occupancy from target service level using Service Level → Occupancy relationship
                rel = service_level_relationship
                popt = rel['parameters']
                model_type = rel['model_type']

                try:
                    if model_type == 'linear' and len(popt) >= 2:
                        predicted_occupancy = max(50, min(95, (popt[0] * target_service_level + popt[1]) * 100))
                    elif model_type == 'polynomial' and len(popt) >= 3:
                        predicted_occupancy = max(50, min(95, (popt[0] * target_service_level**2 + popt[1] * target_service_level + popt[2]) * 100))
                    elif model_type == 'exponential' and len(popt) >= 3:
                        predicted_occupancy = max(50, min(95, (popt[0] * np.exp(popt[1] * target_service_level) + popt[2]) * 100))
                    elif model_type == 'power' and len(popt) >= 3:
                        def power_func(x, a, b, c):
                            return a * np.power(np.abs(x + 0.001), b) + c
                        predicted_occupancy = max(50, min(95, power_func(target_service_level, *popt) * 100))
                    elif model_type == 'logarithmic' and len(popt) >= 2:
                        predicted_occupancy = max(50, min(95, (popt[0] * np.log(target_service_level + 0.001) + popt[1]) * 100))
                    else:
                        st.error(f"❌ Unsupported model type: {model_type}")
                        return

                    relationship_used = f"Service Level→Occupancy ({model_type}, R²={rel['r_squared']:.3f})"

                except Exception as e:
                    st.error(f"❌ Error applying Service Level → Occupancy relationship: {str(e)}")
                    return

                if predicted_occupancy is None:
                    st.error("❌ Failed to predict occupancy from Service Level relationship")
                    return

                st.info(f"🎯 Predicted Occupancy: {predicted_occupancy:.1f}% (from {target_service_level*100:.1f}% Service Level)")

                for idx, row in forecast_data.iterrows():
                    calls = row.get('Calls', 0)
                    base_aht = row.get('AHT', 300)  # Default 5 minutes

                    # Monte Carlo Simulation for FTE calculation
                    # Formula: FTE = (Calls × AHT) / 3600 / Occupancy / (1 - Shrinkage) / Hours_Per_Period

                    sim_results = []
                    hours_per_period = hours_per_day * days_per_week  # Total hours in the period

                    # Run Monte Carlo simulations
                    for sim in range(num_simulations):
                        # Apply random variance to parameters
                        sim_calls = calls * (1 + np.random.uniform(-calls_variance/100, calls_variance/100))
                        sim_aht = base_aht * (1 + np.random.uniform(-aht_variance/100, aht_variance/100))
                        sim_occupancy = predicted_occupancy * (1 + np.random.uniform(-occupancy_variance/100, occupancy_variance/100))
                        sim_occupancy = max(50, min(95, sim_occupancy))  # Keep occupancy in reasonable bounds

                        # Calculate FTE using the correct formula
                        # FTE = (Calls × AHT) / 3600 / Occupancy / (1 - Shrinkage) / Hours_Per_Period
                        sim_fte = (sim_calls * sim_aht) / 3600 / (sim_occupancy/100) / (1 - shrinkage_rate/100) / hours_per_period
                        sim_results.append(sim_fte)

                    # Calculate percentiles (P10, P50, P90)
                    fte_p10 = np.percentile(sim_results, 10)  # Optimistic
                    fte_p50 = np.percentile(sim_results, 50)  # Expected
                    fte_p90 = np.percentile(sim_results, 90)  # Conservative

                    # Use P50 as the required FTE (rounded up)
                    required_fte = max(1, int(np.ceil(fte_p50)))

                    # Calculate resulting occupancy with the P50 FTE
                    total_work_seconds = calls * base_aht
                    actual_occupancy = (total_work_seconds / (required_fte * hours_per_period * 3600 * (1 - shrinkage_rate/100))) * 100 if required_fte > 0 else 0

                    # Calculate meaningful efficiency: calls handled per FTE per hour
                    calls_per_fte_per_hour = calls / (required_fte * hours_per_period) if required_fte > 0 else 0

                    # Round the FTE percentiles for display
                    fte_low = max(1, int(np.ceil(fte_p10)))   # P10 - Optimistic
                    fte_high = max(1, int(np.ceil(fte_p90)))  # P90 - Conservative

                    period_label = row.get('Period', row.get('Date', f"Period {idx+1}"))
                    period_result = {
                        'Period': period_label,
                        'Calls': f"{calls:,}",
                        'AHT (sec)': f"{base_aht:.0f}",
                        'Required FTE': required_fte,
                        'FTE P10 (Optimistic)': fte_low,
                        'FTE P90 (Conservative)': fte_high,
                        'Occupancy %': f"{actual_occupancy:.1f}",
                        'Calls/FTE/Hour': f"{calls_per_fte_per_hour:.1f}"
                    }
                    results.append(period_result)

                    # Store for plotting
                    fte_values.append(required_fte)
                    period_labels.append(period_label)

                # Display results
                st.subheader("📋 FTE Requirements Results")

                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)

                    # Create FTE requirements plot with confidence intervals
                    st.subheader("📊 FTE Requirements by Period")

                    # Extract data for plotting
                    fte_main = [r['Required FTE'] for r in results]
                    fte_low = [r['FTE P10 (Optimistic)'] for r in results]
                    fte_high = [r['FTE P90 (Conservative)'] for r in results]

                    # Create the plot
                    fig_fte = go.Figure()

                    # Add confidence interval area
                    fig_fte.add_trace(go.Scatter(
                        x=period_labels + period_labels[::-1],  # x values for filled area
                        y=fte_high + fte_low[::-1],  # y values for filled area
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='80% Confidence Range',
                        hoverinfo='skip'
                    ))

                    # Add main FTE line
                    fig_fte.add_trace(go.Scatter(
                        x=period_labels,
                        y=fte_main,
                        mode='lines+markers',
                        name='Required FTE',
                        line=dict(color='rgb(0,100,80)', width=3),
                        marker=dict(size=8, color='rgb(0,100,80)'),
                        hovertemplate='Period: %{x}<br>Required FTE: %{y}<extra></extra>'
                    ))

                    # Add low bound line
                    fig_fte.add_trace(go.Scatter(
                        x=period_labels,
                        y=fte_low,
                        mode='lines',
                        name='FTE P10 (Optimistic)',
                        line=dict(color='rgb(0,100,80)', width=1, dash='dash'),
                        hovertemplate='Period: %{x}<br>Low Estimate: %{y}<extra></extra>',
                        showlegend=False
                    ))

                    # Add high bound line
                    fig_fte.add_trace(go.Scatter(
                        x=period_labels,
                        y=fte_high,
                        mode='lines',
                        name='FTE P90 (Conservative)',
                        line=dict(color='rgb(0,100,80)', width=1, dash='dash'),
                        hovertemplate='Period: %{x}<br>High Estimate: %{y}<extra></extra>',
                        showlegend=False
                    ))

                    # Update layout
                    fig_fte.update_layout(
                        title=f"FTE Requirements Forecast (R² = {service_level_relationship['r_squared']:.3f})",
                        xaxis_title="Period",
                        yaxis_title="Required FTE",
                        hovermode='x unified',
                        height=400,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(fig_fte, use_container_width=True)

                    # Add explanation
                    if service_level_relationship and 'r_squared' in service_level_relationship:
                        st.info(f"📈 **Confidence Intervals**: Based on model uncertainty (R² = {service_level_relationship['r_squared']:.3f}). Lower R² values result in wider confidence bands, indicating higher prediction uncertainty.")
                    else:
                        st.info("📈 **Confidence Intervals**: P10 (optimistic), P50 (expected), and P90 (conservative) estimates from Monte Carlo simulation.")

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    total_fte = sum([int(r['Required FTE']) for r in results])
                    avg_occupancy = np.mean([float(r['Occupancy %'].rstrip('%')) for r in results])
                    total_calls = sum([int(r['Calls'].replace(',', '')) for r in results])
                    avg_aht = np.mean([float(r['AHT (sec)']) for r in results])

                    with col1:
                        st.metric("Total FTE Required", total_fte)
                    with col2:
                        st.metric("Average Occupancy", f"{avg_occupancy:.1f}%")
                    with col3:
                        st.metric("Total Calls", f"{total_calls:,}")
                    with col4:
                        st.metric("Average AHT", f"{avg_aht:.0f}s")

                    # Insights and recommendations
                    st.subheader("💡 Insights & Recommendations")

                    insights = []
                    if avg_occupancy > 90:
                        insights.append("⚠️ **High Occupancy Warning**: Average occupancy exceeds 90%. Consider adding buffer capacity.")
                    elif avg_occupancy < 70:
                        insights.append("ℹ️ **Low Occupancy**: Occupancy below 70% may indicate over-staffing opportunity.")
                    else:
                        insights.append("✅ **Optimal Range**: Occupancy levels are within healthy operational range.")

                    if len(st.session_state.fitted_relationships) > 0:
                        insights.append(f"📈 **Model-Based**: Calculations use {len(st.session_state.fitted_relationships)} fitted relationship(s) from historical data.")
                    else:
                        insights.append("📊 **Standard Calculation**: Using base assumptions. Train curve fitting models for more accurate predictions.")

                    for insight in insights:
                        st.write(insight)

                    # Export results
                    if st.button("📥 Export Results to Excel"):
                        # Create Excel file with results
                        from io import BytesIO
                        output = BytesIO()

                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results_df.to_excel(writer, sheet_name='FTE Requirements', index=False)

                            # Add parameters sheet
                            params_df = pd.DataFrame({
                                'Parameter': ['Hours per Day', 'Days per Week', 'Shrinkage %', 'Target Service Level %', 'Simulations', 'Model Used'],
                                'Value': [hours_per_day, days_per_week, shrinkage_rate, target_service_level*100, num_simulations, relationship_used]
                            })
                            params_df.to_excel(writer, sheet_name='Parameters', index=False)

                        st.download_button(
                            label="💾 Download FTE Results.xlsx",
                            data=output.getvalue(),
                            file_name=f"FTE_Requirements_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error("No results calculated. Please check your forecast data.")

    # Help section
    with st.expander("ℹ️ How FTE Calculation Works"):
        st.markdown("""
        **FTE Formula**: `FTE = (Volume × AHT) / (Available Time × Occupancy × (1 - Shrinkage))`

        **Key Components**:
        - **Volume**: Number of calls/contacts to handle
        - **AHT**: Average Handle Time per contact (seconds)
        - **Available Time**: Working hours converted to seconds
        - **Occupancy**: Percentage of time agents are actively handling contacts
        - **Shrinkage**: Percentage for breaks, training, meetings, etc.

        **Curve Fitting Integration**:
        - Uses relationships established in Historical Analysis
        - Applies occupancy→AHT correlations when available
        - Adjusts predictions based on historical patterns

        **Best Practices**:
        - Target occupancy: 80-90% for sustainable performance
        - Include 15-20% shrinkage for breaks and training
        - Round up FTE to ensure service level targets are met
        """)

def show_convergence_analysis():
    """Convergence analysis interface using Service Level Prediction outputs as starting point"""
    st.header("🔄 Convergence Analysis")
    st.write("Advanced convergence algorithms that refine Service Level predictions using dual-loop feedback calculations.")

    st.markdown("""
    ### How Convergence Analysis Works

    This section takes the outputs from **Service Level Prediction** and refines them using sophisticated
    **dual-loop convergence algorithms** that account for feedback loops:

    **🔄 Loop A: Call Volume Feedback**
    - Service Level → Abandon Rate → Retry Calls → Total Calls → New Occupancy → New Service Level

    **🔄 Loop B: Occupancy/AHT Feedback**
    - Occupancy → Adjusted AHT → Final Workload → Final Occupancy → Final Service Level

    These loops iterate until the service level predictions stabilize within tolerance.
    """)

    # ===== COMPREHENSIVE WORKFLOW VALIDATION =====
    st.subheader("🔍 Workflow Prerequisites Validation")

    validation_issues = []

    # Check 1: Service Level Prediction results
    if 'sl_prediction_results' not in st.session_state:
        validation_issues.append("❌ **Service Level Prediction not completed** - Please run the Service Level Prediction section first")
    else:
        st.success("✅ Service Level Prediction results found")

    # Check 2: Historical data and models
    if 'wfm_relationship_models' not in st.session_state or not st.session_state.wfm_relationship_models:
        validation_issues.append("❌ **No trained models found** - Please run Historical Analysis → Generate WFM Relationship Curves first")
    else:
        models = st.session_state.wfm_relationship_models
        st.success(f"✅ Found {len(models)} trained models from Historical Analysis")

        # Check for required relationships
        required_relationships = [
            ('service_level', 'occupancy', 'Service Level → Occupancy'),
            ('occupancy', 'service_level', 'Occupancy → Service Level'),
            ('service_level', 'abandonment_rate', 'Service Level → Abandonment'),
            ('occupancy', 'average_handle_time', 'Occupancy → AHT')
        ]

        found_relationships = []
        missing_relationships = []

        for x_metric, y_metric, display_name in required_relationships:
            found = False
            for model_name, model_data in models.items():
                if (model_data and
                    model_data.get('x_metric') == x_metric and
                    model_data.get('y_metric') == y_metric):
                    found_relationships.append(f"✅ {display_name}")
                    found = True
                    break

            if not found and (x_metric, y_metric) in [('service_level', 'occupancy'), ('occupancy', 'service_level')]:
                missing_relationships.append(f"❌ {display_name} (CRITICAL)")
            elif not found:
                missing_relationships.append(f"⚠️ {display_name} (optional)")

        # Display relationship status
        with st.expander("📊 Available Model Relationships", expanded=bool(missing_relationships)):
            for rel in found_relationships:
                st.write(rel)
            for rel in missing_relationships:
                if "CRITICAL" in rel:
                    st.error(rel)
                else:
                    st.warning(rel)

    # Check 3: Forecast data
    if 'forecast_data' not in st.session_state:
        validation_issues.append("❌ **No forecast data found** - Please complete the Forecasting section first")
    else:
        st.success("✅ Forecast data available")

    # Show validation summary
    if validation_issues:
        st.error("**Convergence Analysis cannot proceed due to missing prerequisites:**")
        for issue in validation_issues:
            st.write(issue)

        st.info("""
        **Required Workflow Order:**
        1. 📁 **Data Upload & Processing** - Load historical data
        2. 📊 **Historical Analysis** - Generate WFM Relationship Curves (especially Service Level ↔ Occupancy)
        3. 📈 **Forecasting** - Load forecast data and calculate FTE requirements
        4. 🎯 **Service Level Prediction** - Generate baseline service level predictions
        5. 🔄 **Convergence Analysis** - Refine predictions with advanced algorithms
        """)
        return

    # Debug model information if all checks pass
    with st.expander("🔧 Debug: Model Information", expanded=False):
        models = st.session_state.wfm_relationship_models
        st.write(f"**Found {len(models)} models in session state:**")
        for model_name, model_data in models.items():
            if model_data:
                st.write(f"- **{model_name}**: {model_data.get('x_metric', 'N/A')} → {model_data.get('y_metric', 'N/A')} "
                        f"({model_data.get('model_type', 'N/A')}, R²={model_data.get('r_squared', 'N/A'):.3f})")
            else:
                st.write(f"- **{model_name}**: ❌ Empty model data")

    # Check for Service Level Prediction results
    if 'sl_prediction_results' not in st.session_state:
        st.error("Critical error: This should not happen after validation above.")
        return

    # Get SL prediction results
    sl_results = st.session_state['sl_prediction_results']
    results_data = sl_results['results_data']
    models = sl_results['models']
    parameters = sl_results['parameters']

    st.success("✅ Service Level Prediction results found! Ready for convergence refinement.")

    # Show summary of starting conditions
    st.subheader("📊 Starting Conditions from Service Level Prediction")

    # Display summary table
    summary_df = pd.DataFrame([
        {
            'Period': r['Period'],
            'FTE': r['Actual FTE'],
            'Calls': r['Call Volume'].replace(',', ''),
            'AHT (s)': r['AHT (sec)'],
            'Occupancy (%)': r['Occupancy %'],
            'Initial SL (%)': r['Predicted SL %'],
            'Abandon (%)': r.get('Abandon %', 'N/A')
        }
        for r in results_data
    ])

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Configuration section (simplified)
    st.subheader("⚙️ Convergence Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        max_iterations = st.slider("Max Iterations", 5, 20, 10,
                                 help="Maximum number of convergence iterations")
        tolerance = st.slider("Tolerance (%)", 0.1, 2.0, 1.0, 0.1,
                            help="Convergence tolerance percentage") / 100

    with col2:
        retry_rate = st.slider("Retry Rate (%)", 10, 50, 30,
                             help="Percentage of abandoned calls that retry") / 100
        damping_factor = st.slider("Damping Factor", 0.5, 1.0, 0.7, 0.1,
                                 help="Damping to prevent oscillation")

    with col3:
        debug_mode = st.checkbox("Debug Mode", help="Show detailed iteration logs")

    # Advanced Configuration
    st.subheader("🎛️ Convergence Mode")
    convergence_mode = st.radio(
        "Choose convergence approach:",
        options=["Conservative (Percentage-based)", "Aggressive (Model-driven)"],
        index=0,
        help="""
        **Conservative**: Applies percentage changes from trained models to starting data. Better for short-term planning with reliable starting conditions.

        **Aggressive**: Uses direct model predictions. Better for long-term planning or when starting data may be outdated.
        """
    )

    use_percentage_based = convergence_mode == "Conservative (Percentage-based)"

    if use_percentage_based:
        max_percentage_change = st.slider(
            "Maximum Change per Iteration (%)",
            10, 50, 30, 5,
            help="Maximum percentage change allowed per iteration in conservative mode"
        ) / 100
    else:
        max_percentage_change = 1.0  # Not used in aggressive mode

    run_all_periods = st.checkbox("Run All Periods", value=True,
                                help="Run convergence for all periods, or select one")

    # Period selection (if not running all)
    selected_periods = []
    if not run_all_periods:
        st.subheader("📅 Period Selection")
        period_options = [r['Period'] for r in results_data]
        selected_period = st.selectbox("Select period to analyze:", period_options)
        selected_periods = [i for i, r in enumerate(results_data) if r['Period'] == selected_period]
    else:
        selected_periods = list(range(len(results_data)))

    # Run convergence analysis
    if st.button("🚀 Run Convergence Analysis", type="primary"):
        try:
            convergence_results = []

            # Create convergence configuration
            config = create_convergence_config('service_level')
            config.max_iterations = max_iterations
            config.tolerance = tolerance
            config.retry_rate = retry_rate
            config.shrinkage = parameters['shrinkage_rate'] / 100
            config.damping_factor = damping_factor
            config.use_percentage_based_changes = use_percentage_based
            config.max_percentage_change = max_percentage_change
            config.debug_enabled = debug_mode
            config.log_iterations = debug_mode

            # ===== ENHANCED MODEL VALIDATION =====
            st.info("🔍 **Validating models before convergence analysis...**")

            # Debug the exact models being passed
            if debug_mode:
                st.write("**Debug: Models being passed to convergence engine:**")
                for model_name, model_data in models.items():
                    if model_data:
                        st.write(f"- {model_name}: x_metric='{model_data.get('x_metric')}', y_metric='{model_data.get('y_metric')}', type='{model_data.get('model_type')}'")
                    else:
                        st.write(f"- {model_name}: ❌ None or empty")

            # Pre-validate critical relationships before creating the predictor
            critical_relationships = [
                ('service_level', 'occupancy'),
                ('occupancy', 'service_level')
            ]

            has_critical_relationship = False
            for x_metric, y_metric in critical_relationships:
                for model_name, model_data in models.items():
                    if (model_data and
                        model_data.get('x_metric') == x_metric and
                        model_data.get('y_metric') == y_metric):
                        has_critical_relationship = True
                        st.success(f"✅ Found critical relationship: {x_metric} → {y_metric}")
                        break

            if not has_critical_relationship:
                st.error("❌ **Critical Error**: No Service Level ↔ Occupancy relationship found in models!")
                st.error("**Available models:**")
                for model_name, model_data in models.items():
                    if model_data:
                        st.write(f"- {model_name}: {model_data.get('x_metric', 'N/A')} → {model_data.get('y_metric', 'N/A')}")
                st.error("""
                **Required Fix**: Please go back to **📊 Historical Analysis** and:
                1. Click "Generate All WFM Relationship Curves"
                2. Ensure the first relationship shows "Service Level → Occupancy"
                3. Then return to **🎯 Service Level Prediction** and re-run it
                4. Finally, try Convergence Analysis again
                """)
                return

            # Create relationship predictor from models
            try:
                relationship_predictor = create_relationship_predictor_from_models(models)
                st.success("✅ Relationship predictor created successfully")
            except ValueError as e:
                st.error(f"❌ **Failed to create relationship predictor**: {str(e)}")
                st.error("""
                **This error indicates that the trained models are not in the expected format.**

                **Troubleshooting Steps:**
                1. Go to **📊 Historical Analysis**
                2. Click "Generate All WFM Relationship Curves" to retrain models
                3. Verify the Service Level → Occupancy relationship is shown
                4. Return to **🎯 Service Level Prediction** and re-run
                5. Try Convergence Analysis again

                **If the problem persists, please check the browser console for additional error details.**
                """)
                return

            # Progress bar for multiple periods
            if len(selected_periods) > 1:
                progress_bar = st.progress(0)
                status_text = st.empty()

            # Run convergence for selected periods
            for idx, period_idx in enumerate(selected_periods):
                period_data = results_data[period_idx]

                if len(selected_periods) > 1:
                    progress = (idx + 1) / len(selected_periods)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {period_data['Period']} ({idx + 1}/{len(selected_periods)})...")

                # Extract values from SL prediction results
                base_calls = float(period_data['Call Volume'].replace(',', ''))
                base_aht = float(period_data['AHT (sec)'])
                planned_fte = period_data['Actual FTE']

                # Create convergence engine
                convergence_engine = ConvergenceEngine(config)

                # Run convergence analysis
                with st.spinner(f"Running convergence for {period_data['Period']}..."):
                    if debug_mode:
                        st.write(f"**Debug - Starting convergence for {period_data['Period']}:**")
                        st.write(f"- Base calls: {base_calls:,.0f}")
                        st.write(f"- Base AHT: {base_aht:.0f}s")
                        st.write(f"- Planned FTE: {planned_fte}")

                    result = convergence_engine.iterate_convergence(
                        base_calls=base_calls,
                        base_aht=base_aht,
                        planned_fte=planned_fte,
                        period="weekly",  # Use the period type from parameters
                        relationship_predictor=relationship_predictor
                    )

                # Store period information with result
                result.period_name = period_data['Period']
                result.initial_sl_prediction = float(period_data['Predicted SL %'].replace('%', '')) / 100

                convergence_results.append(result)

            # Clear progress indicators
            if len(selected_periods) > 1:
                progress_bar.empty()
                status_text.empty()

            # Store results
            st.session_state['convergence_results'] = convergence_results

            # Display results
            st.success(f"✅ Convergence analysis completed for {len(convergence_results)} period(s)")

            # Results comparison table
            st.subheader("📈 Convergence Results Comparison")

            comparison_data = []
            for result in convergence_results:
                comparison_data.append({
                    'Period': result.period_name,
                    'Initial SL (%)': f"{result.initial_sl_prediction:.1%}",
                    'Converged SL (%)': f"{result.final_service_level:.1%}",
                    'SL Change (%)': f"{(result.final_service_level - result.initial_sl_prediction):.1%}",
                    'Final Abandon (%)': f"{result.final_abandon_rate:.2%}",
                    'Final Occupancy (%)': f"{result.final_occupancy:.1%}",
                    'Final Calls': f"{result.final_calls:.0f}",
                    'Final AHT (s)': f"{result.final_aht:.0f}",
                    'Iterations': result.iterations,
                    'Converged': '✅' if result.converged else '❌'
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            # Detailed results for each period
            for i, result in enumerate(convergence_results):
                with st.expander(f"📊 Detailed Results: {result.period_name}", expanded=len(convergence_results) == 1):
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        delta_sl = result.final_service_level - result.initial_sl_prediction
                        st.metric("Final Service Level", f"{result.final_service_level:.1%}",
                                 delta=f"{delta_sl:.1%}")

                    with col2:
                        st.metric("Final Abandon Rate", f"{result.final_abandon_rate:.2%}")

                    with col3:
                        st.metric("Final Occupancy", f"{result.final_occupancy:.1%}")

                    with col4:
                        delta_calls = result.final_calls - result.base_calls
                        st.metric("Final Calls", f"{result.final_calls:.0f}",
                                 delta=f"{delta_calls:.0f}")

                    with col5:
                        delta_aht = result.final_aht - result.base_aht
                        st.metric("Final AHT", f"{result.final_aht:.0f}s",
                                 delta=f"{delta_aht:.0f}s")

                    # Quality indicators
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        stability_color = {"excellent": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}
                        st.write(f"**Stability**: {stability_color.get(result.stability, '⚪')} {result.stability.title()}")

                    with col2:
                        st.write(f"**Confidence**: {result.confidence:.1%}")

                    with col3:
                        st.write(f"**Final Error**: {result.final_error:.3%}")

                    # Convergence progression chart
                    if result.iteration_history:
                        # Create iteration data for plotting
                        iterations = [iter_data.iteration for iter_data in result.iteration_history]
                        service_levels = [iter_data.service_level * 100 for iter_data in result.iteration_history]
                        convergence_errors = [iter_data.convergence_error * 100 for iter_data in result.iteration_history]

                        # Add initial SL prediction as starting point
                        iterations = [0] + iterations
                        service_levels = [result.initial_sl_prediction * 100] + service_levels

                        # Create convergence chart
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=iterations,
                            y=service_levels,
                            mode='lines+markers',
                            name='Service Level (%)',
                            line=dict(color='blue', width=2),
                            marker=dict(size=6)
                        ))

                        # Highlight the improvement from initial prediction
                        fig.add_annotation(
                            x=0, y=result.initial_sl_prediction * 100,
                            text="Initial SL Prediction",
                            showarrow=True, arrowhead=2, arrowcolor="red"
                        )

                        fig.update_layout(
                            title=f"Convergence Progression: {result.period_name}",
                            xaxis_title="Iteration",
                            yaxis_title="Service Level (%)",
                            hovermode='x unified',
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Warnings
                    if result.warnings:
                        st.subheader("⚠️ Analysis Warnings")
                        for warning in result.warnings:
                            st.warning(warning)

            # Summary insights
            st.subheader("💡 Summary Insights")

            # Calculate average improvements
            avg_sl_change = np.mean([r.final_service_level - r.initial_sl_prediction for r in convergence_results])
            max_sl_change = max([abs(r.final_service_level - r.initial_sl_prediction) for r in convergence_results])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average SL Change", f"{avg_sl_change:.1%}")

            with col2:
                st.metric("Largest SL Change", f"{max_sl_change:.1%}")

            with col3:
                converged_count = sum(1 for r in convergence_results if r.converged)
                st.metric("Convergence Rate", f"{converged_count}/{len(convergence_results)}")

            if abs(avg_sl_change) > 0.02:  # More than 2% change
                if avg_sl_change > 0:
                    st.success("✅ **Convergence analysis shows service levels are higher than initial predictions**, indicating the feedback loops have a positive effect.")
                else:
                    st.warning("⚠️ **Convergence analysis shows service levels are lower than initial predictions**, indicating negative feedback from abandon/retry cycles and occupancy stress.")
            else:
                st.info("ℹ️ **Convergence analysis shows minimal change from initial predictions**, suggesting the basic service level calculations were already quite accurate.")

        except Exception as e:
            st.error(f"❌ Convergence analysis failed: {str(e)}")
            if debug_mode:
                st.exception(e)

    # Information section
    st.subheader("📚 Understanding Convergence Analysis")
    st.markdown("""
    **How This Improves Service Level Predictions:**
    - **Accounts for Feedback Loops**: Models how service levels affect abandon rates, creating retry calls
    - **Occupancy Stress Effects**: Models how high occupancy increases AHT due to agent stress
    - **Iterative Refinement**: Continues until predictions stabilize within tolerance
    - **Realistic Scenarios**: Provides more accurate predictions for critical capacity planning

    **When Convergence Makes a Difference:**
    - High abandon rates that generate significant retry volume
    - High occupancy scenarios where stress affects agent performance
    - Contact centers with strong correlations between occupancy and AHT
    - Critical planning scenarios requiring maximum accuracy
    """)

def show_settings():
    """Application settings interface"""
    st.header("� Settings")
    st.write("Configure application assumptions and parameters.")

    st.info("=� Settings functionality will be implemented here")

if __name__ == "__main__":
    main()