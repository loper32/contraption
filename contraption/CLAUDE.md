# Contraption - Workforce Management Analytics Platform

## Project Overview
A comprehensive workforce management analytics application that processes Excel data to create predictive FTE models. The application:

1. **Data Integration**: Loads and merges Excel files based on timestamps
2. **Historical Analysis**: Establishes relationships between workforce metrics
3. **Predictive Modeling**: Uses curve fitting to predict FTE requirements for capacity planning
4. **Forecast Processing**: Applies learned relationships to new forecast data
5. **Convergence Analysis**: Determines expected service levels based on planned staffing
6. **Visualization**: Provides interactive plots and tables for analysis

Target users: Workforce management professionals, capacity planners, operations analysts

## Architecture & Structure
```
src/
├── data/           # Data processing and Excel file handling
├── models/         # Curve fitting and predictive models
├── analysis/       # Historical relationship analysis
├── forecasting/    # Forecast processing and FTE calculation
├── convergence/    # Service level convergence algorithms
├── visualization/  # Plots and dashboard components
├── config/         # Application settings and assumptions
└── main.py         # Main application entry point

tests/              # Test files
docs/               # Documentation
requirements/       # Dependencies
config/             # Configuration files
data/               # Sample data and templates
```

## Development Workflow
```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements/dev.txt

# Run tests
pytest
python -m pytest tests/

# Code quality
black .
flake8 .
mypy .

# Run application
python -m src.main
```

## Key Files & Locations
- `pyproject.toml` / `setup.py` - Project configuration
- `requirements/` - Dependencies (prod, dev, test)
- `src/` - Main source code
- `tests/` - Test suite
- `.env` - Environment variables (not committed)

## Coding Standards
- Use Black for formatting
- Follow PEP 8 style guide
- Type hints for all public functions
- Docstrings in Google/NumPy style
- Max line length: 88 characters

## Current Status
**Latest Version**: Fully functional workforce management analytics platform with advanced curve fitting capabilities.

### Recent Major Improvements (September 2025)
1. **Weighted Auto-Selection Curve Fitting**: Implemented comprehensive algorithm that automatically selects the best curve type (polynomial, exponential, power, logarithmic, linear) based on volume-weighted R² scores for optimal relationship modeling.

2. **Service Level → Occupancy Prediction Fix**: Resolved critical prediction issues where occupancy was incorrectly showing ~50% instead of realistic ~89% values. Fixed decimal/percentage conversion and model application logic.

3. **Target Intersection Visualization**: Enhanced plotting to properly position target markers (stars) on fitted curves using correct model equations rather than linear interpolation.

4. **Simplified Data Presentation**: Streamlined occupancy column naming from confusing multiple columns to single "Occupancy %" column for better user experience.

5. **Volume-Weighted Relationships**: Maintained proper volume weighting throughout curve fitting process to ensure larger volume periods have appropriate influence on model parameters.

6. **Service Level Prediction Feature**: Implemented comprehensive reverse prediction module (`src/service_level_prediction.py`) that calculates expected service levels based on actual FTE staffing plans and forecast workload.

7. **Input Method Persistence**: Fixed session state management to remember user input method selections and pasted data when navigating between Forecasting and Service Level Prediction sections.

### Key Technical Features
- **Monte Carlo FTE Simulation**: Configurable variance parameters for realistic staffing projections
- **Multi-Model Curve Fitting**: Automatic selection from 5+ curve types with weighted optimization
- **Interactive Visualizations**: Plotly-based charts with proper target positioning and model equations
- **Excel Data Integration**: Robust file processing with timestamp-based merging
- **Real-time Forecasting**: Apply learned relationships to new forecast data for capacity planning

## Dependencies & Tools
- **Core**: Streamlit (UI), pandas (data manipulation), numpy (numerical computing)
- **Data Processing**: openpyxl (Excel files), xlsxwriter (Excel output)
- **Analytics**: scipy (curve fitting), scikit-learn (modeling), statsmodels (statistics)
- **Visualization**: plotly (interactive plots), matplotlib (static plots)
- **Testing**: pytest, coverage
- **Quality**: black, flake8, mypy
- **Development**: jupyter (analysis notebooks), ipykernel (notebook support)

## Common Tasks
- Process and merge Excel data files by timestamp
- Develop curve fitting algorithms for metric relationships
- Build forecasting models for FTE requirements
- Implement convergence algorithms for service level prediction
- Create interactive visualizations and dashboards
- Add new workforce management metrics and relationships
- Optimize performance for large datasets
- Write tests for data processing and modeling functions

## Technical Implementation Details

### Weighted Auto-Selection Curve Fitting Algorithm
The core innovation in the latest version is the weighted auto-selection algorithm (`main.py:600-750`):

```python
# Algorithm tries multiple model types and selects best weighted R²
models_to_try = ['polynomial', 'exponential', 'power', 'logarithmic', 'linear']

for test_model_type in models_to_try:
    try:
        # Volume-weighted curve fitting using scipy.optimize.curve_fit
        test_popt, _ = curve_fit(model_func, x_data, y_data, sigma=1/np.sqrt(weights))

        # Calculate weighted R² score
        weighted_ss_res = np.sum(weights * (y_data - y_pred)**2)
        weighted_ss_tot = np.sum(weights * (y_data - weighted_mean)**2)
        test_r_squared = 1 - (weighted_ss_res / weighted_ss_tot)

        # Keep best model
        if test_r_squared > best_r_squared:
            best_model = {'type': test_model_type, 'popt': test_popt, ...}
    except:
        continue
```

### Key Files & Functions
- **`main.py:600-750`**: Weighted auto-selection curve fitting algorithm
- **`main.py:1240-1265`**: Service Level → Occupancy prediction with proper model application
- **`main.py:1146-1159`**: Forecast input method persistence implementation
- **`main.py:850-950`**: Target intersection calculation and visualization
- **`main.py:1100-1200`**: Volume-weighted relationship processing
- **`main.py:300-400`**: Excel data processing and timestamp merging
- **`src/service_level_prediction.py`**: Complete reverse prediction module (FTE → Service Level)
- **`src/service_level_prediction.py:60-73`**: FTE input method persistence implementation

## Notes & Context
- Virtual environment: `contraption/venv/` (to be created)
- Primary workflow: Excel upload → data processing → model training → forecasting → service level prediction → visualization
- Key domain concepts: FTE (Full-Time Equivalent), Service Level, AHT (Average Handle Time), Occupancy
- Data sources: Historical Excel files with timestamp-based metrics
- Output: Predictive models, FTE recommendations, service level projections
- Performance considerations: Large Excel files, real-time convergence calculations