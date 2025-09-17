# Contraption - Workforce Management Analytics Platform

A comprehensive workforce management analytics application that processes Excel data to create predictive FTE models.

## Features

- **Data Integration**: Load and merge Excel files based on timestamps
- **Historical Analysis**: Establish relationships between workforce metrics
- **Predictive Modeling**: Use curve fitting to predict FTE requirements
- **Forecast Processing**: Apply learned relationships to new forecast data
- **Convergence Analysis**: Determine expected service levels based on planned staffing
- **Visualization**: Interactive plots and tables for analysis

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or create the project directory**
   ```bash
   cd contraption
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements/requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   streamlit run src/main.py
   ```

The application will open in your web browser at `http://localhost:8501`

## Development

### Setup Development Environment
```bash
pip install -r requirements/dev.txt
```

### Run Tests
```bash
pytest
```

### Code Quality
```bash
black .          # Format code
flake8 .         # Check style
mypy .           # Type checking
```

## Project Structure

```
contraption/
├── src/
│   ├── data/           # Data processing and Excel handling
│   ├── models/         # Curve fitting and predictive models
│   ├── analysis/       # Historical relationship analysis
│   ├── forecasting/    # Forecast processing and FTE calculation
│   ├── convergence/    # Service level convergence algorithms
│   ├── visualization/  # Plots and dashboard components
│   ├── config/         # Application settings and assumptions
│   └── main.py         # Main application entry point
├── tests/              # Test files
├── data/               # Sample data and templates
├── docs/               # Documentation
└── requirements/       # Dependencies
```

## Usage

1. **Upload Historical Data**: Start by uploading Excel files with historical workforce metrics
2. **Analyze Relationships**: Explore correlations between different metrics
3. **Train Models**: Use curve fitting to establish predictive relationships
4. **Generate Forecasts**: Apply models to new forecast data for FTE predictions
5. **Predict Service Levels**: Run convergence analysis for staffing scenarios

## Target Users

- Workforce management professionals
- Capacity planners
- Operations analysts

## License

[Add your license here]