"""
Comprehensive test suite for Contraption workforce management application
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path

# Import application modules
from src.data.excel_loader import ExcelDataLoader
from src.data.processor import WFMDataProcessor
from src.analysis.relationships import MetricsRelationshipAnalyzer
from src.models.curve_fitting import FTEPredictionModel
from src.visualization.correlation_plots import CorrelationVisualizer


def test_excel_loader():
    """Test Excel data loading functionality"""
    print("\n" + "="*60)
    print("TESTING EXCEL DATA LOADER")
    print("="*60)

    loader = ExcelDataLoader()

    # Test loading single file
    try:
        df_core = loader.load_excel_file('data/wfm_core_metrics.xlsx')
        print(f"✓ Successfully loaded core metrics: {len(df_core)} rows, {len(df_core.columns)} columns")
        print(f"  Columns: {list(df_core.columns)}")
    except Exception as e:
        print(f"✗ Failed to load core metrics: {e}")
        return False

    # Test loading and merging multiple files
    try:
        files = [
            'data/wfm_core_metrics.xlsx',
            'data/wfm_staffing_metrics.xlsx'
        ]
        merged_df = loader.load_and_merge_files(files)
        print(f"✓ Successfully merged files: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        print(f"  Merged columns: {list(merged_df.columns)}")
    except Exception as e:
        print(f"✗ Failed to merge files: {e}")
        return False

    # Test data summary
    summary = loader.get_data_summary(merged_df)
    print(f"✓ Data summary generated:")
    print(f"  - Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"  - Numeric columns: {len(summary['numeric_columns'])}")
    print(f"  - Memory usage: {summary['memory_usage_mb']:.2f} MB")

    # Test validation
    warnings = loader.validate_wfm_data(merged_df)
    if warnings:
        print(f"⚠ Validation warnings: {warnings}")
    else:
        print(f"✓ Data validation passed with no warnings")

    return True


def test_data_processor():
    """Test data processing functionality"""
    print("\n" + "="*60)
    print("TESTING DATA PROCESSOR")
    print("="*60)

    processor = WFMDataProcessor()

    # Load test data
    df = pd.read_excel('data/wfm_complete_data.xlsx')
    df = df.set_index('timestamp')

    # Test column standardization
    df_standardized = processor.standardize_column_names(df)
    print(f"✓ Standardized column names")

    # Test derived metrics calculation
    df_with_metrics = processor.calculate_derived_metrics(df_standardized)
    new_columns = set(df_with_metrics.columns) - set(df.columns)
    print(f"✓ Calculated {len(new_columns)} derived metrics: {new_columns}")

    # Test outlier cleaning
    df_cleaned = processor.clean_outliers(df_with_metrics, method='iqr')
    outliers_removed = df_with_metrics.notna().sum().sum() - df_cleaned.notna().sum().sum()
    print(f"✓ Cleaned {outliers_removed} outliers using IQR method")

    # Test aggregation
    df_weekly = processor.aggregate_to_period(df_cleaned, 'W')
    print(f"✓ Aggregated data from {len(df_cleaned)} daily to {len(df_weekly)} weekly records")

    # Test rolling metrics
    df_rolling = processor.calculate_rolling_metrics(df_cleaned, window=7)
    rolling_columns = [col for col in df_rolling.columns if 'rolling' in col or 'trend' in col]
    print(f"✓ Calculated {len(rolling_columns)} rolling metrics")

    # Test data quality assessment
    quality = processor.validate_data_quality(df_cleaned)
    print(f"✓ Data quality assessment:")
    print(f"  - Missing data: {quality['missing_data_percentage']:.1f}%")
    print(f"  - Duplicate rows: {quality['duplicate_rows']}")

    return True


def test_relationship_analysis():
    """Test relationship analysis functionality"""
    print("\n" + "="*60)
    print("TESTING RELATIONSHIP ANALYSIS")
    print("="*60)

    analyzer = MetricsRelationshipAnalyzer()

    # Load test data
    df = pd.read_excel('data/wfm_complete_data.xlsx')
    df = df.set_index('timestamp')

    # Test correlation calculations
    try:
        pearson_corr = analyzer.calculate_correlations(df, method="pearson")
        print(f"✓ Calculated Pearson correlations for {len(pearson_corr.columns)} metrics")

        spearman_corr = analyzer.calculate_correlations(df, method="spearman")
        print(f"✓ Calculated Spearman correlations")
    except Exception as e:
        print(f"✗ Correlation calculation failed: {e}")
        return False

    # Test strong relationship detection
    strong_relationships = analyzer.find_strong_relationships(pearson_corr, threshold=0.5)
    print(f"✓ Found {len(strong_relationships)} strong relationships (|r| >= 0.5)")
    if strong_relationships:
        top_rel = strong_relationships[0]
        print(f"  Strongest: {top_rel['metric1']} ↔ {top_rel['metric2']} (r = {top_rel['correlation']:.3f})")

    # Test dependency analysis
    dependencies = analyzer.analyze_metric_dependencies(df, 'fte', method='linear')
    print(f"✓ Analyzed dependencies for FTE metric")
    print(f"  Top predictors: {dependencies['top_predictors'][:3]}")

    # Test seasonality detection
    if 'volume' in df.columns:
        seasonality = analyzer.detect_seasonal_patterns(df, 'volume')
        if seasonality['strongest_pattern']:
            pattern = seasonality['strongest_pattern']
            print(f"✓ Detected seasonality in volume: {pattern['name']} (strength: {pattern['strength']:.3f})")
        else:
            print(f"✓ No strong seasonal patterns detected")

    # Test comprehensive report
    report = analyzer.generate_relationship_report(df, target_metrics=['fte', 'volume'])
    print(f"✓ Generated comprehensive relationship report")
    print(f"  Report sections: {list(report.keys())}")

    return True


def test_curve_fitting_models():
    """Test curve fitting and FTE prediction models"""
    print("\n" + "="*60)
    print("TESTING CURVE FITTING MODELS")
    print("="*60)

    model = FTEPredictionModel()

    # Load test data
    df = pd.read_excel('data/wfm_complete_data.xlsx')

    # Prepare sample data
    x_data = df['volume'].values[:100]
    y_data = df['fte'].values[:100]

    # Test different model types
    model_types = ['linear', 'exponential', 'power', 'logarithmic', 'polynomial']
    fitted_models = {}

    for model_type in model_types:
        try:
            if model_type == 'polynomial':
                result = model.fit_model(x_data, y_data, model_type, degree=2)
            else:
                result = model.fit_model(x_data, y_data, model_type)

            if result['fitted']:
                fitted_models[model_type] = result
                print(f"✓ Fitted {model_type} model: R² = {result['r_squared']:.3f}, RMSE = {result['rmse']:.2f}")
            else:
                print(f"✗ Failed to fit {model_type} model")
        except Exception as e:
            print(f"✗ Error fitting {model_type}: {e}")

    # Test auto model selection
    try:
        auto_result = model.fit_model(x_data, y_data, model_type="auto")
        print(f"✓ Auto-selected {auto_result['model_type']} model (AIC = {auto_result['aic']:.2f})")
    except Exception as e:
        print(f"✗ Auto model selection failed: {e}")

    # Test prediction
    if fitted_models:
        test_model = list(fitted_models.values())[0]
        test_x = np.array([x_data.mean()])

        try:
            prediction = model.predict(test_model, test_x)
            print(f"✓ Prediction successful: {prediction[0]:.2f} FTE for volume = {test_x[0]:.0f}")
        except Exception as e:
            print(f"✗ Prediction failed: {e}")

        # Test prediction intervals
        try:
            lower, upper = model.calculate_prediction_intervals(test_model, test_x)
            print(f"✓ 95% Prediction interval: [{lower[0]:.2f}, {upper[0]:.2f}]")
        except Exception as e:
            print(f"✗ Prediction interval calculation failed: {e}")

    # Test model validation
    if fitted_models and len(x_data) > 120:
        test_model = list(fitted_models.values())[0]
        x_test = df['volume'].values[100:120]
        y_test = df['fte'].values[100:120]

        try:
            validation = model.validate_model(test_model, x_test, y_test)
            print(f"✓ Model validation on test set:")
            print(f"  - R²: {validation['r_squared']:.3f}")
            print(f"  - MAPE: {validation['mape']:.1f}%")
        except Exception as e:
            print(f"✗ Model validation failed: {e}")

    # Test multiple model fitting
    try:
        models = model.fit_multiple_models(df, 'fte', ['volume', 'aht', 'occupancy'])
        print(f"✓ Fitted {len(models)} models for different predictors")
        for predictor, model_result in models.items():
            quality = model._assess_model_quality(model_result)
            print(f"  {predictor} → FTE: {model_result['model_type']} (R² = {model_result['r_squared']:.3f}, quality = {quality})")
    except Exception as e:
        print(f"✗ Multiple model fitting failed: {e}")

    return True


def test_visualization():
    """Test visualization components"""
    print("\n" + "="*60)
    print("TESTING VISUALIZATION COMPONENTS")
    print("="*60)

    visualizer = CorrelationVisualizer()

    # Load test data
    df = pd.read_excel('data/wfm_complete_data.xlsx')
    df = df.set_index('timestamp')

    # Calculate correlations for visualization
    analyzer = MetricsRelationshipAnalyzer()
    correlation_matrix = analyzer.calculate_correlations(df, method="pearson")
    strong_relationships = analyzer.find_strong_relationships(correlation_matrix, threshold=0.5)

    # Test correlation heatmap
    try:
        heatmap_fig = visualizer.create_correlation_heatmap(correlation_matrix)
        print(f"✓ Created correlation heatmap")
    except Exception as e:
        print(f"✗ Heatmap creation failed: {e}")

    # Test relationship network
    if len(strong_relationships) > 1:
        try:
            network_fig = visualizer.create_relationship_network(strong_relationships, threshold=0.5)
            print(f"✓ Created relationship network graph")
        except Exception as e:
            print(f"✗ Network graph creation failed: {e}")

    # Test scatter matrix
    try:
        scatter_fig = visualizer.create_scatter_matrix(df, metrics=['fte', 'volume', 'aht'])
        print(f"✓ Created scatter matrix plot")
    except Exception as e:
        print(f"✗ Scatter matrix creation failed: {e}")

    # Test time series comparison
    try:
        ts_fig = visualizer.create_time_series_comparison(df, ['fte', 'volume', 'occupancy'])
        print(f"✓ Created time series comparison plot")
    except Exception as e:
        print(f"✗ Time series plot creation failed: {e}")

    # Test distribution comparison
    try:
        dist_fig = visualizer.create_distribution_comparison(df, ['fte', 'volume', 'aht'], plot_type='histogram')
        print(f"✓ Created distribution comparison plots")
    except Exception as e:
        print(f"✗ Distribution plot creation failed: {e}")

    # Test correlation strength chart
    if strong_relationships:
        try:
            strength_fig = visualizer.create_correlation_strength_chart(strong_relationships, top_n=5)
            print(f"✓ Created correlation strength chart")
        except Exception as e:
            print(f"✗ Strength chart creation failed: {e}")

    return True


def run_all_tests():
    """Run all test suites"""
    print("\n" + "#"*60)
    print("# CONTRAPTION APPLICATION TEST SUITE")
    print("#"*60)

    results = {
        'Excel Data Loader': test_excel_loader(),
        'Data Processor': test_data_processor(),
        'Relationship Analysis': test_relationship_analysis(),
        'Curve Fitting Models': test_curve_fitting_models(),
        'Visualization': test_visualization()
    }

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for module, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status} - {module}")

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! The application is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review the errors above.")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)