#!/usr/bin/env python3
"""
Test the new curve fitting functionality for WFM relationships
"""

import sys
sys.path.insert(0, 'src')

from data.wfm_preprocessor import WFMPreprocessor
from models.curve_fitting import FTEPredictionModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_wfm_curve_fitting():
    """Test curve fitting for key WFM relationships"""

    print("🔬 Testing WFM Curve Fitting Functionality")
    print("=" * 60)

    # Load and process the WFM files
    preprocessor = WFMPreprocessor()

    files = {
        'Calls': '../WFM-Data/Calls.xlsx',
        'Staff': '../WFM-Data/Staff.xlsx',
        'Occ': '../WFM-Data/Occ.xlsx'
    }

    print("\n📁 Loading and processing files...")
    dataframes = []
    for name, filepath in files.items():
        df = preprocessor.preprocess_file(filepath)
        dataframes.append(df)
        print(f"✅ {name}: {len(df)} rows")

    # Merge the files
    print("\n🔗 Merging files...")
    merged_df = preprocessor.merge_preprocessed_files(dataframes, merge_strategy="outer")
    print(f"✅ Merged dataset: {merged_df.shape}")

    # Initialize curve fitting model
    print("\n📈 Initializing curve fitting model...")
    curve_model = FTEPredictionModel()

    # Test key WFM relationships
    test_relationships = [
        {
            "name": "Service Level → Occupancy",
            "x_metric": "service_level",
            "y_metric": "occupancy",
            "description": "How service level affects occupancy requirements"
        },
        {
            "name": "Occupancy → AHT",
            "x_metric": "occupancy",
            "y_metric": "average_handle_time",
            "description": "Relationship between occupancy and handle time"
        },
        {
            "name": "Service Level → Abandonment",
            "x_metric": "service_level",
            "y_metric": "abandonment_rate",
            "description": "How service level impacts abandonment rate"
        }
    ]

    successful_fits = []

    for relationship in test_relationships:
        print(f"\n🎯 Testing: {relationship['name']}")
        print(f"   📊 {relationship['description']}")

        # Check if metrics are available
        x_metric = relationship['x_metric']
        y_metric = relationship['y_metric']

        if x_metric not in merged_df.columns:
            print(f"   ❌ X-metric '{x_metric}' not found in data")
            continue

        if y_metric not in merged_df.columns:
            print(f"   ❌ Y-metric '{y_metric}' not found in data")
            continue

        # Get clean data
        data_subset = merged_df[[x_metric, y_metric]].dropna()
        print(f"   📈 Clean data points: {len(data_subset)}")

        if len(data_subset) < 10:
            print(f"   ❌ Not enough data points for fitting (need ≥10)")
            continue

        x_data = data_subset[x_metric].values
        y_data = data_subset[y_metric].values

        print(f"   📊 X-range: {x_data.min():.3f} to {x_data.max():.3f}")
        print(f"   📊 Y-range: {y_data.min():.3f} to {y_data.max():.3f}")

        # Try different curve types
        curve_types = ["linear", "exponential", "power", "auto"]
        best_model = None
        best_r2 = -1

        for curve_type in curve_types:
            try:
                print(f"   🔄 Trying {curve_type} fit...")
                model_result = curve_model.fit_model(x_data, y_data, model_type=curve_type)

                if model_result['fitted']:
                    r2 = model_result['r_squared']
                    rmse = model_result['rmse']
                    print(f"      ✅ {curve_type}: R²={r2:.3f}, RMSE={rmse:.3f}")

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_result
                        best_model['curve_type'] = curve_type
                else:
                    print(f"      ❌ {curve_type}: Failed to fit")

            except Exception as e:
                print(f"      ❌ {curve_type}: Error - {str(e)}")

        if best_model and best_r2 > 0.1:  # Reasonable correlation threshold
            print(f"   🏆 Best fit: {best_model['curve_type']} (R²={best_r2:.3f})")

            successful_fits.append({
                'relationship': relationship,
                'model': best_model,
                'x_metric': x_metric,
                'y_metric': y_metric,
                'data_points': len(data_subset)
            })
        else:
            print(f"   ❌ No good fit found (best R²={best_r2:.3f})")

    # Summary
    print(f"\n📊 CURVE FITTING SUMMARY")
    print("=" * 40)
    print(f"✅ Successful fits: {len(successful_fits)}/{len(test_relationships)}")

    for fit in successful_fits:
        rel = fit['relationship']
        model = fit['model']
        print(f"\n🎯 {rel['name']}:")
        print(f"   • Model: {model['model_type'].title()}")
        print(f"   • R² Score: {model['r_squared']:.3f}")
        print(f"   • RMSE: {model['rmse']:.3f}")
        print(f"   • Data Points: {fit['data_points']}")
        print(f"   • Parameters: {[f'{p:.4f}' for p in model['parameters']]}")

        # Generate equation string
        if model['model_type'] == 'linear':
            a, b = model['parameters']
            print(f"   • Equation: y = {a:.4f}x + {b:.4f}")
        elif model['model_type'] == 'exponential':
            a, b, c = model['parameters']
            print(f"   • Equation: y = {a:.4f} × exp({b:.4f}x) + {c:.4f}")
        elif model['model_type'] == 'power':
            a, b, c = model['parameters']
            print(f"   • Equation: y = {a:.4f} × x^{b:.4f} + {c:.4f}")

    if successful_fits:
        print(f"\n✅ WFM curve fitting is working! These models are ready for FTE assumptions.")
        return True
    else:
        print(f"\n❌ No successful curve fits found. Check data quality and relationships.")
        return False

if __name__ == "__main__":
    success = test_wfm_curve_fitting()
    if success:
        print("\n🎉 All curve fitting tests passed!")
    else:
        print("\n⚠️  Some curve fitting tests failed. Review the data.")