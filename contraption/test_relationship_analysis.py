#!/usr/bin/env python3
"""
Test the relationship analysis functionality with real WFM data
"""

import sys
sys.path.insert(0, 'src')

from data.wfm_preprocessor import WFMPreprocessor
from analysis.relationships import MetricsRelationshipAnalyzer
import pandas as pd
import numpy as np

def test_relationship_analysis():
    """Test correlation analysis with merged WFM data"""

    print("🔍 Testing Relationship Analysis with Real WFM Data")
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
    print(f"📊 Available columns: {list(merged_df.columns)}")

    # Initialize relationship analyzer
    print("\n📈 Running relationship analysis...")
    analyzer = MetricsRelationshipAnalyzer()

    # Get numeric columns for analysis
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"📊 Numeric metrics found: {numeric_cols}")

    if len(numeric_cols) < 2:
        print("❌ Not enough numeric columns for analysis")
        return

    # Calculate correlations
    try:
        print("\n🔄 Calculating correlations...")
        pearson_corr = analyzer.calculate_correlations(merged_df, method="pearson")
        print("✅ Pearson correlations calculated")
        print("\n📊 Correlation Matrix:")
        print(pearson_corr.round(3))

        # Find strong relationships
        print("\n🔍 Finding strong relationships (threshold: 0.5)...")
        strong_relationships = analyzer.find_strong_relationships(pearson_corr, threshold=0.5)

        if strong_relationships:
            print(f"✅ Found {len(strong_relationships)} strong relationships:")
            for rel in strong_relationships[:5]:  # Show top 5
                print(f"  • {rel['metric1']} ↔ {rel['metric2']}: {rel['correlation']:.3f} ({rel['direction']}, {rel['strength']})")
        else:
            print("📊 No strong relationships found at 0.5 threshold")

        # Try lower threshold
        moderate_relationships = analyzer.find_strong_relationships(pearson_corr, threshold=0.3)
        print(f"\n📊 Moderate relationships (threshold: 0.3): {len(moderate_relationships)}")
        for rel in moderate_relationships[:3]:  # Show top 3
            print(f"  • {rel['metric1']} ↔ {rel['metric2']}: {rel['correlation']:.3f} ({rel['direction']}, {rel['strength']})")

    except Exception as e:
        print(f"❌ Error in correlation analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    # Test dependency analysis for key metrics
    print("\n🎯 Testing dependency analysis...")

    # Focus on key WFM metrics
    key_metrics = ['service_level', 'occupancy', 'offered', 'average_handle_time']
    available_key_metrics = [m for m in key_metrics if m in merged_df.columns]

    if available_key_metrics:
        target_metric = available_key_metrics[0]
        print(f"🎯 Analyzing dependencies for: {target_metric}")

        try:
            dependencies = analyzer.analyze_metric_dependencies(merged_df, target_metric)
            print(f"✅ Dependency analysis completed")
            print(f"📊 Top predictors for {target_metric}:")
            for predictor in dependencies['top_predictors'][:3]:
                dep_info = dependencies['dependencies'][predictor]
                corr = dep_info.get('correlation', 'N/A')
                significance = dep_info.get('significance', 'unknown')
                print(f"  • {predictor}: correlation={corr:.3f} ({significance})")

        except Exception as e:
            print(f"❌ Error in dependency analysis: {str(e)}")

    # Test seasonality detection if we have datetime index
    if isinstance(merged_df.index, pd.DatetimeIndex) and available_key_metrics:
        print(f"\n📅 Testing seasonality detection for: {available_key_metrics[0]}")
        try:
            seasonality = analyzer.detect_seasonal_patterns(merged_df, available_key_metrics[0])
            print(f"✅ Seasonality analysis completed")
            if seasonality['strongest_pattern']:
                pattern = seasonality['strongest_pattern']
                print(f"📊 Strongest pattern: {pattern['name']} (strength: {pattern['strength']:.3f})")
            else:
                print("📊 No clear seasonal patterns detected")
        except Exception as e:
            print(f"❌ Error in seasonality analysis: {str(e)}")

    print("\n✅ Relationship analysis testing completed!")

if __name__ == "__main__":
    test_relationship_analysis()