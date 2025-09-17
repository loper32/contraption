# Contraption Changelog

## Version 1.1.0 - September 17, 2025

### Major Features
- **Weighted Auto-Selection Curve Fitting**: Revolutionary algorithm that automatically selects optimal curve type from polynomial, exponential, power, logarithmic, and linear models based on volume-weighted R² scores
- **Enhanced Service Level → Occupancy Predictions**: Realistic occupancy predictions (~89% for 80% service level targets)
- **Improved Target Visualization**: Target stars properly positioned on fitted curves using mathematical intersection calculations

### Bug Fixes
- **Fixed NameError**: Resolved `j_squared` undefined variable (replaced with `r_squared`)
- **Fixed NameError**: Resolved `adjusted_aht` undefined variable (replaced with `base_aht`)
- **Fixed Occupancy Predictions**: Corrected Service Level → Occupancy predictions from unrealistic ~50% to proper ~89% values
- **Fixed Decimal/Percentage Conversion**: Proper handling of service level decimals (0.8) vs percentages (80%) in curve fitting
- **Fixed Target Positioning**: Target markers now correctly positioned on curves rather than floating off-curve
- **Fixed Column References**: Updated occupancy column naming to eliminate confusion

### Performance Improvements
- **Volume Weighting**: Maintained proper volume weighting throughout curve fitting process
- **Model Selection**: Automatic selection of best-fitting model type based on weighted statistical metrics
- **Optimized Predictions**: Enhanced prediction accuracy through improved model application logic

### User Experience
- **Simplified UI**: Streamlined occupancy columns to single "Occupancy %" column
- **Clean Output**: Removed debug statements for production-ready interface
- **Better Visualizations**: Enhanced plot clarity with proper model equations and R² values

### Technical Implementation
- **Weighted R² Calculation**: `1 - (weighted_ss_res / weighted_ss_tot)` for proper model evaluation
- **Multi-Model Testing**: Systematic evaluation of 5 curve types with automatic best selection
- **Robust Error Handling**: Graceful fallback when curve fitting fails for specific model types
- **Mathematical Accuracy**: Correct model parameter application for predictions

### Code Quality
- **Removed Debug Code**: Clean production codebase without temporary debug statements
- **Improved Documentation**: Updated project documentation with technical implementation details
- **Better Error Messages**: Enhanced user feedback for curve fitting and prediction issues

## Previous Versions
- **Version 1.0.0**: Initial implementation with basic curve fitting and Excel data processing