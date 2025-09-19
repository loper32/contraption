# Service Level Prediction Demo Workflow

## Prerequisites Completed ✅
- Fixed occupancy prediction bug (was showing ~50%, now ~89%)
- Added session state persistence for forecast data
- Implemented Service Level Prediction feature

## To Test the Service Level Prediction:

### Step 1: Upload Data
1. Navigate to **Data Upload & Processing**
2. Upload the 3 Excel files:
   - `Calls.xlsx` - Historical call data
   - `Occ.xlsx` - Occupancy data
   - `Staff.xlsx` - Staffing data

### Step 2: Analyze Historical Data
1. Navigate to **Historical Analysis**
2. Review the merged data
3. Check relationships discovered

### Step 3: Train Models
1. Navigate to **Model Training**
2. Click "Train All Models" or select specific relationships
3. Ensure you have **Service Level → Occupancy** relationship trained

### Step 4: Generate Forecast
1. Navigate to **Forecasting**
2. Use sample data or enter your forecast
3. Set parameters (Days per week, Hours per day, etc.)
4. Click "🚀 Simulate FTE Requirements"
5. This will save the forecast to session state

### Step 5: Test Service Level Prediction
1. Navigate to **Service Level Prediction**
2. Choose input method:
   - **Use Sample Data** - Auto-generates test FTE values
   - **Copy/Paste** - Enter your actual FTE plans
   - **Upload File** - Use Excel/CSV with FTE data

3. View results:
   - Predicted Service Levels based on your FTE
   - Occupancy calculations
   - Visual charts showing trends
   - Insights on staffing adequacy

## Example FTE Input Format

### Tab-separated:
```
Week 1	150
Week 2	155
Week 3	160
Week 4	165
```

### CSV format:
```
Period,FTE
Week 1,150
Week 2,155
Week 3,160
Week 4,165
```

## Key Features:
- **Reverse Prediction**: FTE → Occupancy → Service Level
- **Integration**: Uses forecast workload from previous step
- **Multiple Input Methods**: Sample, paste, or upload
- **Smart Insights**: Warnings for understaffing or burnout risk
- **Visualizations**: Dual-axis chart with SL and Occupancy

## Fixed Issues:
1. ✅ Occupancy prediction now showing correct ~89% instead of 50%
2. ✅ Power model formula consistency fixed
3. ✅ Session state persistence for forecast data
4. ✅ P10/P90 removed from plot legends for clarity