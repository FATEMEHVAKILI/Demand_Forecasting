# PowerBI Dashboards Directory

This directory contains screenshots and exports of the PowerBI dashboards for the AI Assistant Demand Forecasting project.

## 📊 Dashboard Files

### Expected Screenshots

Add your PowerBI dashboard screenshots to this directory with the following filenames:

1. **forecast_overview.png** - Total demand forecasting comparison across all models
2. **abc_analysis.png** - ABC classification visualization with Pareto charts
3. **product_forecast.png** - Product-level forecasts with historical trends
4. **consumption_report.png** - Monthly consumption patterns and trends analysis
5. **model_comparison.png** - Model performance metrics comparison (MAE, RMSE, MAPE)

### How to Add Screenshots

1. Open your PowerBI Desktop file (.pbix)
2. Navigate to each dashboard page
3. Take a screenshot (or use File > Export > Export to PowerPoint/PDF)
4. Save the images in this directory with the names listed above
5. The images will automatically appear in the main README.md file

## 📈 Dashboard Pages Description

### 1. Demand Forecasting Overview
Shows the total demand forecast for the next 3-6 months using different ML models (RF, XGB, Prophet, SARIMA, Bi-LSTM) compared against historical data.

### 2. ABC Analysis Dashboard
Displays monthly ABC classification of products based on consumption value, including:
- Pareto charts showing cumulative percentage
- Distribution of A, B, C class items
- Month-over-month class transitions

### 3. Product-Level Forecasting
Detailed view of individual product forecasts with:
- Historical consumption trends
- Model-specific predictions
- Confidence intervals (where applicable)

### 4. Consumption Report
Analysis of historical consumption patterns including:
- Monthly totals and trends
- Seasonal patterns (Persian calendar)
- Warehouse/consumption location breakdowns

### 5. Model Performance Comparison
Side-by-side evaluation of all models using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Training vs Test performance

## 🔗 Links

- [Main README](../README.md) - Project overview and usage instructions
- [PowerBI_Data.xlsx](../PowerBI_Data.xlsx) - Data source for dashboards (generated after running main.py)

## 📝 Notes

- Supported image formats: PNG, JPG, GIF
- Recommended resolution: 1920x1080 or higher for best quality
- Images are referenced in the main README.md using relative paths
