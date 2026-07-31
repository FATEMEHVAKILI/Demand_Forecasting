# AI Assistant - Demand Forecasting System

A comprehensive machine learning-based demand forecasting system for inventory management with support for Persian (Jalali) calendar, ABC analysis, and multiple forecasting models.

## 📋 Table of Contents

- [Features](#features)
- [Models](#models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
- [Output Files](#output-files)
- [PowerBI Dashboards](#powerbi-dashboards)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Workflow](#workflow)
- [License](#license)

## ✨ Features

- **Multi-Model Forecasting**: Support for multiple ML and statistical models including Random Forest, XGBoost, Prophet, SARIMA, and Bi-LSTM
- **Persian Calendar Support**: Full integration with Jalali (Persian) calendar for date conversions and features
- **ABC Analysis**: Monthly ABC classification of products based on consumption value
- **Feature Engineering**: Automatic generation of lag features, rolling statistics, and seasonal features
- **Feature Importance Analysis**: Comprehensive feature importance evaluation using Correlation, Mutual Information, Random Forest, and XGBoost
- **Outlier Handling**: Winsorization method for outlier detection and treatment
- **Working Day Features**: Integration of working day information for improved accuracy
- **PowerBI Export**: Ready-to-use data export for PowerBI dashboards
- **Flexible Model Selection**: Train specific models or all available models

## 🧠 Models

The system supports the following forecasting models:

| Model | Type | Description |
|-------|------|-------------|
| **Random Forest (RF)** | Ensemble | Tree-based ensemble method with GridSearchCV hyperparameter tuning |
| **XGBoost (XGB)** | Gradient Boosting | Advanced gradient boosting with optimized parameters |
| **Prophet** | Time Series | Facebook's Prophet for additive time series decomposition |
| **SARIMA** | Statistical | Seasonal ARIMA for traditional time series forecasting |
| **Bi-LSTM** | Deep Learning | Bidirectional LSTM neural network using PyTorch |

### Model Evaluation Metrics

All models are evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

## 📦 Requirements

### Core Dependencies

```
numpy
pandas
scikit-learn
jdatetime
openpyxl
```

### Optional Dependencies (for additional models)

```
xgboost          # For XGBoost model
prophet          # For Prophet model
statsmodels      # For SARIMA model
torch            # For Bi-LSTM model
```

### Full Requirements List

See `requirements.txt` for the complete list of dependencies.

## 🔧 Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd <project-directory>
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install optional dependencies for all models**

```bash
pip install xgboost prophet statsmodels torch
```

## 🚀 Usage

### Basic Usage

Run the main script with default models (XGBoost and Random Forest):

```bash
python main.py
```

### Custom Model Selection

To train specific models, modify the `main()` call in `main.py`:

```python
# Train all available models
main(models_to_train=["all"])

# Train specific models
main(models_to_train=["XGB", "RF", "Prophet"])

# Train only Bi-LSTM
main(models_to_train=["Bi-LSTM"])
```

Available model codes:
- `"RF"` - Random Forest
- `"XGB"` - XGBoost
- `"Prophet"` - Facebook Prophet
- `"SARIMA"` - SARIMA
- `"Bi-LSTM"` - Bidirectional LSTM
- `"all"` - All available models

## 📊 Input Data Format

### Required Input Files

1. **Data.xlsx** - Main historical data file
2. **WorkingDay.csv** - Working day information

### Data.xlsx Columns

| Persian Column | English Equivalent | Description |
|----------------|-------------------|-------------|
| تاریخ | Date | Persian date of transaction |
| مقدار (اصلی) | Quantity | Actual quantity/value |
| کد کالا | Product Code | Unique product identifier |
| نام کالا | Product Name | Name of the product |
| نوع سند | Document Type | Type of document |
| وضعیت | Status | Transaction status |
| ماهیت کالا | Product Nature | Nature/category of product |
| واحد سنجش | Unit | Measurement unit |
| انبار | Warehouse | Warehouse location |
| طرف مقابل | Counterparty | Trading partner |
| محل مصرف | Consumption Location | Location of usage |

### WorkingDay.csv Format

```csv
Date,WorkingDay
01/01/2023,1
02/01/2023,1
...
```

## 📁 Output Files

### 1. Forecast_Results.xlsx

Main output file containing multiple sheets:

| Sheet Name | Description |
|------------|-------------|
| **Total_Forecast** | Aggregated forecast values by month and model |
| **ML_Product_Forecast** | Product-level forecasts from ML models (RF, XGB) |
| **BiLSTM_Product_Forecast** | Product-level forecasts from Bi-LSTM model |
| **All_Products_Forecast** | Combined forecasts from all models |
| **Model_Evaluation** | Performance metrics for each model |
| **Feature_Importance** | Feature importance scores from different methods |

### 2. PowerBI_Data.xlsx

Dashboard-ready data with:

| Sheet Name | Description |
|------------|-------------|
| **PowerBI_Data** | Complete processed data with all features |
| **ABC_Analysis** | Monthly ABC classification results |

## 📊 PowerBI Dashboards

The project includes PowerBI dashboards for visualizing demand forecasting results, ABC analysis, and consumption reports. Dashboard screenshots are available in the [`Dashboards`](Dashboards/) directory.

### Available Dashboards

<!-- Add your dashboard images to the /Dashboards folder and reference them below -->

#### 1. Demand Forecasting Overview

![Demand Forecasting Dashboard](Dashboards/forecast_overview.png)

*Forecasting comparison across different models for total demand*

#### 2. ABC Analysis Dashboard

![ABC Analysis Dashboard](Dashboards/abc_analysis.png)

*Monthly ABC classification visualization with Pareto charts*

#### 3. Product-Level Forecasting

![Product Forecast Dashboard](Dashboards/product_forecast.png)

*Detailed product-level forecasts with historical trends*

#### 4. Consumption Report

![Consumption Report](Dashboards/consumption_report.png)

*Monthly consumption patterns and trends analysis*

#### 5. Model Performance Comparison

![Model Comparison](Dashboards/model_comparison.png)

*Side-by-side comparison of MAE, RMSE, and MAPE across all models*

### How to Use the PowerBI Dashboards

1. **Generate the data**: Run `python main.py` to create `PowerBI_Data.xlsx`
2. **Open PowerBI Desktop**: Import the generated Excel file
3. **Load the dashboard**: Open the `.pbix` file (if provided) or create new visualizations
4. **Refresh data**: When new data is available, re-run the script and refresh the PowerBI dataset

### Recommended Visualizations

- **Line Charts**: For forecasting trends comparison
- **Bar Charts**: For ABC classification distribution
- **Matrix Tables**: For detailed product-level data
- **KPI Cards**: For key metrics (MAE, RMSE, MAPE)
- **Slicers**: For filtering by month, product class, or warehouse

## 🗂️ Project Structure

```
├── main.py                 # Main script with all forecasting logic
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── Data.xlsx              # Input: Historical data (required)
├── WorkingDay.csv         # Input: Working days calendar (required)
├── Forecast_Results.xlsx  # Output: Forecasting results (generated)
├── PowerBI_Data.xlsx      # Output: PowerBI-ready data (generated)
└── Dashboards/            # Directory for PowerBI dashboard screenshots
    ├── forecast_overview.png    # Total demand forecasting comparison
    ├── abc_analysis.png         # ABC classification visualization
    ├── product_forecast.png     # Product-level forecasts
    ├── consumption_report.png   # Consumption patterns analysis
    └── model_comparison.png     # Model performance metrics
```

## ⚙️ Configuration

Key configuration parameters in `main.py`:

```python
INPUT_FILE = "Data.xlsx"           # Input data file
WORKING_DAY_FILE = "WorkingDay.csv" # Working days file
POWERBI_OUTPUT = "PowerBI_Data.xlsx" # PowerBI output file
FORECAST_OUTPUT = "Forecast_Results.xlsx" # Forecast output file

FORECAST_HORIZON = 6               # Number of months to forecast
A_THRESHOLD = 70                   # ABC analysis: A class threshold (%)
B_THRESHOLD = 90                   # ABC analysis: B class threshold (%)
```

### Feature Engineering Parameters

- **Lag Features**: 1, 2, 3 month lags
- **Rolling Statistics**: 3-month rolling mean and standard deviation
- **Seasonal Features**: Persian/Gregorian month, season, year
- **Working Day Features**: Binary working day indicator

## 🔄 Workflow

1. **Data Loading & Cleaning**
   - Load Excel data
   - Clean and normalize numeric columns
   - Convert Persian dates to Gregorian
   - Handle missing values

2. **Feature Engineering**
   - Generate Persian calendar features
   - Add working day indicators
   - Create lag and rolling features
   - Encode categorical variables

3. **ABC Analysis**
   - Monthly product classification
   - Cumulative percentage calculation
   - A/B/C class assignment

4. **Outlier Treatment**
   - Log transformation
   - IQR-based winsorization per product

5. **Panel Data Construction**
   - Build monthly product panel
   - Handle missing combinations
   - Generate time-series features

6. **Feature Selection**
   - Calculate feature importance (Correlation, MI, RF, XGB)
   - Select top 10 features for modeling

7. **Model Training**
   - Train/test split (80/20 by time)
   - Hyperparameter tuning with GridSearchCV
   - Model evaluation on test set

8. **Forecasting**
   - Generate product-level forecasts
   - Aggregate to total forecast
   - Export results to Excel

## 📈 ABC Analysis

The system performs monthly ABC analysis based on Pareto principle:

- **Class A**: Top items contributing to 70% of total consumption
- **Class B**: Items contributing to next 20% (70-90%)
- **Class C**: Remaining items (90-100%)

ABC classification is recalculated monthly to reflect changing demand patterns.

## 🛡️ Error Handling

The system includes robust error handling for:
- Missing optional libraries (XGBoost, Prophet, SARIMA, PyTorch)
- Invalid date formats
- Missing data columns
- Insufficient training data
- Model convergence issues

## 📝 Notes

- The system requires at least 6 months of historical data for time series models
- Bi-LSTM requires PyTorch and GPU acceleration is supported if available
- Persian date conversion uses the `jdatetime` library
- All numeric outputs use Int64 to preserve integer precision where possible

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is provided as-is for demand forecasting applications.

## 📧 Support

For questions or issues, please open an issue in the repository.

---

**Author**: AI Assistant Project  
**Version**: 1.0.0  
**Last Updated**: 2024
