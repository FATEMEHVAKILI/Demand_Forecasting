"""
Time Series Forecasting with Multiple ML Algorithms
Includes: Outlier Detection, Noise Checking, Feature Importance
Algorithms: XGBoost, Random Forest, Prophet, SARIMA
Forecasting Horizons: 3 months (90 days) and 6 months (180 days)
"""

from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not installed. Install with: pip install prophet")


class OutlierDetector:
    """Detect and handle outliers using multiple methods"""
    
    def __init__(self, method='iqr', threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.outlier_mask = None
        self.outlier_indices = []
        
    def detect(self, data):
        """Detect outliers in the data"""
        if self.method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            self.outlier_mask = (data < lower_bound) | (data > upper_bound)
            
        elif self.method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            self.outlier_mask = z_scores > self.threshold
            
        elif self.method == 'mad':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            self.outlier_mask = np.abs(modified_z_scores) > self.threshold
            
        self.outlier_indices = np.where(self.outlier_mask)[0]
        return self.outlier_mask
    
    def handle_outliers(self, data, method='clip'):
        """Handle detected outliers"""
        if self.outlier_mask is None:
            self.detect(data)
            
        data_clean = data.copy()
        
        if method == 'clip':
            Q1 = np.percentile(data[~self.outlier_mask], 25)
            Q3 = np.percentile(data[~self.outlier_mask], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data_clean = np.clip(data, lower_bound, upper_bound)
            
        elif method == 'remove':
            data_clean = data[~self.outlier_mask]
            
        elif method == 'impute':
            median_val = np.median(data[~self.outlier_mask])
            data_clean[self.outlier_mask] = median_val
            
        return data_clean
    
    def get_outlier_summary(self, data):
        """Get summary statistics about outliers"""
        if self.outlier_mask is None:
            self.detect(data)
            
        total_points = len(data)
        outlier_count = np.sum(self.outlier_mask)
        outlier_percentage = (outlier_count / total_points) * 100
        
        return {
            'total_points': total_points,
            'outlier_count': int(outlier_count),
            'outlier_percentage': round(outlier_percentage, 2),
            'outlier_indices': self.outlier_indices[:20].tolist()  # First 20
        }


class NoiseChecker:
    """Check for noise in time series data"""
    
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.noise_ratio = None
        self.signal_to_noise_ratio = None
        
    def check_noise(self, data):
        """Calculate noise metrics"""
        # Calculate rolling mean as signal
        rolling_mean = pd.Series(data).rolling(window=self.window_size, center=True).mean()
        
        # Noise is the difference between actual and signal
        noise = data - rolling_mean.values
        
        # Calculate metrics
        signal_std = rolling_mean.std()
        noise_std = noise.std()
        
        self.noise_ratio = noise_std / data.std() if data.std() > 0 else 0
        self.signal_to_noise_ratio = signal_std / noise_std if noise_std > 0 else float('inf')
        
        return {
            'noise_ratio': round(self.noise_ratio, 4),
            'signal_to_noise_ratio': round(self.signal_to_noise_ratio, 4),
            'noise_std': round(noise_std, 4),
            'signal_std': round(signal_std, 4)
        }
    
    def denoise(self, data, method='rolling'):
        """Apply denoising to the data"""
        if method == 'rolling':
            return pd.Series(data).rolling(window=self.window_size, center=True).mean().values
        elif method == 'exponential':
            return pd.Series(data).ewm(span=self.window_size).mean().values
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(data, sigma=self.window_size/2)
        return data


class TimeSeriesPredictor:
    def __init__(self, data_path, working_day_path, input_length=180, output_length=1, 
                 test_size=0.3, random_state=42):
        self.data_path = data_path
        self.working_day_path = working_day_path
        self.input_length = input_length
        self.output_length = output_length
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.df_clean = None
        self.X_all = None
        self.y_all = None
        self.models = {}
        self.best_features = []
        self.feature_importance = {}
        self.metrics = {}
        self.outlier_detector = OutlierDetector(method='iqr', threshold=1.5)
        self.noise_checker = NoiseChecker(window_size=7)
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load data and perform initial preprocessing"""
        start_time = time.time()
        
        self.df = pd.read_csv(self.data_path)
        working_days = pd.read_csv(self.working_day_path)
        
        # Convert dates
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        working_days['Date'] = pd.to_datetime(working_days['Date'])
        
        # Handle missing dates
        if self.df['Date'].isna().any():
            print("Warning: Found missing dates. Dropping rows with missing dates.")
            self.df.dropna(subset=['Date'], inplace=True)
        
        # Merge with working days
        self.df = pd.merge(self.df, working_days, on='Date', how='left')
        
        # Add time features
        self.add_time_related_features()
        
        # Add moving averages
        self.add_moving_averages()
        
        print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        
    def add_time_related_features(self):
        """Add time-based features"""
        self.df['month'] = self.df['Date'].dt.month
        self.df['day_of_month'] = self.df['Date'].dt.day
        self.df['day_of_year'] = self.df['Date'].dt.dayofyear
        self.df['week_of_year'] = self.df['Date'].dt.isocalendar().week
        self.df['day_of_week'] = ((self.df['Date'].dt.dayofweek + 1) % 7) + 1
        self.df['is_month_start'] = self.df['Date'].dt.is_month_start.astype(int)
        self.df['is_month_end'] = self.df['Date'].dt.is_month_end.astype(int)
        self.df['is_weekend'] = ((self.df['WorkingDay'] == 0) & (self.df['day_of_week'] == 7)).astype(int)
        self.df['is_holiday'] = ((self.df['WorkingDay'] == 0) & (self.df['day_of_week'] != 7)).astype(int)
        
    def add_moving_averages(self):
        """Add moving average features"""
        for window in [7, 30, 60, 90]:
            self.df[f'SMA_{window}'] = self.df['Weight'].rolling(window).mean()
        for alpha in [0.9, 0.8]:
            for lag in [7, 30]:
                self.df[f'EMA_alpha_{int(alpha*10)}_lag_{lag}'] = self.df['Weight'].shift(lag).ewm(alpha=alpha).mean()
    
    def detect_and_handle_outliers(self, column='Weight', method='impute'):
        """Detect and handle outliers in specified column"""
        print(f"\n{'='*60}")
        print("OUTLIER DETECTION AND HANDLING")
        print(f"{'='*60}")
        
        data = self.df[column].values
        
        # Detect outliers
        outlier_mask = self.outlier_detector.detect(data)
        outlier_summary = self.outlier_detector.get_outlier_summary(data)
        
        print(f"\nOutlier Summary for '{column}':")
        print(f"  Total data points: {outlier_summary['total_points']}")
        print(f"  Outliers detected: {outlier_summary['outlier_count']} ({outlier_summary['outlier_percentage']}%)")
        print(f"  First 20 outlier indices: {outlier_summary['outlier_indices']}")
        
        # Handle outliers
        self.df[f'{column}_clean'] = self.outlier_detector.handle_outliers(data, method=method)
        
        # Visualize outliers
        self.plot_outliers(column)
        
        return outlier_summary
    
    def check_noise(self, column='Weight'):
        """Check noise levels in the data"""
        print(f"\n{'='*60}")
        print("NOISE ANALYSIS")
        print(f"{'='*60}")
        
        data = self.df[column].values if f'{column}_clean' not in self.df.columns else self.df[f'{column}_clean'].values
        
        noise_metrics = self.noise_checker.check_noise(data)
        
        print(f"\nNoise Metrics for '{column}':")
        print(f"  Noise Ratio: {noise_metrics['noise_ratio']}")
        print(f"  Signal-to-Noise Ratio: {noise_metrics['signal_to_noise_ratio']}")
        print(f"  Noise Std Dev: {noise_metrics['noise_std']}")
        print(f"  Signal Std Dev: {noise_metrics['signal_std']}")
        
        # Interpretation
        if noise_metrics['noise_ratio'] < 0.3:
            print("  → Low noise level - Data is relatively clean")
        elif noise_metrics['noise_ratio'] < 0.6:
            print("  → Moderate noise level - Consider denoising")
        else:
            print("  → High noise level - Denoising recommended")
        
        # Apply denoising
        self.df[f'{column}_denoised'] = self.noise_checker.denoise(data, method='rolling')
        
        return noise_metrics
    
    def plot_outliers(self, column):
        """Plot data with outliers highlighted"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Original data with outliers
            axes[0].plot(self.df['Date'], self.df[column], label='Original', color='blue', alpha=0.7)
            if len(self.outlier_detector.outlier_indices) > 0:
                axes[0].scatter(
                    self.df['Date'].iloc[self.outlier_detector.outlier_indices],
                    self.df[column].iloc[self.outlier_detector.outlier_indices],
                    color='red', s=50, label='Outliers', zorder=5
                )
            axes[0].set_title(f'{column} - Original Data with Outliers')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Cleaned data
            clean_col = f'{column}_clean'
            if clean_col in self.df.columns:
                axes[1].plot(self.df['Date'], self.df[clean_col], label='Cleaned', color='green', alpha=0.7)
                axes[1].set_title(f'{column} - After Outlier Handling')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('outlier_analysis.png', dpi=150)
            print("\nOutlier visualization saved to 'outlier_analysis.png'")
            plt.close()
        except Exception as e:
            print(f"Error plotting outliers: {e}")
    
    def choose_best_features(self):
        """Select best features based on correlation"""
        sma_cols = [col for col in self.df.columns if col.startswith('SMA')]
        ema_cols = [col for col in self.df.columns if col.startswith('EMA')]
        xi_cols = [col for col in self.df.columns if col.startswith('x_')]
        yj_cols = [col for col in self.df.columns if col.startswith('y_')]
        
        time_features = [
            'month', 'day_of_month', 'day_of_year', 'week_of_year',
            'day_of_week', 'is_month_start', 'is_month_end',
            'is_weekend', 'is_holiday'
        ]
        valid_time_features = [feat for feat in time_features if feat in self.df.columns]
        
        # Use cleaned column if available
        target_col = 'Weight_clean' if 'Weight_clean' in self.df.columns else 'Weight'
        
        all_features = sma_cols + ema_cols + xi_cols + yj_cols + valid_time_features
        
        # Filter out non-existent columns
        existing_features = [f for f in all_features if f in self.df.columns]
        
        correlations = self.df[[target_col] + existing_features].corr()[target_col].sort_values(ascending=False)
        
        print(f"\n{'='*60}")
        print("FEATURE CORRELATION ANALYSIS")
        print(f"{'='*60}")
        print("\nTop 15 features based on correlation:")
        print(correlations.head(15))
        
        # Select top features (excluding target itself)
        self.best_features = [f for f in correlations.nlargest(15).index.tolist() if f != target_col][:10]
        
        print(f"\nSelected {len(self.best_features)} best features:")
        for i, feature in enumerate(self.best_features, 1):
            print(f"  {i}. {feature}: {correlations[feature]:.4f}")
        
        return self.best_features
    
    def _transform_dataset(self):
        """Transform dataset for time series prediction"""
        data = self.df.copy()
        
        # Use cleaned column if available
        target_col = 'Weight_clean' if 'Weight_clean' in self.df.columns else 'Weight'
        
        for i in range(1, self.input_length + 1):
            data[f'x_{i}'] = data[target_col].shift(-i)
        for j in range(self.output_length):
            data[f'y_{j}'] = data[target_col].shift(-self.output_length - j)
        
        return data.dropna().reset_index(drop=True)
    
    def prepare_features_and_target(self):
        """Prepare features and target variables"""
        transformed_df = self._transform_dataset()
        self.choose_best_features()
        
        feature_columns = [col for col in transformed_df.columns 
                          if col.startswith('x_') or col in self.best_features]
        
        self.X = transformed_df[feature_columns].values
        self.X = np.nan_to_num(self.X, nan=0.0)
        
        y_columns = [f'y_{j}' for j in range(self.output_length)]
        self.y = transformed_df[y_columns].values
        self.y = np.nan_to_num(self.y, nan=0.0)
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f'\nShape of X: {self.X.shape}')
        print(f'Shape of y: {self.y.shape}')
        print(f'Shape of X_scaled: {self.X_scaled.shape}')
    
    def adfuller_test(self):
        """Perform ADF stationarity test"""
        result = adfuller(self.df['Weight'].dropna())
        print(f'\n{"="*60}')
        print("ADF STATIONARITY TEST")
        print(f"{'='*60}")
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        print(f'Critical Values:')
        for key, value in result[4].items():
            print(f'  {key}: {value:.3f}')
        
        if result[1] < 0.05:
            print('\n→ Time series is STATIONARY (p < 0.05)')
        else:
            print('\n→ Time series is NON-STATIONARY (p >= 0.05)')
        
        return result[1] < 0.05
    
    def train_xgboost_model(self, param_grid=None):
        """Train XGBoost model with optional hyperparameter tuning"""
        print(f'\n{"="*60}')
        print("TRAINING XGBOOST MODEL")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = self.calculate_metrics(y_train.flatten(), y_train_pred.flatten())
        test_metrics = self.calculate_metrics(y_test.flatten(), y_test_pred.flatten())
        
        print(f"\nTraining Metrics: MAE={train_metrics['mae']:.3f}, RMSE={train_metrics['rmse']:.3f}, R²={train_metrics['r2']:.4f}")
        print(f"Testing Metrics: MAE={test_metrics['mae']:.3f}, RMSE={test_metrics['rmse']:.3f}, R²={test_metrics['r2']:.4f}")
        
        # Feature importance
        self.feature_importance['xgboost'] = model.feature_importances_
        
        self.models['xgboost'] = model
        self.metrics['xgboost'] = {'train': train_metrics, 'test': test_metrics}
        
        print(f"XGBoost training completed in {time.time() - start_time:.2f} seconds")
        
        return model, test_metrics
    
    def train_random_forest_model(self):
        """Train Random Forest model"""
        print(f'\n{"="*60}')
        print("TRAINING RANDOM FOREST MODEL")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = self.calculate_metrics(y_train.flatten(), y_train_pred.flatten())
        test_metrics = self.calculate_metrics(y_test.flatten(), y_test_pred.flatten())
        
        print(f"\nTraining Metrics: MAE={train_metrics['mae']:.3f}, RMSE={train_metrics['rmse']:.3f}, R²={train_metrics['r2']:.4f}")
        print(f"Testing Metrics: MAE={test_metrics['mae']:.3f}, RMSE={test_metrics['rmse']:.3f}, R²={test_metrics['r2']:.4f}")
        
        # Feature importance
        self.feature_importance['random_forest'] = model.feature_importances_
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = {'train': train_metrics, 'test': test_metrics}
        
        print(f"Random Forest training completed in {time.time() - start_time:.2f} seconds")
        
        return model, test_metrics
    
    def train_prophet_model(self, forecast_days=180):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Skipping...")
            return None, None
        
        print(f'\n{"="*60}')
        print("TRAINING PROPHET MODEL")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Prepare data for Prophet
        df_prophet = self.df[['Date', 'Weight']].copy()
        df_prophet.columns = ['ds', 'y']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(df_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Evaluate on historical data
        train_size = int(len(df_prophet) * (1 - self.test_size))
        train = df_prophet.iloc[:train_size]
        test = df_prophet.iloc[train_size:]
        
        # Make predictions on test set
        test_forecast = model.predict(test[['ds']])
        
        test_metrics = self.calculate_metrics(test['y'].values, test_forecast['yhat'].values)
        
        print(f"\nTesting Metrics: MAE={test_metrics['mae']:.3f}, RMSE={test_metrics['rmse']:.3f}, R²={test_metrics['r2']:.4f}")
        
        self.models['prophet'] = model
        self.metrics['prophet'] = {'test': test_metrics}
        
        print(f"Prophet training completed in {time.time() - start_time:.2f} seconds")
        
        return model, forecast
    
    def train_sarima_model(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """Train SARIMA model"""
        print(f'\n{"="*60}')
        print("TRAINING SARIMA MODEL")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Use cleaned data if available
        data = self.df['Weight_clean'].values if 'Weight_clean' in self.df.columns else self.df['Weight'].values
        
        # Fit SARIMA
        model = ARIMA(
            data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit()
        
        # Make predictions
        train_size = int(len(data) * (1 - self.test_size))
        train = data[:train_size]
        test = data[train_size:]
        
        # Forecast
        forecast = fitted_model.forecast(steps=len(test))
        
        test_metrics = self.calculate_metrics(test, forecast)
        
        print(f"\nTesting Metrics: MAE={test_metrics['mae']:.3f}, RMSE={test_metrics['rmse']:.3f}, R²={test_metrics['r2']:.4f}")
        
        self.models['sarima'] = fitted_model
        self.metrics['sarima'] = {'test': test_metrics}
        
        print(f"SARIMA training completed in {time.time() - start_time:.2f} seconds")
        
        return fitted_model, forecast
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mae': mae,
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': rmse,
            'r2': r2
        }
    
    def plot_feature_importance(self, model_name='xgboost', top_n=15):
        """Plot feature importance for a model"""
        if model_name not in self.feature_importance:
            print(f"No feature importance available for {model_name}")
            return
        
        importances = self.feature_importance[model_name]
        feature_names = [col for col in self.df.columns 
                        if col.startswith('x_') or col in self.best_features]
        
        if len(feature_names) != len(importances):
            feature_names = feature_names[:len(importances)]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices][::-1])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {model_name.upper()}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}.png', dpi=150)
        print(f"Feature importance plot saved to 'feature_importance_{model_name}.png'")
        plt.close()
    
    def forecast_future(self, model_name='xgboost', forecast_days=180, working_days_path=None):
        """Generate future forecasts"""
        print(f'\n{"="*60}')
        print(f"FUTURE FORECASTING ({forecast_days} days) - {model_name.upper()}")
        print(f"{'='*60}")
        
        if working_days_path:
            working_days = pd.read_csv(working_days_path, parse_dates=['Date'])
            working_days.set_index('Date', inplace=True)
        
        model = self.models.get(model_name)
        if model is None:
            print(f"Model {model_name} not found!")
            return None
        
        last_date = self.df['Date'].iloc[-1]
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days)
        
        if model_name in ['xgboost', 'random_forest']:
            # Use recent features for forecasting
            if len(self.X) >= forecast_days:
                future_features = self.X[-forecast_days:, :]
                future_features_scaled = self.scaler.transform(future_features)
                predictions = model.predict(future_features_scaled)
            else:
                print("Insufficient data for forecasting")
                return None
                
        elif model_name == 'prophet':
            future_df = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future_df)
            predictions = forecast['yhat'].values[-forecast_days:]
            
        elif model_name == 'sarima':
            predictions = model.forecast(steps=forecast_days)
        
        # Adjust for working days
        forecast_values = []
        for date, pred in zip(future_dates, predictions):
            if working_days_path and date in working_days.index:
                if working_days.loc[date, 'WorkingDay'] == 0:
                    forecast_values.append(0)
                else:
                    forecast_values.append(pred)
            else:
                forecast_values.append(pred)
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted': forecast_values
        })
        
        print(f"\nForecast Summary:")
        print(f"  Period: {future_dates[0].date()} to {future_dates[-1].date()}")
        print(f"  Total days: {forecast_days}")
        print(f"  Mean forecast: {np.mean(forecast_values):.2f}")
        print(f"  Std forecast: {np.std(forecast_values):.2f}")
        print(f"  Min forecast: {np.min(forecast_values):.2f}")
        print(f"  Max forecast: {np.max(forecast_values):.2f}")
        
        return forecast_df
    
    def compare_models(self):
        """Compare all trained models"""
        print(f'\n{"="*60}')
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        comparison_df = pd.DataFrame()
        
        for model_name, metrics in self.metrics.items():
            test_metrics = metrics.get('test', {})
            row = pd.DataFrame({
                'Model': [model_name],
                'MAE': [test_metrics.get('mae', np.nan)],
                'RMSE': [test_metrics.get('rmse', np.nan)],
                'R²': [test_metrics.get('r2', np.nan)]
            })
            comparison_df = pd.concat([comparison_df, row], ignore_index=True)
        
        comparison_df = comparison_df.sort_values('MAE')
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        print("\nModel Performance Ranking:")
        print(comparison_df.to_string(index=False))
        
        # Save to CSV
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("\nModel comparison saved to 'model_comparison.csv'")
        
        return comparison_df
    
    def plot_all_forecasts(self, forecast_3m=None, forecast_6m=None):
        """Plot all forecasts together"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot historical data
        ax.plot(self.df['Date'], self.df['Weight'], label='Historical', color='blue', linewidth=2)
        
        # Plot 3-month forecast
        if forecast_3m is not None:
            ax.plot(forecast_3m['Date'], forecast_3m['Forecasted'], 
                   label='3-Month Forecast', color='green', linestyle='--', linewidth=2)
        
        # Plot 6-month forecast
        if forecast_6m is not None:
            ax.plot(forecast_6m['Date'], forecast_6m['Forecasted'], 
                   label='6-Month Forecast', color='red', linestyle='-.', linewidth=2)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gcf().autofmt_xdate()
        
        ax.set_title('Time Series Forecast - 3 Months & 6 Months', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('all_forecasts.png', dpi=150)
        print("Combined forecast plot saved to 'all_forecasts.png'")
        plt.close()
    
    def run_complete_analysis(self, forecast_3m_days=90, forecast_6m_days=180):
        """Run complete forecasting analysis"""
        print(f'\n{"#"*80}')
        print("# COMPLETE TIME SERIES FORECASTING ANALYSIS")
        print('#'*80 + '\n')
        
        # Step 1: Load data
        self.load_and_preprocess_data()
        
        # Step 2: Outlier detection
        self.detect_and_handle_outliers('Weight', method='impute')
        
        # Step 3: Noise checking
        self.check_noise('Weight')
        
        # Step 4: Stationarity test
        self.adfuller_test()
        
        # Step 5: Feature selection
        self.prepare_features_and_target()
        
        # Step 6: Train models
        print(f'\n{"#"*80}')
        print("# TRAINING ML MODELS")
        print(f"{"#"*80}")
        
        self.train_xgboost_model()
        self.train_random_forest_model()
        
        if PROPHET_AVAILABLE:
            self.train_prophet_model(forecast_days=forecast_6m_days)
        
        self.train_sarima_model()
        
        # Step 7: Feature importance
        print(f'\n{"#"*80}')
        print("# FEATURE IMPORTANCE ANALYSIS")
        print(f"{"#"*80}")
        
        self.plot_feature_importance('xgboost')
        self.plot_feature_importance('random_forest')
        
        # Step 8: Model comparison
        self.compare_models()
        
        # Step 9: Generate forecasts
        print(f'\n{"#"*80}')
        print("# GENERATING FORECASTS")
        print(f"{"#"*80}")
        
        # Choose best model based on MAE
        best_model = min(self.metrics.keys(), 
                        key=lambda k: self.metrics[k].get('test', {}).get('mae', float('inf')))
        print(f"\nBest model: {best_model.upper()}")
        
        forecast_3m = self.forecast_future(best_model, forecast_3m_days, self.working_day_path)
        forecast_6m = self.forecast_future(best_model, forecast_6m_days, self.working_day_path)
        
        # Save forecasts
        if forecast_3m is not None:
            forecast_3m.to_csv('forecast_3months.csv', index=False)
            print("3-month forecast saved to 'forecast_3months.csv'")
        
        if forecast_6m is not None:
            forecast_6m.to_csv('forecast_6months.csv', index=False)
            print("6-month forecast saved to 'forecast_6months.csv'")
        
        # Step 10: Visualization
        self.plot_all_forecasts(forecast_3m, forecast_6m)
        
        print(f'\n{"#"*80}')
        print("# ANALYSIS COMPLETE")
        print(f"{"#"*80}")
        print("\nGenerated files:")
        print("  - outlier_analysis.png")
        print("  - feature_importance_xgboost.png")
        print("  - feature_importance_random_forest.png")
        print("  - model_comparison.csv")
        print("  - forecast_3months.csv")
        print("  - forecast_6months.csv")
        print("  - all_forecasts.png")
        
        return {
            'models': self.models,
            'metrics': self.metrics,
            'forecast_3m': forecast_3m,
            'forecast_6m': forecast_6m
        }


# Main execution
if __name__ == "__main__":
    predictor = TimeSeriesPredictor(
        data_path='Data.csv',
        working_day_path='WorkingDay.csv',
        input_length=180,
        output_length=1,
        test_size=0.3,
        random_state=42
    )
    
    results = predictor.run_complete_analysis(
        forecast_3m_days=90,
        forecast_6m_days=180
    )
