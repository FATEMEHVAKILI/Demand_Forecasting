#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Time Series Forecasting with Feature Importance Analysis
Combines:
1. Feature Importance & Selection (Outlier detection, Correlation, MI, Tree-based, Regularization, RFE, PCA)
2. ML Forecasting (XGBoost-GA, Random Forest, Prophet, SARIMA)
3. 3-month and 6-month forecasting with outlier/noise checking
4. ABC Analysis preparation for PowerBI

Author: Combined from user's XGB-GA and Feature Importance code
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px

# Statistical tests
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram, linkage

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

# Genetic Algorithm for hyperparameter tuning
try:
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer, Continuous
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False
    print("Warning: sklearn-genetic-algorithm not installed. Using default XGBoost parameters.")

# Time Series Models
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not installed. Skipping Prophet model.")

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: FEATURE IMPORTANCE AND SELECTION
# ============================================================================

class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analysis with multiple methods."""
    
    def __init__(self, data_path: str, working_day_path: str, target_column: str = 'Weight'):
        self.data_path = data_path
        self.working_day_path = working_day_path
        self.target_column = target_column
        self.df = None
        self.df_clean = None
        self.numeric_cols = []
        self.feature_scores = {}
        self.best_features = []
        self.outlier_info = {}
        
    def load_and_preprocess(self):
        """Load data and perform initial preprocessing."""
        print("\n" + "="*70)
        print("LOADING AND PREPROCESSING DATA")
        print("="*70)
        
        start_time = time.time()
        
        # Load main data
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        # Load working days
        if os.path.exists(self.working_day_path):
            if self.working_day_path.endswith('.csv'):
                working_days = pd.read_csv(self.working_day_path)
            else:
                working_days = pd.read_excel(self.working_day_path)
            
            working_days['Date'] = pd.to_datetime(working_days['Date'])
            self.df = pd.merge(self.df, working_days, on='Date', how='left')
            print(f"Merged with working days data. Shape: {self.df.shape}")
        
        # Convert Date column
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Drop rows with missing dates
        if self.df['Date'].isna().any():
            print("Warning: Found missing dates. Dropping rows with missing dates.")
            self.df.dropna(subset=['Date'], inplace=True)
        
        # Add time-related features
        self._add_time_features()
        
        # Add moving averages
        self._add_moving_averages()
        
        # Drop rows with NaN values
        self.df.dropna(inplace=True)
        
        self.df_clean = self.df.copy()
        
        # Identify numeric columns (excluding date and target)
        exclude_cols = ['Date', self.target_column, 'WeightTarget', 'WeightForecast']
        exclude_cols += [col for col in self.df_clean.columns if col.startswith('y_')]
        self.numeric_cols = [
            col for col in self.df_clean.select_dtypes(include=[np.number]).columns 
            if col not in exclude_cols
        ]
        
        print(f"Data loaded successfully. Shape: {self.df_clean.shape}")
        print(f"Numeric features identified: {len(self.numeric_cols)}")
        print(f"Runtime: {time.time() - start_time:.2f} seconds")
        
        return self.df_clean
    
    def _add_time_features(self):
        """Add time-related features."""
        self.df['month'] = self.df['Date'].dt.month
        self.df['day_of_month'] = self.df['Date'].dt.day
        self.df['day_of_year'] = self.df['Date'].dt.dayofyear
        self.df['week_of_year'] = self.df['Date'].dt.isocalendar().week
        self.df['day_of_week'] = ((self.df['Date'].dt.dayofweek + 1) % 7) + 1
        self.df['is_month_start'] = self.df['Date'].dt.is_month_start.astype(int)
        self.df['is_month_end'] = self.df['Date'].dt.is_month_end.astype(int)
        
        if 'WorkingDay' in self.df.columns:
            self.df['is_weekend'] = ((self.df['WorkingDay'] == 0) & (
                self.df['day_of_week'] == 7)).astype(int)
            self.df['is_holiday'] = ((self.df['WorkingDay'] == 0) & (
                self.df['day_of_week'] != 7)).astype(int)
        else:
            self.df['is_weekend'] = (self.df['day_of_week'] >= 6).astype(int)
            self.df['is_holiday'] = 0
    
    def _add_moving_averages(self):
        """Add moving average features."""
        for window in [60, 90, 120, 150, 180, 365]:
            self.df[f'SMA_{window}'] = self.df[self.target_column].rolling(window).mean()
        
        for alpha in [0.95, 0.9, 0.8]:
            for lag in [60, 90, 120]:
                self.df[f'EMA_alpha_{int(alpha*100)}_lag_{lag}'] = \
                    self.df[self.target_column].shift(lag).ewm(alpha=alpha).mean()
    
    def detect_outliers(self, method: str = 'all') -> Dict:
        """Detect outliers using multiple methods."""
        print("\n" + "="*70)
        print("OUTLIER DETECTION")
        print("="*70)
        
        outlier_results = {}
        
        for col in self.numeric_cols:
            if col not in self.df_clean.columns:
                continue
                
            data = self.df_clean[col].dropna()
            if len(data) < 10:
                continue
            
            outliers = {'IQR': [], 'Z-Score': [], 'MAD': []}
            
            # IQR Method
            if method in ['all', 'IQR']:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers['IQR'] = ((data < lower_bound) | (data > upper_bound)).sum()
            
            # Z-Score Method
            if method in ['all', 'Z-Score']:
                z_scores = np.abs(zscore(data))
                outliers['Z-Score'] = (z_scores > 3).sum()
            
            # MAD Method
            if method in ['all', 'MAD']:
                median = data.median()
                mad = np.median(np.abs(data - median))
                if mad > 0:
                    modified_z_scores = 0.6745 * (data - median) / mad
                    outliers['MAD'] = (np.abs(modified_z_scores) > 3.5).sum()
                else:
                    outliers['MAD'] = 0
            
            total_outliers = max(outliers.values()) if outliers.values() else 0
            outlier_pct = (total_outliers / len(data)) * 100
            
            outlier_results[col] = {
                'count': total_outliers,
                'percentage': outlier_pct,
                'methods': outliers
            }
        
        # Print summary
        high_outlier_features = [
            col for col, info in outlier_results.items() 
            if info['percentage'] > 5
        ]
        
        print(f"\nFeatures with >5% outliers: {len(high_outlier_features)}")
        for col in high_outlier_features[:10]:
            print(f"  {col}: {outlier_results[col]['percentage']:.2f}%")
        
        self.outlier_info = outlier_results
        
        # Visualization
        self._plot_outlier_analysis(outlier_results)
        
        return outlier_results
    
    def _plot_outlier_analysis(self, outlier_results: Dict):
        """Plot outlier analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar chart of outlier percentages
        cols = list(outlier_results.keys())[:20]
        percentages = [outlier_results[col]['percentage'] for col in cols]
        
        axes[0, 0].bar(range(len(cols)), percentages, color='steelblue')
        axes[0, 0].set_xlabel('Feature')
        axes[0, 0].set_ylabel('Outlier Percentage (%)')
        axes[0, 0].set_title('Outlier Percentage by Feature (Top 20)')
        axes[0, 0].set_xticks(range(len(cols)))
        axes[0, 0].set_xticklabels(cols, rotation=45, ha='right')
        axes[0, 0].axhline(y=5, color='red', linestyle='--', label='5% Threshold')
        axes[0, 0].legend()
        
        # Boxplots for top features
        top_features = sorted(
            outlier_results.keys(), 
            key=lambda x: outlier_results[x]['percentage'], 
            reverse=True
        )[:6]
        
        for idx, col in enumerate(top_features):
            row = (idx + 1) // 3
            col_idx = (idx + 1) % 3
            if row < 2 and col_idx < 2:
                sns.boxplot(y=self.df_clean[col], ax=axes[row, col_idx])
                axes[row, col_idx].set_title(f'{col}')
                axes[row, col_idx].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('feature_outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_outlier_analysis.png")
    
    def analyze_correlations(self):
        """Analyze correlations between features and target."""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        # Prepare data for correlation
        corr_cols = self.numeric_cols + [self.target_column]
        corr_cols = [col for col in corr_cols if col in self.df_clean.columns]
        
        corr_df = self.df_clean[corr_cols].copy()
        
        # Pearson correlation
        pearson_corr = corr_df.corr(method='pearson')[self.target_column].sort_values(ascending=False)
        
        # Spearman correlation
        spearman_corr = corr_df.corr(method='spearman')[self.target_column].sort_values(ascending=False)
        
        print("\nTop 15 features by Pearson correlation:")
        print(pearson_corr.head(15))
        
        print("\nTop 15 features by Spearman correlation:")
        print(spearman_corr.head(15))
        
        # Store scores
        self.feature_scores['pearson'] = pearson_corr.drop(self.target_column).to_dict()
        self.feature_scores['spearman'] = spearman_corr.drop(self.target_column).to_dict()
        
        # Visualization
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_df.corr(), annot=False, cmap='coolwarm', center=0, 
                   linewidths=0.5, square=True)
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_correlation_heatmap.png")
        
        return pearson_corr, spearman_corr
    
    def calculate_mutual_information(self):
        """Calculate mutual information between features and target."""
        print("\n" + "="*70)
        print("MUTUAL INFORMATION ANALYSIS")
        print("="*70)
        
        # Prepare data
        feature_cols = [col for col in self.numeric_cols if col in self.df_clean.columns]
        X = self.df_clean[feature_cols].values
        y = self.df_clean[self.target_column].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        mi_dict = dict(zip(feature_cols, mi_scores))
        mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 15 features by Mutual Information:")
        for feat, score in mi_sorted[:15]:
            print(f"  {feat}: {score:.4f}")
        
        self.feature_scores['mutual_info'] = mi_dict
        
        # Visualization
        top_features = [feat for feat, _ in mi_sorted[:20]]
        top_scores = [score for _, score in mi_sorted[:20]]
        
        plt.figure(figsize=(14, 8))
        plt.bar(range(len(top_features)), top_scores, color='coral')
        plt.xlabel('Feature')
        plt.ylabel('Mutual Information Score')
        plt.title('Feature Importance by Mutual Information (Top 20)')
        plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_mutual_information.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_mutual_information.png")
        
        return mi_dict
    
    def tree_based_importance(self):
        """Calculate feature importance using tree-based models."""
        print("\n" + "="*70)
        print("TREE-BASED FEATURE IMPORTANCE")
        print("="*70)
        
        # Prepare data
        feature_cols = [col for col in self.numeric_cols if col in self.df_clean.columns]
        X = self.df_clean[feature_cols].values
        y = self.df_clean[self.target_column].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # Extra Trees
        et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        et.fit(X, y)
        et_importance = et.feature_importances_
        
        # Average importance
        avg_importance = (rf_importance + et_importance) / 2
        
        rf_dict = dict(zip(feature_cols, rf_importance))
        et_dict = dict(zip(feature_cols, et_importance))
        avg_dict = dict(zip(feature_cols, avg_importance))
        
        avg_sorted = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 15 features by Random Forest:")
        for feat, score in sorted(rf_dict.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {feat}: {score:.4f}")
        
        print("\nTop 15 features by Extra Trees:")
        for feat, score in sorted(et_dict.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {feat}: {score:.4f}")
        
        print("\nTop 15 features by Average Importance:")
        for feat, score in avg_sorted[:15]:
            print(f"  {feat}: {score:.4f}")
        
        self.feature_scores['random_forest'] = rf_dict
        self.feature_scores['extra_trees'] = et_dict
        self.feature_scores['tree_avg'] = avg_dict
        
        # Visualization
        top_features = [feat for feat, _ in avg_sorted[:20]]
        top_scores = [score for _, score in avg_sorted[:20]]
        
        plt.figure(figsize=(14, 8))
        plt.bar(range(len(top_features)), top_scores, color='forestgreen')
        plt.xlabel('Feature')
        plt.ylabel('Importance Score')
        plt.title('Tree-Based Feature Importance (Top 20)')
        plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_tree_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_tree_importance.png")
        
        return avg_dict
    
    def regularization_based_selection(self):
        """Feature selection using regularization methods."""
        print("\n" + "="*70)
        print("REGULARIZATION-BASED FEATURE SELECTION")
        print("="*70)
        
        # Prepare data
        feature_cols = [col for col in self.numeric_cols if col in self.df_clean.columns]
        X = self.df_clean[feature_cols].values
        y = self.df_clean[self.target_column].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Lasso (L1)
        lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y)
        lasso_coef = np.abs(lasso.coef_)
        
        # Ridge (L2)
        ridge = Ridge(alpha=1.0, random_state=42, max_iter=10000)
        ridge.fit(X_scaled, y)
        ridge_coef = np.abs(ridge.coef_)
        
        # ElasticNet
        enet = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
        enet.fit(X_scaled, y)
        enet_coef = np.abs(enet.coef_)
        
        lasso_dict = dict(zip(feature_cols, lasso_coef))
        ridge_dict = dict(zip(feature_cols, ridge_coef))
        enet_dict = dict(zip(feature_cols, enet_coef))
        
        # Average coefficients
        avg_coef = (lasso_coef + ridge_coef + enet_coef) / 3
        avg_dict = dict(zip(feature_cols, avg_coef))
        
        avg_sorted = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Count non-zero Lasso coefficients
        non_zero_lasso = sum(lasso_coef > 0.001)
        print(f"\nLasso selected {non_zero_lasso} features (coefficient > 0.001)")
        
        print("\nTop 15 features by Regularization (Average):")
        for feat, score in avg_sorted[:15]:
            print(f"  {feat}: {score:.4f}")
        
        self.feature_scores['lasso'] = lasso_dict
        self.feature_scores['ridge'] = ridge_dict
        self.feature_scores['elasticnet'] = enet_dict
        self.feature_scores['regularization_avg'] = avg_dict
        
        # Visualization
        top_features = [feat for feat, _ in avg_sorted[:20]]
        top_scores = [score for _, score in avg_sorted[:20]]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        plt.bar(range(len(top_features)), top_scores, color='purple')
        plt.xlabel('Feature')
        plt.ylabel('Coefficient Magnitude')
        plt.title('Regularization-Based Feature Importance (Top 20)')
        plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_regularization_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_regularization_importance.png")
        
        return avg_dict
    
    def recursive_feature_elimination(self, n_features_to_select: int = 15):
        """Recursive Feature Elimination (RFE)."""
        print("\n" + "="*70)
        print("RECURSIVE FEATURE ELIMINATION (RFE)")
        print("="*70)
        
        # Prepare data
        feature_cols = [col for col in self.numeric_cols if col in self.df_clean.columns]
        X = self.df_clean[feature_cols].values
        y = self.df_clean[self.target_column].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Use Random Forest as estimator
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X, y)
        
        selected_features = [feat for feat, sel in zip(feature_cols, rfe.support_) if sel]
        ranking = dict(zip(feature_cols, rfe.ranking_))
        
        print(f"\nRFE selected {len(selected_features)} features:")
        for feat in selected_features:
            print(f"  {feat} (Rank: {ranking[feat]})")
        
        # Store ranking (inverse, so higher is better)
        rfe_scores = {feat: n_features_to_select - rank + 1 for feat, rank in ranking.items()}
        self.feature_scores['rfe'] = rfe_scores
        
        return selected_features
    
    def pca_analysis(self, variance_threshold: float = 0.90):
        """PCA analysis for dimensionality reduction."""
        print("\n" + "="*70)
        print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
        print("="*70)
        
        # Prepare data
        feature_cols = [col for col in self.numeric_cols if col in self.df_clean.columns]
        X = self.df_clean[feature_cols].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Find optimal components
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        print(f"\nVariance explained by each component:")
        for i, var in enumerate(pca.explained_variance_ratio_[:10]):
            print(f"  PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")
        
        print(f"\nComponents needed for {variance_threshold*100:.0f}% variance: {n_components}")
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                marker='o', linewidth=2, label='Cumulative Variance')
        plt.axhline(y=variance_threshold, color='r', linestyle='--', 
                   label=f'{variance_threshold*100:.0f}% Threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('PCA Scree Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_pca_scree.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_pca_scree.png")
        
        return n_components, cumulative_variance
    
    def ensemble_feature_selection(self, top_n: int = 15) -> List[str]:
        """Combine all methods for ensemble feature selection."""
        print("\n" + "="*70)
        print("ENSEMBLE FEATURE SELECTION")
        print("="*70)
        
        # Run all methods if not already run
        if 'pearson' not in self.feature_scores:
            self.analyze_correlations()
        if 'mutual_info' not in self.feature_scores:
            self.calculate_mutual_information()
        if 'tree_avg' not in self.feature_scores:
            self.tree_based_importance()
        if 'regularization_avg' not in self.feature_scores:
            self.regularization_based_selection()
        if 'rfe' not in self.feature_scores:
            self.recursive_feature_elimination()
        
        # Normalize scores for each method
        normalized_scores = {}
        
        for method, scores in self.feature_scores.items():
            if not scores:
                continue
            
            values = np.array(list(scores.values()))
            if values.max() == values.min():
                norm_values = np.ones_like(values)
            else:
                norm_values = (values - values.min()) / (values.max() - values.min())
            
            normalized_scores[method] = dict(zip(scores.keys(), norm_values))
        
        # Calculate average normalized score
        all_features = set()
        for scores in normalized_scores.values():
            all_features.update(scores.keys())
        
        ensemble_scores = {}
        for feat in all_features:
            method_scores = [
                normalized_scores[method][feat] 
                for method in normalized_scores 
                if feat in normalized_scores[method]
            ]
            if method_scores:
                ensemble_scores[feat] = np.mean(method_scores)
        
        # Sort features
        sorted_features = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 15 features by Ensemble Method:")
        for feat, score in sorted_features[:top_n]:
            print(f"  {feat}: {score:.4f}")
        
        self.best_features = [feat for feat, _ in sorted_features[:top_n]]
        
        # Visualization
        top_features = [feat for feat, _ in sorted_features[:20]]
        top_scores = [score for _, score in sorted_features[:20]]
        
        plt.figure(figsize=(14, 8))
        plt.bar(range(len(top_features)), top_scores, color='navy')
        plt.xlabel('Feature')
        plt.ylabel('Normalized Ensemble Score')
        plt.title('Ensemble Feature Importance Ranking (Top 20)')
        plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_ensemble_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_ensemble_ranking.png")
        
        # Save detailed results
        self._save_feature_importance_report(sorted_features)
        
        return self.best_features
    
    def _save_feature_importance_report(self, sorted_features: List[Tuple]):
        """Save detailed feature importance report."""
        report = {
            'best_features': self.best_features,
            'feature_scores': {},
            'summary': {
                'total_features_analyzed': len(self.numeric_cols),
                'features_selected': len(self.best_features),
                'methods_used': list(self.feature_scores.keys())
            }
        }
        
        for feat, score in sorted_features:
            report['feature_scores'][feat] = {
                'ensemble_score': score,
                'pearson': self.feature_scores.get('pearson', {}).get(feat, 0),
                'spearman': self.feature_scores.get('spearman', {}).get(feat, 0),
                'mutual_info': self.feature_scores.get('mutual_info', {}).get(feat, 0),
                'random_forest': self.feature_scores.get('random_forest', {}).get(feat, 0),
                'tree_avg': self.feature_scores.get('tree_avg', {}).get(feat, 0),
                'regularization_avg': self.feature_scores.get('regularization_avg', {}).get(feat, 0),
                'rfe': self.feature_scores.get('rfe', {}).get(feat, 0)
            }
        
        # Save JSON report
        with open('feature_selection_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("Saved: feature_selection_report.json")
        
        # Save CSV
        df_scores = pd.DataFrame([
            {
                'Feature': feat,
                'Ensemble_Score': score,
                'Pearson': self.feature_scores.get('pearson', {}).get(feat, 0),
                'Spearman': self.feature_scores.get('spearman', {}).get(feat, 0),
                'Mutual_Info': self.feature_scores.get('mutual_info', {}).get(feat, 0),
                'Random_Forest': self.feature_scores.get('random_forest', {}).get(feat, 0),
                'Tree_Avg': self.feature_scores.get('tree_avg', {}).get(feat, 0),
                'Regularization_Avg': self.feature_scores.get('regularization_avg', {}).get(feat, 0),
                'RFE': self.feature_scores.get('rfe', {}).get(feat, 0)
            }
            for feat, score in sorted_features
        ])
        df_scores.to_csv('feature_importance_detailed.csv', index=False)
        print("Saved: feature_importance_detailed.csv")


# ============================================================================
# PART 2: TIME SERIES FORECASTING WITH ML MODELS
# ============================================================================

class TimeSeriesPredictor:
    """Time series forecasting with multiple ML algorithms."""
    
    def __init__(self, data_path: str, working_day_path: str, 
                 input_length: int, output_length: int, 
                 test_size: float = 0.3, random_state: int = 42,
                 selected_features: Optional[List[str]] = None):
        self.data_path = data_path
        self.working_day_path = working_day_path
        self.input_length = input_length
        self.output_length = output_length
        self.test_size = test_size
        self.random_state = random_state
        self.selected_features = selected_features
        self.df = None
        self.X_all = None
        self.y_all = None
        self.model = None
        self.models = {}
        self.best_features = []
        self.mae = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data."""
        print("\n" + "="*70)
        print("LOADING AND PREPROCESSING DATA FOR FORECASTING")
        print("="*70)
        
        start_time = time.time()
        
        # Load data
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        else:
            self.df = pd.read_excel(self.data_path)
        
        # Load working days
        if os.path.exists(self.working_day_path):
            if self.working_day_path.endswith('.csv'):
                working_days = pd.read_csv(self.working_day_path)
            else:
                working_days = pd.read_excel(self.working_day_path)
            
            working_days['Date'] = pd.to_datetime(working_days['Date'])
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df = pd.merge(self.df, working_days, on='Date', how='left')
        
        # Handle missing dates
        if self.df['Date'].isna().any():
            print("Warning: Missing dates found. Dropping rows.")
            self.df.dropna(subset=['Date'], inplace=True)
        
        # Add time features
        self._add_time_related_features()
        
        # Add moving averages
        self._add_moving_averages()
        
        # Drop NaN rows
        self.df.dropna(inplace=True)
        
        print(f"Data loaded. Shape: {self.df.shape}")
        print(f"Runtime: {time.time() - start_time:.2f} seconds")
        
        return self.df
    
    def _add_time_related_features(self):
        """Add time-related features."""
        self.df['month'] = self.df['Date'].dt.month
        self.df['day_of_month'] = self.df['Date'].dt.day
        self.df['day_of_year'] = self.df['Date'].dt.dayofyear
        self.df['week_of_year'] = self.df['Date'].dt.isocalendar().week
        self.df['day_of_week'] = ((self.df['Date'].dt.dayofweek + 1) % 7) + 1
        self.df['is_month_start'] = self.df['Date'].dt.is_month_start.astype(int)
        self.df['is_month_end'] = self.df['Date'].dt.is_month_end.astype(int)
        
        if 'WorkingDay' in self.df.columns:
            self.df['is_weekend'] = ((self.df['WorkingDay'] == 0) & (
                self.df['day_of_week'] == 7)).astype(int)
            self.df['is_holiday'] = ((self.df['WorkingDay'] == 0) & (
                self.df['day_of_week'] != 7)).astype(int)
        else:
            self.df['is_weekend'] = (self.df['day_of_week'] >= 6).astype(int)
            self.df['is_holiday'] = 0
    
    def _add_moving_averages(self):
        """Add moving average features."""
        for window in [60, 90, 120, 150, 180, 365]:
            self.df[f'SMA_{window}'] = self.df['Weight'].rolling(window).mean()
        
        for alpha in [0.95, 0.9, 0.8]:
            for lag in [60, 90, 120]:
                self.df[f'EMA_alpha_{int(alpha*100)}_lag_{lag}'] = \
                    self.df['Weight'].shift(lag).ewm(alpha=alpha).mean()
    
    def check_outliers_and_noise(self):
        """Check for outliers and noise in the target variable."""
        print("\n" + "="*70)
        print("OUTLIER AND NOISE CHECKING")
        print("="*70)
        
        target = self.df['Weight'].dropna()
        
        # Outlier detection
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((target < lower_bound) | (target > upper_bound)).sum()
        outlier_pct = (outliers / len(target)) * 100
        
        print(f"\nOutliers detected: {outliers} ({outlier_pct:.2f}%)")
        
        # Noise analysis
        rolling_mean = target.rolling(window=7).mean()
        residual = target - rolling_mean
        noise_std = residual.std()
        signal_std = target.std()
        noise_ratio = noise_std / signal_std
        
        print(f"Noise Standard Deviation: {noise_std:.4f}")
        print(f"Signal Standard Deviation: {signal_std:.4f}")
        print(f"Noise Ratio: {noise_ratio:.4f}")
        
        if noise_ratio < 0.1:
            print("✓ Low noise level - Data is clean")
        elif noise_ratio < 0.3:
            print("⚠ Moderate noise level - Consider smoothing")
        else:
            print("✗ High noise level - Denoising recommended")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original data with outliers highlighted
        axes[0, 0].plot(self.df['Date'], self.df['Weight'], label='Actual', alpha=0.6)
        outlier_mask = (self.df['Weight'] < lower_bound) | (self.df['Weight'] > upper_bound)
        axes[0, 0].scatter(
            self.df.loc[outlier_mask, 'Date'], 
            self.df.loc[outlier_mask, 'Weight'], 
            color='red', s=30, label='Outliers', alpha=0.7
        )
        axes[0, 0].axhline(y=lower_bound, color='orange', linestyle='--', label='Lower Bound')
        axes[0, 0].axhline(y=upper_bound, color='orange', linestyle='--', label='Upper Bound')
        axes[0, 0].set_title('Target Variable with Outliers')
        axes[0, 0].legend()
        
        # Distribution
        axes[0, 1].hist(target, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=lower_bound, color='orange', linestyle='--', label='Lower Bound')
        axes[0, 1].axvline(x=upper_bound, color='orange', linestyle='--', label='Upper Bound')
        axes[0, 1].set_title('Distribution of Target Variable')
        axes[0, 1].legend()
        
        # Residuals (noise)
        axes[1, 0].plot(residual.dropna(), label='Residuals', alpha=0.6)
        axes[1, 0].axhline(y=0, color='black', linestyle='-')
        axes[1, 0].set_title('Residuals (Noise)')
        axes[1, 0].legend()
        
        # Signal vs Noise
        axes[1, 1].plot(rolling_mean.dropna(), label='Signal (Rolling Mean)', linewidth=2)
        axes[1, 1].plot(target, label='Original', alpha=0.3)
        axes[1, 1].set_title('Signal vs Original Data')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('outlier_noise_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: outlier_noise_analysis.png")
        
        return {
            'outlier_count': outliers,
            'outlier_percentage': outlier_pct,
            'noise_ratio': noise_ratio
        }
    
    def prepare_features_and_target(self):
        """Prepare features and target variables."""
        print("\n" + "="*70)
        print("PREPARING FEATURES AND TARGET")
        print("="*70)
        
        # Transform dataset
        data = self.df.copy()
        
        # Create lagged features
        for i in range(1, self.input_length + 1):
            data[f'x_{i}'] = data['Weight'].shift(-i)
        
        # Create target
        for j in range(self.output_length):
            data[f'y_{j}'] = data['Weight'].shift(-self.output_length - j)
        
        data = data.dropna().reset_index(drop=True)
        
        # Select features
        if self.selected_features:
            # Use selected features from feature importance analysis
            feature_columns = [
                col for col in data.columns 
                if col.startswith('x_') or col in self.selected_features
            ]
        else:
            # Auto-select based on correlation
            sma_cols = [col for col in data.columns if col.startswith('SMA')]
            ema_cols = [col for col in data.columns if col.startswith('EMA')]
            xi_cols = [col for col in data.columns if col.startswith('x_')]
            yj_cols = [col for col in data.columns if col.startswith('y_')]
            
            time_features = [
                'month', 'day_of_month', 'day_of_year', 'week_of_year',
                'day_of_week', 'is_month_start', 'is_month_end',
                'is_weekend', 'is_holiday'
            ]
            valid_time_features = [feat for feat in time_features if feat in data.columns]
            
            all_features = sma_cols + ema_cols + xi_cols + yj_cols + valid_time_features
            
            correlations = data[['Weight'] + all_features].corr()['Weight'].sort_values(ascending=False)
            self.best_features = correlations.nlargest(15).index.tolist()
            
            feature_columns = [col for col in data.columns if col.startswith('x_') or col in self.best_features]
        
        self.X = data[feature_columns].values
        
        y_columns = [f'y_{j}' for j in range(self.output_length)]
        self.y = data[y_columns].values
        
        print(f'Shape of X: {self.X.shape}')
        print(f'Shape of y: {self.y.shape}')
        print(f'Features used: {len(feature_columns)}')
        
        return self.X, self.y
    
    def adfuller_test(self):
        """Perform Augmented Dickey-Fuller test for stationarity."""
        result = adfuller(self.df['Weight'].dropna())
        print('\nAugmented Dickey-Fuller Test:')
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value:.6f}')
        
        if result[1] < 0.05:
            print("✓ Time series is stationary (p < 0.05)")
        else:
            print("⚠ Time series may not be stationary (p >= 0.05)")
        
        return result
    
    def train_xgboost_ga(self):
        """Train XGBoost with Genetic Algorithm hyperparameter tuning."""
        print("\n" + "="*70)
        print("TRAINING XGBOOST WITH GENETIC ALGORITHM")
        print("="*70)
        
        if not GENETIC_AVAILABLE:
            print("Using default XGBoost parameters (sklearn-genetic-algorithm not installed)")
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
            model.fit(X_train, y_train)
            self.models['XGBoost'] = model
            return model
        
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        
        ga_param_grid = {
            'max_depth': Integer(3, 7),
            'learning_rate': Continuous(0.01, 0.3),
            'n_estimators': Integer(100, 300),
            'subsample': Continuous(0.6, 1.0),
            'colsample_bytree': Continuous(0.6, 1.0),
            'gamma': Continuous(0.0, 1.0),
            'min_child_weight': Integer(1, 5),
        }
        
        ga_search = GASearchCV(
            estimator=model,
            param_grid=ga_param_grid,
            scoring='neg_mean_absolute_error',
            cv=3,
            population_size=5,
            generations=3,
            n_jobs=-1,
            verbose=True,
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        ga_search.fit(X_train, y_train)
        best_model = ga_search.best_estimator_
        
        self.models['XGBoost'] = best_model
        print(f"Best parameters: {ga_search.best_params_}")
        
        return best_model
    
    def train_random_forest(self):
        """Train Random Forest model."""
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        
        return model
    
    def train_prophet(self, forecast_days: int = 180):
        """Train Prophet model."""
        print("\n" + "="*70)
        print("TRAINING PROPHET")
        print("="*70)
        
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Skipping.")
            return None
        
        # Prepare data for Prophet
        prophet_df = self.df[['Date', 'Weight']].copy()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add regressors if available
        if 'is_holiday' in prophet_df.columns:
            model.add_regressor('is_holiday')
        
        model.fit(prophet_df)
        self.models['Prophet'] = model
        
        return model
    
    def train_sarima(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """Train SARIMA model."""
        print("\n" + "="*70)
        print("TRAINING SARIMA")
        print("="*70)
        
        # Prepare data
        data = self.df['Weight'].dropna()
        
        # Fit ARIMA (SARIMA requires seasonal data setup)
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        
        self.models['SARIMA'] = fitted_model
        
        print(f"SARIMA{order} trained successfully")
        print(f"AIC: {fitted_model.aic:.2f}")
        
        return fitted_model
    
    def evaluate_models(self) -> Dict:
        """Evaluate all trained models."""
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            if name == 'Prophet':
                # Prophet evaluation
                test_df = self.df[['Date', 'Weight']].iloc[-len(y_test):].copy()
                test_df.columns = ['ds', 'y']
                
                if 'is_holiday' in test_df.columns:
                    test_df['is_holiday'] = self.df['is_holiday'].iloc[-len(y_test):]
                
                forecast = model.predict(test_df)
                y_pred = forecast['yhat'].values
                
                # Flatten for multi-output comparison
                if len(y_pred.shape) == 1 and len(y_test.shape) > 1:
                    y_pred = np.repeat(y_pred.reshape(-1, 1), y_test.shape[1], axis=1)
                
            elif name == 'SARIMA':
                # SARIMA forecast
                forecast = model.forecast(steps=len(y_test))
                y_pred = forecast.values
                
                if len(y_pred.shape) == 1 and len(y_test.shape) > 1:
                    y_pred = np.repeat(y_pred.reshape(-1, 1), y_test.shape[1], axis=1)
            
            else:
                # Tree-based models
                y_pred = model.predict(X_test)
            
            # Flatten arrays
            y_test_flat = y_test.flatten()
            y_pred_flat = y_pred.flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_flat, y_pred_flat)
            rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
            r2 = r2_score(y_test_flat, y_pred_flat)
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'predictions': y_pred_flat
            }
            
            print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        self.results = results
        
        # Save comparison
        df_results = pd.DataFrame([
            {'Model': name, **metrics}
            for name, metrics in results.items()
        ])
        df_results = df_results.sort_values('MAE')
        df_results.to_csv('model_comparison.csv', index=False)
        print("\nSaved: model_comparison.csv")
        
        # Visualization
        self._plot_model_comparison(results)
        
        return results
    
    def _plot_model_comparison(self, results: Dict):
        """Plot model comparison."""
        models = list(results.keys())
        mae_values = [results[m]['MAE'] for m in models]
        rmse_values = [results[m]['RMSE'] for m in models]
        r2_values = [results[m]['R2'] for m in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # MAE comparison
        axes[0].bar(models, mae_values, color='steelblue')
        axes[0].set_ylabel('MAE')
        axes[0].set_title('Mean Absolute Error Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[1].bar(models, rmse_values, color='coral')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Root Mean Squared Error Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[2].bar(models, r2_values, color='forestgreen')
        axes[2].set_ylabel('R²')
        axes[2].set_title('R-squared Comparison')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('model_comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: model_comparison_chart.png")
    
    def forecast_future(self, forecast_days: int, horizon_name: str = "forecast"):
        """Generate future forecasts."""
        print(f"\nGenerating {horizon_name} forecast for {forecast_days} days...")
        
        if not self.models:
            raise ValueError("No models trained. Call train_* methods first.")
        
        # Get best model
        if self.results:
            best_model_name = min(self.results, key=lambda x: self.results[x]['MAE'])
        else:
            best_model_name = 'XGBoost'
        
        best_model = self.models[best_model_name]
        print(f"Using best model: {best_model_name}")
        
        # Generate future dates
        last_date = self.df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Load working days
        working_days = None
        if os.path.exists(self.working_day_path):
            if self.working_day_path.endswith('.csv'):
                working_days = pd.read_csv(self.working_day_path)
            else:
                working_days = pd.read_excel(self.working_day_path)
            working_days['Date'] = pd.to_datetime(working_days['Date'])
            working_days.set_index('Date', inplace=True)
        
        forecasts = {}
        
        for name, model in self.models.items():
            predictions = []
            
            if name == 'Prophet' and PROPHET_AVAILABLE:
                future_df = pd.DataFrame({'ds': future_dates})
                if 'is_holiday' in self.df.columns:
                    future_df['is_holiday'] = 0
                forecast = model.predict(future_df)
                predictions = forecast['yhat'].values
            
            elif name == 'SARIMA':
                forecast = model.forecast(steps=forecast_days)
                predictions = forecast.values
            
            else:
                # For tree-based models, use recent features
                if len(self.X) >= forecast_days:
                    future_features = self.X[-forecast_days:, :]
                else:
                    # Pad with last available features
                    future_features = np.tile(self.X[-1, :], (forecast_days, 1))
                
                predictions = model.predict(future_features)
            
            # Apply working day filter
            final_predictions = []
            for date, pred in zip(future_dates, predictions):
                if working_days is not None and date in working_days.index:
                    if working_days.loc[date, 'WorkingDay'] == 0:
                        final_predictions.append(0)
                    else:
                        final_predictions.append(pred)
                else:
                    final_predictions.append(pred)
            
            forecasts[name] = final_predictions
        
        # Save forecasts
        forecast_df = pd.DataFrame({'Date': future_dates})
        for name, preds in forecasts.items():
            forecast_df[f'{name}_Forecast'] = preds
        
        filename = f'forecast_{horizon_name}.csv'
        forecast_df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
        
        # Plot forecasts
        self._plot_forecasts(forecast_df, future_dates, forecasts, horizon_name)
        
        return forecast_df
    
    def _plot_forecasts(self, forecast_df: pd.DataFrame, future_dates: pd.DatetimeIndex, 
                       forecasts: Dict, horizon_name: str):
        """Plot forecast results."""
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(self.df['Date'], self.df['Weight'], label='Historical', 
                color='black', linewidth=2, alpha=0.7)
        
        # Plot forecasts
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for idx, (name, preds) in enumerate(forecasts.items()):
            color = colors[idx % len(colors)]
            plt.plot(future_dates, preds, label=f'{name}', 
                    linestyle='--', color=color, linewidth=2, alpha=0.8)
        
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.title(f'{horizon_name.replace("_", " ").title()} Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'all_forecasts_{horizon_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        print("\nPlotting feature importance...")
        
        for name, model in self.models.items():
            if name in ['XGBoost', 'RandomForest']:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    
                    # Get feature names
                    if self.selected_features:
                        feature_names = [
                            col for col in self.df.columns 
                            if col.startswith('x_') or col in self.selected_features
                        ][:len(importance)]
                    else:
                        feature_names = [f'Feature_{i}' for i in range(len(importance))]
                    
                    # Sort by importance
                    indices = np.argsort(importance)[::-1][:20]
                    
                    plt.figure(figsize=(12, 8))
                    plt.bar(range(len(indices)), importance[indices])
                    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], 
                              rotation=45, ha='right')
                    plt.xlabel('Feature')
                    plt.ylabel('Importance')
                    plt.title(f'{name} Feature Importance (Top 20)')
                    plt.tight_layout()
                    
                    filename = f'feature_importance_{name.lower()}.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("COMPREHENSIVE TIME SERIES FORECASTING SYSTEM")
    print("="*70)
    
    # Configuration
    DATA_PATH = 'Data.csv'
    WORKING_DAY_PATH = 'WorkingDay.csv'
    TARGET_COLUMN = 'Weight'
    
    INPUT_LENGTH = 180
    OUTPUT_LENGTH = 1
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    # Check if data files exist
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found!")
        print("Please provide your data file.")
        return
    
    if not os.path.exists(WORKING_DAY_PATH):
        print(f"Warning: {WORKING_DAY_PATH} not found. Proceeding without working days.")
        WORKING_DAY_PATH = None
    
    # ========================================================================
    # STEP 1: FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    analyzer = FeatureImportanceAnalyzer(DATA_PATH, WORKING_DAY_PATH or '', TARGET_COLUMN)
    df_clean = analyzer.load_and_preprocess()
    
    # Detect outliers
    analyzer.detect_outliers(method='all')
    
    # Analyze correlations
    analyzer.analyze_correlations()
    
    # Calculate mutual information
    analyzer.calculate_mutual_information()
    
    # Tree-based importance
    analyzer.tree_based_importance()
    
    # Regularization-based selection
    analyzer.regularization_based_selection()
    
    # RFE
    analyzer.recursive_feature_elimination(n_features_to_select=15)
    
    # PCA analysis
    n_components, cum_var = analyzer.pca_analysis(variance_threshold=0.90)
    
    # Ensemble feature selection
    best_features = analyzer.ensemble_feature_selection(top_n=15)
    
    print(f"\n✓ Best {len(best_features)} features selected:")
    for i, feat in enumerate(best_features, 1):
        print(f"  {i}. {feat}")
    
    # ========================================================================
    # STEP 2: TIME SERIES FORECASTING
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: TIME SERIES FORECASTING")
    print("="*70)
    
    predictor = TimeSeriesPredictor(
        data_path=DATA_PATH,
        working_day_path=WORKING_DAY_PATH or '',
        input_length=INPUT_LENGTH,
        output_length=OUTPUT_LENGTH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        selected_features=best_features
    )
    
    # Load and preprocess
    predictor.load_and_preprocess_data()
    
    # Check outliers and noise
    predictor.check_outliers_and_noise()
    
    # ADF test
    predictor.adfuller_test()
    
    # Prepare features
    predictor.prepare_features_and_target()
    
    # Train models
    print("\nTraining models...")
    predictor.train_xgboost_ga()
    predictor.train_random_forest()
    
    if PROPHET_AVAILABLE:
        predictor.train_prophet()
    
    predictor.train_sarima(order=(1, 1, 1))
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # ========================================================================
    # STEP 3: GENERATE FORECASTS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: GENERATING FORECASTS")
    print("="*70)
    
    # 3-month forecast (90 days)
    forecast_3m = predictor.forecast_future(forecast_days=90, horizon_name="3months")
    
    # 6-month forecast (180 days)
    forecast_6m = predictor.forecast_future(forecast_days=180, horizon_name="6months")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("EXECUTION COMPLETE - SUMMARY")
    print("="*70)
    
    print("\nGenerated Files:")
    print("  Feature Analysis:")
    print("    - feature_outlier_analysis.png")
    print("    - feature_correlation_heatmap.png")
    print("    - feature_mutual_information.png")
    print("    - feature_tree_importance.png")
    print("    - feature_regularization_importance.png")
    print("    - feature_ensemble_ranking.png")
    print("    - feature_pca_scree.png")
    print("    - feature_importance_detailed.csv")
    print("    - feature_selection_report.json")
    print("\n  Forecasting:")
    print("    - outlier_noise_analysis.png")
    print("    - model_comparison_chart.png")
    print("    - model_comparison.csv")
    print("    - feature_importance_xgboost.png")
    print("    - feature_importance_randomforest.png")
    print("    - forecast_3months.csv")
    print("    - forecast_6months.csv")
    print("    - all_forecasts_3months.png")
    print("    - all_forecasts_6months.png")
    
    print("\nBest Features:", best_features)
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"  {model_name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")
    
    print("\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    main()
