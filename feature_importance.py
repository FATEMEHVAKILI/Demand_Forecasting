"""
Feature Importance & Selection Module for Time Series Forecasting
Integrates multiple feature selection methods to choose optimal features before ML modeling
"""

from termcolor import cprint
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis with multiple selection methods
    """
    
    def __init__(self, data_path, working_day_path=None, target_column='Weight'):
        self.data_path = data_path
        self.working_day_path = working_day_path
        self.target_column = target_column
        self.df = None
        self.df_clean = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.feature_importance_results = {}
        self.selected_features = []
        self.scaler = RobustScaler()
        
    def load_and_preprocess(self):
        """Load data and perform initial preprocessing"""
        print("="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        start_time = time.time()
        
        # Load main data
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        print(f"Original data shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Convert Date column
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Load working days if provided
        if self.working_day_path and os.path.exists(self.working_day_path):
            if self.working_day_path.endswith('.csv'):
                working_days = pd.read_csv(self.working_day_path)
            else:
                working_days = pd.read_excel(self.working_day_path)
            
            working_days['Date'] = pd.to_datetime(working_days['Date'])
            self.df = pd.merge(self.df, working_days, on='Date', how='left')
            print(f"Merged with working days data. New shape: {self.df.shape}")
        
        # Separate numeric and categorical columns
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from feature lists temporarily
        if self.target_column in self.numeric_cols:
            self.numeric_cols.remove(self.target_column)
        if self.target_column in self.categorical_cols:
            self.categorical_cols.remove(self.target_column)
        
        print(f"\nNumeric features ({len(self.numeric_cols)}): {self.numeric_cols}")
        print(f"Categorical features ({len(self.categorical_cols)}): {self.categorical_cols}")
        
        # Handle missing values
        print("\n--- Handling Missing Values ---")
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        cols_with_missing = missing_percent[missing_percent > 0]
        
        if len(cols_with_missing) > 0:
            print(f"Columns with missing values:\n{cols_with_missing}")
            
            # Fill numeric columns with median
            for col in self.numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  {col}: filled {self.df[col].isnull().sum()} missing values with median {median_val:.2f}")
            
            # Fill categorical columns with mode
            for col in self.categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    print(f"  {col}: filled with mode '{mode_val}'")
        else:
            print("No missing values found!")
        
        self.df_clean = self.df.copy()
        print(f"\nData loading and preprocessing completed in {time.time() - start_time:.2f} seconds")
        return self.df_clean
    
    def outlier_analysis(self, method='all', threshold=1.5):
        """
        Detect outliers using multiple methods without removing them
        Methods: 'iqr', 'zscore', 'mad', 'all'
        """
        print("\n" + "="*60)
        print("OUTLIER ANALYSIS")
        print("="*60)
        
        outlier_summary = {}
        
        for col in self.numeric_cols:
            data = self.df_clean[col].dropna()
            
            # IQR Method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((data < Q1 - threshold * IQR) | (data > Q3 + threshold * IQR)).sum()
            
            # Z-Score Method
            mean_val = data.mean()
            std_val = data.std()
            z_scores = np.abs((data - mean_val) / std_val)
            zscore_outliers = (z_scores > 3).sum()
            
            # MAD Method
            median_val = data.median()
            mad = np.median(np.abs(data - median_val))
            mad_outliers = (np.abs(data - median_val) > threshold * mad * 1.4826).sum()
            
            outlier_summary[col] = {
                'IQR_Outliers': iqr_outliers,
                'ZScore_Outliers': zscore_outliers,
                'MAD_Outliers': mad_outliers,
                'Total_Rows': len(data),
                'IQR_Percent': (iqr_outliers / len(data)) * 100,
                'ZScore_Percent': (zscore_outliers / len(data)) * 100,
                'MAD_Percent': (mad_outliers / len(data)) * 100
            }
        
        # Create summary DataFrame
        outlier_df = pd.DataFrame(outlier_summary).T
        print("\nOutlier Detection Summary:")
        print(outlier_df[['IQR_Outliers', 'ZScore_Outliers', 'MAD_Outliers', 
                         'IQR_Percent', 'ZScore_Percent', 'MAD_Percent']].round(2))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Boxplots for top features
        top_features = self.numeric_cols[:min(6, len(self.numeric_cols))]
        for i, col in enumerate(top_features):
            row = i // 3
            col_idx = i % 3
            if row < 2 and col_idx < 2:
                sns.boxplot(y=self.df_clean[col], ax=axes[row, col_idx])
                axes[row, col_idx].set_title(f'Boxplot: {col}')
        
        # Hide unused subplots
        for i in range(len(top_features), 4):
            row = i // 2
            col_idx = i % 2
            if row < 2 and col_idx < 2:
                fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        plt.savefig('feature_outlier_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Outlier visualization saved to 'feature_outlier_analysis.png'")
        plt.show()
        
        # Identify high-outlier features
        high_outlier_features = outlier_df[outlier_df['IQR_Percent'] > 5].index.tolist()
        if high_outlier_features:
            print(f"\n⚠ Features with >5% outliers (consider transformation): {high_outlier_features}")
        else:
            print("\n✓ No features with excessive outliers detected")
        
        return outlier_df
    
    def correlation_analysis(self):
        """Analyze correlations between features and target"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Prepare data for correlation
        corr_data = self.df_clean[self.numeric_cols + [self.target_column]].copy()
        
        # Pearson correlation
        pearson_corr = corr_data.corr(method='pearson')[self.target_column].sort_values(ascending=False)
        
        # Spearman correlation (rank-based)
        spearman_corr = corr_data.corr(method='spearman')[self.target_column].sort_values(ascending=False)
        
        print("\nTop 15 Features by Pearson Correlation:")
        print(pearson_corr.head(15))
        
        print("\nTop 15 Features by Spearman Correlation:")
        print(spearman_corr.head(15))
        
        # Heatmap
        plt.figure(figsize=(14, 12))
        correlation_matrix = corr_data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   linewidths=0.5, fmt='.2f', square=True)
        plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("\n✓ Correlation heatmap saved to 'feature_correlation_heatmap.png'")
        plt.show()
        
        # Store results
        self.feature_importance_results['pearson_correlation'] = pearson_corr.to_dict()
        self.feature_importance_results['spearman_correlation'] = spearman_corr.to_dict()
        
        return pearson_corr, spearman_corr
    
    def mutual_information(self):
        """Calculate mutual information between features and target"""
        print("\n" + "="*60)
        print("MUTUAL INFORMATION ANALYSIS")
        print("="*60)
        
        X = self.df_clean[self.numeric_cols].values
        y = self.df_clean[self.target_column].values
        
        # Handle any remaining NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
        
        mi_df = pd.DataFrame({
            'Feature': self.numeric_cols,
            'Mutual_Information': mi_scores
        }).sort_values('Mutual_Information', ascending=False)
        
        print("\nMutual Information Scores:")
        print(mi_df)
        
        # Visualization
        plt.figure(figsize=(12, 8))
        plt.barh(mi_df['Feature'], mi_df['Mutual_Information'])
        plt.xlabel('Mutual Information Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance by Mutual Information', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_mutual_information.png', dpi=300, bbox_inches='tight')
        print("\n✓ Mutual information plot saved to 'feature_mutual_information.png'")
        plt.show()
        
        self.feature_importance_results['mutual_information'] = mi_df.set_index('Feature')['Mutual_Information'].to_dict()
        
        return mi_df
    
    def tree_based_importance(self, n_estimators=100):
        """Calculate feature importance using Random Forest and Extra Trees"""
        print("\n" + "="*60)
        print("TREE-BASED FEATURE IMPORTANCE")
        print("="*60)
        
        X = self.df_clean[self.numeric_cols].values
        y = self.df_clean[self.target_column].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y_clean)
        rf_importance = rf.feature_importances_
        
        # Extra Trees
        print("Training Extra Trees...")
        et = ExtraTreesRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        et.fit(X_scaled, y_clean)
        et_importance = et.feature_importances_
        
        # Create DataFrames
        rf_df = pd.DataFrame({
            'Feature': self.numeric_cols,
            'RF_Importance': rf_importance
        }).sort_values('RF_Importance', ascending=False)
        
        et_df = pd.DataFrame({
            'Feature': self.numeric_cols,
            'ET_Importance': et_importance
        }).sort_values('ET_Importance', ascending=False)
        
        # Combined importance
        combined_df = pd.DataFrame({
            'Feature': self.numeric_cols,
            'RF_Importance': rf_importance,
            'ET_Importance': et_importance,
            'Average_Importance': (rf_importance + et_importance) / 2
        }).sort_values('Average_Importance', ascending=False)
        
        print("\nTop 15 Features by Random Forest Importance:")
        print(rf_df.head(15))
        
        print("\nTop 15 Features by Extra Trees Importance:")
        print(et_df.head(15))
        
        print("\nCombined Feature Importance (Average):")
        print(combined_df.head(15))
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        # RF Importance
        top_n = min(15, len(self.numeric_cols))
        sns.barplot(data=rf_df.head(top_n), x='RF_Importance', y='Feature', ax=axes[0], palette='viridis')
        axes[0].set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Importance Score')
        axes[0].set_ylabel('Feature')
        
        # ET Importance
        sns.barplot(data=et_df.head(top_n), x='ET_Importance', y='Feature', ax=axes[1], palette='plasma')
        axes[1].set_title('Extra Trees Feature Importance', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Importance Score')
        axes[1].set_ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig('feature_tree_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Tree-based importance plots saved to 'feature_tree_importance.png'")
        plt.show()
        
        # Store results
        self.feature_importance_results['random_forest'] = rf_df.set_index('Feature')['RF_Importance'].to_dict()
        self.feature_importance_results['extra_trees'] = et_df.set_index('Feature')['ET_Importance'].to_dict()
        self.feature_importance_results['tree_average'] = combined_df.set_index('Feature')['Average_Importance'].to_dict()
        
        return combined_df
    
    def regularization_based_selection(self):
        """Use Lasso, Ridge, and ElasticNet for feature selection"""
        print("\n" + "="*60)
        print("REGULARIZATION-BASED FEATURE SELECTION")
        print("="*60)
        
        X = self.df_clean[self.numeric_cols].values
        y = self.df_clean[self.target_column].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Lasso (L1) - performs feature selection
        print("\nTraining Lasso (L1)...")
        lasso = Lasso(alpha=0.1, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y_clean)
        lasso_coef = lasso.coef_
        
        # Ridge (L2)
        print("Training Ridge (L2)...")
        ridge = Ridge(alpha=1.0, random_state=42, max_iter=10000)
        ridge.fit(X_scaled, y_clean)
        ridge_coef = ridge.coef_
        
        # ElasticNet
        print("Training ElasticNet...")
        enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)
        enet.fit(X_scaled, y_clean)
        enet_coef = enet.coef_
        
        # Create DataFrames
        lasso_df = pd.DataFrame({
            'Feature': self.numeric_cols,
            'Lasso_Coefficient': lasso_coef,
            'Absolute_Coefficient': np.abs(lasso_coef)
        }).sort_values('Absolute_Coefficient', ascending=False)
        
        ridge_df = pd.DataFrame({
            'Feature': self.numeric_cols,
            'Ridge_Coefficient': ridge_coef,
            'Absolute_Coefficient': np.abs(ridge_coef)
        }).sort_values('Absolute_Coefficient', ascending=False)
        
        enet_df = pd.DataFrame({
            'Feature': self.numeric_cols,
            'ElasticNet_Coefficient': enet_coef,
            'Absolute_Coefficient': np.abs(enet_coef)
        }).sort_values('Absolute_Coefficient', ascending=False)
        
        # Count non-zero coefficients in Lasso
        lasso_non_zero = (lasso_coef != 0).sum()
        print(f"\nLasso selected {lasso_non_zero} features (non-zero coefficients)")
        
        print("\nTop 15 Features by Lasso Coefficient Magnitude:")
        print(lasso_df.head(15))
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        top_n = min(15, len(self.numeric_cols))
        
        sns.barplot(data=lasso_df.head(top_n), x='Absolute_Coefficient', y='Feature', 
                   ax=axes[0], palette='Reds')
        axes[0].set_title('Lasso (L1) Coefficients', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Absolute Coefficient')
        axes[0].set_ylabel('Feature')
        
        sns.barplot(data=ridge_df.head(top_n), x='Absolute_Coefficient', y='Feature', 
                   ax=axes[1], palette='Blues')
        axes[1].set_title('Ridge (L2) Coefficients', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Absolute Coefficient')
        
        sns.barplot(data=enet_df.head(top_n), x='Absolute_Coefficient', y='Feature', 
                   ax=axes[2], palette='Greens')
        axes[2].set_title('ElasticNet Coefficients', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Absolute Coefficient')
        
        plt.tight_layout()
        plt.savefig('feature_regularization_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Regularization-based importance plots saved to 'feature_regularization_importance.png'")
        plt.show()
        
        # Store results
        self.feature_importance_results['lasso'] = lasso_df.set_index('Feature')['Absolute_Coefficient'].to_dict()
        self.feature_importance_results['ridge'] = ridge_df.set_index('Feature')['Absolute_Coefficient'].to_dict()
        self.feature_importance_results['elasticnet'] = enet_df.set_index('Feature')['Absolute_Coefficient'].to_dict()
        
        return lasso_df, ridge_df, enet_df
    
    def recursive_feature_elimination(self, n_features_to_select=10):
        """Perform Recursive Feature Elimination (RFE)"""
        print("\n" + "="*60)
        print("RECURSIVE FEATURE ELIMINATION (RFE)")
        print("="*60)
        
        X = self.df_clean[self.numeric_cols].values
        y = self.df_clean[self.target_column].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Use Random Forest as estimator
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Check if we have enough features for RFE
        if X_scaled.shape[1] < 2:
            print("RFE requires at least 2 features. Skipping...")
            rfe_df = pd.DataFrame({
                'Feature': self.numeric_cols,
                'Selected': [True]*len(self.numeric_cols),
                'Ranking': [1]*len(self.numeric_cols)
            })
            return rfe_df, self.numeric_cols
        
        # Adjust n_features_to_select if needed
        actual_n_features = min(n_features_to_select, X_scaled.shape[1])
        
        rfe = RFE(estimator=estimator, n_features_to_select=actual_n_features, step=1)
        rfe.fit(X_scaled, y_clean)
        
        # Create ranking DataFrame
        rfe_df = pd.DataFrame({
            'Feature': self.numeric_cols,
            'Selected': rfe.support_,
            'Ranking': rfe.ranking_
        }).sort_values('Ranking')
        
        selected_features = rfe_df[rfe_df['Selected']]['Feature'].tolist()
        
        print(f"\nSelected {n_features_to_select} features:")
        for i, feat in enumerate(selected_features, 1):
            print(f"  {i}. {feat}")
        
        print(f"\nRFE Ranking:")
        print(rfe_df)
        
        # Cross-validation score for selected features
        cv_scores = cross_val_score(estimator, X_scaled[:, rfe.support_], y_clean, 
                                   cv=5, scoring='neg_mean_absolute_error')
        print(f"\nCV MAE with selected features: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        self.feature_importance_results['rfe_selected'] = selected_features
        self.feature_importance_results['rfe_ranking'] = rfe_df.set_index('Feature')['Ranking'].to_dict()
        
        return rfe_df, selected_features
    
    def ensemble_feature_selection(self, top_k=15):
        """
        Combine all feature selection methods to get final feature ranking
        """
        print("\n" + "="*60)
        print("ENSEMBLE FEATURE SELECTION")
        print("="*60)
        
        # Collect all rankings
        all_rankings = {}
        
        # Pearson correlation
        if 'pearson_correlation' in self.feature_importance_results:
            pearson = self.feature_importance_results['pearson_correlation']
            pearson_series = pd.Series(pearson).abs().sort_values(ascending=False)
            all_rankings['Pearson'] = pearson_series
        
        # Spearman correlation
        if 'spearman_correlation' in self.feature_importance_results:
            spearman = self.feature_importance_results['spearman_correlation']
            spearman_series = pd.Series(spearman).abs().sort_values(ascending=False)
            all_rankings['Spearman'] = spearman_series
        
        # Mutual Information
        if 'mutual_information' in self.feature_importance_results:
            mi = self.feature_importance_results['mutual_information']
            mi_series = pd.Series(mi).sort_values(ascending=False)
            all_rankings['Mutual_Info'] = mi_series
        
        # Random Forest
        if 'random_forest' in self.feature_importance_results:
            rf = self.feature_importance_results['random_forest']
            rf_series = pd.Series(rf).sort_values(ascending=False)
            all_rankings['Random_Forest'] = rf_series
        
        # Extra Trees
        if 'extra_trees' in self.feature_importance_results:
            et = self.feature_importance_results['extra_trees']
            et_series = pd.Series(et).sort_values(ascending=False)
            all_rankings['Extra_Trees'] = et_series
        
        # Lasso
        if 'lasso' in self.feature_importance_results:
            lasso = self.feature_importance_results['lasso']
            lasso_series = pd.Series(lasso).sort_values(ascending=False)
            all_rankings['Lasso'] = lasso_series
        
        # Normalize and combine rankings
        normalized_scores = {}
        for method, scores in all_rankings.items():
            # Normalize to 0-1 range
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = scores
            normalized_scores[method] = normalized
        
        # Create combined DataFrame
        combined_df = pd.DataFrame(normalized_scores)
        combined_df['Average_Score'] = combined_df.mean(axis=1)
        combined_df = combined_df.sort_values('Average_Score', ascending=False)
        
        print("\nCombined Feature Rankings (Normalized Scores):")
        print(combined_df.round(4))
        
        # Select top-k features
        self.selected_features = combined_df.head(top_k).index.tolist()
        
        print(f"\n*** FINAL SELECTED FEATURES (Top {top_k}) ***")
        for i, feat in enumerate(self.selected_features, 1):
            print(f"  {i}. {feat}")
        
        # Visualization
        plt.figure(figsize=(14, 10))
        top_n = min(20, len(combined_df))
        
        # Create stacked bar chart
        combined_df.head(top_n).plot(kind='barh', stacked=True, figsize=(14, 10), 
                                     colormap='tab20')
        plt.title(f'Top {top_n} Features by Ensemble Method', fontsize=14, fontweight='bold')
        plt.xlabel('Normalized Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig('feature_ensemble_ranking.png', dpi=300, bbox_inches='tight')
        print("\n✓ Ensemble ranking plot saved to 'feature_ensemble_ranking.png'")
        plt.show()
        
        # Save detailed results
        combined_df.to_csv('feature_importance_detailed.csv')
        print("✓ Detailed feature importance saved to 'feature_importance_detailed.csv'")
        
        return combined_df, self.selected_features
    
    def pca_analysis(self, variance_threshold=0.95):
        """Perform PCA to understand feature redundancy"""
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
        print("="*60)
        
        X = self.df_clean[self.numeric_cols].values
        y = self.df_clean[self.target_column].values
        
        # Handle NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        # Cumulative variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Find optimal components
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        print(f"\nVariance explained by each component:")
        for i, var in enumerate(pca.explained_variance_ratio_[:10], 1):
            print(f"  PC{i}: {var:.4f} ({var*100:.2f}%)")
        
        print(f"\nCumulative variance:")
        for i, cum_var in enumerate(cumulative_variance[:10], 1):
            print(f"  PC{i}: {cum_var:.4f} ({cum_var*100:.2f}%)")
        
        print(f"\nComponents needed for {variance_threshold*100}% variance: {n_components}")
        
        # Scree plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                marker='o', linewidth=2, label='Cumulative Variance')
        plt.axhline(y=variance_threshold, color='r', linestyle='--', 
                   label=f'{variance_threshold*100}% Threshold')
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Variance Explained', fontsize=12)
        plt.title('PCA Scree Plot', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_pca_scree.png', dpi=300, bbox_inches='tight')
        print("\n✓ PCA scree plot saved to 'feature_pca_scree.png'")
        plt.show()
        
        # Component loadings
        if len(self.numeric_cols) <= 15:
            loadings_df = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=self.numeric_cols
            )
            print("\nPrincipal Component Loadings:")
            print(loadings_df.round(3))
        
        return n_components, cumulative_variance
    
    def generate_report(self):
        """Generate comprehensive feature importance report"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE SUMMARY REPORT")
        print("="*60)
        
        report = {
            'total_features_analyzed': len(self.numeric_cols),
            'selected_features_count': len(self.selected_features),
            'selected_features': self.selected_features,
            'methods_used': list(self.feature_importance_results.keys()),
            'feature_rankings': {}
        }
        
        # Add top features from each method
        for method, results in self.feature_importance_results.items():
            if isinstance(results, dict):
                sorted_features = sorted(results.items(), key=lambda x: x[1], reverse=True)
                report['feature_rankings'][method] = [f[0] for f in sorted_features[:10]]
        
        # Print summary
        print(f"\nTotal Features Analyzed: {report['total_features_analyzed']}")
        print(f"Final Selected Features: {report['selected_features_count']}")
        print(f"\nMethods Applied:")
        for method in report['methods_used']:
            print(f"  ✓ {method.replace('_', ' ').title()}")
        
        print(f"\nFinal Selected Features for ML Modeling:")
        for i, feat in enumerate(self.selected_features, 1):
            print(f"  {i}. {feat}")
        
        # Save report
        import json
        with open('feature_selection_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("\n✓ Report saved to 'feature_selection_report.json'")
        
        return report


def main():
    """Main execution function"""
    print("="*60)
    print("FEATURE IMPORTANCE & SELECTION ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(
        data_path='Data.csv',
        working_day_path='WorkingDay.csv',
        target_column='Weight'
    )
    
    # Step 1: Load and preprocess
    df_clean = analyzer.load_and_preprocess()
    
    # Step 2: Outlier analysis
    analyzer.outlier_analysis(method='all')
    
    # Step 3: Correlation analysis
    pearson_corr, spearman_corr = analyzer.correlation_analysis()
    
    # Step 4: Mutual information
    mi_df = analyzer.mutual_information()
    
    # Step 5: Tree-based importance
    tree_df = analyzer.tree_based_importance(n_estimators=100)
    
    # Step 6: Regularization-based selection
    lasso_df, ridge_df, enet_df = analyzer.regularization_based_selection()
    
    # Step 7: RFE
    rfe_df, rfe_selected = analyzer.recursive_feature_elimination(n_features_to_select=15)
    
    # Step 8: Ensemble selection
    combined_df, selected_features = analyzer.ensemble_feature_selection(top_k=15)
    
    # Step 9: PCA analysis
    n_components, cumulative_variance = analyzer.pca_analysis(variance_threshold=0.95)
    
    # Step 10: Generate report
    report = analyzer.generate_report()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nRecommended features for ML modeling: {selected_features}")
    print("\nGenerated files:")
    print("  - feature_outlier_analysis.png")
    print("  - feature_correlation_heatmap.png")
    print("  - feature_mutual_information.png")
    print("  - feature_tree_importance.png")
    print("  - feature_regularization_importance.png")
    print("  - feature_ensemble_ranking.png")
    print("  - feature_pca_scree.png")
    print("  - feature_importance_detailed.csv")
    print("  - feature_selection_report.json")
    
    return selected_features


if __name__ == "__main__":
    selected_features = main()
