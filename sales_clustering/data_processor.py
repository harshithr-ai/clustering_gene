"""
Data Processing Module
Handles data loading, quality assessment, preprocessing, and validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from typing import Dict, List, Tuple, Optional
import config


class DataProcessor:
    """Enterprise-grade data processor with quality assessment"""
    
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.quality_report = {}
        self.preprocessing_log = []
        self.encoders = {}
        self.scaler = None
        self.feature_statistics = {}
        
    def load_data(self, df: pd.DataFrame) -> Dict:
        """Load and perform initial quality assessment"""
        self.original_data = df.copy()
        
        # Comprehensive quality assessment
        quality_metrics = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'missing_rate': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'duplicate_records': df.duplicated().sum(),
            'data_quality_score': self._calculate_quality_score(df),
            'feature_variance_issues': self._check_low_variance_features(df),
            'outlier_percentage': self._estimate_outliers(df)
        }
        
        self.quality_report = quality_metrics
        return quality_metrics
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-1)"""
        scores = []
        
        # Completeness score
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        scores.append(completeness)
        
        # Uniqueness score
        uniqueness = 1 - (df.duplicated().sum() / len(df))
        scores.append(uniqueness)
        
        # Variance score (for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            variances = df[numeric_cols].var()
            variance_score = (variances > config.MIN_FEATURE_VARIANCE).mean()
            scores.append(variance_score)
        
        return np.mean(scores)
    
    def _check_low_variance_features(self, df: pd.DataFrame) -> List[str]:
        """Identify features with low variance"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_variance_features = []
        
        for col in numeric_cols:
            if df[col].var() < config.MIN_FEATURE_VARIANCE:
                low_variance_features.append(col)
        
        return low_variance_features
    
    def _estimate_outliers(self, df: pd.DataFrame) -> float:
        """Estimate percentage of outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0
        
        outlier_mask = pd.Series([False] * len(df))
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask |= (df[col] < (Q1 - config.OUTLIER_IQR_MULTIPLIER * IQR)) | \
                           (df[col] > (Q3 + config.OUTLIER_IQR_MULTIPLIER * IQR))
        
        return outlier_mask.sum() / len(df)
    
    def preprocess(self, 
                   imputation_strategy: str = 'advanced',
                   scaling_method: str = 'robust',
                   handle_outliers: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Advanced preprocessing pipeline
        
        Args:
            imputation_strategy: 'simple', 'advanced' (KNN), or 'statistical'
            scaling_method: 'standard', 'robust', or 'minmax'
            handle_outliers: Whether to cap outliers
        """
        df = self.original_data.copy()
        self.preprocessing_log = []
        
        # Step 1: Handle missing values
        df, missing_log = self._handle_missing_values(df, imputation_strategy)
        self.preprocessing_log.extend(missing_log)
        
        # Step 2: Encode categorical variables
        df, encoding_log = self._encode_categorical(df)
        self.preprocessing_log.extend(encoding_log)
        
        # Step 3: Handle outliers
        if handle_outliers:
            df, outlier_log = self._handle_outliers(df)
            self.preprocessing_log.extend(outlier_log)
        
        # Step 4: Remove low-variance features
        df, variance_log = self._remove_low_variance_features(df)
        self.preprocessing_log.extend(variance_log)
        
        # Step 5: Calculate feature statistics
        self._calculate_feature_statistics(df)
        
        self.processed_data = df
        
        # Return processed data and numeric feature names
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return df, numeric_features
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced missing value imputation"""
        log = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    if strategy == 'advanced' and missing_pct < 30:
                        # Use KNN imputation for better accuracy
                        imputer = KNNImputer(n_neighbors=5)
                        df[col] = imputer.fit_transform(df[[col]])
                        log.append(f"Advanced KNN imputation applied to '{col}' "
                                 f"({missing_count} values, {missing_pct:.1f}%)")
                    else:
                        # Use median for robustness
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        log.append(f"Median imputation applied to '{col}' "
                                 f"({missing_count} values, {missing_pct:.1f}%)")
                else:
                    # Categorical: use mode
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    log.append(f"Mode imputation applied to categorical '{col}' "
                             f"({missing_count} values, {missing_pct:.1f}%)")
        
        return df, log
    
    def _encode_categorical(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Encode categorical variables with label encoding"""
        log = []
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            # Convert to string to handle any remaining NaN
            df[col] = df[col].astype(str)
            
            # Label encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
            
            log.append(f"Label encoding applied to '{col}' ({unique_count} unique categories)")
        
        return df, log
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Cap outliers using IQR method"""
        log = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - config.OUTLIER_IQR_MULTIPLIER * IQR
            upper_bound = Q3 + config.OUTLIER_IQR_MULTIPLIER * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outlier_count > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_pct = (outlier_count / len(df)) * 100
                log.append(f"Outliers capped in '{col}' ({outlier_count} values, {outlier_pct:.1f}%)")
        
        return df, log
    
    def _remove_low_variance_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with insufficient variance"""
        log = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        to_remove = []
        
        for col in numeric_cols:
            variance = df[col].var()
            if variance < config.MIN_FEATURE_VARIANCE:
                to_remove.append(col)
                log.append(f"Removed low-variance feature '{col}' (variance: {variance:.6f})")
        
        if to_remove:
            df = df.drop(columns=to_remove)
        
        return df, log
    
    def _calculate_feature_statistics(self, df: pd.DataFrame):
        """Calculate comprehensive statistics for each feature"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.feature_statistics[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'cv': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0,  # Coefficient of variation
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
            }
    
    def scale_features(self, features: List[str], method: str = 'robust') -> np.ndarray:
        """
        Scale selected features
        
        Args:
            features: List of feature names to scale
            method: 'standard' or 'robust' (recommended for outliers)
        """
        if method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        X = self.processed_data[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.preprocessing_log.append(f"{method.capitalize()} scaling applied to {len(features)} features")
        
        return X_scaled
    
    def get_feature_correlations(self, features: List[str]) -> pd.DataFrame:
        """Calculate feature correlation matrix"""
        return self.processed_data[features].corr()
    
    def detect_multicollinearity(self, features: List[str], threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """Detect highly correlated feature pairs"""
        corr_matrix = self.get_feature_correlations(features)
        high_corr = []
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr.append((features[i], features[j], corr_val))
        
        return high_corr
