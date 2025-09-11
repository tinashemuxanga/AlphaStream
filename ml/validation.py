"""
Data Validation and Quality Module

Ensures data integrity and quality before model training/inference.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    outliers: Dict[str, int]
    duplicates: int
    data_types: Dict[str, str]
    statistics: Dict[str, Dict[str, float]]
    issues: List[str]
    warnings: List[str]
    passed: bool


class DataValidator:
    """Validate and check data quality."""
    
    def __init__(
        self,
        missing_threshold: float = 0.1,
        outlier_std: float = 4.0,
        min_samples: int = 100
    ):
        """
        Initialize data validator.
        
        Args:
            missing_threshold: Maximum allowed missing data ratio
            outlier_std: Standard deviations for outlier detection
            min_samples: Minimum required samples
        """
        self.missing_threshold = missing_threshold
        self.outlier_std = outlier_std
        self.min_samples = min_samples
        
    def validate_ohlcv(self, data: pd.DataFrame) -> DataQualityReport:
        """
        Validate OHLCV data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Data quality report
        """
        issues = []
        warnings = []
        
        # Basic shape validation
        total_rows = len(data)
        total_columns = len(data.columns)
        
        if total_rows < self.min_samples:
            issues.append(f"Insufficient data: {total_rows} rows < {self.min_samples} minimum")
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check missing values
        missing_values = {}
        for col in data.columns:
            missing_count = data[col].isna().sum()
            missing_ratio = missing_count / total_rows
            missing_values[col] = missing_count
            
            if missing_ratio > self.missing_threshold:
                issues.append(f"Column '{col}' has {missing_ratio:.1%} missing values")
        
        # Check data consistency
        if all(col in data.columns for col in ['high', 'low', 'close']):
            # High should be >= Low
            invalid_hl = (data['high'] < data['low']).sum()
            if invalid_hl > 0:
                issues.append(f"Found {invalid_hl} rows where High < Low")
            
            # Close should be between High and Low
            invalid_close = ((data['close'] > data['high']) | (data['close'] < data['low'])).sum()
            if invalid_close > 0:
                warnings.append(f"Found {invalid_close} rows where Close outside High-Low range")
        
        # Check for outliers
        outliers = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                z_scores = np.abs(stats.zscore(col_data))
                outlier_count = (z_scores > self.outlier_std).sum()
                outliers[col] = int(outlier_count)
                
                if outlier_count > total_rows * 0.05:  # More than 5% outliers
                    warnings.append(f"Column '{col}' has {outlier_count} potential outliers")
        
        # Check for duplicates
        duplicates = data.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")
        
        # Check data types
        data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Calculate statistics
        statistics = {}
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                statistics[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'skew': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis())
                }
        
        # Check for zero or negative prices
        if 'close' in data.columns:
            zero_prices = (data['close'] <= 0).sum()
            if zero_prices > 0:
                issues.append(f"Found {zero_prices} zero or negative prices")
        
        # Check for zero volume
        if 'volume' in data.columns:
            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > total_rows * 0.1:  # More than 10% zero volume
                warnings.append(f"Found {zero_volume} periods with zero volume")
        
        # Check time series continuity
        if isinstance(data.index, pd.DatetimeIndex):
            time_diff = data.index.to_series().diff()
            expected_freq = time_diff.mode()[0] if len(time_diff.mode()) > 0 else None
            
            if expected_freq:
                gaps = time_diff[time_diff > expected_freq * 2]
                if len(gaps) > 0:
                    warnings.append(f"Found {len(gaps)} time gaps in data")
        
        passed = len(issues) == 0
        
        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            outliers=outliers,
            duplicates=duplicates,
            data_types=data_types,
            statistics=statistics,
            issues=issues,
            warnings=warnings,
            passed=passed
        )
    
    def validate_features(self, features: pd.DataFrame) -> DataQualityReport:
        """
        Validate feature matrix.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Data quality report
        """
        issues = []
        warnings = []
        
        total_rows = len(features)
        total_columns = len(features.columns)
        
        # Check for infinite values
        inf_counts = {}
        for col in features.columns:
            inf_count = np.isinf(features[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
                issues.append(f"Column '{col}' has {inf_count} infinite values")
        
        # Check for constant features
        constant_features = []
        for col in features.columns:
            if features[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            warnings.append(f"Found {len(constant_features)} constant features")
        
        # Check feature correlation
        correlation_matrix = features.corr()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            warnings.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.95)")
        
        # Check missing values
        missing_values = {}
        for col in features.columns:
            missing_count = features[col].isna().sum()
            missing_values[col] = missing_count
            
            if missing_count > total_rows * self.missing_threshold:
                issues.append(f"Feature '{col}' has too many missing values: {missing_count}/{total_rows}")
        
        # Check for outliers
        outliers = {}
        for col in features.columns:
            col_data = features[col].dropna()
            if len(col_data) > 0:
                z_scores = np.abs(stats.zscore(col_data))
                outlier_count = (z_scores > self.outlier_std).sum()
                outliers[col] = int(outlier_count)
        
        # Calculate statistics
        statistics = {}
        for col in features.columns:
            col_data = features[col].dropna()
            if len(col_data) > 0:
                statistics[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max())
                }
        
        passed = len(issues) == 0
        
        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            outliers=outliers,
            duplicates=0,
            data_types={col: str(dtype) for col, dtype in features.dtypes.items()},
            statistics=statistics,
            issues=issues,
            warnings=warnings,
            passed=passed
        )
    
    def clean_data(
        self,
        data: pd.DataFrame,
        handle_missing: str = 'forward_fill',
        remove_outliers: bool = False,
        clip_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Clean and preprocess data.
        
        Args:
            data: Input DataFrame
            handle_missing: Method to handle missing values
            remove_outliers: Whether to remove outliers
            clip_outliers: Whether to clip outliers
            
        Returns:
            Cleaned DataFrame
        """
        cleaned = data.copy()
        
        # Handle missing values
        if handle_missing == 'forward_fill':
            cleaned = cleaned.fillna(method='ffill')
        elif handle_missing == 'backward_fill':
            cleaned = cleaned.fillna(method='bfill')
        elif handle_missing == 'interpolate':
            cleaned = cleaned.interpolate(method='linear')
        elif handle_missing == 'drop':
            cleaned = cleaned.dropna()
        elif handle_missing == 'mean':
            for col in cleaned.select_dtypes(include=[np.number]).columns:
                cleaned[col].fillna(cleaned[col].mean(), inplace=True)
        
        # Handle outliers
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = cleaned[col].dropna()
            if len(col_data) > 0:
                z_scores = np.abs(stats.zscore(cleaned[col]))
                
                if remove_outliers:
                    # Remove outliers
                    cleaned = cleaned[z_scores <= self.outlier_std]
                elif clip_outliers:
                    # Clip outliers
                    lower = col_data.quantile(0.001)
                    upper = col_data.quantile(0.999)
                    cleaned[col] = cleaned[col].clip(lower, upper)
        
        # Remove duplicates
        cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
        
        # Sort by index
        cleaned = cleaned.sort_index()
        
        return cleaned


class ModelValidator:
    """Validate model predictions and performance."""
    
    def __init__(self):
        """Initialize model validator."""
        self.validation_results = []
        
    def validate_predictions(
        self,
        predictions: np.ndarray,
        expected_classes: Optional[List] = None,
        check_probability: bool = False
    ) -> Dict[str, Any]:
        """
        Validate model predictions.
        
        Args:
            predictions: Model predictions
            expected_classes: Expected class labels
            check_probability: Whether to check probability values
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Check shape
        if len(predictions) == 0:
            results['valid'] = False
            results['issues'].append("Empty predictions")
            return results
        
        # Check for NaN
        nan_count = np.isnan(predictions).sum()
        if nan_count > 0:
            results['valid'] = False
            results['issues'].append(f"Found {nan_count} NaN predictions")
        
        # Check for infinite values
        inf_count = np.isinf(predictions).sum()
        if inf_count > 0:
            results['valid'] = False
            results['issues'].append(f"Found {inf_count} infinite predictions")
        
        # Check classes
        if expected_classes:
            unique_preds = np.unique(predictions[~np.isnan(predictions)])
            unexpected = set(unique_preds) - set(expected_classes)
            if unexpected:
                results['valid'] = False
                results['issues'].append(f"Unexpected predictions: {unexpected}")
        
        # Check probabilities
        if check_probability:
            if np.any((predictions < 0) | (predictions > 1)):
                results['valid'] = False
                results['issues'].append("Probability values outside [0, 1] range")
            
            # Check if probabilities sum to 1 (for multi-class)
            if len(predictions.shape) > 1:
                row_sums = predictions.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-6):
                    results['issues'].append("Probabilities don't sum to 1")
        
        # Calculate statistics
        results['stats'] = {
            'count': len(predictions),
            'unique_values': len(np.unique(predictions[~np.isnan(predictions)])),
            'distribution': pd.Series(predictions).value_counts().to_dict()
        }
        
        return results
    
    def validate_performance(
        self,
        metrics: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Validate model performance metrics.
        
        Args:
            metrics: Performance metrics
            thresholds: Minimum acceptable thresholds
            
        Returns:
            Validation results
        """
        default_thresholds = {
            'accuracy': 0.5,
            'f1': 0.3,
            'sharpe_ratio': 0.0,
            'max_drawdown': -0.3
        }
        
        if thresholds:
            default_thresholds.update(thresholds)
        
        results = {
            'passed': True,
            'failures': [],
            'warnings': []
        }
        
        for metric, threshold in default_thresholds.items():
            if metric in metrics:
                value = metrics[metric]
                
                if metric == 'max_drawdown':
                    # Drawdown is negative, so check if it's worse than threshold
                    if value < threshold:
                        results['passed'] = False
                        results['failures'].append(
                            f"{metric}: {value:.3f} < {threshold:.3f} (threshold)"
                        )
                else:
                    # Higher is better for other metrics
                    if value < threshold:
                        results['passed'] = False
                        results['failures'].append(
                            f"{metric}: {value:.3f} < {threshold:.3f} (threshold)"
                        )
                
                # Add warnings for borderline performance
                if metric == 'sharpe_ratio' and 0 < value < 0.5:
                    results['warnings'].append(f"Low Sharpe ratio: {value:.3f}")
                elif metric == 'accuracy' and 0.5 < value < 0.55:
                    results['warnings'].append(f"Near-random accuracy: {value:.3f}")
        
        return results


if __name__ == "__main__":
    print("AlphaStream Data Validation Module")
    print("Components: DataValidator, ModelValidator")
    print("Ensures data quality and model reliability")
