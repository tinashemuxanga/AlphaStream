"""
Model Monitoring and Drift Detection Module

Monitors model performance, detects drift, and triggers retraining.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics snapshot."""
    timestamp: datetime
    model_name: str
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    mse: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    prediction_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'mse': self.mse,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'prediction_count': self.prediction_count
        }


@dataclass
class DriftReport:
    """Data/concept drift detection report."""
    timestamp: datetime
    drift_detected: bool
    drift_score: float
    drift_type: str  # 'data', 'concept', 'performance'
    features_drifted: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(
        self,
        model_name: str,
        baseline_metrics: Optional[Dict[str, float]] = None,
        drift_threshold: float = 0.1,
        performance_window: int = 100,
        storage_path: Optional[str] = None
    ):
        """
        Initialize model monitor.
        
        Args:
            model_name: Name of the model to monitor
            baseline_metrics: Baseline performance metrics
            drift_threshold: Threshold for drift detection
            performance_window: Window size for performance monitoring
            storage_path: Path to store monitoring data
        """
        self.model_name = model_name
        self.baseline_metrics = baseline_metrics or {}
        self.drift_threshold = drift_threshold
        self.performance_window = performance_window
        self.storage_path = Path(storage_path) if storage_path else Path('monitoring')
        
        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[ModelMetrics] = []
        self.predictions_buffer = []
        self.features_buffer = []
        self.actuals_buffer = []
        
        # Load existing metrics
        self._load_metrics_history()
        
    def record_prediction(
        self,
        features: pd.Series,
        prediction: Any,
        actual: Optional[Any] = None
    ):
        """
        Record a prediction for monitoring.
        
        Args:
            features: Input features
            prediction: Model prediction
            actual: Actual value (if available)
        """
        self.predictions_buffer.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual
        })
        
        self.features_buffer.append(features)
        
        if actual is not None:
            self.actuals_buffer.append(actual)
        
        # Check if we should evaluate performance
        if len(self.predictions_buffer) >= self.performance_window:
            self._evaluate_performance()
    
    def _evaluate_performance(self):
        """Evaluate recent model performance."""
        # Get predictions with actuals
        recent_preds = [p for p in self.predictions_buffer[-self.performance_window:] 
                       if p['actual'] is not None]
        
        if len(recent_preds) < 10:  # Need minimum samples
            return
        
        predictions = [p['prediction'] for p in recent_preds]
        actuals = [p['actual'] for p in recent_preds]
        
        # Calculate metrics
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            model_name=self.model_name,
            prediction_count=len(predictions)
        )
        
        # Classification metrics
        try:
            metrics.accuracy = accuracy_score(actuals, predictions)
            metrics.f1 = f1_score(actuals, predictions, average='weighted')
        except:
            pass
        
        # Regression metrics
        try:
            metrics.mse = mean_squared_error(actuals, predictions)
        except:
            pass
        
        # Store metrics
        self.metrics_history.append(metrics)
        self._save_metrics(metrics)
        
        # Check for performance degradation
        self._check_performance_drift(metrics)
        
        # Clear old buffer entries
        if len(self.predictions_buffer) > self.performance_window * 2:
            self.predictions_buffer = self.predictions_buffer[-self.performance_window:]
    
    def _check_performance_drift(self, current_metrics: ModelMetrics) -> Optional[DriftReport]:
        """
        Check for performance drift.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Drift report if drift detected
        """
        if not self.baseline_metrics:
            return None
        
        drift_report = DriftReport(
            timestamp=datetime.now(),
            drift_detected=False,
            drift_score=0.0,
            drift_type='performance'
        )
        
        # Compare with baseline
        for metric_name in ['accuracy', 'f1', 'mse']:
            baseline_value = self.baseline_metrics.get(metric_name)
            current_value = getattr(current_metrics, metric_name)
            
            if baseline_value is not None and current_value is not None:
                if metric_name == 'mse':
                    # For MSE, higher is worse
                    degradation = (current_value - baseline_value) / baseline_value
                else:
                    # For accuracy/f1, lower is worse
                    degradation = (baseline_value - current_value) / baseline_value
                
                if degradation > self.drift_threshold:
                    drift_report.drift_detected = True
                    drift_report.drift_score = max(drift_report.drift_score, degradation)
                    drift_report.details[metric_name] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation': degradation
                    }
        
        if drift_report.drift_detected:
            drift_report.recommendations.append("Consider retraining the model")
            drift_report.recommendations.append("Review recent data quality")
            drift_report.recommendations.append("Check for concept drift")
            
            logger.warning(f"Performance drift detected for {self.model_name}: {drift_report.drift_score:.2%}")
            self._save_drift_report(drift_report)
        
        return drift_report if drift_report.drift_detected else None
    
    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        method: str = 'ks'
    ) -> DriftReport:
        """
        Detect data drift between reference and current data.
        
        Args:
            reference_data: Reference/training data
            current_data: Current/production data
            method: Drift detection method ('ks', 'chi2', 'psi')
            
        Returns:
            Drift report
        """
        drift_report = DriftReport(
            timestamp=datetime.now(),
            drift_detected=False,
            drift_score=0.0,
            drift_type='data'
        )
        
        drifted_features = []
        
        for column in reference_data.columns:
            if column not in current_data.columns:
                continue
            
            ref_col = reference_data[column].dropna()
            curr_col = current_data[column].dropna()
            
            if len(ref_col) == 0 or len(curr_col) == 0:
                continue
            
            # Kolmogorov-Smirnov test
            if method == 'ks':
                if ref_col.dtype in ['float64', 'int64']:
                    statistic, p_value = stats.ks_2samp(ref_col, curr_col)
                    
                    if p_value < 0.05:  # Significant drift
                        drifted_features.append(column)
                        drift_report.details[column] = {
                            'method': 'ks',
                            'statistic': statistic,
                            'p_value': p_value
                        }
            
            # Population Stability Index (PSI)
            elif method == 'psi':
                psi = self._calculate_psi(ref_col, curr_col)
                if psi > 0.2:  # Significant drift
                    drifted_features.append(column)
                    drift_report.details[column] = {
                        'method': 'psi',
                        'psi': psi
                    }
        
        if drifted_features:
            drift_report.drift_detected = True
            drift_report.features_drifted = drifted_features
            drift_report.drift_score = len(drifted_features) / len(reference_data.columns)
            
            drift_report.recommendations.append(f"Retrain with recent data")
            drift_report.recommendations.append(f"Review features: {', '.join(drifted_features[:5])}")
            
            logger.warning(f"Data drift detected in {len(drifted_features)} features")
        
        return drift_report
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """
        Calculate Population Stability Index.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins
            
        Returns:
            PSI value
        """
        # Create bins from reference data
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize
        ref_dist = (ref_counts + 1) / (len(reference) + bins)
        curr_dist = (curr_counts + 1) / (len(current) + bins)
        
        # Calculate PSI
        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
        
        return psi
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """
        Get monitoring dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        # Recent metrics
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        
        # Performance trend
        performance_trend = {}
        if recent_metrics:
            for metric_name in ['accuracy', 'f1', 'mse']:
                values = [getattr(m, metric_name) for m in recent_metrics 
                         if getattr(m, metric_name) is not None]
                if values:
                    performance_trend[metric_name] = {
                        'current': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'trend': 'improving' if len(values) > 1 and values[-1] > values[-2] else 'declining'
                    }
        
        # Prediction statistics
        recent_predictions = self.predictions_buffer[-100:]
        pred_stats = {}
        if recent_predictions:
            preds = [p['prediction'] for p in recent_predictions]
            pred_stats = {
                'total_predictions': len(self.predictions_buffer),
                'recent_predictions': len(recent_predictions),
                'prediction_distribution': pd.Series(preds).value_counts().to_dict()
            }
        
        return {
            'model_name': self.model_name,
            'last_updated': datetime.now().isoformat(),
            'baseline_metrics': self.baseline_metrics,
            'performance_trend': performance_trend,
            'prediction_statistics': pred_stats,
            'recent_metrics': [m.to_dict() for m in recent_metrics],
            'alerts': self._get_alerts()
        }
    
    def _get_alerts(self) -> List[Dict[str, str]]:
        """Get current alerts."""
        alerts = []
        
        # Check recent performance
        if self.metrics_history:
            latest = self.metrics_history[-1]
            
            if latest.accuracy and latest.accuracy < 0.5:
                alerts.append({
                    'level': 'critical',
                    'message': f'Accuracy below 50%: {latest.accuracy:.2%}',
                    'timestamp': latest.timestamp.isoformat()
                })
            
            if latest.f1 and latest.f1 < 0.3:
                alerts.append({
                    'level': 'warning',
                    'message': f'F1 score below 0.3: {latest.f1:.3f}',
                    'timestamp': latest.timestamp.isoformat()
                })
        
        # Check prediction volume
        recent_count = len([p for p in self.predictions_buffer 
                          if datetime.now() - p['timestamp'] < timedelta(hours=1)])
        if recent_count == 0:
            alerts.append({
                'level': 'info',
                'message': 'No predictions in the last hour',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def _save_metrics(self, metrics: ModelMetrics):
        """Save metrics to storage."""
        metrics_file = self.storage_path / f"{self.model_name}_metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def _save_drift_report(self, report: DriftReport):
        """Save drift report."""
        drift_file = self.storage_path / f"{self.model_name}_drift.jsonl"
        with open(drift_file, 'a') as f:
            f.write(json.dumps({
                'timestamp': report.timestamp.isoformat(),
                'drift_detected': report.drift_detected,
                'drift_score': report.drift_score,
                'drift_type': report.drift_type,
                'features_drifted': report.features_drifted,
                'recommendations': report.recommendations,
                'details': report.details
            }) + '\n')
    
    def _load_metrics_history(self):
        """Load historical metrics."""
        metrics_file = self.storage_path / f"{self.model_name}_metrics.jsonl"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    metrics = ModelMetrics(
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        model_name=data['model_name'],
                        accuracy=data.get('accuracy'),
                        f1=data.get('f1'),
                        precision=data.get('precision'),
                        recall=data.get('recall'),
                        mse=data.get('mse'),
                        sharpe_ratio=data.get('sharpe_ratio'),
                        max_drawdown=data.get('max_drawdown'),
                        prediction_count=data.get('prediction_count', 0)
                    )
                    self.metrics_history.append(metrics)


class MonitoringService:
    """Central monitoring service for all models."""
    
    def __init__(self, storage_path: str = 'monitoring'):
        """Initialize monitoring service."""
        self.monitors: Dict[str, ModelMonitor] = {}
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def register_model(
        self,
        model_name: str,
        baseline_metrics: Dict[str, float]
    ) -> ModelMonitor:
        """
        Register a model for monitoring.
        
        Args:
            model_name: Name of the model
            baseline_metrics: Baseline performance metrics
            
        Returns:
            Model monitor instance
        """
        monitor = ModelMonitor(
            model_name=model_name,
            baseline_metrics=baseline_metrics,
            storage_path=str(self.storage_path)
        )
        self.monitors[model_name] = monitor
        logger.info(f"Registered model {model_name} for monitoring")
        return monitor
    
    def get_monitor(self, model_name: str) -> Optional[ModelMonitor]:
        """Get monitor for a model."""
        return self.monitors.get(model_name)
    
    def get_all_dashboards(self) -> Dict[str, Any]:
        """Get dashboards for all monitored models."""
        return {
            name: monitor.get_monitoring_dashboard()
            for name, monitor in self.monitors.items()
        }
    
    def check_all_models(self) -> List[Tuple[str, DriftReport]]:
        """Check all models for issues."""
        issues = []
        for name, monitor in self.monitors.items():
            if monitor.metrics_history:
                latest = monitor.metrics_history[-1]
                drift_report = monitor._check_performance_drift(latest)
                if drift_report and drift_report.drift_detected:
                    issues.append((name, drift_report))
        return issues


if __name__ == "__main__":
    print("AlphaStream Model Monitoring Module")
    print("Components: ModelMonitor, MonitoringService")
    print("Tracks performance, detects drift, and triggers alerts")
