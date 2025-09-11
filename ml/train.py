"""
Training Pipeline Module

Handles model training, hyperparameter tuning, and experiment tracking
with Weights & Biases integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import optuna
from optuna.samplers import TPESampler

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Experiment tracking
import wandb
import mlflow
from datetime import datetime
import hashlib
import pickle
import joblib

# Internal imports
from .models import ModelFactory
from .dataset import DataPipeline, WalkForwardSplitter
from .features import FeatureEngineer


class ExperimentTracker:
    """Manage experiment tracking with W&B and MLflow."""
    
    def __init__(
        self,
        project_name: str = "alphastream",
        experiment_name: str = None,
        tracking_uri: str = None,
        use_wandb: bool = True,
        use_mlflow: bool = False
    ):
        """
        Initialize experiment tracker.
        
        Args:
            project_name: Project name for tracking
            experiment_name: Specific experiment name
            tracking_uri: MLflow tracking URI
            use_wandb: Whether to use Weights & Biases
            use_mlflow: Whether to use MLflow
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=self.experiment_name,
                reinit=True
            )
            
        if self.use_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(project_name)
            mlflow.start_run(run_name=self.experiment_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.use_wandb:
            wandb.config.update(params)
        if self.use_mlflow:
            mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        if self.use_wandb:
            wandb.log(metrics, step=step)
        if self.use_mlflow:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, file_path: str, artifact_type: str = None):
        """Log artifact file."""
        if self.use_wandb:
            wandb.save(file_path)
        if self.use_mlflow:
            mlflow.log_artifact(file_path, artifact_type)
    
    def log_model(self, model, model_name: str):
        """Log trained model."""
        # Save model locally first
        model_path = f"models/{model_name}.pkl"
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        
        if self.use_wandb:
            wandb.save(model_path)
        if self.use_mlflow:
            mlflow.sklearn.log_model(model, model_name)
    
    def finish(self):
        """Finish tracking."""
        if self.use_wandb:
            wandb.finish()
        if self.use_mlflow:
            mlflow.end_run()


class ModelTrainer:
    """Train and evaluate ML models."""
    
    def __init__(
        self,
        model_type: str,
        model_params: Optional[Dict] = None,
        task_type: str = 'classification',
        experiment_tracker: Optional[ExperimentTracker] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model to train
            model_params: Model hyperparameters
            task_type: Task type ('classification' or 'regression')
            experiment_tracker: Experiment tracking instance
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.task_type = task_type
        self.tracker = experiment_tracker
        
        # Initialize model
        self.model = ModelFactory.create(model_type, task_type, **self.model_params)
        
        # Scaler for features
        self.scaler = None
        self.feature_selector = None
        
    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        scale: bool = True,
        select_features: bool = False,
        n_features: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features.
        
        Args:
            X_train: Training features
            X_test: Test features
            scale: Whether to scale features
            select_features: Whether to select top features
            n_features: Number of features to select
            
        Returns:
            Tuple of processed (X_train, X_test)
        """
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Scale features
        if scale:
            if self.scaler is None:
                self.scaler = RobustScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
            else:
                X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert back to DataFrame
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Feature selection
        if select_features and n_features < X_train.shape[1]:
            if self.feature_selector is None:
                selector_func = f_classif if self.task_type == 'classification' else mutual_info_classif
                self.feature_selector = SelectKBest(selector_func, k=n_features)
                X_train = self.feature_selector.fit_transform(X_train, y_train)
            else:
                X_train = self.feature_selector.transform(X_train)
            X_test = self.feature_selector.transform(X_test)
        
        return X_train, X_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training metrics
        """
        # Prepare validation data
        val_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train model
        self.model.train(X_train, y_train, val_data)
        
        # Get training predictions
        train_preds = self.model.predict(X_train)
        
        # Calculate metrics
        metrics = self.evaluate(y_train, train_preds, prefix='train')
        
        # Validation metrics if available
        if val_data:
            val_preds = self.model.predict(X_val)
            val_metrics = self.evaluate(y_val, val_preds, prefix='val')
            metrics.update(val_metrics)
        
        # Log metrics
        if self.tracker:
            self.tracker.log_metrics(metrics)
        
        return metrics
    
    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        prefix: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate predictions.
        
        Args:
            y_true: True values
            y_pred: Predictions
            prefix: Metric prefix
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if self.task_type == 'classification':
            # Binary classification metrics
            metrics[f'{prefix}_accuracy'] = accuracy_score(y_true, y_pred)
            metrics[f'{prefix}_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics[f'{prefix}_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics[f'{prefix}_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC if binary
            if len(np.unique(y_true)) == 2:
                try:
                    if hasattr(self.model, 'predict_proba'):
                        y_proba = self.model.model.predict_proba(self.model.last_X)[:, 1]
                        metrics[f'{prefix}_roc_auc'] = roc_auc_score(y_true, y_proba)
                except:
                    pass
                    
        else:
            # Regression metrics
            metrics[f'{prefix}_mse'] = mean_squared_error(y_true, y_pred)
            metrics[f'{prefix}_rmse'] = np.sqrt(metrics[f'{prefix}_mse'])
            metrics[f'{prefix}_mae'] = mean_absolute_error(y_true, y_pred)
            
            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics[f'{prefix}_r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance."""
        importance = self.model.get_feature_importance()
        if importance is not None:
            return pd.DataFrame({
                'feature': feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)
        return pd.DataFrame()


class HyperparameterTuner:
    """Hyperparameter tuning with Optuna."""
    
    def __init__(
        self,
        model_type: str,
        task_type: str = 'classification',
        n_trials: int = 50,
        experiment_tracker: Optional[ExperimentTracker] = None
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            model_type: Type of model
            task_type: Task type
            n_trials: Number of optimization trials
            experiment_tracker: Experiment tracker
        """
        self.model_type = model_type
        self.task_type = task_type
        self.n_trials = n_trials
        self.tracker = experiment_tracker
        self.best_params = None
        self.best_score = None
        
    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space."""
        params = {}
        
        if self.model_type == 'random_forest':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
            params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7])
            
        elif self.model_type == 'xgboost':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            params['gamma'] = trial.suggest_float('gamma', 0, 5)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0, 2)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0, 2)
            
        elif self.model_type == 'lightgbm':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 300)
            params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.5, 1.0)
            params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.5, 1.0)
            params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 10)
            params['min_child_samples'] = trial.suggest_int('min_child_samples', 5, 100)
            
        elif self.model_type == 'lstm':
            params['hidden_size'] = trial.suggest_int('hidden_size', 32, 256)
            params['n_layers'] = trial.suggest_int('n_layers', 1, 4)
            params['dropout'] = trial.suggest_float('dropout', 0, 0.5)
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
        elif self.model_type == 'transformer':
            params['d_model'] = trial.suggest_categorical('d_model', [64, 128, 256])
            params['n_heads'] = trial.suggest_categorical('n_heads', [4, 8])
            params['n_layers'] = trial.suggest_int('n_layers', 1, 4)
            params['dropout'] = trial.suggest_float('dropout', 0, 0.5)
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            
        return params
    
    def objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:
        """Objective function for optimization."""
        # Get hyperparameters
        params = self.get_param_space(trial)
        
        # Train model
        trainer = ModelTrainer(
            self.model_type,
            params,
            self.task_type
        )
        
        # Preprocess
        X_train_proc, X_val_proc = trainer.preprocess(X_train, X_val)
        
        # Train
        trainer.train(X_train_proc, y_train, X_val_proc, y_val)
        
        # Evaluate
        val_preds = trainer.predict(X_val_proc)
        metrics = trainer.evaluate(y_val, val_preds)
        
        # Get optimization metric
        if self.task_type == 'classification':
            score = metrics.get('test_f1', 0)
        else:
            score = -metrics.get('test_rmse', float('inf'))
        
        # Log to tracker
        if self.tracker:
            trial_metrics = {f'trial_{trial.number}_{k}': v for k, v in metrics.items()}
            self.tracker.log_metrics(trial_metrics)
        
        return score
    
    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Best parameters
        """
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials
        )
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Log best params
        if self.tracker:
            self.tracker.log_params({'best_' + k: v for k, v in self.best_params.items()})
            self.tracker.log_metrics({'best_score': self.best_score})
        
        return self.best_params


class TrainingPipeline:
    """Complete training pipeline."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for experiment
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.experiment_name = experiment_name
        
        # Initialize tracker
        self.tracker = ExperimentTracker(
            project_name=self.config.get('project_name', 'alphastream'),
            experiment_name=experiment_name,
            use_wandb=self.config.get('use_wandb', True),
            use_mlflow=self.config.get('use_mlflow', False)
        )
        
    def _load_config(self, path: str) -> Dict:
        """Load configuration from file."""
        path = Path(path)
        if path.suffix == '.json':
            with open(path) as f:
                return json.load(f)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path) as f:
                return yaml.safe_load(f)
        return {}
    
    def run(
        self,
        data_pipeline: DataPipeline,
        model_configs: List[Dict[str, Any]],
        use_walk_forward: bool = True,
        tune_hyperparameters: bool = False,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            data_pipeline: Data pipeline instance
            model_configs: List of model configurations
            use_walk_forward: Whether to use walk-forward validation
            tune_hyperparameters: Whether to tune hyperparameters
            n_trials: Number of tuning trials
            
        Returns:
            Results dictionary
        """
        results = {
            'models': {},
            'metrics': {},
            'best_model': None
        }
        
        # Get data splits
        if use_walk_forward:
            splits = data_pipeline.get_walk_forward_splits(
                n_splits=self.config.get('n_splits', 5)
            )
        else:
            X_train, X_test, y_train, y_test = data_pipeline.get_train_test_split()
            splits = [(X_train, X_test, y_train, y_test)]
        
        best_score = -float('inf')
        
        for model_config in model_configs:
            model_type = model_config['type']
            model_params = model_config.get('params', {})
            task_type = model_config.get('task_type', 'classification')
            
            print(f"\nTraining {model_type}...")
            
            # Track model config
            self.tracker.log_params({
                'model_type': model_type,
                'task_type': task_type,
                'n_splits': len(splits)
            })
            
            fold_metrics = []
            
            for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
                print(f"  Fold {i+1}/{len(splits)}")
                
                # Split train into train/val for tuning
                val_size = int(len(X_train) * 0.2)
                X_tr = X_train.iloc[:-val_size]
                X_val = X_train.iloc[-val_size:]
                y_tr = y_train.iloc[:-val_size]
                y_val = y_train.iloc[-val_size:]
                
                # Tune hyperparameters
                if tune_hyperparameters and i == 0:  # Tune only on first fold
                    tuner = HyperparameterTuner(
                        model_type,
                        task_type,
                        n_trials,
                        self.tracker
                    )
                    model_params = tuner.optimize(X_tr, y_tr, X_val, y_val)
                    print(f"  Best params: {model_params}")
                
                # Train model
                trainer = ModelTrainer(
                    model_type,
                    model_params,
                    task_type,
                    self.tracker
                )
                
                # Preprocess
                X_train_proc, X_test_proc = trainer.preprocess(X_train, X_test)
                
                # Train
                train_metrics = trainer.train(X_train_proc, y_train)
                
                # Test
                test_preds = trainer.predict(X_test_proc)
                test_metrics = trainer.evaluate(y_test, test_preds)
                
                # Log fold metrics
                fold_metric = {f'fold_{i+1}_{k}': v for k, v in test_metrics.items()}
                self.tracker.log_metrics(fold_metric)
                
                fold_metrics.append(test_metrics)
                
                # Save model if best
                score = test_metrics.get('test_f1', test_metrics.get('test_r2', 0))
                if score > best_score:
                    best_score = score
                    results['best_model'] = {
                        'type': model_type,
                        'params': model_params,
                        'trainer': trainer,
                        'score': score
                    }
            
            # Calculate average metrics
            avg_metrics = {}
            for key in fold_metrics[0].keys():
                values = [m[key] for m in fold_metrics]
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
            
            results['models'][model_type] = {
                'params': model_params,
                'fold_metrics': fold_metrics,
                'avg_metrics': avg_metrics
            }
            
            # Log average metrics
            self.tracker.log_metrics(avg_metrics)
            
            print(f"  Average test F1: {avg_metrics.get('avg_test_f1', 0):.4f}")
        
        # Save best model
        if results['best_model']:
            trainer = results['best_model']['trainer']
            self.tracker.log_model(trainer.model, f"best_{results['best_model']['type']}")
        
        # Finish tracking
        self.tracker.finish()
        
        return results


if __name__ == "__main__":
    print("AlphaStream Training Pipeline")
    print("Components: ExperimentTracker, ModelTrainer, HyperparameterTuner, TrainingPipeline")
    print("Ready to train and optimize trading models")
