"""
Unit tests for ML models module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.models import ModelFactory, RandomForestModel, XGBoostModel
from ml.features import FeatureEngineer, TargetGenerator
from ml.dataset import DataLoader, DatasetBuilder


class TestModels:
    """Test ML models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(500).cumsum(),
            'high': 102 + np.random.randn(500).cumsum(),
            'low': 98 + np.random.randn(500).cumsum(),
            'close': 100 + np.random.randn(500).cumsum(),
            'volume': np.random.randint(1000000, 10000000, 500)
        }, index=dates)
        
        # Ensure high/low are correct
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    @pytest.fixture
    def features_and_targets(self, sample_data):
        """Generate features and targets from sample data."""
        feature_engineer = FeatureEngineer()
        target_generator = TargetGenerator(target_type='classification')
        
        features = feature_engineer.transform(sample_data)
        targets = target_generator.generate(sample_data)
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | targets.isna())
        features = features[valid_idx]
        targets = targets[valid_idx]
        
        return features, targets
    
    def test_model_factory(self):
        """Test ModelFactory creates correct model types."""
        # Test RandomForest
        model = ModelFactory.create('random_forest', 'classification')
        assert isinstance(model, RandomForestModel)
        
        # Test XGBoost
        model = ModelFactory.create('xgboost', 'classification')
        assert isinstance(model, XGBoostModel)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            ModelFactory.create('invalid_model', 'classification')
    
    def test_random_forest_training(self, features_and_targets):
        """Test RandomForest model training."""
        features, targets = features_and_targets
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        y_train = targets.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = targets.iloc[split_idx:]
        
        # Create and train model
        model = ModelFactory.create('random_forest', 'classification')
        model.train(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert all(p in [-1, 0, 1] for p in predictions)
        
        # Check metrics
        metrics = model.evaluate(X_test, y_test)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_xgboost_training(self, features_and_targets):
        """Test XGBoost model training."""
        features, targets = features_and_targets
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        y_train = targets.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = targets.iloc[split_idx:]
        
        # Create and train model
        model = ModelFactory.create('xgboost', 'classification')
        model.train(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert all(p in [-1, 0, 1] for p in predictions)
        
        # Check feature importance
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == X_train.shape[1]
    
    def test_ensemble_model(self, features_and_targets):
        """Test ensemble model."""
        features, targets = features_and_targets
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        y_train = targets.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        
        # Create base models
        rf_model = ModelFactory.create('random_forest', 'classification')
        rf_model.train(X_train, y_train)
        
        xgb_model = ModelFactory.create('xgboost', 'classification')
        xgb_model.train(X_train, y_train)
        
        # Create ensemble
        ensemble = ModelFactory.create('ensemble', 'classification')
        ensemble.models = [('rf', rf_model), ('xgb', xgb_model)]
        
        # Make predictions
        predictions = ensemble.predict(X_test)
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert all(p in [-1, 0, 1] for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
