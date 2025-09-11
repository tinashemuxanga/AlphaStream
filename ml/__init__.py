"""
AlphaStream ML Package

Machine learning components for trading signal generation.
"""

from .models import (
    BaseModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    LSTMModel,
    TransformerModel,
    EnsembleModel,
    ModelFactory
)

from .features import (
    FeatureEngineer,
    TargetGenerator
)

from .dataset import (
    DataLoader,
    DatasetBuilder,
    WalkForwardSplitter,
    DataPipeline
)

from .train import (
    ExperimentTracker,
    ModelTrainer,
    HyperparameterTuner,
    TrainingPipeline
)

__all__ = [
    # Models
    'BaseModel',
    'RandomForestModel',
    'XGBoostModel',
    'LightGBMModel',
    'LSTMModel',
    'TransformerModel',
    'EnsembleModel',
    'ModelFactory',
    
    # Features
    'FeatureEngineer',
    'TargetGenerator',
    
    # Dataset
    'DataLoader',
    'DatasetBuilder',
    'WalkForwardSplitter',
    'DataPipeline',
    
    # Training
    'ExperimentTracker',
    'ModelTrainer',
    'HyperparameterTuner',
    'TrainingPipeline'
]

__version__ = '1.0.0'
