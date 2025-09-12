"""
AlphaStream ML Models Module

Comprehensive collection of machine learning models for trading signal generation.
Includes traditional ML, deep learning, and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)

import xgboost as xgb
import lightgbm as lgb
# import catboost as cb  # Optional: Install if needed

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Type hints
ModelType = Union[Any, nn.Module]


@dataclass
class SignalPrediction:
    """Trading signal prediction with metadata."""
    symbol: str
    timestamp: pd.Timestamp
    signal: int  # -1: sell, 0: hold, 1: buy
    confidence: float  # 0-1 confidence score
    model_name: str
    features_used: List[str]
    metadata: Dict[str, Any]


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: str, model_type: str = 'classification'):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        self.training_history = []
        
    @abstractmethod
    def build_model(self, **kwargs):
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Preprocess features."""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        if self.model_type == 'classification':
            return {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='weighted'),
                'recall': recall_score(y, predictions, average='weighted'),
                'f1': f1_score(y, predictions, average='weighted')
            }
        else:
            return {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'r2': r2_score(y, predictions)
            }


# Traditional ML Models

class RandomForestModel(BaseModel):
    """Random Forest model for signal generation."""
    
    def build_model(self, **kwargs):
        """Build Random Forest model."""
        if self.model_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                max_features=kwargs.get('max_features', 'sqrt'),
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                max_features=kwargs.get('max_features', 'sqrt'),
                random_state=42,
                n_jobs=-1
            )
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train Random Forest model."""
        if self.model is None:
            self.build_model(**kwargs)
        
        X_scaled = self.preprocess(X, fit=True)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Random Forest."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.preprocess(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.preprocess(X)
        if self.model_type == 'classification':
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict(X_scaled)


class XGBoostModel(BaseModel):
    """XGBoost model for signal generation."""
    
    def build_model(self, **kwargs):
        """Build XGBoost model."""
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'random_state': 42,
            'n_jobs': -1
        }
        
        if self.model_type == 'classification':
            params['objective'] = 'multi:softprob'
            params['num_class'] = 3  # buy, hold, sell
            self.model = xgb.XGBClassifier(**params)
        else:
            params['objective'] = 'reg:squarederror'
            self.model = xgb.XGBRegressor(**params)
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train XGBoost model."""
        if self.model is None:
            self.build_model(**kwargs)
        
        X_scaled = self.preprocess(X, fit=True)
        
        # Early stopping
        eval_set = kwargs.get('eval_set', None)
        if eval_set:
            X_val, y_val = eval_set
            X_val_scaled = self.preprocess(X_val)
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X_scaled, y)
        
        self.is_trained = True
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.preprocess(X)
        return self.model.predict(X_scaled)


class LightGBMModel(BaseModel):
    """LightGBM model for signal generation."""
    
    def build_model(self, **kwargs):
        """Build LightGBM model."""
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', -1),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'num_leaves': kwargs.get('num_leaves', 31),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        if self.model_type == 'classification':
            params['objective'] = 'multiclass'
            params['num_class'] = 3
            self.model = lgb.LGBMClassifier(**params)
        else:
            params['objective'] = 'regression'
            self.model = lgb.LGBMRegressor(**params)
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train LightGBM model."""
        if self.model is None:
            self.build_model(**kwargs)
        
        X_scaled = self.preprocess(X, fit=True)
        
        # Categorical features support
        categorical_features = kwargs.get('categorical_features', None)
        self.model.fit(
            X_scaled, y,
            categorical_feature=categorical_features
        )
        
        self.is_trained = True
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.preprocess(X)
        return self.model.predict(X_scaled)


# Deep Learning Models

class LSTMModel(BaseModel):
    """LSTM model for time series prediction."""
    
    def __init__(self, name: str, input_dim: int, sequence_length: int = 20):
        super().__init__(name, 'regression')
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_model(self, **kwargs):
        """Build LSTM model."""
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_layers = kwargs.get('num_layers', 2)
        dropout = kwargs.get('dropout', 0.2)
        output_dim = kwargs.get('output_dim', 3)  # 3 classes: buy, hold, sell
        
        self.model = LSTMNetwork(
            self.input_dim,
            hidden_dim,
            num_layers,
            output_dim,
            dropout
        ).to(self.device)
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train LSTM model."""
        if self.model is None:
            self.build_model(**kwargs)
        
        # Prepare sequences
        X_sequences = self._prepare_sequences(X)
        y_tensor = torch.FloatTensor(y.values).to(self.device)
        
        # Training parameters
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 32)
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        # Create data loader
        dataset = TimeSeriesDataset(X_sequences, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss() if self.model_type == 'classification' else nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            self.training_history.append({'epoch': epoch, 'loss': epoch_loss})
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        X_sequences = self._prepare_sequences(X)
        
        with torch.no_grad():
            predictions = self.model(X_sequences)
            if self.model_type == 'classification':
                predictions = torch.argmax(predictions, dim=1)
        
        return predictions.cpu().numpy()
    
    def _prepare_sequences(self, X: pd.DataFrame) -> torch.Tensor:
        """Prepare sequences for LSTM input."""
        X_scaled = self.preprocess(X, fit=not self.is_trained)
        sequences = []
        
        for i in range(len(X_scaled) - self.sequence_length + 1):
            sequences.append(X_scaled[i:i + self.sequence_length])
        
        return torch.FloatTensor(np.array(sequences)).to(self.device)


class LSTMNetwork(nn.Module):
    """LSTM neural network architecture."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMNetwork, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last output
        last_out = lstm_out[:, -1, :]
        
        # Dropout and fully connected layer
        out = self.dropout(last_out)
        out = self.fc(out)
        
        return out


class TransformerModel(BaseModel):
    """Transformer model for signal generation."""
    
    def __init__(self, name: str, input_dim: int, sequence_length: int = 20):
        super().__init__(name, 'classification')
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self, **kwargs):
        """Build Transformer model."""
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('num_layers', 3)
        dropout = kwargs.get('dropout', 0.1)
        output_dim = kwargs.get('output_dim', 3)
        
        self.model = TransformerNetwork(
            self.input_dim,
            d_model,
            nhead,
            num_layers,
            output_dim,
            dropout,
            self.sequence_length
        ).to(self.device)
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train Transformer model."""
        if self.model is None:
            self.build_model(**kwargs)
        
        # Similar to LSTM training
        X_sequences = self._prepare_sequences(X)
        y_tensor = torch.LongTensor(y.values).to(self.device)
        
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 32)
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        dataset = TimeSeriesDataset(X_sequences, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            self.training_history.append({'epoch': epoch, 'loss': epoch_loss})
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Transformer."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        X_sequences = self._prepare_sequences(X)
        
        with torch.no_grad():
            predictions = self.model(X_sequences)
            predictions = torch.argmax(predictions, dim=1)
        
        return predictions.cpu().numpy()
    
    def _prepare_sequences(self, X: pd.DataFrame) -> torch.Tensor:
        """Prepare sequences for Transformer input."""
        X_scaled = self.preprocess(X, fit=not self.is_trained)
        sequences = []
        
        for i in range(len(X_scaled) - self.sequence_length + 1):
            sequences.append(X_scaled[i:i + self.sequence_length])
        
        return torch.FloatTensor(np.array(sequences)).to(self.device)


class TransformerNetwork(nn.Module):
    """Transformer neural network architecture."""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout, seq_len):
        super(TransformerNetwork, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use the last output
        x = x[:, -1, :]
        
        # Final classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Ensemble Models

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(self, name: str, base_models: List[BaseModel], ensemble_type: str = 'voting'):
        super().__init__(name, 'classification')
        self.base_models = base_models
        self.ensemble_type = ensemble_type
        self.weights = None
        
    def build_model(self, **kwargs):
        """Build ensemble model."""
        if self.ensemble_type == 'voting':
            self._build_voting_ensemble(**kwargs)
        elif self.ensemble_type == 'stacking':
            self._build_stacking_ensemble(**kwargs)
        elif self.ensemble_type == 'blending':
            self._build_blending_ensemble(**kwargs)
        elif self.ensemble_type == 'bayesian':
            self._build_bayesian_ensemble(**kwargs)
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def _build_voting_ensemble(self, **kwargs):
        """Build voting ensemble."""
        voting = kwargs.get('voting', 'soft')  # 'hard' or 'soft'
        weights = kwargs.get('weights', None)
        
        if self.model_type == 'classification':
            estimators = [(model.name, model.model) for model in self.base_models]
            self.model = VotingClassifier(estimators, voting=voting, weights=weights)
        else:
            estimators = [(model.name, model.model) for model in self.base_models]
            self.model = VotingRegressor(estimators, weights=weights)
    
    def _build_stacking_ensemble(self, **kwargs):
        """Build stacking ensemble."""
        meta_learner = kwargs.get('meta_learner', LogisticRegression())
        
        if self.model_type == 'classification':
            estimators = [(model.name, model.model) for model in self.base_models]
            self.model = StackingClassifier(estimators, final_estimator=meta_learner)
        else:
            estimators = [(model.name, model.model) for model in self.base_models]
            self.model = StackingRegressor(estimators, final_estimator=meta_learner)
    
    def _build_blending_ensemble(self, **kwargs):
        """Build blending ensemble (custom implementation)."""
        # Blending uses a holdout validation set
        self.blend_features = []
        self.meta_model = kwargs.get('meta_learner', LogisticRegression())
    
    def _build_bayesian_ensemble(self, **kwargs):
        """Build Bayesian model averaging ensemble."""
        # Initialize with equal weights
        self.weights = np.ones(len(self.base_models)) / len(self.base_models)
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train ensemble model."""
        if self.ensemble_type in ['voting', 'stacking']:
            # Train all base models first
            for model in self.base_models:
                if not model.is_trained:
                    model.train(X, y, **kwargs)
            
            # Build and train ensemble
            self.build_model(**kwargs)
            X_scaled = self.preprocess(X, fit=True)
            self.model.fit(X_scaled, y)
            
        elif self.ensemble_type == 'blending':
            # Split data for blending
            blend_split = kwargs.get('blend_split', 0.2)
            split_idx = int(len(X) * (1 - blend_split))
            
            X_train, X_blend = X[:split_idx], X[split_idx:]
            y_train, y_blend = y[:split_idx], y[split_idx:]
            
            # Train base models on training set
            blend_predictions = []
            for model in self.base_models:
                model.train(X_train, y_train, **kwargs)
                pred = model.predict(X_blend)
                blend_predictions.append(pred)
            
            # Train meta model on blend set
            blend_features = np.column_stack(blend_predictions)
            self.meta_model.fit(blend_features, y_blend)
            
        elif self.ensemble_type == 'bayesian':
            # Train all base models
            for model in self.base_models:
                if not model.is_trained:
                    model.train(X, y, **kwargs)
            
            # Calculate weights based on validation performance
            val_split = kwargs.get('val_split', 0.2)
            split_idx = int(len(X) * (1 - val_split))
            X_val, y_val = X[split_idx:], y[split_idx:]
            
            performances = []
            for model in self.base_models:
                metrics = model.evaluate(X_val, y_val)
                performance = metrics.get('accuracy', metrics.get('r2', 0))
                performances.append(performance)
            
            # Convert to weights (softmax)
            performances = np.array(performances)
            self.weights = np.exp(performances) / np.sum(np.exp(performances))
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with ensemble."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.ensemble_type in ['voting', 'stacking']:
            X_scaled = self.preprocess(X)
            return self.model.predict(X_scaled)
            
        elif self.ensemble_type == 'blending':
            # Get predictions from all base models
            blend_predictions = []
            for model in self.base_models:
                pred = model.predict(X)
                blend_predictions.append(pred)
            
            # Use meta model for final prediction
            blend_features = np.column_stack(blend_predictions)
            return self.meta_model.predict(blend_features)
            
        elif self.ensemble_type == 'bayesian':
            # Weighted average of predictions
            predictions = []
            for model, weight in zip(self.base_models, self.weights):
                pred = model.predict(X)
                predictions.append(pred * weight)
            
            weighted_pred = np.sum(predictions, axis=0)
            
            # For classification, round to nearest class
            if self.model_type == 'classification':
                return np.round(weighted_pred).astype(int)
            return weighted_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from ensemble."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.ensemble_type in ['voting', 'stacking'] and self.model_type == 'classification':
            X_scaled = self.preprocess(X)
            return self.model.predict_proba(X_scaled)
        
        elif self.ensemble_type == 'bayesian':
            # Weighted average of probabilities
            all_probas = []
            for model, weight in zip(self.base_models, self.weights):
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    all_probas.append(proba * weight)
            
            if all_probas:
                return np.sum(all_probas, axis=0)
        
        # Fallback to regular predictions
        predictions = self.predict(X)
        n_classes = 3  # buy, hold, sell
        probas = np.zeros((len(predictions), n_classes))
        probas[np.arange(len(predictions)), predictions] = 1
        return probas


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data."""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Model Factory

class ModelFactory:
    """Factory for creating and managing models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """Create a model instance."""
        model_map = {
            'random_forest': RandomForestModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel,
            'ensemble': EnsembleModel
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_map[model_type]
        
        # Special handling for deep learning models
        if model_type in ['lstm', 'transformer']:
            input_dim = kwargs.pop('input_dim', 50)
            sequence_length = kwargs.pop('sequence_length', 20)
            return model_class(model_type, input_dim, sequence_length)
        
        # Special handling for ensemble
        if model_type == 'ensemble':
            base_models = kwargs.pop('base_models', [])
            ensemble_type = kwargs.pop('ensemble_type', 'voting')
            return model_class(model_type, base_models, ensemble_type)
        
        return model_class(model_type)
    
    @staticmethod
    def create_ensemble(base_model_types: List[str], ensemble_type: str = 'voting', **kwargs) -> EnsembleModel:
        """Create an ensemble model with specified base models."""
        base_models = []
        
        for model_type in base_model_types:
            model = ModelFactory.create_model(model_type, **kwargs)
            base_models.append(model)
        
        return EnsembleModel('ensemble', base_models, ensemble_type)


if __name__ == "__main__":
    # Example usage
    print("AlphaStream ML Models Module Loaded")
    print(f"Available models: Random Forest, XGBoost, LightGBM, LSTM, Transformer, Ensemble")
    print(f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
