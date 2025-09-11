"""
Dataset Module

Handles data loading, preprocessing, and train/test splitting
with walk-forward analysis support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import joblib
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from .features import FeatureEngineer, TargetGenerator


class DataLoader:
    """Load market data from various sources."""
    
    def __init__(self, source: str = 'yfinance', cache_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            source: Data source ('yfinance', 'polygon', 'alpha_vantage', 'csv')
            cache_dir: Directory for caching downloaded data
        """
        self.source = source
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = '1d',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load OHLCV data for symbols.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        all_data = {}
        
        for symbol in symbols:
            # Check cache first
            if use_cache:
                cached_data = self._load_from_cache(symbol, start_date, end_date, interval)
                if cached_data is not None:
                    all_data[symbol] = cached_data
                    continue
            
            # Load fresh data
            if self.source == 'yfinance':
                data = self._load_yfinance(symbol, start_date, end_date, interval)
            elif self.source == 'csv':
                data = self._load_csv(symbol)
            else:
                raise ValueError(f"Unsupported data source: {self.source}")
            
            if data is not None and not data.empty:
                all_data[symbol] = data
                if use_cache:
                    self._save_to_cache(symbol, data, start_date, end_date, interval)
        
        if len(symbols) == 1:
            return all_data[symbols[0]] if symbols[0] in all_data else pd.DataFrame()
        
        return all_data
    
    def _load_yfinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )
            
            # Rename columns to lowercase
            data.columns = data.columns.str.lower()
            
            # Add symbol column
            data['symbol'] = symbol
            
            return data
            
        except Exception as e:
            print(f"Error loading {symbol} from yfinance: {e}")
            return pd.DataFrame()
    
    def _load_csv(self, symbol: str) -> pd.DataFrame:
        """Load data from CSV file."""
        file_path = self.cache_dir / f"{symbol}.csv"
        if file_path.exists():
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            data['symbol'] = symbol
            return data
        return pd.DataFrame()
    
    def _load_from_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}.pkl"
        if cache_file.exists():
            try:
                return pd.read_pickle(cache_file)
            except:
                return None
        return None
    
    def _save_to_cache(
        self,
        symbol: str,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}.pkl"
        data.to_pickle(cache_file)


class DatasetBuilder:
    """Build ML-ready datasets with features and targets."""
    
    def __init__(
        self,
        feature_engineer: Optional[FeatureEngineer] = None,
        target_generator: Optional[TargetGenerator] = None
    ):
        """
        Initialize dataset builder.
        
        Args:
            feature_engineer: Feature engineering instance
            target_generator: Target generation instance
        """
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.target_generator = target_generator or TargetGenerator()
        self.raw_data = None
        self.features = None
        self.targets = None
        
    def build(
        self,
        data: pd.DataFrame,
        feature_groups: Optional[List[str]] = None,
        min_history: int = 200
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix and target vector.
        
        Args:
            data: Raw OHLCV data
            feature_groups: List of feature groups to generate
            min_history: Minimum history required for feature calculation
            
        Returns:
            Tuple of (features, targets)
        """
        self.raw_data = data.copy()
        
        # Generate features
        self.features = self.feature_engineer.transform(data, feature_groups)
        
        # Generate targets
        self.targets = self.target_generator.generate(data)
        
        # Remove rows with insufficient history
        if min_history > 0:
            self.features = self.features.iloc[min_history:]
            self.targets = self.targets.iloc[min_history:]
        
        # Remove rows with NaN targets (end of series)
        valid_idx = ~self.targets.isna()
        self.features = self.features[valid_idx]
        self.targets = self.targets[valid_idx]
        
        return self.features, self.targets
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.features.columns.tolist() if self.features is not None else []
    
    def save(self, path: str):
        """Save dataset to disk."""
        data = {
            'features': self.features,
            'targets': self.targets,
            'feature_names': self.get_feature_names(),
            'raw_data': self.raw_data
        }
        joblib.dump(data, path)
    
    def load(self, path: str):
        """Load dataset from disk."""
        data = joblib.load(path)
        self.features = data['features']
        self.targets = data['targets']
        self.raw_data = data.get('raw_data')
        return self.features, self.targets


class WalkForwardSplitter:
    """Walk-forward analysis splitter for time series."""
    
    def __init__(
        self,
        n_splits: int = 5,
        train_period: Optional[int] = None,
        test_period: Optional[int] = None,
        gap: int = 0,
        expanding: bool = False
    ):
        """
        Initialize walk-forward splitter.
        
        Args:
            n_splits: Number of splits
            train_period: Fixed training period length (if None, uses expanding window)
            test_period: Test period length
            gap: Gap between train and test sets
            expanding: Whether to use expanding window for training
        """
        self.n_splits = n_splits
        self.train_period = train_period
        self.test_period = test_period
        self.gap = gap
        self.expanding = expanding or (train_period is None)
        
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Generate train/test splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        splits = []
        
        if self.test_period is None:
            # Calculate test period based on number of splits
            total_test_size = n_samples // (self.n_splits + 1)
            test_size = total_test_size // self.n_splits
        else:
            test_size = self.test_period
        
        if self.train_period is None:
            # Use expanding window
            min_train_size = max(test_size * 2, 100)  # Minimum training size
        else:
            min_train_size = self.train_period
        
        for i in range(self.n_splits):
            if self.expanding:
                # Expanding window
                train_start = 0
                train_end = min_train_size + i * test_size
            else:
                # Rolling window
                train_start = i * test_size
                train_end = train_start + self.train_period
            
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)
            
            if test_end > n_samples:
                break
                
            train_idx = X.index[train_start:train_end]
            test_idx = X.index[test_start:test_end]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits
    
    def get_fold_dates(
        self,
        X: pd.DataFrame,
        splits: List[Tuple[pd.Index, pd.Index]]
    ) -> List[Dict[str, Any]]:
        """Get date ranges for each fold."""
        fold_info = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            info = {
                'fold': i + 1,
                'train_start': X.loc[train_idx].index.min(),
                'train_end': X.loc[train_idx].index.max(),
                'test_start': X.loc[test_idx].index.min(),
                'test_end': X.loc[test_idx].index.max(),
                'train_size': len(train_idx),
                'test_size': len(test_idx)
            }
            fold_info.append(info)
        
        return fold_info


class DataPipeline:
    """Complete data pipeline from raw data to ML-ready datasets."""
    
    def __init__(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        feature_groups: Optional[List[str]] = None,
        target_type: str = 'classification',
        target_horizon: int = 1,
        target_threshold: float = 0.002
    ):
        """
        Initialize data pipeline.
        
        Args:
            symbols: Trading symbols
            start_date: Start date for data
            end_date: End date for data
            feature_groups: Feature groups to generate
            target_type: Type of target ('classification' or 'regression')
            target_horizon: Prediction horizon
            target_threshold: Threshold for classification
        """
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.feature_groups = feature_groups
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.target_generator = TargetGenerator(
            target_type=target_type,
            horizon=target_horizon,
            threshold=target_threshold
        )
        self.dataset_builder = DatasetBuilder(
            self.feature_engineer,
            self.target_generator
        )
        
        self.raw_data = None
        self.features = None
        self.targets = None
        
    def run(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Run complete pipeline."""
        # Load data
        print(f"Loading data for {self.symbols}...")
        self.raw_data = self.data_loader.load(
            self.symbols,
            self.start_date,
            self.end_date
        )
        
        if isinstance(self.raw_data, dict):
            # Multiple symbols - concatenate
            all_data = []
            for symbol, data in self.raw_data.items():
                data['symbol'] = symbol
                all_data.append(data)
            self.raw_data = pd.concat(all_data, axis=0).sort_index()
        
        print(f"Loaded {len(self.raw_data)} rows of data")
        
        # Build dataset
        print("Generating features...")
        self.features, self.targets = self.dataset_builder.build(
            self.raw_data,
            self.feature_groups
        )
        
        print(f"Generated {self.features.shape[1]} features")
        print(f"Dataset shape: {self.features.shape}")
        
        return self.features, self.targets
    
    def get_train_test_split(
        self,
        test_size: float = 0.2,
        gap: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Get simple train/test split."""
        if self.features is None or self.targets is None:
            self.run()
        
        n_samples = len(self.features)
        split_idx = int(n_samples * (1 - test_size)) - gap
        
        X_train = self.features.iloc[:split_idx]
        X_test = self.features.iloc[split_idx + gap:]
        y_train = self.targets.iloc[:split_idx]
        y_test = self.targets.iloc[split_idx + gap:]
        
        return X_train, X_test, y_train, y_test
    
    def get_walk_forward_splits(
        self,
        n_splits: int = 5,
        train_period: Optional[int] = None,
        test_period: Optional[int] = None,
        gap: int = 0
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Get walk-forward splits."""
        if self.features is None or self.targets is None:
            self.run()
        
        splitter = WalkForwardSplitter(
            n_splits=n_splits,
            train_period=train_period,
            test_period=test_period,
            gap=gap
        )
        
        splits = splitter.split(self.features, self.targets)
        
        result = []
        for train_idx, test_idx in splits:
            X_train = self.features.loc[train_idx]
            X_test = self.features.loc[test_idx]
            y_train = self.targets.loc[train_idx]
            y_test = self.targets.loc[test_idx]
            result.append((X_train, X_test, y_train, y_test))
        
        return result


if __name__ == "__main__":
    print("AlphaStream Dataset Module")
    print("Components: DataLoader, DatasetBuilder, WalkForwardSplitter, DataPipeline")
    print("Ready to build ML-ready datasets from market data")
