"""
Feature Engineering Module

Comprehensive feature extraction for trading signals including:
- Technical indicators
- Market microstructure features  
- Rolling statistics
- Price patterns
- Volume analysis
- Sentiment features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import ta
import pandas_ta as ta_extended
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Main feature engineering class."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature engineer with configuration."""
        self.config = config or {}
        self.feature_groups = {
            'technical': TechnicalFeatures(),
            'microstructure': MicrostructureFeatures(),
            'rolling': RollingFeatures(),
            'pattern': PatternFeatures(),
            'volume': VolumeFeatures(),
            'derived': DerivedFeatures()
        }
        self.scaler = None
        self.feature_names = []
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit the feature engineer on training data."""
        # Fit any scalers or transformers if needed
        features = self.transform(df)
        self.scaler = RobustScaler()
        self.scaler.fit(features)
        return self
        
    def transform(self, df: pd.DataFrame, feature_groups: Optional[List[str]] = None) -> pd.DataFrame:
        """Transform raw OHLCV data into features."""
        if feature_groups is None:
            feature_groups = list(self.feature_groups.keys())
            
        all_features = []
        
        for group_name in feature_groups:
            if group_name in self.feature_groups:
                group_features = self.feature_groups[group_name].generate(df)
                all_features.append(group_features)
                
        # Combine all features
        features_df = pd.concat(all_features, axis=1)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        return features_df
    
    def fit_transform(self, df: pd.DataFrame, feature_groups: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df, feature_groups)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Fill any remaining NaN with 0
        df = df.fillna(0)
        
        # Replace infinities with large values
        df = df.replace([np.inf, -np.inf], [1e10, -1e10])
        
        return df
    
    def get_feature_importance(self, model: Any) -> pd.DataFrame:
        """Get feature importance from a trained model."""
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return pd.DataFrame()


class TechnicalFeatures:
    """Technical indicator features."""
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features."""
        features = pd.DataFrame(index=df.index)
        
        # Price-based indicators
        features['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()
        features['macd_cross'] = np.where(
            (features['macd'] > features['macd_signal']) & 
            (features['macd'].shift(1) <= features['macd_signal'].shift(1)), 1,
            np.where(
                (features['macd'] < features['macd_signal']) & 
                (features['macd'].shift(1) >= features['macd_signal'].shift(1)), -1, 0
            )
        )
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        features['bb_high'] = bb.bollinger_hband()
        features['bb_low'] = bb.bollinger_lband()
        features['bb_mid'] = bb.bollinger_mavg()
        features['bb_width'] = features['bb_high'] - features['bb_low']
        features['bb_position'] = (df['close'] - features['bb_low']) / (features['bb_high'] - features['bb_low'])
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
            
        # Moving average crossovers
        features['golden_cross'] = np.where(
            (features['sma_50'] > features['sma_200']) & 
            (features['sma_50'].shift(1) <= features['sma_200'].shift(1)), 1, 0
        )
        features['death_cross'] = np.where(
            (features['sma_50'] < features['sma_200']) & 
            (features['sma_50'].shift(1) >= features['sma_200'].shift(1)), 1, 0
        )
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        features['stoch_k'] = stoch.stoch()
        features['stoch_d'] = stoch.stoch_signal()
        
        # ATR (Average True Range)
        features['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        features['atr_percent'] = features['atr'] / df['close']
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        features['adx'] = adx.adx()
        features['adx_pos'] = adx.adx_pos()
        features['adx_neg'] = adx.adx_neg()
        
        # CCI (Commodity Channel Index)
        features['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        # Williams %R
        features['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        features['ichimoku_a'] = ichimoku.ichimoku_a()
        features['ichimoku_b'] = ichimoku.ichimoku_b()
        features['ichimoku_base'] = ichimoku.ichimoku_base_line()
        features['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ta.momentum.ROCIndicator(df['close'], window=period).roc()
        
        # Parabolic SAR
        psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
        features['psar'] = psar.psar()
        features['psar_up'] = psar.psar_up()
        features['psar_down'] = psar.psar_down()
        
        return features


class MicrostructureFeatures:
    """Market microstructure features."""
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate microstructure features."""
        features = pd.DataFrame(index=df.index)
        
        # Spread metrics
        features['spread'] = df['high'] - df['low']
        features['spread_percent'] = features['spread'] / df['close']
        features['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        features['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        
        # Gaps
        features['gap'] = df['open'] - df['close'].shift(1)
        features['gap_percent'] = features['gap'] / df['close'].shift(1)
        features['gap_up'] = (features['gap'] > 0).astype(int)
        features['gap_down'] = (features['gap'] < 0).astype(int)
        
        # Price location within bar
        features['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        features['high_close_ratio'] = df['high'] / df['close']
        features['low_close_ratio'] = df['low'] / df['close']
        
        # Volatility measures
        features['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        features['true_range_percent'] = features['true_range'] / df['close']
        
        # Garman-Klass volatility
        features['garman_klass_vol'] = np.sqrt(
            (np.log(df['high'] / df['low']) ** 2) / 2 - 
            (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)
        )
        
        # Parkinson volatility
        features['parkinson_vol'] = np.sqrt(
            np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
        )
        
        # Rogers-Satchell volatility
        features['rogers_satchell_vol'] = np.sqrt(
            np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
            np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
        )
        
        # Efficiency ratio (signal to noise)
        period = 10
        change = abs(df['close'] - df['close'].shift(period))
        volatility = df['close'].diff().abs().rolling(period).sum()
        features['efficiency_ratio'] = change / volatility
        
        # Amihud illiquidity
        if 'volume' in df.columns:
            features['amihud_illiquidity'] = abs(df['close'].pct_change()) / (df['volume'] + 1e-10)
            features['amihud_illiquidity_ma'] = features['amihud_illiquidity'].rolling(20).mean()
        
        return features


class RollingFeatures:
    """Rolling statistical features."""
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling statistical features."""
        features = pd.DataFrame(index=df.index)
        
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Returns
            returns = df['close'].pct_change()
            
            # Rolling statistics
            features[f'return_{window}d'] = returns.rolling(window).sum()
            features[f'volatility_{window}d'] = returns.rolling(window).std()
            features[f'skew_{window}d'] = returns.rolling(window).skew()
            features[f'kurtosis_{window}d'] = returns.rolling(window).apply(lambda x: stats.kurtosis(x))
            
            # Price statistics
            features[f'min_{window}d'] = df['low'].rolling(window).min()
            features[f'max_{window}d'] = df['high'].rolling(window).max()
            features[f'range_{window}d'] = features[f'max_{window}d'] - features[f'min_{window}d']
            features[f'close_to_max_{window}d'] = df['close'] / features[f'max_{window}d']
            features[f'close_to_min_{window}d'] = df['close'] / features[f'min_{window}d']
            
            # Drawdown
            rolling_max = df['close'].rolling(window, min_periods=1).max()
            features[f'drawdown_{window}d'] = (df['close'] - rolling_max) / rolling_max
            
            # Momentum
            features[f'momentum_{window}d'] = df['close'] / df['close'].shift(window) - 1
            
            # Correlation with volume (if available)
            if 'volume' in df.columns:
                features[f'price_volume_corr_{window}d'] = (
                    df['close'].rolling(window).corr(df['volume'])
                )
        
        # Expanding window features
        expanding_returns = returns.expanding(min_periods=20)
        features['expanding_mean'] = expanding_returns.mean()
        features['expanding_std'] = expanding_returns.std()
        features['expanding_sharpe'] = features['expanding_mean'] / (features['expanding_std'] + 1e-10)
        
        return features


class PatternFeatures:
    """Price pattern recognition features."""
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate pattern-based features."""
        features = pd.DataFrame(index=df.index)
        
        # Candlestick patterns
        features['doji'] = (
            abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10) < 0.1
        ).astype(int)
        
        features['hammer'] = (
            ((df['high'] - df['low']) > 3 * abs(df['close'] - df['open'])) &
            ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10) > 0.6) &
            ((df['open'] - df['low']) / (df['high'] - df['low'] + 1e-10) > 0.6)
        ).astype(int)
        
        features['shooting_star'] = (
            ((df['high'] - df['low']) > 3 * abs(df['close'] - df['open'])) &
            ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10) > 0.6) &
            ((df['high'] - df['open']) / (df['high'] - df['low'] + 1e-10) > 0.6)
        ).astype(int)
        
        features['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)
        
        features['bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        ).astype(int)
        
        # Support and resistance levels
        window = 20
        features['resistance'] = df['high'].rolling(window).max()
        features['support'] = df['low'].rolling(window).min()
        features['close_to_resistance'] = df['close'] / features['resistance']
        features['close_to_support'] = df['close'] / features['support']
        
        # Breakout detection
        features['breakout_up'] = (
            (df['close'] > features['resistance'].shift(1)) &
            (df['close'].shift(1) <= features['resistance'].shift(1))
        ).astype(int)
        
        features['breakout_down'] = (
            (df['close'] < features['support'].shift(1)) &
            (df['close'].shift(1) >= features['support'].shift(1))
        ).astype(int)
        
        # Higher highs and lower lows
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        features['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        features['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        
        # Consecutive patterns
        features['consecutive_ups'] = (df['close'] > df['close'].shift(1)).astype(int)
        features['consecutive_ups'] = features['consecutive_ups'].groupby(
            (features['consecutive_ups'] != features['consecutive_ups'].shift()).cumsum()
        ).cumsum() * features['consecutive_ups']
        
        features['consecutive_downs'] = (df['close'] < df['close'].shift(1)).astype(int)
        features['consecutive_downs'] = features['consecutive_downs'].groupby(
            (features['consecutive_downs'] != features['consecutive_downs'].shift()).cumsum()
        ).cumsum() * features['consecutive_downs']
        
        return features


class VolumeFeatures:
    """Volume-based features."""
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        features = pd.DataFrame(index=df.index)
        
        if 'volume' not in df.columns:
            return features
            
        # Basic volume features
        features['volume'] = df['volume']
        features['volume_sma_10'] = df['volume'].rolling(10).mean()
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        
        # On-Balance Volume (OBV)
        obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
        features['obv'] = obv.on_balance_volume()
        features['obv_sma'] = features['obv'].rolling(20).mean()
        features['obv_signal'] = features['obv'] - features['obv_sma']
        
        # Volume Weighted Average Price (VWAP)
        features['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            df['high'], df['low'], df['close'], df['volume']
        ).volume_weighted_average_price()
        features['price_to_vwap'] = df['close'] / features['vwap']
        
        # Accumulation/Distribution Index
        features['adi'] = ta.volume.AccDistIndexIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).acc_dist_index()
        
        # Money Flow Index
        features['mfi'] = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).money_flow_index()
        
        # Chaikin Money Flow
        features['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).chaikin_money_flow()
        
        # Ease of Movement
        features['eom'] = ta.volume.EaseOfMovementIndicator(
            df['high'], df['low'], df['volume']
        ).ease_of_movement()
        
        # Force Index
        features['force_index'] = ta.volume.ForceIndexIndicator(
            df['close'], df['volume']
        ).force_index()
        
        # Volume Price Trend
        features['vpt'] = ta.volume.VolumePriceTrendIndicator(
            df['close'], df['volume']
        ).volume_price_trend()
        
        # Negative Volume Index
        features['nvi'] = ta.volume.NegativeVolumeIndexIndicator(
            df['close'], df['volume']
        ).negative_volume_index()
        
        # Volume momentum
        features['volume_momentum_5'] = df['volume'] / df['volume'].shift(5)
        features['volume_momentum_10'] = df['volume'] / df['volume'].shift(10)
        
        # Price-volume divergence
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        features['price_volume_divergence'] = price_change - volume_change
        
        # Volume spikes
        volume_std = df['volume'].rolling(20).std()
        volume_mean = df['volume'].rolling(20).mean()
        features['volume_spike'] = (df['volume'] - volume_mean) / volume_std
        
        return features


class DerivedFeatures:
    """Derived and composite features."""
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate derived features from basic OHLCV."""
        features = pd.DataFrame(index=df.index)
        
        # Returns at different scales
        for lag in [1, 2, 3, 5, 10, 20]:
            features[f'return_{lag}'] = df['close'].pct_change(lag)
            features[f'log_return_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['day_of_month'] = df.index.day
            features['month'] = df.index.month
            features['quarter'] = df.index.quarter
            features['is_month_start'] = df.index.is_month_start.astype(int)
            features['is_month_end'] = df.index.is_month_end.astype(int)
            features['is_quarter_start'] = df.index.is_quarter_start.astype(int)
            features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Lag features
        for col in ['close', 'volume', 'high', 'low']:
            if col in df.columns:
                for lag in [1, 2, 3, 5, 10]:
                    features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Cumulative features
        returns = df['close'].pct_change()
        features['cumulative_return'] = (1 + returns).cumprod() - 1
        features['cumulative_volume'] = df['volume'].cumsum() if 'volume' in df.columns else 0
        
        # Z-scores
        for col in ['close', 'volume']:
            if col in df.columns:
                mean = df[col].rolling(20).mean()
                std = df[col].rolling(20).std()
                features[f'{col}_zscore'] = (df[col] - mean) / (std + 1e-10)
        
        # Interaction features
        if 'volume' in df.columns:
            features['price_x_volume'] = df['close'] * df['volume']
            features['return_x_volume'] = returns * df['volume']
        
        return features


class TargetGenerator:
    """Generate target variables for supervised learning."""
    
    def __init__(self, target_type: str = 'classification', horizon: int = 1, threshold: float = 0.002):
        """
        Initialize target generator.
        
        Args:
            target_type: 'classification' or 'regression'
            horizon: Prediction horizon in periods
            threshold: Threshold for classification (e.g., 0.002 = 0.2%)
        """
        self.target_type = target_type
        self.horizon = horizon
        self.threshold = threshold
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Generate target variable."""
        if self.target_type == 'classification':
            return self._generate_classification_target(df)
        else:
            return self._generate_regression_target(df)
    
    def _generate_classification_target(self, df: pd.DataFrame) -> pd.Series:
        """Generate classification target (buy/hold/sell)."""
        # Calculate forward returns
        forward_returns = df['close'].shift(-self.horizon) / df['close'] - 1
        
        # Create labels
        # 0: Sell (return < -threshold)
        # 1: Hold (-threshold <= return <= threshold)
        # 2: Buy (return > threshold)
        target = pd.Series(1, index=df.index, name='target')  # Default to hold
        target[forward_returns < -self.threshold] = 0  # Sell
        target[forward_returns > self.threshold] = 2   # Buy
        
        return target
    
    def _generate_regression_target(self, df: pd.DataFrame) -> pd.Series:
        """Generate regression target (continuous returns)."""
        # Calculate forward returns
        forward_returns = df['close'].shift(-self.horizon) / df['close'] - 1
        forward_returns.name = 'target'
        
        return forward_returns


if __name__ == "__main__":
    print("AlphaStream Feature Engineering Module")
    print(f"Available feature groups: technical, microstructure, rolling, pattern, volume, derived")
    print("Ready to generate 200+ features from OHLCV data")
