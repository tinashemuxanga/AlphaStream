"""
AlphaStream Example Usage Notebook

This file can be converted to a Jupyter notebook or run as a Python script.
Demonstrates how to use AlphaStream for trading signal generation.
"""

# %% [markdown]
# # AlphaStream - Trading Signal Generation Example
# 
# This notebook demonstrates how to:
# 1. Load market data
# 2. Generate features
# 3. Train ML models
# 4. Generate trading signals
# 5. Backtest strategies
# 6. Visualize results

# %% Import libraries
import sys
from pathlib import Path
sys.path.append(str(Path().absolute().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import AlphaStream modules
from ml.dataset import DataPipeline, DataLoader
from ml.features import FeatureEngineer, TargetGenerator
from ml.models import ModelFactory
from ml.train import ModelTrainer, TrainingPipeline
from backtesting.engine import BacktestEngine

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %% [markdown]
# ## 1. Load and Explore Data

# %% Load data
print("Loading market data...")

# Initialize data loader
data_loader = DataLoader()

# Load data for multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT']
start_date = '2020-01-01'
end_date = '2024-01-01'

data = {}
for symbol in symbols:
    data[symbol] = data_loader.load(symbol, start_date, end_date)
    print(f"Loaded {len(data[symbol])} days of data for {symbol}")

# %% Visualize price data
fig, axes = plt.subplots(len(symbols), 1, figsize=(12, 4*len(symbols)))

for i, symbol in enumerate(symbols):
    ax = axes[i] if len(symbols) > 1 else axes
    ax.plot(data[symbol].index, data[symbol]['close'], label=f'{symbol} Close Price')
    ax.set_title(f'{symbol} Stock Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Feature Engineering

# %% Generate features
print("\nGenerating features...")

feature_engineer = FeatureEngineer()

# Generate features for each symbol
features = {}
for symbol in symbols:
    features[symbol] = feature_engineer.transform(data[symbol])
    print(f"Generated {features[symbol].shape[1]} features for {symbol}")

# Display feature names
feature_names = features['AAPL'].columns.tolist()
print(f"\nSample features (first 20):")
for i, name in enumerate(feature_names[:20]):
    print(f"  {i+1:2d}. {name}")

# %% Feature correlation analysis
# Calculate correlation matrix for first symbol
corr_matrix = features['AAPL'].iloc[-100:].corr()

# Find highly correlated features
high_corr = np.where(np.abs(corr_matrix) > 0.9)
high_corr_features = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                       for x, y in zip(*high_corr) if x != y and x < y]

print(f"\nHighly correlated features (|corr| > 0.9):")
for feat1, feat2, corr in high_corr_features[:10]:
    print(f"  {feat1[:30]:30s} <-> {feat2[:30]:30s}: {corr:.3f}")

# %% [markdown]
# ## 3. Model Training

# %% Prepare data pipeline
print("\nPreparing data pipeline...")

# Initialize data pipeline
data_pipeline = DataPipeline(
    symbols=['AAPL'],  # Start with single symbol
    start_date='2020-01-01',
    end_date='2023-01-01',
    target_type='classification',
    target_horizon=1,
    target_threshold=0.002  # 0.2% threshold
)

# Run pipeline
X, y = data_pipeline.run()
print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# %% Train models
print("\nTraining models...")

# Split data
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_val = X.iloc[split_idx:]
y_val = y.iloc[split_idx:]

# Train different models
models = {}
model_types = ['random_forest', 'xgboost', 'lightgbm']

for model_type in model_types:
    print(f"\nTraining {model_type}...")
    
    # Create trainer
    trainer = ModelTrainer(
        model_type=model_type,
        task_type='classification'
    )
    
    # Preprocess data
    X_train_proc, X_val_proc = trainer.preprocess(X_train, X_val)
    
    # Train model
    metrics = trainer.train(X_train_proc, y_train, X_val_proc, y_val)
    
    # Store model
    models[model_type] = trainer
    
    # Print metrics
    print(f"  Validation F1: {metrics.get('val_f1', 0):.4f}")
    print(f"  Validation Accuracy: {metrics.get('val_accuracy', 0):.4f}")

# %% Feature importance
# Get feature importance from best model
best_model = models['xgboost']
importance_df = best_model.get_feature_importance(X_train.columns.tolist())

if not importance_df.empty:
    # Plot top 20 features
    top_features = importance_df.head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Generate Trading Signals

# %% Generate predictions
print("\nGenerating trading signals...")

# Use best model for predictions
best_model = models['xgboost']

# Generate predictions on validation set
predictions = best_model.predict(X_val_proc)

# Create signals DataFrame
signals_df = pd.DataFrame({
    'date': X_val.index,
    'actual': y_val.values,
    'predicted': predictions
})

# Calculate accuracy
accuracy = (signals_df['actual'] == signals_df['predicted']).mean()
print(f"Signal accuracy: {accuracy:.4f}")

# Signal distribution
print(f"\nSignal distribution:")
print(signals_df['predicted'].value_counts())

# %% Visualize signals
# Load price data for visualization
test_data = data_loader.load('AAPL', 
                            X_val.index[0].strftime('%Y-%m-%d'),
                            X_val.index[-1].strftime('%Y-%m-%d'))

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Plot price
ax1.plot(test_data.index, test_data['close'], label='Close Price', color='black', alpha=0.7)

# Add buy/sell signals
buy_signals = signals_df[signals_df['predicted'] == 1]
sell_signals = signals_df[signals_df['predicted'] == -1]

ax1.scatter(buy_signals['date'], test_data.loc[buy_signals['date'], 'close'], 
           color='green', marker='^', s=100, label='Buy Signal', alpha=0.7)
ax1.scatter(sell_signals['date'], test_data.loc[sell_signals['date'], 'close'], 
           color='red', marker='v', s=100, label='Sell Signal', alpha=0.7)

ax1.set_ylabel('Price ($)')
ax1.set_title('AAPL Price with Trading Signals')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot returns
returns = test_data['close'].pct_change()
ax2.plot(test_data.index, returns, label='Daily Returns', alpha=0.5)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.axhline(y=0.002, color='green', linestyle='--', alpha=0.3, label='Buy Threshold')
ax2.axhline(y=-0.002, color='red', linestyle='--', alpha=0.3, label='Sell Threshold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Returns')
ax2.set_title('Daily Returns')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Backtesting

# %% Run backtest
print("\nRunning backtest...")

# Prepare signals for backtesting
signals = pd.Series(predictions, index=X_val.index)

# Initialize backtest engine
backtest_engine = BacktestEngine(
    test_data,
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005,
    position_size_method='equal_weight'
)

# Add signals
backtest_engine.add_signals(signals)

# Run backtest
results = backtest_engine.run(
    stop_loss=0.02,  # 2% stop loss
    take_profit=0.05  # 5% take profit
)

# Print metrics
metrics = results['metrics']
print("\nBacktest Results:")
print(f"  Total Return: {metrics['total_return']:.2%}")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"  Win Rate: {metrics['win_rate']:.2%}")
print(f"  Total Trades: {metrics['total_trades']}")
print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

# %% Plot backtest results
# Create comprehensive backtest visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Equity Curve
ax1 = axes[0]
equity_curve = results['equity_curve']
ax1.plot(equity_curve.index, equity_curve.values, label='Portfolio Value', linewidth=2)
ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax1.set_ylabel('Portfolio Value ($)')
ax1.set_title('Equity Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Drawdown
ax2 = axes[1]
running_max = equity_curve.expanding().max()
drawdown = (equity_curve - running_max) / running_max * 100
ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
ax2.set_ylabel('Drawdown (%)')
ax2.set_title('Drawdown')
ax2.grid(True, alpha=0.3)

# Plot 3: Trade Distribution
ax3 = axes[2]
if not results['trades'].empty:
    trades_df = results['trades']
    returns = trades_df['return_pct'].dropna() * 100
    ax3.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Trade Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Trade Returns Distribution (n={len(returns)})')
    ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Summary
# 
# This example demonstrated:
# - Loading market data for multiple symbols
# - Generating 200+ technical features
# - Training multiple ML models (RandomForest, XGBoost, LightGBM)
# - Generating trading signals
# - Running comprehensive backtests
# - Visualizing results
# 
# Next steps:
# - Try different hyperparameters
# - Test on different time periods
# - Implement ensemble methods
# - Add more sophisticated risk management

print("\n✅ Example completed successfully!")
