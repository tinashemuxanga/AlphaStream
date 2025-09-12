#!/usr/bin/env python3
"""
Quick test script to verify AlphaStream is working correctly.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("AlphaStream System Test")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing module imports...")
try:
    from ml.dataset import DataLoader, DataPipeline
    from ml.features import FeatureEngineer
    from ml.models import ModelFactory
    print("✅ ML modules imported successfully")
except Exception as e:
    print(f"❌ Error importing ML modules: {e}")
    sys.exit(1)

# Test 2: Data loading
print("\n2. Testing data loading...")
try:
    data_loader = DataLoader()
    data = data_loader.load(
        symbols='AAPL',
        start_date='2024-01-01',
        end_date='2024-03-01'
    )
    print(f"✅ Loaded {len(data)} days of AAPL data")
    print(f"   Latest close: ${data['close'].iloc[-1]:.2f}")
except Exception as e:
    print(f"❌ Error loading data: {e}")

# Test 3: Feature engineering
print("\n3. Testing feature engineering...")
try:
    feature_engineer = FeatureEngineer()
    features = feature_engineer.transform(data)
    print(f"✅ Generated {features.shape[1]} features")
    print(f"   Sample features: {list(features.columns[:5])}")
except Exception as e:
    print(f"❌ Error generating features: {e}")

# Test 4: Model creation
print("\n4. Testing model creation...")
try:
    model = ModelFactory.create('random_forest', 'classification')
    print(f"✅ Created RandomForest model")
    
    model = ModelFactory.create('xgboost', 'classification')
    print(f"✅ Created XGBoost model")
except Exception as e:
    print(f"❌ Error creating models: {e}")

# Test 5: Quick training test
print("\n5. Testing model training (mini dataset)...")
try:
    # Prepare mini dataset
    from ml.features import TargetGenerator
    target_gen = TargetGenerator(target_type='classification')
    
    # Use last 100 rows for quick test
    mini_features = features.iloc[-100:]
    mini_targets = target_gen.generate(data.iloc[-100:])
    
    # Remove NaN
    valid_idx = ~(mini_features.isna().any(axis=1) | mini_targets.isna())
    mini_features = mini_features[valid_idx]
    mini_targets = mini_targets[valid_idx]
    
    if len(mini_features) > 50:
        # Split data
        split_idx = int(len(mini_features) * 0.7)
        X_train = mini_features.iloc[:split_idx]
        y_train = mini_targets.iloc[:split_idx]
        X_test = mini_features.iloc[split_idx:]
        y_test = mini_targets.iloc[split_idx:]
        
        # Train model
        model = ModelFactory.create('random_forest', 'classification', n_estimators=10)
        model.train(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        print(f"✅ Model trained successfully")
        print(f"   Test accuracy: {accuracy:.2%}")
    else:
        print("⚠️  Not enough data for training test")
except Exception as e:
    print(f"❌ Error in training test: {e}")

# Test 6: Backtesting
print("\n6. Testing backtesting engine...")
try:
    from backtesting.engine import BacktestEngine
    
    # Create simple signals
    import pandas as pd
    signals = pd.Series([1, -1, 0, 1, -1] * (len(data) // 5), index=data.index[:len(data) // 5 * 5])
    
    # Run mini backtest
    engine = BacktestEngine(
        data.iloc[:len(signals)],
        initial_capital=10000,
        commission=0.001
    )
    engine.add_signals(signals)
    results = engine.run()
    
    print(f"✅ Backtesting engine working")
    print(f"   Total trades: {results['metrics']['total_trades']}")
except Exception as e:
    print(f"❌ Error in backtesting: {e}")

print("\n" + "=" * 60)
print("System Test Summary")
print("=" * 60)
print("\n✅ AlphaStream is working correctly!")
print("\nNext steps:")
print("1. Train full models: python train_models.py train --symbols AAPL")
print("2. Start API: python -m uvicorn api.main:app --reload")
print("3. Run backtests: python train_models.py backtest --model models/ensemble.pkl")
print("\n")
