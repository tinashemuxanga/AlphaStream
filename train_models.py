#!/usr/bin/env python3
"""
AlphaStream Main Training Script

CLI interface for training and evaluating trading models.
"""

import argparse
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Internal imports
from ml.dataset import DataPipeline
from ml.train import TrainingPipeline
from ml.models import ModelFactory
from backtesting.engine import BacktestEngine, WalkForwardBacktest


def load_config(config_path: str) -> dict:
    """Load configuration from file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if path.suffix == '.json':
        with open(path) as f:
            return json.load(f)
    elif path.suffix in ['.yml', '.yaml']:
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def train_models(args):
    """Train models based on configuration."""
    print("=" * 50)
    print("AlphaStream Model Training")
    print("=" * 50)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'symbols': args.symbols.split(',') if args.symbols else ['AAPL', 'GOOGL', 'MSFT'],
            'start_date': args.start_date or '2020-01-01',
            'end_date': args.end_date or datetime.now().strftime('%Y-%m-%d'),
            'models': [
                {'type': 'random_forest', 'task_type': 'classification'},
                {'type': 'xgboost', 'task_type': 'classification'},
                {'type': 'lightgbm', 'task_type': 'classification'}
            ],
            'target': {
                'type': 'classification',
                'horizon': 1,
                'threshold': 0.002
            },
            'use_walk_forward': True,
            'n_splits': 5,
            'tune_hyperparameters': args.tune,
            'n_trials': 50
        }
    
    print(f"\nConfiguration:")
    print(f"  Symbols: {config['symbols']}")
    print(f"  Period: {config['start_date']} to {config['end_date']}")
    print(f"  Models: {[m['type'] for m in config['models']]}")
    print(f"  Walk-forward splits: {config.get('n_splits', 5)}")
    print(f"  Hyperparameter tuning: {config.get('tune_hyperparameters', False)}")
    
    # Initialize data pipeline
    print("\nInitializing data pipeline...")
    data_pipeline = DataPipeline(
        symbols=config['symbols'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        target_type=config['target']['type'],
        target_horizon=config['target']['horizon'],
        target_threshold=config['target']['threshold']
    )
    
    # Run data pipeline
    print("Loading and processing data...")
    features, targets = data_pipeline.run()
    print(f"  Dataset shape: {features.shape}")
    print(f"  Features: {features.shape[1]}")
    print(f"  Target distribution: {targets.value_counts().to_dict()}")
    
    # Initialize training pipeline
    print("\nInitializing training pipeline...")
    training_pipeline = TrainingPipeline(
        experiment_name=args.experiment or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Train models
    print("\nTraining models...")
    results = training_pipeline.run(
        data_pipeline=data_pipeline,
        model_configs=config['models'],
        use_walk_forward=config.get('use_walk_forward', True),
        tune_hyperparameters=config.get('tune_hyperparameters', False),
        n_trials=config.get('n_trials', 50)
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Training Results")
    print("=" * 50)
    
    for model_type, model_results in results['models'].items():
        print(f"\n{model_type.upper()}:")
        avg_metrics = model_results['avg_metrics']
        for metric, value in avg_metrics.items():
            if 'avg_' in metric:
                print(f"  {metric.replace('avg_', '').replace('_', ' ').title()}: {value:.4f}")
    
    # Save best model
    if results['best_model']:
        best_model = results['best_model']
        print(f"\nBest Model: {best_model['type']}")
        print(f"  Score: {best_model['score']:.4f}")
        
        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{best_model['type']}_best.pkl"
        joblib.dump(best_model['trainer'].model, model_path)
        print(f"  Saved to: {model_path}")
        
        # Save ensemble model if multiple models
        if len(results['models']) > 1:
            print("\nCreating ensemble model...")
            ensemble_models = []
            for model_type, model_results in results['models'].items():
                # Get the trained model from the last fold
                # In production, you'd retrain on full data
                ensemble_models.append((model_type, best_model['trainer'].model))
            
            # Create ensemble
            ensemble = ModelFactory.create('ensemble', 'classification')
            ensemble.models = ensemble_models
            
            # Save ensemble
            ensemble_path = model_dir / "ensemble.pkl"
            joblib.dump(ensemble, ensemble_path)
            print(f"  Ensemble saved to: {ensemble_path}")
    
    print("\nTraining complete!")
    
    return results


def backtest_strategy(args):
    """Run backtesting on trained models."""
    print("=" * 50)
    print("AlphaStream Backtesting")
    print("=" * 50)
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    print(f"\nLoading model: {model_path}")
    model = joblib.load(model_path)
    
    # Load data
    from ml.dataset import DataLoader
    data_loader = DataLoader()
    
    symbols = args.symbols.split(',') if args.symbols else ['AAPL']
    
    for symbol in symbols:
        print(f"\nBacktesting {symbol}...")
        
        # Load data
        data = data_loader.load(
            symbol,
            args.start_date or '2022-01-01',
            args.end_date or datetime.now().strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print(f"  No data found for {symbol}")
            continue
        
        # Generate features
        from ml.features import FeatureEngineer
        feature_engineer = FeatureEngineer()
        features = feature_engineer.transform(data)
        
        # Generate predictions
        predictions = model.predict(features)
        signals = pd.Series(predictions, index=features.index)
        
        # Run backtest
        engine = BacktestEngine(
            data,
            initial_capital=args.capital or 100000,
            commission=0.001,
            slippage=0.0005,
            position_size_method=args.position_size or 'equal_weight'
        )
        
        engine.add_signals(signals)
        results = engine.run(
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            trailing_stop=args.trailing_stop
        )
        
        # Print metrics
        metrics = results['metrics']
        print(f"\n  Performance Metrics:")
        print(f"    Total Return: {metrics['total_return']:.2%}")
        print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"    Win Rate: {metrics['win_rate']:.2%}")
        print(f"    Total Trades: {metrics['total_trades']}")
        print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Save report
        if args.output:
            report_path = Path(args.output) / f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path.parent.mkdir(exist_ok=True)
            report = engine.generate_report(str(report_path))
            print(f"  Report saved to: {report_path}")
        
        # Plot results
        if args.plot:
            plot_path = Path(args.output) / f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig = engine.plot_results(save_path=str(plot_path))
            print(f"  Plot saved to: {plot_path}")


def evaluate_models(args):
    """Evaluate and compare models."""
    print("=" * 50)
    print("AlphaStream Model Evaluation")
    print("=" * 50)
    
    # Load models
    model_dir = Path("models")
    if not model_dir.exists():
        print("No models directory found. Please train models first.")
        return
    
    models = {}
    for model_file in model_dir.glob("*.pkl"):
        model_name = model_file.stem
        models[model_name] = joblib.load(model_file)
        print(f"Loaded model: {model_name}")
    
    if not models:
        print("No models found.")
        return
    
    # Load test data
    from ml.dataset import DataPipeline
    
    data_pipeline = DataPipeline(
        symbols=args.symbols.split(',') if args.symbols else ['AAPL'],
        start_date=args.start_date or '2023-01-01',
        end_date=args.end_date or datetime.now().strftime('%Y-%m-%d'),
        target_type='classification'
    )
    
    X_test, y_test = data_pipeline.run()
    
    print(f"\nTest data shape: {X_test.shape}")
    
    # Evaluate each model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            results[model_name] = metrics
            
            # Print metrics
            for metric, value in metrics.items():
                print(f"  {metric.title()}: {value:.4f}")
                
        except Exception as e:
            print(f"  Error evaluating model: {e}")
    
    # Compare models
    if len(results) > 1:
        print("\n" + "=" * 50)
        print("Model Comparison")
        print("=" * 50)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('f1', ascending=False)
        
        print("\nRanked by F1 Score:")
        print(comparison_df.to_string())
        
        # Save comparison
        if args.output:
            comparison_path = Path(args.output) / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            comparison_path.parent.mkdir(exist_ok=True)
            comparison_df.to_csv(comparison_path)
            print(f"\nComparison saved to: {comparison_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AlphaStream - ML Trading Signal Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models with default config
  python train_models.py train --symbols AAPL,GOOGL,MSFT
  
  # Train with custom config
  python train_models.py train --config config/training.yaml
  
  # Train with hyperparameter tuning
  python train_models.py train --symbols AAPL --tune
  
  # Backtest a model
  python train_models.py backtest --model models/xgboost_best.pkl --symbols AAPL
  
  # Evaluate models
  python train_models.py evaluate --symbols AAPL,GOOGL
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    train_parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    train_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    train_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    train_parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    train_parser.add_argument('--experiment', type=str, help='Experiment name for tracking')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--model', type=str, required=True, help='Path to model file')
    backtest_parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    backtest_parser.add_argument('--start-date', type=str, help='Start date')
    backtest_parser.add_argument('--end-date', type=str, help='End date')
    backtest_parser.add_argument('--capital', type=float, help='Initial capital')
    backtest_parser.add_argument('--position-size', type=str, help='Position sizing method')
    backtest_parser.add_argument('--stop-loss', type=float, help='Stop loss percentage')
    backtest_parser.add_argument('--take-profit', type=float, help='Take profit percentage')
    backtest_parser.add_argument('--trailing-stop', type=float, help='Trailing stop percentage')
    backtest_parser.add_argument('--output', type=str, help='Output directory')
    backtest_parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    evaluate_parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    evaluate_parser.add_argument('--start-date', type=str, help='Start date')
    evaluate_parser.add_argument('--end-date', type=str, help='End date')
    evaluate_parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'train':
        train_models(args)
    elif args.command == 'backtest':
        backtest_strategy(args)
    elif args.command == 'evaluate':
        evaluate_models(args)


if __name__ == "__main__":
    main()
