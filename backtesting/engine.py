"""
Backtesting Engine Module

Comprehensive backtesting with performance metrics, risk analysis,
and walk-forward validation support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Performance metrics
from scipy import stats
import quantstats as qs
import empyrical as ep

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Internal imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class Trade:
    """Single trade representation."""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'long' or 'short'
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    
    def close(self, exit_time: datetime, exit_price: float):
        """Close the trade."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.return_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.return_pct = (self.entry_price - exit_price) / self.entry_price
        
        # Apply costs
        self.pnl -= (self.commission + self.slippage)
        
    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_time is None


@dataclass
class Position:
    """Current position tracking."""
    symbol: str
    quantity: float
    avg_price: float
    side: str
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L."""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_price - current_price) * self.quantity


class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_positions: int = 10,
        position_size_method: str = 'equal_weight'
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
            slippage: Slippage per trade (as fraction)
            max_positions: Maximum number of concurrent positions
            position_size_method: Method for position sizing
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_positions = max_positions
        self.position_size_method = position_size_method
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []
        
    def calculate_position_size(
        self,
        signal_strength: float = 1.0,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on method.
        
        Args:
            signal_strength: Strength of signal (0-1)
            volatility: Current volatility for risk-based sizing
            
        Returns:
            Position size as fraction of capital
        """
        if self.position_size_method == 'equal_weight':
            return 1.0 / self.max_positions
            
        elif self.position_size_method == 'signal_weighted':
            base_size = 1.0 / self.max_positions
            return base_size * signal_strength
            
        elif self.position_size_method == 'volatility_scaled':
            if volatility is None:
                return 1.0 / self.max_positions
            target_vol = 0.02  # 2% target volatility
            return min(target_vol / volatility, 1.0 / self.max_positions)
            
        elif self.position_size_method == 'kelly':
            # Simplified Kelly Criterion
            win_prob = 0.5 + signal_strength * 0.1
            win_loss_ratio = 1.5
            kelly_pct = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            return max(0, min(kelly_pct, 0.25))  # Cap at 25%
            
        else:
            return 1.0 / self.max_positions
    
    def open_position(
        self,
        symbol: str,
        price: float,
        quantity: float,
        side: str,
        timestamp: datetime
    ) -> Trade:
        """Open a new position."""
        # Check if we can open more positions
        if len(self.positions) >= self.max_positions:
            return None
            
        # Calculate costs
        trade_value = price * quantity
        commission = trade_value * self.commission
        slippage = trade_value * self.slippage
        
        # Check if we have enough capital
        required_capital = trade_value + commission + slippage
        if required_capital > self.capital:
            return None
        
        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_time=timestamp,
            exit_time=None,
            entry_price=price * (1 + self.slippage if side == 'long' else 1 - self.slippage),
            exit_price=None,
            quantity=quantity,
            side=side,
            commission=commission,
            slippage=slippage
        )
        
        # Update position
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.quantity += quantity
            pos.avg_price = (pos.avg_price * (pos.quantity - quantity) + trade.entry_price * quantity) / pos.quantity
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=trade.entry_price,
                side=side
            )
        
        # Update capital
        self.capital -= required_capital
        
        self.trades.append(trade)
        return trade
    
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        quantity: Optional[float] = None
    ) -> Optional[Trade]:
        """Close a position."""
        if symbol not in self.positions:
            return None
            
        pos = self.positions[symbol]
        
        # Determine quantity to close
        if quantity is None:
            quantity = pos.quantity
        else:
            quantity = min(quantity, pos.quantity)
        
        # Find open trade
        open_trade = None
        for trade in reversed(self.trades):
            if trade.symbol == symbol and trade.is_open:
                open_trade = trade
                break
        
        if open_trade is None:
            return None
        
        # Close trade
        exit_price = price * (1 - self.slippage if pos.side == 'long' else 1 + self.slippage)
        open_trade.close(timestamp, exit_price)
        
        # Update capital
        self.capital += exit_price * quantity - open_trade.commission
        
        # Update or remove position
        pos.quantity -= quantity
        if pos.quantity == 0:
            del self.positions[symbol]
        
        return open_trade
    
    def update_equity(self, prices: Dict[str, float], timestamp: datetime):
        """Update equity curve with current prices."""
        total_value = self.capital
        
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_unrealized_pnl(prices[symbol])
                total_value += prices[symbol] * pos.quantity
        
        self.equity_curve.append(total_value)
        self.timestamps.append(timestamp)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if len(self.equity_curve) < 2:
            return {}
        
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        metrics = {
            'total_return': (self.equity_curve[-1] - self.initial_capital) / self.initial_capital,
            'total_trades': len(self.trades),
            'winning_trades': sum(1 for t in self.trades if t.pnl and t.pnl > 0),
            'losing_trades': sum(1 for t in self.trades if t.pnl and t.pnl < 0),
            'avg_win': np.mean([t.pnl for t in self.trades if t.pnl and t.pnl > 0]) if any(t.pnl > 0 for t in self.trades if t.pnl) else 0,
            'avg_loss': np.mean([t.pnl for t in self.trades if t.pnl and t.pnl < 0]) if any(t.pnl < 0 for t in self.trades if t.pnl) else 0,
            'win_rate': sum(1 for t in self.trades if t.pnl and t.pnl > 0) / len(self.trades) if self.trades else 0,
            'profit_factor': abs(sum(t.pnl for t in self.trades if t.pnl and t.pnl > 0) / sum(t.pnl for t in self.trades if t.pnl and t.pnl < 0)) if any(t.pnl < 0 for t in self.trades if t.pnl) else 0,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'sortino_ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'calmar_ratio': (self.equity_curve[-1] / self.initial_capital - 1) / abs(self._calculate_max_drawdown()) if self._calculate_max_drawdown() != 0 else 0
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        equity = pd.Series(self.equity_curve)
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return drawdown.min()


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_positions: int = 10,
        position_size_method: str = 'equal_weight'
    ):
        """
        Initialize backtest engine.
        
        Args:
            data: OHLCV data with signals
            initial_capital: Starting capital
            commission: Commission per trade
            slippage: Slippage per trade
            max_positions: Maximum concurrent positions
            position_size_method: Position sizing method
        """
        self.data = data
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            max_positions=max_positions,
            position_size_method=position_size_method
        )
        
        self.signals = pd.DataFrame(index=data.index)
        self.results = {}
        
    def add_signals(
        self,
        signals: pd.Series,
        signal_type: str = 'discrete'
    ):
        """
        Add trading signals.
        
        Args:
            signals: Trading signals (1=buy, 0=hold, -1=sell)
            signal_type: Type of signal ('discrete' or 'continuous')
        """
        self.signals['signal'] = signals
        self.signals['signal_type'] = signal_type
        
    def run(
        self,
        strategy: Optional[Callable] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run backtest.
        
        Args:
            strategy: Custom strategy function
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            trailing_stop: Trailing stop percentage
            
        Returns:
            Backtest results
        """
        print("Running backtest...")
        
        # Track trailing stops
        trailing_stops = {}
        
        for i, (timestamp, row) in enumerate(self.data.iterrows()):
            # Get current prices
            current_prices = {'symbol': row['close']}
            
            # Check for exit signals on existing positions
            for symbol in list(self.portfolio.positions.keys()):
                pos = self.portfolio.positions[symbol]
                current_price = row['close']
                
                # Check stop loss
                if stop_loss and pos.side == 'long':
                    if current_price <= pos.avg_price * (1 - stop_loss):
                        self.portfolio.close_position(symbol, current_price, timestamp)
                        continue
                
                # Check take profit
                if take_profit and pos.side == 'long':
                    if current_price >= pos.avg_price * (1 + take_profit):
                        self.portfolio.close_position(symbol, current_price, timestamp)
                        continue
                
                # Update trailing stop
                if trailing_stop and symbol in trailing_stops:
                    if pos.side == 'long':
                        trailing_stops[symbol] = max(trailing_stops[symbol], current_price * (1 - trailing_stop))
                        if current_price <= trailing_stops[symbol]:
                            self.portfolio.close_position(symbol, current_price, timestamp)
                            del trailing_stops[symbol]
                            continue
                
                # Check for exit signal
                if 'signal' in self.signals.columns:
                    signal = self.signals.loc[timestamp, 'signal']
                    if signal == -1:  # Sell signal
                        self.portfolio.close_position(symbol, current_price, timestamp)
                        if symbol in trailing_stops:
                            del trailing_stops[symbol]
            
            # Check for entry signals
            if 'signal' in self.signals.columns:
                signal = self.signals.loc[timestamp, 'signal']
                
                if signal == 1:  # Buy signal
                    # Calculate position size
                    volatility = self.data['close'].pct_change().rolling(20).std().iloc[i] if i >= 20 else 0.02
                    position_size = self.portfolio.calculate_position_size(
                        signal_strength=abs(signal) if self.signals['signal_type'].iloc[0] == 'continuous' else 1.0,
                        volatility=volatility
                    )
                    
                    # Calculate quantity
                    capital_to_invest = self.portfolio.capital * position_size
                    quantity = capital_to_invest / row['close']
                    
                    # Open position
                    trade = self.portfolio.open_position(
                        symbol='symbol',
                        price=row['close'],
                        quantity=quantity,
                        side='long',
                        timestamp=timestamp
                    )
                    
                    # Initialize trailing stop
                    if trailing_stop and trade:
                        trailing_stops['symbol'] = row['close'] * (1 - trailing_stop)
            
            # Update portfolio equity
            self.portfolio.update_equity(current_prices, timestamp)
        
        # Close any remaining positions
        for symbol in list(self.portfolio.positions.keys()):
            self.portfolio.close_position(
                symbol,
                self.data.iloc[-1]['close'],
                self.data.index[-1]
            )
        
        # Calculate results
        self.results = {
            'metrics': self.portfolio.get_performance_metrics(),
            'equity_curve': pd.Series(
                self.portfolio.equity_curve,
                index=pd.DatetimeIndex(self.portfolio.timestamps + [self.data.index[-1]])[:len(self.portfolio.equity_curve)]
            ),
            'trades': pd.DataFrame([
                {
                    'symbol': t.symbol,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'side': t.side,
                    'pnl': t.pnl,
                    'return_pct': t.return_pct
                }
                for t in self.portfolio.trades
            ]),
            'signals': self.signals
        }
        
        print(f"Backtest complete. Total return: {self.results['metrics']['total_return']:.2%}")
        
        return self.results
    
    def plot_results(
        self,
        show_trades: bool = True,
        show_drawdown: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot backtest results.
        
        Args:
            show_trades: Whether to show trade markers
            show_drawdown: Whether to show drawdown chart
            save_path: Path to save plot
        """
        fig = make_subplots(
            rows=3 if show_drawdown else 2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Signals', 'Equity Curve', 'Drawdown') if show_drawdown else ('Price & Signals', 'Equity Curve'),
            row_heights=[0.4, 0.3, 0.3] if show_drawdown else [0.5, 0.5]
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['open'],
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add trade markers
        if show_trades and not self.results['trades'].empty:
            trades_df = self.results['trades']
            
            # Buy markers
            buys = trades_df[trades_df['side'] == 'long']
            if not buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buys['entry_time'],
                        y=buys['entry_price'],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name='Buy'
                    ),
                    row=1, col=1
                )
            
            # Sell markers
            sells = trades_df[trades_df['exit_time'].notna()]
            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells['exit_time'],
                        y=sells['exit_price'],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name='Sell'
                    ),
                    row=1, col=1
                )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.results['equity_curve'].index,
                y=self.results['equity_curve'].values,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # Drawdown
        if show_drawdown:
            equity = self.results['equity_curve']
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    fill='tozeroy',
                    name='Drawdown %',
                    line=dict(color='red', width=1)
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Backtest Results',
            xaxis_title='Date',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return fig
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate detailed backtest report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report as string
        """
        metrics = self.results['metrics']
        
        report = f"""
        =====================================
        BACKTEST PERFORMANCE REPORT
        =====================================
        
        RETURNS
        -------
        Total Return: {metrics['total_return']:.2%}
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Sortino Ratio: {metrics['sortino_ratio']:.2f}
        Calmar Ratio: {metrics['calmar_ratio']:.2f}
        
        RISK METRICS
        ------------
        Maximum Drawdown: {metrics['max_drawdown']:.2%}
        
        TRADE STATISTICS
        ----------------
        Total Trades: {metrics['total_trades']}
        Winning Trades: {metrics['winning_trades']}
        Losing Trades: {metrics['losing_trades']}
        Win Rate: {metrics['win_rate']:.2%}
        
        Average Win: ${metrics['avg_win']:.2f}
        Average Loss: ${metrics['avg_loss']:.2f}
        Profit Factor: {metrics['profit_factor']:.2f}
        
        EQUITY STATISTICS
        -----------------
        Starting Capital: ${self.portfolio.initial_capital:,.2f}
        Ending Capital: ${self.portfolio.equity_curve[-1]:,.2f}
        Peak Equity: ${max(self.portfolio.equity_curve):,.2f}
        
        =====================================
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report


class WalkForwardBacktest:
    """Walk-forward backtesting."""
    
    def __init__(
        self,
        backtest_params: Dict[str, Any],
        n_splits: int = 5,
        train_ratio: float = 0.8
    ):
        """
        Initialize walk-forward backtest.
        
        Args:
            backtest_params: Parameters for BacktestEngine
            n_splits: Number of walk-forward splits
            train_ratio: Ratio of data for training
        """
        self.backtest_params = backtest_params
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.results = []
        
    def run(
        self,
        data: pd.DataFrame,
        model_trainer: Callable,
        signal_generator: Callable
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            data: Complete dataset
            model_trainer: Function to train model on data
            signal_generator: Function to generate signals from model
            
        Returns:
            Aggregated results
        """
        n_samples = len(data)
        test_size = int(n_samples * (1 - self.train_ratio) / self.n_splits)
        
        fold_results = []
        
        for i in range(self.n_splits):
            print(f"\nWalk-forward fold {i+1}/{self.n_splits}")
            
            # Define train/test split
            train_end = int(n_samples * self.train_ratio) + i * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Train model
            model = model_trainer(train_data)
            
            # Generate signals
            signals = signal_generator(model, test_data)
            
            # Run backtest
            engine = BacktestEngine(test_data, **self.backtest_params)
            engine.add_signals(signals)
            results = engine.run()
            
            fold_results.append({
                'fold': i + 1,
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'metrics': results['metrics'],
                'equity_curve': results['equity_curve'],
                'trades': results['trades']
            })
        
        # Aggregate results
        aggregated = self._aggregate_results(fold_results)
        
        return aggregated
    
    def _aggregate_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across folds."""
        # Combine equity curves
        combined_equity = pd.concat([r['equity_curve'] for r in fold_results])
        
        # Average metrics
        metric_keys = fold_results[0]['metrics'].keys()
        avg_metrics = {}
        for key in metric_keys:
            values = [r['metrics'][key] for r in fold_results]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
        
        # Combine all trades
        all_trades = pd.concat([r['trades'] for r in fold_results if not r['trades'].empty])
        
        return {
            'fold_results': fold_results,
            'aggregated_metrics': avg_metrics,
            'combined_equity': combined_equity,
            'all_trades': all_trades
        }


if __name__ == "__main__":
    print("AlphaStream Backtesting Engine")
    print("Components: Portfolio, BacktestEngine, WalkForwardBacktest")
    print("Ready to backtest trading strategies with comprehensive metrics")
