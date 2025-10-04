"""
Base Strategy Abstract Class
Provides common interface for all trading strategies with Optuna-ready architecture

Author: HFT System
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies

    All strategies must implement:
    - vectorized_process_dataset: Main backtesting logic
    - get_default_params: Default strategy parameters
    - get_param_space: Parameter space for optimization (Optuna-ready)
    """

    def __init__(self, symbol: str, **params):
        """
        Initialize base strategy

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            **params: Strategy-specific parameters
        """
        self.symbol = symbol
        self.params = params

    @abstractmethod
    def vectorized_process_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process entire dataset using vectorized operations

        Args:
            df: DataFrame with OHLCV data
                Required columns: time, open, high, low, close, Volume

        Returns:
            Dictionary with backtest results:
            {
                'trades': List[Dict],  # List of trade records
                'total': int,  # Total trades
                'win_rate': float,  # Win rate (0-1)
                'net_pnl': float,  # Net profit/loss
                'sharpe_ratio': float,  # Sharpe ratio
                'max_drawdown': float,  # Max drawdown %
                'bb_data': Dict,  # Chart data
                ...
            }
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default parameters for the strategy

        Returns:
            Dictionary with default parameter values
            Example: {'period': 50, 'std_dev': 2.0, 'stop_loss_pct': 0.5}
        """
        pass

    @classmethod
    @abstractmethod
    def get_param_space(cls) -> Dict[str, tuple]:
        """
        Get parameter space for optimization (Optuna-ready)

        Returns:
            Dictionary mapping parameter names to (type, bounds) tuples

            Format:
            {
                'param_name': ('int', min_value, max_value),
                'param_name': ('float', min_value, max_value),
                'param_name': ('categorical', [option1, option2, ...])
            }

            Example:
            {
                'period': ('int', 10, 100),
                'std_dev': ('float', 1.0, 3.0),
                'stop_loss_pct': ('float', 0.1, 2.0)
            }
        """
        pass

    @property
    def name(self) -> str:
        """Strategy name (class name by default)"""
        return self.__class__.__name__

    @staticmethod
    def calculate_performance_metrics(trades: List[Dict], initial_capital: float) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from trades

        Common method for all strategies to ensure consistent metrics calculation

        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital amount

        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                'total': 0, 'win_rate': 0, 'net_pnl': 0, 'net_pnl_percentage': 0,
                'max_drawdown': 0, 'sharpe_ratio': 0, 'profit_factor': 0,
                'total_winning_trades': 0, 'total_losing_trades': 0,
                'average_win': 0, 'average_loss': 0, 'largest_win': 0, 'largest_loss': 0,
                'loose_streak': 0
            }

        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_pnl = sum(t.get('pnl', 0) for t in trades)

        # Win/Loss statistics
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        largest_win = max([t['pnl'] for t in winning_trades], default=0)
        largest_loss = min([t['pnl'] for t in losing_trades], default=0)

        # Return percentage
        return_pct = (total_pnl / initial_capital * 10) if initial_capital > 0 else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Simplified Sharpe ratio (using daily returns approximation)
        trade_returns = [t.get('pnl', 0) / initial_capital for t in trades]
        if len(trade_returns) > 1:
            # Calculate Sharpe ratio based on trade returns
            mean_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)
            sharpe_ratio = mean_return / std_return if std_return != 0 else 0
            # Annualize the ratio (assuming ~252 trading days)
            sharpe_ratio *= np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Calculate max drawdown
        equity_curve = [initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.get('pnl', 0))

        peak = initial_capital
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Calculate loose streak (максимальное количество стоп-лоссов подряд)
        loose_streak = 0
        max_loose_streak = 0
        for trade in trades:
            exit_reason = trade.get('exit_reason', '')
            if 'stop_loss' in exit_reason:
                loose_streak += 1
                max_loose_streak = max(max_loose_streak, loose_streak)
            else:
                loose_streak = 0

        return {
            'total': total_trades,
            'win_rate': win_rate,
            'net_pnl': total_pnl,
            'net_pnl_percentage': return_pct,
            'max_drawdown': max_dd * 100,  # As percentage
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'total_winning_trades': len(winning_trades),
            'total_losing_trades': len(losing_trades),
            'average_win': avg_win,
            'average_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'loose_streak': max_loose_streak
        }
