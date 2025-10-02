"""
Base Strategy Abstract Class
Provides common interface for all trading strategies with Optuna-ready architecture

Author: HFT System
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic strategy statistics (optional override)

        Returns:
            Dictionary with strategy statistics
        """
        return {
            'symbol': self.symbol,
            'params': self.params
        }

    @property
    def name(self) -> str:
        """Strategy name (class name by default)"""
        return self.__class__.__name__
