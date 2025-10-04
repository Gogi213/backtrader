"""
Vectorized Bollinger Bands Mean Reversion Strategy
High-frequency trading focused, fully vectorized implementation
Processing OHLCV data efficiently with numpy/numba optimization

Author: HFT System
"""
import numpy as np
import pandas as pd
from numba import njit, prange
from datetime import datetime
from typing import List, Dict, Optional, Any
from ..data.technical_indicators import vectorized_bb_calculation, vectorized_signal_generation
from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry
import warnings


@StrategyRegistry.register('bollinger')
class VectorizedBollingerStrategy(BaseStrategy):
    """
    High-frequency Bollinger Bands strategy using fully vectorized operations
    Processes entire OHLCV datasets at once
    """

    def __init__(self,
                 symbol: str,
                 period: int = 50,
                 std_dev: float = 2.0,
                 stop_loss_pct: float = 0.005,
                 initial_capital: float = 10000.0,
                 commission_pct: float = 0.0005):
        """
        Initialize vectorized strategy

        Args:
            symbol: Trading symbol
            period: Bollinger Bands period
            std_dev: Standard deviation multiplier
            stop_loss_pct: Stop loss percentage (decimal)
            initial_capital: Initial capital
            commission_pct: Commission percentage (decimal)
        """
        super().__init__(symbol, period=period, std_dev=std_dev,
                        stop_loss_pct=stop_loss_pct, initial_capital=initial_capital,
                        commission_pct=commission_pct)

        self.period = period
        self.std_dev = std_dev
        self.stop_loss_pct = stop_loss_pct
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

        # Performance tracking
        self.current_capital = initial_capital
        self.completed_trades = []
        
    def vectorized_process_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process entire dataset using vectorized operations

        Args:
            df: DataFrame with OHLCV data containing columns: time, open, high, low, close, Volume

        Returns:
            Dictionary with backtest results including trades and performance metrics
        """
        # Validate input
        required_cols = ['time', 'price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract data arrays
        times = df['time'].values
        prices = df['price'].values.astype(np.float64)

        # Extract OHLC data for candlestick charts (if available)
        ohlc_data = {}
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_data = {
                'open': df['open'].values.astype(np.float64),
                'high': df['high'].values.astype(np.float64),
                'low': df['low'].values.astype(np.float64),
                'close': df['close'].values.astype(np.float64)
            }
            print(f"DEBUG STRATEGY: OHLC data extracted for candlestick charts")

        # CRITICAL FIX: Проверка что у нас достаточно данных для BB
        if len(prices) < self.period:
            print(f"WARNING: Недостаточно данных для BB период {self.period}. Есть только {len(prices)} точек.")
            print(f"Уменьшаем период BB до {len(prices)//2}")
            effective_period = max(5, len(prices)//2)  # Минимум 5, максимум половина данных
        else:
            effective_period = self.period

        # Perform all BB calculations at once using vectorized function
        sma, upper_band, lower_band = vectorized_bb_calculation(prices, effective_period, self.std_dev)
        
        # Generate all signals at once using vectorized function
        entry_signals, exit_signals, position_status = vectorized_signal_generation(
            prices, sma, upper_band, lower_band, self.stop_loss_pct
        )
        
        # Process the signals to generate actual trades
        trades = self._generate_trades_vectorized(times, prices, entry_signals, exit_signals, position_status)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(trades)
        
        # CRITICAL FIX: Properly format bb_data for chart (times remain in SECONDS here)
        valid_mask = ~np.isnan(sma)
        bb_data = {
            'times': times[valid_mask],  # Times in SECONDS (will be converted to ms in backtest)
            'prices': prices[valid_mask],
            'bb_middle': sma[valid_mask],
            'bb_upper': upper_band[valid_mask],
            'bb_lower': lower_band[valid_mask],
            'bb_period': self.period,
            'bb_std': self.std_dev
        }

        # Add OHLC data for candlestick charts (if available)
        if ohlc_data:
            bb_data.update({
                'open': ohlc_data['open'][valid_mask],
                'high': ohlc_data['high'][valid_mask],
                'low': ohlc_data['low'][valid_mask],
                'close': ohlc_data['close'][valid_mask]
            })
            print(f"DEBUG STRATEGY: Added OHLC data to bb_data for candlestick charts")

        print(f"DEBUG STRATEGY: Created bb_data with {len(bb_data['times'])} points")
        if len(bb_data['times']) > 0:
            print(f"DEBUG STRATEGY: Time range: {bb_data['times'][0]:.0f} to {bb_data['times'][-1]:.0f} seconds")
            print(f"DEBUG STRATEGY: Price range: {bb_data['prices'].min():.4f} to {bb_data['prices'].max():.4f}")
        else:
            print("DEBUG STRATEGY: WARNING - bb_data пустая! График не будет отображаться.")

        # Add vectorized results to metrics
        performance_metrics.update({
            'symbol': self.symbol,
            'trades': trades,
            'total_ticks': len(df),
            'bb_calculations': len(sma) - np.isnan(sma).sum(),
            'prices_array_size': len(prices),
            'current_capital': self.current_capital,
            'bb_data': bb_data
        })
        
        return performance_metrics
    
    def _generate_trades_vectorized(self, times: np.ndarray, prices: np.ndarray, entry_signals: np.ndarray, exit_signals: np.ndarray, position_status: np.ndarray) -> List[Dict]:
        """
        Generate actual trades from vectorized signals using fully vectorized operations
        
        Args:
            times: Array of timestamps
            prices: Array of prices
            entry_signals: Array of entry signals (1: long, -1: short, 0: none)
            exit_signals: Array of exit signals (1: exit long, -1: exit short, 0: none)
            position_status: Array of position status (1: long, -1: short, 0: none)
            
        Returns:
            List of trade dictionaries
        """
        # Find entry and exit points
        entry_indices = np.where(entry_signals != 0)[0]
        exit_indices = np.where(exit_signals != 0)[0]
        
        # If no entries or exits, return empty list
        if len(entry_indices) == 0 or len(exit_indices) == 0:
            return []
        
        # Create all possible entry-exit pairs where exit comes after entry
        valid_pairs = []
        exit_idx = 0
        
        for entry_idx in entry_indices:
            # Find the first exit that occurs after this entry
            while exit_idx < len(exit_indices) and exit_indices[exit_idx] <= entry_idx:
                exit_idx += 1
            
            if exit_idx < len(exit_indices):
                valid_pairs.append((entry_idx, exit_indices[exit_idx]))
                exit_idx += 1  # Move to next exit for subsequent entries
        
        if not valid_pairs:
            return []
        
        # Process all valid pairs at once
        trades = []
        current_capital = self.initial_capital
        
        for entry_idx, exit_idx in valid_pairs:
            entry_signal = entry_signals[entry_idx]
            exit_signal = exit_signals[exit_idx]
            
            side = 'long' if entry_signal == 1 else 'short'
            entry_price = prices[entry_idx]
            exit_price = prices[exit_idx]
            
            # Calculate P&L with commission
            position_size = 100.0  # Fixed position size in dollars
            commission_cost = position_size * self.commission_pct * 2 / 100  # Entry + Exit commission

            if side == 'long':
                pnl = (exit_price - entry_price) * (position_size / entry_price) - commission_cost
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:  # short
                pnl = (entry_price - exit_price) * (position_size / entry_price) - commission_cost
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            # Update local capital
            current_capital += pnl
            
            # FIXED: Calculate duration in minutes (timestamps are in SECONDS from dataset)
            duration_minutes = (times[exit_idx] - times[entry_idx]) / 60.0
            
            # Create trade record
            trade = {
                'timestamp': times[entry_idx],
                'exit_timestamp': times[exit_idx],
                'symbol': self.symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': 100.0,  # Fixed position size in dollars
                'pnl': pnl,
                'pnl_percentage': pnl_pct,
                'duration': duration_minutes,
                'exit_reason': self._determine_exit_reason(exit_signal), # Detailed exit reason based on signal type
                'capital_before': current_capital - pnl,
                'capital_after': current_capital
            }
            
            trades.append(trade)
        
        # Update instance capital after processing all trades
        self.current_capital = current_capital
        
        return trades

    def _determine_exit_reason(self, exit_signal: int) -> str:
        """
        Determine exit reason based on new exit signal codes

        Args:
            exit_signal: Exit signal from vectorized_signal_generation
                        1: stop loss long, 2: take profit long (SMA)
                        -1: stop loss short, -2: take profit short (SMA)

        Returns:
            String description of exit reason
        """
        exit_reasons = {
            1: 'stop_loss_long',
            2: 'take_profit_sma_long',
            -1: 'stop_loss_short',
            -2: 'take_profit_sma_short'
        }

        return exit_reasons.get(exit_signal, f'unknown_exit_{exit_signal}')

    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics from trades"""
        return BaseStrategy.calculate_performance_metrics(trades, self.initial_capital)

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default parameters for Bollinger Bands strategy

        Returns:
            Dictionary with default parameter values
        """
        return {
            'period': 50,
            'std_dev': 2.0,
            'stop_loss_pct': 0.005,  # 0.5%
            'initial_capital': 10000.0,
            'commission_pct': 0.0005  # 0.05%
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, tuple]:
        """
        Get parameter space for optimization (Optuna-ready)

        Returns:
            Dictionary mapping parameter names to (type, bounds) tuples
        """
        return {
            'period': ('int', 10, 200),
            'std_dev': ('float', 1.0, 3.5),
            'stop_loss_pct': ('float', 0.001, 0.02),  # 0.1% to 2%
            'initial_capital': ('float', 1000.0, 100000.0),
            'commission_pct': ('float', 0.0001, 0.001)  # 0.01% to 0.1%
        }


if __name__ == "__main__":
    # Test the strategy
    from .strategy_registry import StrategyRegistry
    from .strategy_factory import StrategyFactory

    print("Available strategies:", StrategyRegistry.list_strategies())
    strategy_class = StrategyRegistry.get('bollinger')
    strategy = StrategyFactory.create('bollinger', 'TESTUSDT', **strategy_class.get_default_params())
    print(f"Vectorized Bollinger Strategy initialized: {strategy.name}")
    print(f"Default params: {strategy.get_default_params()}")
    print(f"Param space: {strategy.get_param_space()}")
