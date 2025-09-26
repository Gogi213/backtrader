"""
Pure Tick Bollinger Bands Mean Reversion Strategy
High-frequency trading focused, numpy/numba optimized

Author: HFT System
"""
import numpy as np
import pandas as pd
from numba import jit, njit
from datetime import datetime
from typing import List, Dict, Optional, Any
from ..data.pure_tick_handler import PureTickHandler, fast_rolling_mean, fast_rolling_std


class PureTickBollingerStrategy:
    """
    High-frequency Bollinger Bands strategy working on pure tick data (DEL)
    This file is marked for deletion - use vectorized_bollinger_strategy.py instead
    No candle aggregation, operates on individual ticks
    """

    def __init__(self,
                 symbol: str,
                 period: int = 50,  # Smaller period for HFT
                 std_dev: float = 2.0,  # Tighter bands for HFT
                 stop_loss_pct: float = 0.005,  # 0.5% for HFT
                 initial_capital: float = 10000.0):
        """
        Initialize pure tick strategy

        Args:
            symbol: Trading symbol
            period: Bollinger Bands period (lower for HFT)
            std_dev: Standard deviation multiplier
            stop_loss_pct: Stop loss percentage (decimal)
            initial_capital: Initial capital
        """
        self.symbol = symbol
        self.period = period
        self.std_dev = std_dev
        self.stop_loss_pct = stop_loss_pct
        self.initial_capital = initial_capital

        # Strategy state
        self.position = None
        self.prices = []  # Price history for BB calculation
        self.completed_trades = []
        self.tick_count = 0

        # Performance tracking
        self.current_capital = initial_capital

    def process_tick(self, tick_time: pd.Timestamp, price: float, qty: float = 0.0) -> List[Dict]:
        """
        Process individual tick and generate signals

        Args:
            tick_time: Tick timestamp
            price: Tick price
            qty: Tick quantity (optional)

        Returns:
            List of completed trades (if any)
        """
        self.tick_count += 1
        self.prices.append(price)

        # Keep only necessary history for BB calculation
        max_history = self.period * 2  # Keep 2x period for safety
        if len(self.prices) > max_history:
            self.prices = self.prices[-max_history:]

        trades = []

        # Need minimum data for BB calculation
        if len(self.prices) < self.period:
            return trades

        # Calculate Bollinger Bands for current state
        prices_array = np.array(self.prices, dtype=np.float64)
        sma, upper_band, lower_band = self._calculate_bb_fast(prices_array)

        # Get current BB values (last value in arrays)
        current_sma = sma[-1]
        current_upper = upper_band[-1]
        current_lower = lower_band[-1]

        # Skip if BB values are NaN
        if np.isnan(current_sma) or np.isnan(current_upper) or np.isnan(current_lower):
            return trades

        # Trading logic
        if self.position is None:
            # Entry signals
            trade = self._check_entry_signals(tick_time, price, current_sma, current_upper, current_lower)
            if trade:
                self.position = trade

        else:
            # Exit signals
            completed_trade = self._check_exit_signals(tick_time, price, current_sma, current_upper, current_lower)
            if completed_trade:
                trades.append(completed_trade)
                self.completed_trades.append(completed_trade)
                self.position = None

        return trades

    def _calculate_bb_fast(self, prices: np.ndarray) -> tuple:
        """Fast BB calculation using numba"""
        sma = fast_rolling_mean(prices, self.period)
        rolling_std = fast_rolling_std(prices, self.period)

        upper_band = sma + (rolling_std * self.std_dev)
        lower_band = sma - (rolling_std * self.std_dev)

        return sma, upper_band, lower_band

    def _check_entry_signals(self, tick_time: pd.Timestamp, price: float, sma: float, upper: float, lower: float) -> Optional[Dict]:
        """Check for entry signals"""

        # Long signal: price touches or goes below lower band
        if price <= lower:
            return {
                'entry_time': tick_time,
                'entry_price': price,
                'side': 'long',
                'size': 1000.0,  # Fixed position size in dollars
                'stop_loss_price': price * (1 - self.stop_loss_pct),
                'target_price': sma  # Target SMA (mean reversion)
            }

        # Short signal: price touches or goes above upper band
        elif price >= upper:
            return {
                'entry_time': tick_time,
                'entry_price': price,
                'side': 'short',
                'size': 1000.0,
                'stop_loss_price': price * (1 + self.stop_loss_pct),
                'target_price': sma
            }

        return None

    def _check_exit_signals(self, tick_time: pd.Timestamp, price: float, sma: float, upper: float, lower: float) -> Optional[Dict]:
        """Check for exit signals"""
        if not self.position:
            return None

        entry_price = self.position['entry_price']
        side = self.position['side']
        size = self.position['size']
        stop_loss_price = self.position['stop_loss_price']

        exit_reason = None
        should_exit = False

        if side == 'long':
            # Long exit conditions
            if price >= sma:  # Target hit (mean reversion)
                exit_reason = 'target_hit'
                should_exit = True
            elif price <= stop_loss_price:  # Stop loss
                exit_reason = 'stop_loss'
                should_exit = True

        elif side == 'short':
            # Short exit conditions
            if price <= sma:  # Target hit (mean reversion)
                exit_reason = 'target_hit'
                should_exit = True
            elif price >= stop_loss_price:  # Stop loss
                exit_reason = 'stop_loss'
                should_exit = True

        if should_exit:
            # Calculate P&L
            if side == 'long':
                pnl = (price - entry_price) * (size / entry_price)  # Size is in dollars
                pnl_pct = (price - entry_price) / entry_price
            else:  # short
                pnl = (entry_price - price) * (size / entry_price)
                pnl_pct = (entry_price - price) / entry_price

            # Update capital
            self.current_capital += pnl

            # Duration
            duration_ms = (tick_time - self.position['entry_time']).total_seconds() * 1000

            return {
                'timestamp': self.position['entry_time'].value // 1000000,  # Convert to ms
                'exit_timestamp': tick_time.value // 1000000,
                'symbol': self.symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': price,
                'size': size,
                'pnl': pnl,
                'pnl_percentage': pnl_pct * 100,
                'duration': duration_ms,
                'exit_reason': exit_reason,
                'capital_before': self.current_capital - pnl,
                'capital_after': self.current_capital
            }

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'ticks_processed': self.tick_count,
            'total_trades': len(self.completed_trades),
            'current_capital': self.current_capital,
            'has_position': self.position is not None,
            'price_history_size': len(self.prices)
        }


if __name__ == "__main__":
    # Test the strategy
    strategy = PureTickBollingerStrategy("TESTUSDT")
    print("Pure Tick Bollinger Strategy initialized for HFT")