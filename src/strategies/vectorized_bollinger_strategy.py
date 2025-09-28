"""
Vectorized Bollinger Bands Mean Reversion Strategy
High-frequency trading focused, fully vectorized implementation
Processing 4+ million ticks efficiently with numpy/numba optimization

Author: HFT System
"""
import numpy as np
import pandas as pd
from numba import njit, prange
from datetime import datetime
from typing import List, Dict, Optional, Any
from ..data.technical_indicators import vectorized_bb_calculation, vectorized_signal_generation
import warnings


class VectorizedBollingerStrategy:
    """
    High-frequency Bollinger Bands strategy using fully vectorized operations
    Processes entire datasets at once instead of individual ticks
    """
    
    def __init__(self,
                 symbol: str,
                 period: int = 50,  # Smaller period for HFT
                 std_dev: float = 2.0,  # Tighter bands for HFT
                 stop_loss_pct: float = 0.005,  # 0.5% for HFT
                 initial_capital: float = 10000.0,
                 commission_pct: float = 0.0005):  # 0.05% commission
        """
        Initialize vectorized strategy
        
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
        self.commission_pct = commission_pct

        # Performance tracking
        self.current_capital = initial_capital
        self.completed_trades = []
        
    def vectorized_process_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process entire dataset using vectorized operations
        
        Args:
            df: DataFrame with tick data containing columns: time, price, qty
            
        Returns:
            Dictionary with backtest results including trades and performance metrics
        """
        # Validate input
        required_cols = ['time', 'price', 'qty']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract data arrays
        times = df['time'].values
        prices = df['price'].values.astype(np.float64)
        qtys = df['qty'].values

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
        return_pct = (total_pnl / self.initial_capital * 10) if self.initial_capital > 0 else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Simplified Sharpe ratio (using daily returns approximation)
        trade_returns = [t.get('pnl', 0) / self.initial_capital for t in trades]
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
        equity_curve = [self.initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.get('pnl', 0))

        peak = self.initial_capital
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
            'loose_streak': max_loose_streak  # Максимальное количество стоп-лоссов подряд
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'total_trades': len(self.completed_trades),
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'capital_growth': (self.current_capital - self.initial_capital) / self.initial_capital * 100
        }


if __name__ == "__main__":
    # Test the strategy
    strategy = VectorizedBollingerStrategy("TESTUSDT")
    print("Vectorized Bollinger Strategy initialized for HFT with full vectorization")
