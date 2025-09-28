"""
Technical Indicators for HFT Strategies
Centralized module for all technical analysis calculations
Following HFT principles: maximum performance, minimal complexity

Author: HFT System
"""
import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(parallel=True)
def vectorized_bb_calculation(prices: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands for entire price array at once using parallel processing

    Args:
        prices: Array of prices (close prices from klines)
        period: BB period
        std_dev: Standard deviation multiplier

    Returns:
        Tuple of (sma, upper_band, lower_band)
    """
    n = len(prices)
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)

    # Calculate rolling means and stds for entire array using parallel processing
    for i in prange(period-1, n):
        sma[i] = np.mean(prices[i-period+1:i+1])
        std[i] = np.std(prices[i-period+1:i+1])

    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return sma, upper_band, lower_band


@njit(parallel=True)
def vectorized_signal_generation(prices: np.ndarray, sma: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray, stop_loss_pct: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate all signals for the entire dataset at once using vectorized operations

    Args:
        prices: Array of close prices
        sma: Simple moving average array
        upper_band: Upper Bollinger Band array
        lower_band: Lower Bollinger Band array
        stop_loss_pct: Stop loss percentage as decimal

    Returns:
        Tuple of (entry_signals, exit_signals, position_status)
    """
    n = len(prices)
    entry_signals = np.zeros(n, dtype=np.int8)  # 0: no signal, 1: long, -1: short
    exit_signals = np.zeros(n, dtype=np.int8)   # 0: no signal, 1: exit long, -1: exit short
    position_status = np.zeros(n, dtype=np.int8)  # 0: no position, 1: long, -1: short

    # Calculate price crosses for entry signals
    lower_cross = np.zeros(n-1, dtype=np.bool_)
    upper_cross = np.zeros(n-1, dtype=np.bool_)

    for i in prange(n-1):
        lower_cross[i] = (prices[i+1] <= lower_band[i+1]) & (prices[i] > lower_band[i])
        upper_cross[i] = (prices[i+1] >= upper_band[i+1]) & (prices[i] < upper_band[i])

    # First pass: Calculate entry signals and initial position status
    current_pos = 0
    for i in range(1, n):
        if current_pos == 0:  # No current position
            if lower_cross[i-1]:  # Price crossed below lower band
                entry_signals[i] = 1 # Long signal
                current_pos = 1
            elif upper_cross[i-1]:  # Price crossed above upper band
                entry_signals[i] = -1  # Short signal
                current_pos = -1
        else:
            # Check for exit conditions with DISTINCT exit signal types
            if current_pos == 1:  # Long position
                stop_loss = prices[i-1] * (1 - stop_loss_pct)
                if prices[i] <= stop_loss:
                    exit_signals[i] = 1  # STOP LOSS long
                    current_pos = 0
                elif prices[i] >= sma[i]:
                    exit_signals[i] = 2  # TAKE PROFIT long (SMA reversal)
                    current_pos = 0
            elif current_pos == -1:  # Short position
                stop_loss = prices[i-1] * (1 + stop_loss_pct)
                if prices[i] >= stop_loss:
                    exit_signals[i] = -1  # STOP LOSS short
                    current_pos = 0
                elif prices[i] <= sma[i]:
                    exit_signals[i] = -2  # TAKE PROFIT short (SMA reversal)
                    current_pos = 0

        position_status[i] = current_pos

    return entry_signals, exit_signals, position_status


if __name__ == "__main__":
    # Test the indicators
    print("Technical Indicators module loaded successfully")
    print("Available functions: vectorized_bb_calculation, vectorized_signal_generation")