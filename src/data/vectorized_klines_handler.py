"""
Vectorized Klines Data Handler - High Frequency Trading focused
Full vectorization for processing klines efficiently

Author: HFT System
"""
import numpy as np
import pandas as pd
from numba import njit, prange
import os
from typing import Tuple, Optional


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
            # Check for exit conditions
            if current_pos == 1:  # Long position
                stop_loss = prices[i-1] * (1 - stop_loss_pct)
                if prices[i] <= stop_loss or prices[i] >= sma[i]:
                    exit_signals[i] = 1  # Exit long
                    current_pos = 0
            elif current_pos == -1:  # Short position
                stop_loss = prices[i-1] * (1 + stop_loss_pct)
                if prices[i] >= stop_loss or prices[i] <= sma[i]:
                    exit_signals[i] = -1 # Exit short
                    current_pos = 0

        position_status[i] = current_pos

    return entry_signals, exit_signals, position_status


class VectorizedKlinesHandler:
    """
    Vectorized klines data handler for HFT strategies
    Processes entire datasets at once instead of individual candles
    """

    def __init__(self):
        """Initialize handler"""
        self.required_columns = ['Symbol', 'time', 'open', 'high', 'low', 'close', 'Volume']

    def load_klines(self, csv_path: str) -> pd.DataFrame:
        """
        Load klines data from CSV without any processing

        Args:
            csv_path: Path to CSV file with klines data

        Returns:
            DataFrame with raw klines data sorted by time
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Klines data file not found: {csv_path}")

        # Load raw CSV
        df = pd.read_csv(csv_path)

        # Validate columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Basic data validation
        if df.empty:
            raise ValueError("Empty klines data file")

        if (df['close'] <= 0).any():
            raise ValueError("Invalid close price values found")

        if (df['Volume'] < 0).any():
            raise ValueError("Invalid volume values found")

        # Sort by time and reset index
        df = df.sort_values('time').reset_index(drop=True)

        # Add derived fields for HFT
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])

        print(f"Loaded {len(df):,} klines")
        return df

    def prepare_price_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare numpy price array for fast calculations (using close prices)

        Args:
            df: Klines DataFrame

        Returns:
            Numpy array of close prices
        """
        return df['close'].values.astype(np.float64)

    def vectorized_backtest(self, df: pd.DataFrame, bb_period: int = 50, bb_std: float = 2.0, stop_loss_pct: float = 0.005) -> pd.DataFrame:
        """
        Perform complete backtest on entire dataset using vectorized operations

        Args:
            df: DataFrame with klines data
            bb_period: Bollinger Bands period
            bb_std: BB standard deviation multiplier
            stop_loss_pct: Stop loss percentage as decimal

        Returns:
            DataFrame with backtest results including signals and positions
        """
        # Extract numpy arrays for vectorized operations
        prices = df['close'].values.astype(np.float64)
        times = df['time'].values
        volumes = df['Volume'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values

        # Calculate all BB values at once using vectorized function
        sma, upper_band, lower_band = vectorized_bb_calculation(prices, bb_period, bb_std)

        # Generate all signals at once using vectorized function
        entry_signals, exit_signals, position_status = vectorized_signal_generation(
            prices, sma, upper_band, lower_band, stop_loss_pct
        )

        # Create results dataframe with all computed values
        results_df = pd.DataFrame({
            'time': times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'entry_signal': entry_signals,
            'exit_signal': exit_signals,
            'position_status': position_status
        })

        # Add datetime column
        results_df['datetime'] = pd.to_datetime(results_df['time'], unit='s')

        return results_df

    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Get basic klines data statistics"""
        return {
            'total_klines': len(df),
            'price_range': (df['close'].min(), df['close'].max()),
            'time_range': (df['time'].min(), df['time'].max()),
            'volume_total': df['Volume'].sum(),
            'high_max': df['high'].max(),
            'low_min': df['low'].min(),
            'avg_body_size': df['body_size'].mean(),
            'avg_price_range': df['price_range'].mean()
        }


if __name__ == "__main__":
    # Test the handler
    handler = VectorizedKlinesHandler()
    print("Vectorized Klines Handler initialized for HFT processing with full vectorization")