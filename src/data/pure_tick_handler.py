"""
Pure Tick Data Handler - High Frequency Trading focused
No candle conversion, only raw tick processing with numpy/numba optimization

Author: HFT System
"""
import numpy as np
import pandas as pd
from numba import jit, njit
import os
from typing import Tuple, Optional


@njit
def fast_rolling_mean(prices: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean calculation using numba"""
    n = len(prices)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        result[i] = np.mean(prices[i - window + 1:i + 1])

    return result


@njit
def fast_rolling_std(prices: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation using numba"""
    n = len(prices)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = prices[i - window + 1:i + 1]
        mean_val = np.mean(window_data)
        variance = np.mean((window_data - mean_val) ** 2)
        result[i] = np.sqrt(max(variance, 0.0))  # Prevent sqrt of negative

    return result


class PureTickHandler:
    """
    Pure tick data handler for HFT strategies
    No aggregation, no candles - only raw tick processing
    """

    def __init__(self):
        """Initialize handler"""
        self.required_columns = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']

    def load_ticks(self, csv_path: str) -> pd.DataFrame:
        """
        Load pure tick data from CSV without any processing

        Args:
            csv_path: Path to CSV file with tick data

        Returns:
            DataFrame with raw tick data sorted by time
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Tick data file not found: {csv_path}")

        # Load raw CSV
        df = pd.read_csv(csv_path)

        # Validate columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Basic data validation
        if df.empty:
            raise ValueError("Empty tick data file")

        if (df['price'] <= 0).any():
            raise ValueError("Invalid price values found")

        if (df['qty'] <= 0).any():
            raise ValueError("Invalid quantity values found")

        # Sort by time and reset index
        df = df.sort_values('time').reset_index(drop=True)

        # Add derived fields for HFT
        df['side'] = df['is_buyer_maker'].apply(lambda x: 'sell' if x else 'buy')
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')

        print(f"Loaded {len(df):,} pure ticks")
        return df

    def prepare_price_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare numpy price array for fast calculations

        Args:
            df: Tick DataFrame

        Returns:
            Numpy array of prices
        """
        return df['price'].values.astype(np.float64)

    def calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast Bollinger Bands calculation using numba

        Args:
            prices: Array of prices
            period: BB period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (sma, upper_band, lower_band)
        """
        if len(prices) < period:
            # Not enough data
            n = len(prices)
            return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

        # Calculate using numba-optimized functions
        sma = fast_rolling_mean(prices, period)
        rolling_std = fast_rolling_std(prices, period)

        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)

        return sma, upper_band, lower_band

    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Get basic tick data statistics"""
        return {
            'total_ticks': len(df),
            'price_range': (df['price'].min(), df['price'].max()),
            'time_range': (df['time'].min(), df['time'].max()),
            'volume_total': df['qty'].sum(),
            'quote_volume_total': df['quote_qty'].sum(),
            'buy_ticks': len(df[df['side'] == 'buy']),
            'sell_ticks': len(df[df['side'] == 'sell'])
        }


if __name__ == "__main__":
    # Test the handler
    handler = PureTickHandler()
    print("Pure Tick Handler initialized for HFT processing")