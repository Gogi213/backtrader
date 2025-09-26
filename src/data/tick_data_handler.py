"""
Data handling module for converting tick data to Jesse-compatible format

This module provides utilities for processing specific tick data format 
and converting it to the OHLCV format required by Jesse.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime


class TickDataHandler:
    """
    Handles the processing of tick data and conversion to Jesse-compatible format
    """
    def __init__(self, timeframe='1m'):
        """
        Initialize the handler with a specific timeframe
        :param timeframe: Desired timeframe for candles ('1m', '5m', '15m', etc.)
        """
        self.timeframe = timeframe
        self.timeframe_multiplier = self._get_timeframe_multiplier(timeframe)
    
    def _get_timeframe_multiplier(self, timeframe):
        """
        Get the multiplier for the timeframe in minutes
        """
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 24 * 60
        else:
            return 1  # Default to 1 minute
    
    def process_csv_to_jesse_format(self, csv_path):
        """
        Process a CSV file containing tick data in the specific format:
        id,price,qty,quote_qty,time,is_buyer_maker
        
        :param csv_path: Path to the CSV file
        :return: DataFrame in Jesse-compatible format
        """
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Validate required columns exist
        required_columns = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file")
        
        # Convert time from milliseconds to datetime
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        
        # Group by time period to create candles
        df.set_index('datetime', inplace=True)
        
        # Resample to the desired timeframe and aggregate
        candle_df = df.resample(f'{self.timeframe_multiplier}T').agg({
            'price': ['first', 'max', 'min', 'last'],  # open, high, low, close
            'qty': 'sum',  # volume (quantity)
            'quote_qty': 'sum'  # quote volume
        }).round(8)
        
        # Flatten column names
        candle_df.columns = ['_'.join(col).strip() for col in candle_df.columns]
        
        # Rename columns to standard format
        candle_df = candle_df.rename(columns={
            'price_first': 'open',
            'price_max': 'high', 
            'price_min': 'low',
            'price_last': 'close',
            'qty_sum': 'volume'
        })
        
        # Reset index to have datetime as a column
        candle_df.reset_index(inplace=True)
        
        # Create timestamp in milliseconds (Jesse format)
        candle_df['timestamp'] = (candle_df['datetime'].astype(np.int64) // 1000000).astype(int)
        
        # Select only the required columns for Jesse
        result = candle_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Ensure the DataFrame is sorted by timestamp
        result = result.sort_values('timestamp').reset_index(drop=True)
        
        return result

    def load_raw_ticks(self, csv_path):
        """
        Load raw tick data without any aggregation/candle conversion
        For HFT strategies that need to work with individual ticks

        :param csv_path: Path to the CSV file
        :return: DataFrame with raw tick data, properly formatted
        """
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Validate required columns exist
        required_columns = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file")

        # Convert time from milliseconds to datetime for easier handling
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')

        # Sort by timestamp to ensure chronological order
        df = df.sort_values('time').reset_index(drop=True)

        # Add side information (buy/sell from market perspective)
        df['side'] = df['is_buyer_maker'].apply(lambda x: 'sell' if x else 'buy')

        # Keep original timestamp format for consistency
        df['timestamp'] = df['time']

        return df

    def validate_data(self, df):
        """
        Validate the loaded tick data
        :param df: DataFrame with tick data
        :return: True if valid, False otherwise
        """
        required_columns = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
        for col in required_columns:
            if col not in df.columns:
                return False
        
        # Check if prices are positive
        if (df['price'] <= 0).any():
            return False
        
        # Check if quantities are positive
        if (df['qty'] <= 0).any():
            return False
        
        return True


def save_jesse_candles(candles_df, output_path):
    """
    Save processed candles to a CSV file in Jesse format
    :param candles_df: DataFrame with Jesse-compatible candle data
    :param output_path: Path to save the file
    """
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in candles_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain required columns: {required_cols}")
    
    candles_df.to_csv(output_path, index=False)
    print(f"Saved Jesse-compatible candles to {output_path}")


if __name__ == "__main__":
    # Example usage
    handler = TickDataHandler(timeframe='1m')
    # For demonstration purposes only - this would require an actual CSV file
    # df = handler.process_csv_to_jesse_format('path/to/your/tick_data.csv')