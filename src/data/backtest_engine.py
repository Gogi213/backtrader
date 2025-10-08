"""
Vectorized Backtest Engine - Unified Interface for Strategy Backtesting
Optimized for HFT systems with numpy-based processing

Author: HFT System
"""
import numpy as np
from typing import Dict, Any, Optional
from .klines_handler import UltraFastKlinesHandler


def run_vectorized_klines_backtest(strategy, data_path: str, 
                                 sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Unified interface to run vectorized backtest for any strategy
    
    Args:
        strategy: Strategy instance with vectorized_process_dataset method
        data_path: Path to klines data file
        sample_size: Optional sample size for testing
        
    Returns:
        Dictionary with backtest results
    """
    # Load data
    handler = UltraFastKlinesHandler()
    klines_data = handler.load_klines(data_path)
    
    # Convert to dictionary format
    if hasattr(klines_data, 'data'):
        data_dict = {
            'time': klines_data.data['time'],
            'open': klines_data.data['open'],
            'high': klines_data.data['high'],
            'low': klines_data.data['low'],
            'close': klines_data.data['close'],
            'volume': klines_data.data['volume']
        }
    else:
        data_dict = {
            'time': klines_data['time'].values,
            'open': klines_data['open'].values,
            'high': klines_data['high'].values,
            'low': klines_data['low'].values,
            'close': klines_data['close'].values,
            'volume': klines_data['Volume'].values
        }
    
    # Apply sample size if specified
    if sample_size is not None and sample_size < len(data_dict['time']):
        for key in data_dict:
            data_dict[key] = data_dict[key][:sample_size]
    
    # Run strategy backtest
    try:
        results = strategy.vectorized_process_dataset(data_dict)
        return results
    except Exception as e:
        return {
            'error': str(e),
            'total': 0,
            'profit': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }