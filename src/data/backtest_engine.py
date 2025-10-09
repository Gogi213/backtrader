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
    
    # Apply sample size if specified
    if sample_size is not None and sample_size < len(klines_data):
        processed_data = klines_data.head(sample_size)
    else:
        processed_data = klines_data
    
    # Run strategy backtest
    try:
        results = strategy.vectorized_process_dataset(processed_data)
        return results
    except Exception as e:
        return {
            'error': str(e),
            'total': 0,
            'profit': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }