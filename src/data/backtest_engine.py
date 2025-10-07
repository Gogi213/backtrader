"""
Vectorized Klines Backtesting Engine
High-frequency trading focused, fully vectorized implementation for klines data
Processing klines efficiently with numpy/numba optimization

Author: HFT System
"""
import argparse
import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.data.klines_handler import UltraFastKlinesHandler, VectorizedKlinesHandler
from src.strategies import StrategyRegistry

# Import unified backtesting components
try:
    from src.core import BacktestManager, BacktestConfig, BacktestResults
    UNIFIED_BACKTEST_AVAILABLE = True
except ImportError:
    UNIFIED_BACKTEST_AVAILABLE = False


def run_vectorized_klines_backtest(csv_path: str = None,
                                  times: np.ndarray = None,
                                  prices: np.ndarray = None,
                                  opens: np.ndarray = None,
                                  highs: np.ndarray = None,
                                  lows: np.ndarray = None,
                                  closes: np.ndarray = None,
                                  volumes: np.ndarray = None,
                                  symbol: str = 'BTCUSDT',
                                  strategy_name: str = 'hierarchical_mean_reversion',
                                  strategy_params: dict = None,
                                  initial_capital: float = 10000.0,
                                  commission_pct: float = 0.05,
                                  max_klines: int = None) -> dict:
    """
    Run fully vectorized backtest on klines data using Strategy Factory

    Args:
        csv_path: Path to CSV file with klines data (optional if numpy arrays provided)
        times: Time values array (optional if csv_path provided)
        prices: Price values array (optional if csv_path provided)
        opens: Open values array (optional if csv_path provided)
        highs: High values array (optional if csv_path provided)
        lows: Low values array (optional if csv_path provided)
        closes: Close values array (optional if csv_path provided)
        volumes: Volume values array (optional if csv_path provided)
        symbol: Trading symbol
        strategy_name: Name of strategy to use (default: 'hierarchical_mean_reversion')
        strategy_params: Dictionary with strategy-specific parameters
        initial_capital: Initial capital for backtest (default: 10000.0)
        commission_pct: Commission percentage (default: 0.05)
        max_klines: Maximum klines to process (for testing)

    Returns:
        Dictionary with backtest results
    """
    # Get default parameters if none provided
    if strategy_params is None:
        strategy_class = StrategyRegistry.get(strategy_name)
        strategy_params = strategy_class.get_default_params() if strategy_class else {}
    
    print(f"VECTORIZED KLINES BACKTEST: {symbol}")
    print(f"Strategy: {strategy_name}")
    print(f"Parameters: {strategy_params}")

    try:
        # Load klines data (from CSV or use provided numpy arrays)
        if times is not None and prices is not None:
            print(f"Using provided numpy arrays with {len(times):,} rows")
            # Use provided numpy arrays directly
            klines_times = times
            klines_prices = prices
            klines_opens = opens
            klines_highs = highs
            klines_lows = lows
            klines_closes = closes
            klines_volumes = volumes
        else:
            # Load from CSV using ULTRA-FAST handler
            handler = UltraFastKlinesHandler()
            klines_data = handler.load_klines(csv_path)
            
            # Extract numpy arrays from NumpyKlinesData
            klines_times = klines_data['time']
            klines_prices = klines_data['close']
            klines_opens = klines_data['open']
            klines_highs = klines_data['high']
            klines_lows = klines_data['low']
            klines_closes = klines_data['close']
            klines_volumes = klines_data['volume']

        # Limit klines for testing if requested
        if max_klines and len(klines_times) > max_klines:
            klines_times = klines_times[:max_klines]
            klines_prices = klines_prices[:max_klines]
            if klines_opens is not None:
                klines_opens = klines_opens[:max_klines]
            if klines_highs is not None:
                klines_highs = klines_highs[:max_klines]
            if klines_lows is not None:
                klines_lows = klines_lows[:max_klines]
            if klines_closes is not None:
                klines_closes = klines_closes[:max_klines]
            if klines_volumes is not None:
                klines_volumes = klines_volumes[:max_klines]
            print(f"Limited to {max_klines:,} klines for testing")

        print(f"Processing {len(klines_times):,} klines using vectorized operations...")
        start_time = datetime.now()

        # For hierarchical_mean_reversion strategy, pass numpy arrays directly
        if strategy_name == 'hierarchical_mean_reversion':
            # Create strategy instance with parameters
            strategy = StrategyRegistry.create(
                name=strategy_name,
                symbol=symbol,
                initial_capital=initial_capital,
                commission_pct=commission_pct,
                **strategy_params
            )
            
            # Run backtest with numpy arrays directly
            results = strategy.turbo_process_dataset(
                times=klines_times,
                prices=klines_prices,
                opens=klines_opens,
                highs=klines_highs,
                lows=klines_lows,
                closes=klines_closes
            )
            
            # Add metadata
            results['symbol'] = symbol
        else:
            # For non-turbo strategies, we need to create a simple DataFrame-like structure
            # but we'll minimize pandas usage
            
            # Create a simple dictionary structure that mimics a DataFrame
            # This avoids creating an actual pandas DataFrame
            klines_data = {
                'time': klines_times,
                'price': klines_closes,  # Add 'price' for strategies that expect it
                'close': klines_closes,
                'open': klines_opens,
                'high': klines_highs,
                'low': klines_lows,
                'volume': klines_volumes
            }
            
            # Create a simple wrapper class that mimics DataFrame behavior
            class SimpleDataFrame:
                def __init__(self, data):
                    self.data = data
                    self.columns = list(data.keys())
                
                def __getitem__(self, key):
                    return SimpleSeries(self.data[key])
                
                def __len__(self):
                    return len(self.data['time'])
                
                def head(self, n):
                    new_data = {}
                    for key, value in self.data.items():
                        if value is not None:
                            new_data[key] = value[:n]
                        else:
                            new_data[key] = None
                    return SimpleDataFrame(new_data)
            
            class SimpleSeries:
                def __init__(self, data):
                    self.data = data
                
                @property
                def values(self):
                    return self.data
            
            # Create simple DataFrame wrapper
            klines_simple_df = SimpleDataFrame(klines_data)
            
            # Initialize strategy using Registry with dynamic parameters
            strategy = StrategyRegistry.create(
                name=strategy_name,
                symbol=symbol,
                initial_capital=initial_capital,
                commission_pct=commission_pct,
                **strategy_params
            )
    
            # Perform complete backtest using vectorized operations
            # Note: This may still require some modifications to strategy classes
            # to work with our simple DataFrame wrapper
            results = strategy.vectorized_process_dataset(klines_simple_df)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        print(f"Vectorized backtest completed in {processing_time:.2f}s")
        if processing_time > 0:
            print(f"Processing rate: {len(klines_times)/processing_time:,.0f} klines/sec")
        else:
            print("Processing rate: N/A (completed in less than 0.01s)")
        print(f"Total trades generated: {results.get('total', 0)}")

        # Add klines statistics
        results.update({
            'total_klines': len(klines_times),
            'klines_processed': len(klines_times),
            'processing_time_seconds': processing_time,
            'data_type': 'klines'
        })

        if len(klines_times) > 0:
            results['started_at'] = klines_times[0]
            results['finished_at'] = klines_times[-1]

        # CRITICAL FIX: Ensure ALL timestamps are in milliseconds for chart compatibility
        if 'bb_data' in results:
            bb_data = results['bb_data']
            if 'times' in bb_data:
                # Convert from seconds to milliseconds (Unix timestamp * 1000)
                bb_data['times'] = bb_data['times'] * 1000
                print(f"DEBUG: Converted bb_data times to milliseconds. First few: {bb_data['times'][:3]}")

        if 'indicator_data' in results:
            indicator_data = results['indicator_data']
            if 'times' in indicator_data:
                indicator_data['times'] = indicator_data['times'] * 1000
                print(f"DEBUG: Converted indicator_data times to milliseconds")

        # CRITICAL FIX: Convert trade timestamps to milliseconds
        if 'trades' in results:
            for trade in results['trades']:
                if 'timestamp' in trade:
                    trade['timestamp'] = trade['timestamp'] * 1000
                if 'exit_timestamp' in trade:
                    trade['exit_timestamp'] = trade['exit_timestamp'] * 1000
            print(f"DEBUG: Converted {len(results['trades'])} trade timestamps to milliseconds")

        return results

    except Exception as e:
        print(f"Klines backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def run_unified_backtest(csv_path: str = None,
                        symbol: str = 'BTCUSDT',
                        strategy_name: str = 'hierarchical_mean_reversion',
                        strategy_params: dict = None,
                        initial_capital: float = 10000.0,
                        commission_pct: float = 0.05,
                        max_klines: int = None,
                        output_file: str = None,
                        verbose: bool = False) -> dict:
    """
    Run backtest using unified BacktestManager
    
    This function provides a compatibility layer between the legacy
    interface and the new unified backtesting architecture.
    
    Args:
        csv_path: Path to CSV file with klines data
        symbol: Trading symbol
        strategy_name: Name of strategy to use
        strategy_params: Dictionary with strategy-specific parameters
        initial_capital: Initial capital for backtest
        commission_pct: Commission percentage
        max_klines: Maximum klines to process
        output_file: Output file for results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with backtest results
    """
    if not UNIFIED_BACKTEST_AVAILABLE:
        print("Warning: Unified backtest not available, falling back to legacy implementation")
        return run_vectorized_klines_backtest(
            csv_path=csv_path,
            symbol=symbol,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            max_klines=max_klines
        )
    
    # Create configuration
    config = BacktestConfig(
        strategy_name=strategy_name,
        strategy_params=strategy_params or {},
        symbol=symbol,
        data_path=csv_path,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        max_klines=max_klines,
        output_file=output_file,
        verbose=verbose
    )
    
    # Create manager and run backtest
    manager = BacktestManager()
    results = manager.run_backtest(config)
    
    # Convert to dictionary for compatibility
    return results.to_dict()


def run_batch_backtest(batch_config_path: str,
                      parallel: bool = True,
                      max_workers: Optional[int] = None,
                      verbose: bool = False) -> Dict[str, Any]:
    """
    Run batch backtests from configuration file
    
    Args:
        batch_config_path: Path to JSON configuration file with batch backtest settings
        parallel: Whether to run backtests in parallel
        max_workers: Maximum number of parallel workers
        verbose: Enable verbose output
        
    Returns:
        Dictionary with batch results
    """
    if not UNIFIED_BACKTEST_AVAILABLE:
        return {'error': 'Unified backtest not available for batch processing'}
    
    try:
        # Load batch configuration
        with open(batch_config_path, 'r') as f:
            batch_config = json.load(f)
        
        # Extract configurations
        configs = []
        for config_data in batch_config.get('backtests', []):
            config = BacktestConfig.from_dict(config_data)
            configs.append(config)
        
        if not configs:
            return {'error': 'No backtest configurations found in batch file'}
        
        # Create manager and run batch
        manager = BacktestManager()
        results_list = manager.run_batch_backtest(configs, parallel, max_workers)
        
        # Convert results to dictionaries
        results_dicts = [result.to_dict() for result in results_list]
        
        # Get execution statistics
        stats = manager.get_execution_stats()
        
        return {
            'results': results_dicts,
            'stats': stats,
            'total_backtests': len(configs),
            'successful_backtests': sum(1 for r in results_dicts if not r.get('error')),
            'failed_backtests': sum(1 for r in results_dicts if r.get('error'))
        }
        
    except Exception as e:
        return {'error': f'Batch backtest failed: {str(e)}'}


# Note: CLI functionality has been removed from this module.
# Use the GUI interface for backtesting operations.