"""
Vectorized Klines Backtesting Engine
High-frequency trading focused, fully vectorized implementation for klines data
Processing klines efficiently with numpy/numba optimization

Author: HFT System
"""
import argparse
import os
import sys
import numpy as np
from datetime import datetime
from src.data.klines_handler import VectorizedKlinesHandler
from src.strategies import StrategyRegistry


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
            # Load from CSV
            handler = VectorizedKlinesHandler()
            klines_df = handler.load_klines(csv_path)
            
            # Extract numpy arrays from DataFrame
            klines_times = klines_df['time'].values
            klines_prices = klines_df['close'].values
            klines_opens = klines_df['open'].values if 'open' in klines_df.columns else None
            klines_highs = klines_df['high'].values if 'high' in klines_df.columns else None
            klines_lows = klines_df['low'].values if 'low' in klines_df.columns else None
            klines_closes = klines_df['close'].values
            klines_volumes = klines_df['Volume'].values if 'Volume' in klines_df.columns else None

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

        # Remove initial_capital and commission_pct from strategy_params to avoid conflicts
        filtered_params = {k: v for k, v in strategy_params.items()
                          if k not in ['initial_capital', 'commission_pct']}
        
        # For hierarchical_mean_reversion strategy, pass numpy arrays directly
        if strategy_name == 'hierarchical_mean_reversion':
            # Create strategy instance with parameters
            strategy = StrategyRegistry.create(
                name=strategy_name,
                symbol=symbol,
                initial_capital=initial_capital,
                commission_pct=commission_pct,
                **filtered_params
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
                **filtered_params
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


def main():
    """CLI entry point for vectorized klines backtesting"""
    parser = argparse.ArgumentParser(description='Vectorized Klines Strategy Backtester')
    parser.add_argument('--csv', required=False, help='CSV file path with klines data')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--strategy', default='hierarchical_mean_reversion', help='Strategy name (default: hierarchical_mean_reversion)')
    parser.add_argument('--max-klines', type=int, help='Limit klines for testing')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--list-strategies', action='store_true', help='List available strategies')

    # Parse known arguments first to get strategy name
    known_args, _ = parser.parse_known_args()
    
    # Add strategy-specific arguments
    if known_args.strategy and not known_args.list_strategies:
        try:
            strategy_class = StrategyRegistry.get(known_args.strategy)
            default_params = strategy_class.get_default_params() if strategy_class else {}

            # Add arguments for each parameter
            for param_name, param_value in default_params.items():
                # Convert parameter name to CLI argument format
                cli_name = f"--{param_name.replace('_', '-')}"
                
                # Determine type based on parameter value
                param_type = type(param_value)
                if param_type == bool:
                    parser.add_argument(cli_name, action='store_true',
                                      help=f'{param_name} (default: {param_value})')
                else:
                    parser.add_argument(cli_name, type=param_type, default=param_value,
                                      help=f'{param_name} (default: {param_value})')
        except Exception as e:
            print(f"Warning: Could not load strategy parameters: {e}")

    args = parser.parse_args()

    # List available strategies if requested
    if args.list_strategies:
        print("Available strategies:")
        for strategy_name in StrategyRegistry.list_strategies():
            strategy_class = StrategyRegistry.get(strategy_name)
            print(f"  - {strategy_name}: {strategy_class.__name__}")
        sys.exit(0)

    # Check if CSV file is provided when not listing strategies
    if not args.csv:
        print("Error: --csv is required when not using --list-strategies")
        sys.exit(1)

    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} not found")
        sys.exit(1)

    # Build strategy parameters dictionary
    strategy_params = {}
    if args.strategy:
        try:
            strategy_class = StrategyRegistry.get(args.strategy)
            default_params = strategy_class.get_default_params() if strategy_class else {}

            # Override defaults with CLI arguments
            for param_name in default_params.keys():
                cli_name = param_name.replace('-', '_')
                if hasattr(args, cli_name):
                    strategy_params[param_name] = getattr(args, cli_name)
        except Exception as e:
            print(f"Warning: Could not load strategy parameters: {e}")

    print("Vectorized Klines Backtest Configuration:")
    print(f"  File: {args.csv}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Strategy: {args.strategy}")
    for param_name, param_value in strategy_params.items():
        print(f"  {param_name}: {param_value}")
    if args.max_klines:
        print(f"  Max Klines: {args.max_klines:,}")

    results = run_vectorized_klines_backtest(
        csv_path=args.csv,
        symbol=args.symbol,
        strategy_name=args.strategy,
        strategy_params=strategy_params,
        max_klines=args.max_klines
    )

    if 'error' in results:
        print(f"Backtest failed: {results['error']}")
        sys.exit(1)

    # Display results
    print("\n" + "="*60)
    print("VECTORIZED KLINES BACKTEST RESULTS")
    print("="*60)
    print(f"Symbol: {results.get('symbol', 'N/A')}")
    print(f"Total Trades: {results.get('total', 0)}")
    print(f"Win Rate: {results.get('win_rate', 0):.1%}")
    print(f"Net P&L: ${results.get('net_pnl', 0):,.2f}")
    print(f"Return: {results.get('net_pnl_percentage', 0):.2f}%")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"Winners: {results.get('total_winning_trades', 0)}")
    print(f"Losers: {results.get('total_losing_trades', 0)}")
    print(f"Avg Win: ${results.get('average_win', 0):.2f}")
    print(f"Avg Loss: ${results.get('average_loss', 0):.2f}")
    print(f"Best Trade: ${results.get('largest_win', 0):.2f}")
    print(f"Worst Trade: ${results.get('largest_loss', 0):.2f}")
    print(f"Klines Processed: {results.get('klines_processed', 0):,}")
    print(f"Processing Time: {results.get('processing_time_seconds', 0):.2f}s")
    processing_time_seconds = results.get('processing_time_seconds', 1)
    if processing_time_seconds > 0:
        print(f"Klines per Second: {results.get('klines_processed', 0)/processing_time_seconds:,.0f}")
    else:
        print(f"Klines per Second: N/A (completed in less than 0.01s)")
    print("="*60)

    if args.output:
        # Save results
        with open(args.output, 'w') as f:
            f.write(f"Vectorized Klines Backtest Results - {args.symbol}\n")
            f.write("="*50 + "\n")
            f.write(f"Total Trades: {results.get('total', 0)}\n")
            f.write(f"Win Rate: {results.get('win_rate', 0):.1%}\n")
            f.write(f"Net P&L: ${results.get('net_pnl', 0):,.2f}\n")
            f.write(f"Return: {results.get('net_pnl_percentage', 0):.2f}%\n")
            f.write(f"Klines Processed: {results.get('klines_processed', 0):,}\n")
            f.write(f"Processing Time: {results.get('processing_time_seconds', 0):.2f}s\n")
        print(f"Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()