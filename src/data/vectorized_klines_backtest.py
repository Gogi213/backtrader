"""
Vectorized Klines Backtesting Engine
High-frequency trading focused, fully vectorized implementation for klines data
Processing klines efficiently with numpy/numba optimization

Author: HFT System
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.vectorized_klines_handler import VectorizedKlinesHandler
from src.strategies.strategy_factory import StrategyFactory


def run_vectorized_klines_backtest(csv_path: str,
                                  symbol: str = 'BTCUSDT',
                                  strategy_name: str = 'bollinger',
                                  strategy_params: dict = None,
                                  initial_capital: float = 10000.0,
                                  commission_pct: float = 0.05,
                                  max_klines: int = None) -> dict:
    """
    Run fully vectorized backtest on klines data using Strategy Factory

    Args:
        csv_path: Path to CSV file with klines data
        symbol: Trading symbol
        strategy_name: Name of strategy to use (default: 'bollinger')
        strategy_params: Dictionary with strategy-specific parameters
        initial_capital: Initial capital for backtest (default: 10000.0)
        commission_pct: Commission percentage (default: 0.05)
        max_klines: Maximum klines to process (for testing)

    Returns:
        Dictionary with backtest results
    """
    # Get default parameters if none provided
    if strategy_params is None:
        strategy_params = StrategyFactory.get_strategy_info(strategy_name)['default_params']
    
    print(f"VECTORIZED KLINES BACKTEST: {symbol}")
    print(f"Strategy: {strategy_name}")
    print(f"Parameters: {strategy_params}")

    try:
        # Load klines data
        handler = VectorizedKlinesHandler()
        klines_df = handler.load_klines(csv_path)

        # Limit klines for testing if requested
        if max_klines and len(klines_df) > max_klines:
            klines_df = klines_df.head(max_klines)
            print(f"Limited to {max_klines:,} klines for testing")

        print(f"Processing {len(klines_df):,} klines using vectorized operations...")
        start_time = datetime.now()

        # Prepare for strategy processing - convert klines to tick-like format
        # Include full OHLC data for candlestick charts
        tick_like_df = pd.DataFrame({
            'price': klines_df['close'].values,  # Primary price (close)
            'open': klines_df['open'].values,    # OHLC data for candlesticks
            'high': klines_df['high'].values,
            'low': klines_df['low'].values,
            'close': klines_df['close'].values,
            'time': klines_df['time'].values,
            'qty': klines_df['Volume'].values,  # Use volume as quantity
            'quote_qty': klines_df['Volume'].values * klines_df['close'].values,
            'is_buyer_maker': np.random.choice([True, False], size=len(klines_df)),  # Placeholder
            'id': range(len(klines_df))
        })

        # Initialize strategy using Factory with dynamic parameters
        # Remove initial_capital and commission_pct from strategy_params to avoid conflicts
        filtered_params = {k: v for k, v in strategy_params.items()
                          if k not in ['initial_capital', 'commission_pct']}
        
        strategy = StrategyFactory.create(
            strategy_name=strategy_name,
            symbol=symbol,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            **filtered_params
        )

        # Perform complete backtest using vectorized operations
        results = strategy.vectorized_process_dataset(tick_like_df)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        print(f"Vectorized backtest completed in {processing_time:.2f}s")
        if processing_time > 0:
            print(f"Processing rate: {len(klines_df)/processing_time:,.0f} klines/sec")
        else:
            print("Processing rate: N/A (completed in less than 0.01s)")
        print(f"Total trades generated: {results.get('total', 0)}")

        # Add klines statistics
        stats = handler.get_statistics(klines_df)
        results.update({
            'total_klines': stats['total_klines'],
            'klines_processed': len(klines_df),
            'processing_time_seconds': processing_time,
            'data_type': 'klines'
        })

        if len(klines_df) > 0:
            results['started_at'] = klines_df.iloc[0]['time']
            results['finished_at'] = klines_df.iloc[-1]['time']

        # CRITICAL FIX: Ensure ALL timestamps are in milliseconds for chart compatibility
        if 'bb_data' in results:
            bb_data = results['bb_data']
            if 'times' in bb_data:
                # Convert from seconds to milliseconds (Unix timestamp * 1000)
                bb_data['times'] = bb_data['times'] * 1000
                print(f"DEBUG: Converted bb_data times to milliseconds. First few: {bb_data['times'][:3]}")

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
    parser.add_argument('--strategy', default='bollinger', help='Strategy name (default: bollinger)')
    parser.add_argument('--max-klines', type=int, help='Limit klines for testing')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--list-strategies', action='store_true', help='List available strategies')

    # Parse known arguments first to get strategy name
    known_args, _ = parser.parse_known_args()
    
    # Add strategy-specific arguments
    if known_args.strategy and not known_args.list_strategies:
        try:
            strategy_info = StrategyFactory.get_strategy_info(known_args.strategy)
            default_params = strategy_info['default_params']
            
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
        for strategy_name in StrategyFactory.list_available_strategies():
            info = StrategyFactory.get_strategy_info(strategy_name)
            print(f"  - {strategy_name}: {info['class'].__name__}")
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
            strategy_info = StrategyFactory.get_strategy_info(args.strategy)
            default_params = strategy_info['default_params']
            
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