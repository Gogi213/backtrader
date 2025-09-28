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
from src.strategies.vectorized_bollinger_strategy import VectorizedBollingerStrategy


def run_vectorized_klines_backtest(csv_path: str,
                                  symbol: str = 'BTCUSDT',
                                  bb_period: int = 50,
                                  bb_std: float = 2.0,
                                  stop_loss_pct: float = 0.5,
                                  initial_capital: float = 10000.0,
                                  max_klines: int = None) -> dict:
    """
    Run fully vectorized backtest on klines data

    Args:
        csv_path: Path to CSV file with klines data
        symbol: Trading symbol
        bb_period: Bollinger Bands period
        bb_std: BB standard deviation multiplier
        stop_loss_pct: Stop loss percentage
        initial_capital: Initial capital
        max_klines: Maximum klines to process (for testing)

    Returns:
        Dictionary with backtest results
    """
    print(f"VECTORIZED KLINES BACKTEST: {symbol}")
    print(f"Parameters: BB({bb_period}, {bb_std:.1f}), SL: {stop_loss_pct}%")

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
        # Use close prices as our primary data points
        tick_like_df = pd.DataFrame({
            'price': klines_df['close'].values,
            'time': klines_df['time'].values,
            'qty': klines_df['Volume'].values,  # Use volume as quantity
            'quote_qty': klines_df['Volume'].values * klines_df['close'].values,
            'is_buyer_maker': np.random.choice([True, False], size=len(klines_df)),  # Placeholder
            'id': range(len(klines_df))
        })

        # Initialize strategy
        strategy = VectorizedBollingerStrategy(
            symbol=symbol,
            period=bb_period,
            std_dev=bb_std,
            stop_loss_pct=stop_loss_pct / 100,  # Convert to decimal
            initial_capital=initial_capital
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

        # Update bb_data format for chart compatibility
        if 'bb_data' in results:
            bb_data = results['bb_data']
            # Convert EXISTING timestamps from seconds to milliseconds for chart compatibility
            # DO NOT overwrite bb_data - it contains filtered BB values that match each other
            if 'times' in bb_data:
                bb_data['times'] = bb_data['times'] * 1000  # Convert existing times to ms
            # Keep the existing prices and BB data that are already filtered and aligned

        # Convert trade timestamps to milliseconds for chart compatibility
        if 'trades' in results:
            for trade in results['trades']:
                if 'timestamp' in trade:
                    trade['timestamp'] = trade['timestamp'] * 1000  # Convert to ms
                if 'exit_timestamp' in trade:
                    trade['exit_timestamp'] = trade['exit_timestamp'] * 1000  # Convert to ms

        return results

    except Exception as e:
        print(f"Klines backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """CLI entry point for vectorized klines backtesting"""
    parser = argparse.ArgumentParser(description='Vectorized Klines Bollinger Bands Backtester')
    parser.add_argument('--csv', required=True, help='CSV file path with klines data')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--bb-period', type=int, default=50, help='BB period')
    parser.add_argument('--bb-std', type=float, default=2.0, help='BB std dev')
    parser.add_argument('--stop-loss', type=float, default=0.5, help='Stop loss %')
    parser.add_argument('--capital', type=float, default=1000, help='Initial capital')
    parser.add_argument('--max-klines', type=int, help='Limit klines for testing')
    parser.add_argument('--output', help='Output file for results')

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} not found")
        sys.exit(1)

    print("Vectorized Klines Backtest Configuration:")
    print(f"  File: {args.csv}")
    print(f"  Symbol: {args.symbol}")
    print(f"  BB Period: {args.bb_period}")
    print(f"  BB Std Dev: {args.bb_std}")
    print(f"  Stop Loss: {args.stop_loss}%")
    print(f"  Capital: ${args.capital:,.0f}")
    if args.max_klines:
        print(f"  Max Klines: {args.max_klines:,}")

    results = run_vectorized_klines_backtest(
        csv_path=args.csv,
        symbol=args.symbol,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        stop_loss_pct=args.stop_loss,
        initial_capital=args.capital,
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