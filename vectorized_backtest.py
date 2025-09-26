"""
Vectorized Pure Tick Backtesting Engine
High-frequency trading focused, fully vectorized implementation
Processing 4+ million ticks efficiently with numpy/numba optimization

Author: HFT System
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.vectorized_tick_handler import VectorizedTickHandler
from src.strategies.vectorized_bollinger_strategy import VectorizedBollingerStrategy


def run_vectorized_backtest(csv_path: str,
                           symbol: str = 'BTCUSDT',
                           bb_period: int = 50,
                           bb_std: float = 2.0,
                           stop_loss_pct: float = 0.5,
                           initial_capital: float = 10000.0,
                           max_ticks: int = None) -> dict:
    """
    Run fully vectorized backtest on pure tick data

    Args:
        csv_path: Path to CSV file with tick data
        symbol: Trading symbol
        bb_period: Bollinger Bands period
        bb_std: BB standard deviation multiplier
        stop_loss_pct: Stop loss percentage
        initial_capital: Initial capital
        max_ticks: Maximum ticks to process (for testing)

    Returns:
        Dictionary with backtest results
    """
    print(f"VECTORIZED PURE TICK BACKTEST: {symbol}")
    print(f"Parameters: BB({bb_period}, {bb_std:.1f}), SL: {stop_loss_pct}%")
    
    try:
        # Load tick data
        handler = VectorizedTickHandler()
        tick_df = handler.load_ticks(csv_path)
        
        # Limit ticks for testing if requested
        if max_ticks and len(tick_df) > max_ticks:
            tick_df = tick_df.head(max_ticks)
            print(f"Limited to {max_ticks:,} ticks for testing")
        
        print(f"Processing {len(tick_df):,} ticks using vectorized operations...")
        start_time = datetime.now()
        
        # Initialize strategy
        strategy = VectorizedBollingerStrategy(
            symbol=symbol,
            period=bb_period,
            std_dev=bb_std,
            stop_loss_pct=stop_loss_pct / 10,  # Convert to decimal
            initial_capital=initial_capital
        )
        
        # Perform complete backtest using vectorized operations
        results = strategy.vectorized_process_dataset(tick_df)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"Vectorized backtest completed in {processing_time:.2f}s")
        if processing_time > 0:
            print(f"Processing rate: {len(tick_df)/processing_time:,.0f} ticks/sec")
        else:
            print("Processing rate: N/A (completed in less than 0.01s)")
        print(f"Total trades generated: {results.get('total', 0)}")
        
        # Add tick statistics
        stats = strategy.get_stats()
        results.update(stats)
        
        if tick_df is not None and len(tick_df) > 0:
            results['started_at'] = tick_df.iloc[0]['time']
            results['finished_at'] = tick_df.iloc[-1]['time']
            results['processing_time_seconds'] = processing_time
        
        return results

    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """CLI entry point for vectorized pure tick backtesting"""
    parser = argparse.ArgumentParser(description='Vectorized Pure Tick Bollinger Bands Backtester')
    parser.add_argument('--csv', required=True, help='CSV file path with tick data')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--bb-period', type=int, default=50, help='BB period (HFT optimized)')
    parser.add_argument('--bb-std', type=float, default=2.0, help='BB std dev')
    parser.add_argument('--stop-loss', type=float, default=0.5, help='Stop loss %')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--max-ticks', type=int, help='Limit ticks for testing')
    parser.add_argument('--output', help='Output file for results')

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} not found")
        sys.exit(1)

    print("Vectorized Pure Tick Backtest Configuration:")
    print(f"  File: {args.csv}")
    print(f"  Symbol: {args.symbol}")
    print(f"  BB Period: {args.bb_period}")
    print(f"  BB Std Dev: {args.bb_std}")
    print(f"  Stop Loss: {args.stop_loss}%")
    print(f"  Capital: ${args.capital:,.0f}")
    if args.max_ticks:
        print(f"  Max Ticks: {args.max_ticks:,}")

    results = run_vectorized_backtest(
        csv_path=args.csv,
        symbol=args.symbol,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        stop_loss_pct=args.stop_loss,
        initial_capital=args.capital,
        max_ticks=args.max_ticks
    )

    if 'error' in results:
        print(f"Backtest failed: {results['error']}")
        sys.exit(1)

    # Display results
    print("\n" + "="*60)
    print("VECTORIZED PURE TICK BACKTEST RESULTS")
    print("="*60)
    print(f"Symbol: {results['symbol']}")
    print(f"Total Trades: {results['total']}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Net P&L: ${results['net_pnl']:,.2f}")
    print(f"Return: {results['net_pnl_percentage']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Winners: {results['total_winning_trades']}")
    print(f"Losers: {results['total_losing_trades']}")
    print(f"Avg Win: ${results['average_win']:.2f}")
    print(f"Avg Loss: ${results['average_loss']:.2f}")
    print(f"Best Trade: ${results['largest_win']:.2f}")
    print(f"Worst Trade: ${results['largest_loss']:.2f}")
    print(f"Ticks Processed: {results.get('total_ticks', 0):,}")
    print(f"Processing Time: {results.get('processing_time_seconds', 0):.2f}s")
    processing_time_seconds = results.get('processing_time_seconds', 1)
    if processing_time_seconds > 0:
        print(f"Ticks per Second: {results.get('total_ticks', 0)/processing_time_seconds:,.0f}")
    else:
        print(f"Ticks per Second: N/A (completed in less than 0.01s)")
    print("="*60)

    if args.output:
        # Save results
        with open(args.output, 'w') as f:
            f.write(f"Vectorized Pure Tick Backtest Results - {args.symbol}\n")
            f.write("="*50 + "\n")
            f.write(f"Total Trades: {results['total']}\n")
            f.write(f"Win Rate: {results['win_rate']:.1%}\n")
            f.write(f"Net P&L: ${results['net_pnl']:,.2f}\n")
            f.write(f"Return: {results['net_pnl_percentage']:.2f}%\n")
            f.write(f"Ticks Processed: {results.get('total_ticks', 0):,}\n")
            f.write(f"Processing Time: {results.get('processing_time_seconds', 0):.2f}s\n")
            f.write(f"Ticks per Second: {results.get('total_ticks', 0)/results.get('processing_time_seconds', 1):,.0f}\n")
        print(f"Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()