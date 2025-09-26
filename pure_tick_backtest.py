"""
Pure Tick Backtesting Engine
High-frequency trading focused, no candle aggregation

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


def calculate_performance_metrics(trades: list, initial_capital: float = 10000.0) -> dict:
    """Calculate comprehensive performance metrics from trades"""
    if not trades:
        return {
            'total': 0, 'win_rate': 0, 'net_pnl': 0, 'net_pnl_percentage': 0,
            'max_drawdown': 0, 'sharpe_ratio': 0, 'profit_factor': 0,
            'total_winning_trades': 0, 'total_losing_trades': 0,
            'average_win': 0, 'average_loss': 0, 'largest_win': 0, 'largest_loss': 0
        }

    # Basic metrics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    total_pnl = sum(t.get('pnl', 0) for t in trades)

    # Win/Loss statistics
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    largest_win = max([t['pnl'] for t in winning_trades], default=0)
    largest_loss = min([t['pnl'] for t in losing_trades], default=0)

    # Return percentage
    return_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0

    # Profit factor
    gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Simplified Sharpe ratio
    trade_returns = [t.get('pnl', 0) / initial_capital for t in trades]
    if len(trade_returns) > 1:
        sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) if np.std(trade_returns) > 0 else 0
        sharpe_ratio *= np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0

    # Max drawdown
    equity_curve = [initial_capital]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade.get('pnl', 0))

    peak = initial_capital
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        'total': total_trades,
        'win_rate': win_rate,
        'net_pnl': total_pnl,
        'net_pnl_percentage': return_pct,
        'max_drawdown': max_dd * 100,  # As percentage
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'total_winning_trades': len(winning_trades),
        'total_losing_trades': len(losing_trades),
        'average_win': avg_win,
        'average_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss
    }


def run_pure_tick_backtest(csv_path: str,
                          symbol: str = 'BTCUSDT',
                          bb_period: int = 50,
                          bb_std: float = 2.0,
                          stop_loss_pct: float = 0.5,
                          initial_capital: float = 10000.0,
                          max_ticks: int = None) -> dict:
    """
    Run backtest on pure tick data using vectorized engine

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
    print(f"PURE TICK BACKTEST: {symbol}")
    print(f"Parameters: BB({bb_period}, {bb_std:.1f}), SL: {stop_loss_pct}%")

    try:
        # Load tick data using vectorized handler
        handler = VectorizedTickHandler()
        tick_data = handler.load_ticks(csv_path)

        # Limit ticks for testing if requested
        if max_ticks and len(tick_data) > max_ticks:
            tick_data = tick_data.head(max_ticks)
            print(f"Limited to {max_ticks:,} ticks for testing")

        # Initialize vectorized strategy
        strategy = VectorizedBollingerStrategy(
            symbol=symbol,
            period=bb_period,
            std_dev=bb_std,
            stop_loss_pct=stop_loss_pct / 100,  # Convert to decimal
            initial_capital=initial_capital
        )

        print(f"Processing {len(tick_data):,} ticks with vectorized engine...")
        
        # Process all ticks at once using vectorized approach
        results = strategy.vectorized_process_dataset(tick_data)
        all_trades = results.get('trades', [])
        print(f"Backtest completed - Generated {len(all_trades)} trades")

        # Add additional metadata
        results['symbol'] = symbol
        results['ticks_processed'] = len(tick_data)
        if tick_data is not None and len(tick_data) > 0:
            results['started_at'] = tick_data.iloc[0]['time']
            results['finished_at'] = tick_data.iloc[-1]['time']

        return results

    except Exception as e:
        print(f"Backtest failed: {e}")
        return {'error': str(e)}


def main():
    """CLI entry point for pure tick backtesting"""
    parser = argparse.ArgumentParser(description='Pure Tick Bollinger Bands Backtester')
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

    print("Pure Tick Backtest Configuration:")
    print(f"  File: {args.csv}")
    print(f"  Symbol: {args.symbol}")
    print(f"  BB Period: {args.bb_period}")
    print(f"  BB Std Dev: {args.bb_std}")
    print(f"  Stop Loss: {args.stop_loss}%")
    print(f"  Capital: ${args.capital:,.0f}")
    if args.max_ticks:
        print(f"  Max Ticks: {args.max_ticks:,}")

    results = run_pure_tick_backtest(
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
    print("PURE TICK BACKTEST RESULTS")
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
    print(f"Ticks Processed: {results.get('ticks_processed', 0):,}")
    print("="*60)

    if args.output:
        # Save results
        with open(args.output, 'w') as f:
            f.write(f"Pure Tick Backtest Results - {args.symbol}\n")
            f.write("="*50 + "\n")
            f.write(f"Total Trades: {results['total']}\n")
            f.write(f"Win Rate: {results['win_rate']:.1%}\n")
            f.write(f"Net P&L: ${results['net_pnl']:,.2f}\n")
            f.write(f"Return: {results['net_pnl_percentage']:.2f}%\n")
            f.write(f"Ticks Processed: {results.get('ticks_processed', 0):,}\n")
        print(f"Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()