"""
Simplified CLI Backtesting Tool without Jesse dependency
Following HFT principles: minimal complexity, high performance
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.tick_data_handler import TickDataHandler
from src.strategies.bollinger_bands_strategy import BollingerBandsMeanReversionStrategy


def run_backtest(csv_path, symbol='1INCHUSDT', exchange='Binance', timeframe='1m',
                bb_period=200, bb_std=3.0, stop_loss_pct=1.0, sma_tp_period=20,
                initial_capital=10000.0, **kwargs):
    """
    Run backtest using simplified strategy without Jesse
    """
    print(f"Starting backtest for {symbol}")
    print(f"Parameters: BB({bb_period}, {bb_std:.1f}), SL: {stop_loss_pct}%, TP: SMA({sma_tp_period})")

    # Load and process tick data
    handler = TickDataHandler(timeframe=timeframe)
    try:
        df = handler.process_csv_to_jesse_format(csv_path)
        print(f"Processed {len(df)} candles from {csv_path}")
    except Exception as e:
        print(f"Error processing data: {e}")
        return {'error': str(e)}

    if df.empty:
        print("No data to backtest")
        return {'error': 'No data'}

    # Initialize strategy
    strategy = BollingerBandsMeanReversionStrategy(
        symbol=symbol,
        period=bb_period,
        std_dev=bb_std,
        stop_loss_pct=stop_loss_pct / 100,  # Convert to decimal
        take_profit_sma_period=sma_tp_period,
        initial_capital=initial_capital
    )

    # Run backtest tick by tick (simplified)
    print("Running backtest...")
    all_trades = []

    # Convert OHLCV to price ticks for simplified processing
    # In real implementation, you'd use actual tick data
    for i, row in df.iterrows():
        timestamp = pd.to_datetime(row['timestamp'], unit='ms')
        # Use close price as tick price
        price = row['close']

        trades = strategy.process_tick(timestamp, price)
        all_trades.extend(trades)

    # Calculate results
    results = calculate_performance_metrics(all_trades, strategy, df)
    results['symbol'] = symbol
    results['trades'] = all_trades
    results['started_at'] = df.iloc[0]['timestamp']
    results['finished_at'] = df.iloc[-1]['timestamp']

    print("Backtest completed")
    return results


def calculate_performance_metrics(trades, strategy, price_data):
    """Calculate comprehensive performance metrics"""
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
    initial_capital = strategy.initial_capital
    return_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0

    # Profit factor
    gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Simplified Sharpe ratio (using trade returns)
    trade_returns = [t.get('pnl', 0) / initial_capital for t in trades]
    if len(trade_returns) > 1:
        sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) if np.std(trade_returns) > 0 else 0
        sharpe_ratio *= np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0

    # Max drawdown (simplified)
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
        'largest_loss': largest_loss,
        'total_fees': 0,  # Not implemented
        'sortino_ratio': 0,  # Not implemented
        'total_open_trades': 0,  # Not applicable
        'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0
    }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='Simplified BB Strategy Backtester')
    parser.add_argument('--csv', required=True, help='CSV file path')
    parser.add_argument('--symbol', default='1INCHUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='1m', help='Timeframe')
    parser.add_argument('--bb-period', type=int, default=200, help='BB period')
    parser.add_argument('--bb-std', type=float, default=3.0, help='BB std dev')
    parser.add_argument('--stop-loss', type=float, default=1.0, help='Stop loss %')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--output', help='Output file')

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} not found")
        sys.exit(1)

    print("Backtest Configuration:")
    print(f"  File: {args.csv}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  BB Period: {args.bb_period}")
    print(f"  BB Std Dev: {args.bb_std}")
    print(f"  Stop Loss: {args.stop_loss}%")
    print(f"  Capital: ${args.capital:,.0f}")

    results = run_backtest(
        csv_path=args.csv,
        symbol=args.symbol,
        timeframe=args.timeframe,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        stop_loss_pct=args.stop_loss,
        initial_capital=args.capital
    )

    if 'error' in results:
        print(f"Backtest failed: {results['error']}")
        sys.exit(1)

    # Display results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
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

    if args.output:
        # Save results
        with open(args.output, 'w') as f:
            f.write(f"Backtest Results - {args.symbol}\n")
            f.write("="*50 + "\n")
            f.write(f"Total Trades: {results['total']}\n")
            f.write(f"Win Rate: {results['win_rate']:.1%}\n")
            f.write(f"Net P&L: ${results['net_pnl']:,.2f}\n")
            f.write(f"Return: {results['net_pnl_percentage']:.2f}%\n")
            f.write(f"Max Drawdown: {results['max_drawdown']:.2f}%\n")
            f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            f.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()