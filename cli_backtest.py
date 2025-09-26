"""
Simplified CLI Backtesting Tool with Vectorized Strategy
Following HFT principles: minimal complexity, high performance
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.vectorized_tick_handler import VectorizedTickHandler
from src.strategies.vectorized_bollinger_strategy import VectorizedBollingerStrategy

def load_trades_from_csv(csv_path):
    """
    Load real trades from CSV file and convert to standard format
    Expected CSV format: id,price,qty,quote_qty,time,is_buyer_maker
    """
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file")
        
        # Convert to standard trade format
        trades = []
        for _, row in df.iterrows():
            # Determine side based on is_buyer_maker
            # If is_buyer_maker is True, it means the buyer initiated the trade (buy order)
            # If is_buyer_maker is False, it means the seller initiated the trade (sell order)
            side = 'long' if row['is_buyer_maker'] else 'short'
            
            trade = {
                'timestamp': int(row['time']),  # Keep original timestamp format
                'entry_price': float(row['price']),
                'exit_price': float(row['price']),  # Same as entry for tick data
                'qty': float(row['qty']),
                'side': side,
                'pnl': 0.0,  # Placeholder, actual PnL would need more complex calculation
                'pnl_percentage': 0.0,
                'duration': 0
            }
            trades.append(trade)
        
        return trades
    except Exception as e:
        print(f"Error loading trades from CSV: {e}")
        return []



def run_backtest(csv_path, symbol='BTCUSDT', exchange='Binance', timeframe='1m',
                bb_period=50, bb_std=2.0, stop_loss_pct=0.5, sma_tp_period=20,
                initial_capital=10000.0, use_real_trades=False, use_tick_mode=True, **kwargs):
    """
    Run backtest using vectorized strategy
    If use_real_trades is True, load trades directly from CSV instead of running strategy
    If use_tick_mode is True, work with raw ticks instead of aggregated candles
    """
    print(f"Starting vectorized backtest for {symbol}")

    if use_tick_mode:
        print(f"VECTORIZED TICK MODE: Processing raw ticks with BB({bb_period}, {bb_std:.1f}), SL: {stop_loss_pct}%")
    else:
        print(f"Parameters: BB({bb_period}, {bb_std:.1f}), SL: {stop_loss_pct}%, TP: SMA({sma_tp_period})")

    if use_real_trades:
        # Load real trades from CSV
        print("Loading real trades from CSV...")
        all_trades = load_trades_from_csv(csv_path)
        print(f"Loaded {len(all_trades)} trades from {csv_path}")
        
        if not all_trades:
            return {'error': 'No trades loaded'}
        
        # Calculate results based on real trades
        results = calculate_performance_metrics(all_trades, None, None)
        results['symbol'] = symbol
        results['trades'] = all_trades
        if all_trades:
            results['started_at'] = all_trades[0]['timestamp']
            results['finished_at'] = all_trades[-1]['timestamp']
        
        print("Trade loading completed")
        return results
    else:
        # Load data based on mode
        handler = VectorizedTickHandler()
        try:
            # Load raw ticks for vectorized processing
            df = handler.load_ticks(csv_path)
            print(f"Loaded {len(df):,} raw ticks from {csv_path}")
        except Exception as e:
            print(f"Error processing data: {e}")
            return {'error': str(e)}

        if df.empty:
            print("No data to backtest")
            return {'error': 'No data'}

        # Initialize vectorized strategy
        strategy = VectorizedBollingerStrategy(
            symbol=symbol,
            period=bb_period,
            std_dev=bb_std,
            stop_loss_pct=stop_loss_pct / 100,  # Convert to decimal
            initial_capital=initial_capital
        )

        # Run vectorized backtest
        print("Running vectorized backtest...")
        results = strategy.vectorized_process_dataset(df)
        
        # Add additional metadata
        results['symbol'] = symbol
        results['started_at'] = df.iloc[0]['time'] if len(df) > 0 else None
        results['finished_at'] = df.iloc[-1]['time'] if len(df) > 0 else None

        print("Vectorized backtest completed")
        return results


def calculate_performance_metrics(trades, strategy=None, price_data=None):
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
    # Use initial capital of 10000 as default if strategy is not provided
    initial_capital = strategy.initial_capital if strategy else 10000.0
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
        'max_drawdown': max_dd * 10,  # As percentage
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
    parser.add_argument('--use-real-trades', action='store_true', help='Load real trades from CSV instead of running backtest')
    parser.add_argument('--tick-mode', action='store_true', help='Process raw ticks instead of converting to candles')

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
    print(f"  Use Real Trades: {args.use_real_trades}")
    print(f"  Tick Mode: {args.tick_mode}")

    results = run_backtest(
        csv_path=args.csv,
        symbol=args.symbol,
        timeframe=args.timeframe,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        stop_loss_pct=args.stop_loss,
        initial_capital=args.capital,
        use_real_trades=args.use_real_trades,
        use_tick_mode=args.tick_mode
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