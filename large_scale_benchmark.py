import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from src.strategies.vectorized_bollinger_strategy import VectorizedBollingerStrategy

def create_large_benchmark_dataset(n_ticks: int = 4_000_000):
    """Create a large benchmark dataset with realistic price movements"""
    print(f"Creating large benchmark dataset with {n_ticks:,} ticks")
    
    # Generate realistic timestamps (every few milliseconds for HFT)
    start_time = int(datetime.now().timestamp() * 1000)  # Current time in milliseconds
    # Simulate high-frequency ticks (every 1-10 ms)
    time_intervals = np.random.choice([1, 2, 3, 4, 5], size=n_ticks)  # Random intervals in ms
    times = np.cumsum(time_intervals) + start_time
    
    # Generate realistic price movements around a base price
    base_price = 50000.0  # Starting price around $50k
    # Small random walk with occasional jumps to simulate real market
    returns = np.random.normal(0, 0.0001, n_ticks)  # Very small random movements
    # Add some larger movements to simulate volatility
    large_moves = np.random.choice([0, 0.001, -0.001], size=n_ticks, p=[0.94, 0.03, 0.03])
    returns += large_moves
    
    # Calculate prices using cumulative returns
    log_prices = np.log(base_price) + np.cumsum(returns)
    prices = np.exp(log_prices)
    
    # Generate realistic quantities
    qtys = np.random.lognormal(mean=3, sigma=1, size=n_ticks)  # Lognormal distribution for quantities
    qtys = np.clip(qtys, 0.001, 100)  # Clamp to reasonable values
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': times,
        'price': prices,
        'qty': qtys,
        'quote_qty': prices * qtys,
        'is_buyer_maker': np.random.choice([True, False], size=n_ticks)
    })
    
    # Save to CSV for benchmarking
    df.to_csv('large_benchmark_ticks.csv', index=False)
    print(f"Created large benchmark dataset with {len(df):,} ticks")
    return df

def benchmark_vectorized_system(df: pd.DataFrame):
    """Benchmark the vectorized system with large dataset"""
    print(f"VECTORIZED PURE TICK BACKTEST: BTCUSDT")
    print(f"Parameters: BB(20, 1.5), SL: 1.0%")
    print(f"Loaded {len(df):,} pure ticks")
    
    # Initialize strategy
    strategy = VectorizedBollingerStrategy(
        symbol="BTCUSDT",
        period=20,
        std_dev=1.5,
        stop_loss_pct=0.01,  # 1% stop loss
        initial_capital=10000.0
    )
    
    # Measure processing time
    start_time = time.time()
    print(f"Processing {len(df):,} ticks using vectorized operations...")
    
    results = strategy.vectorized_process_dataset(df)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate metrics
    ticks_per_sec = len(df) / processing_time
    
    print(f"Vectorized backtest completed in {processing_time:.2f}s")
    print(f"Processing rate: {ticks_per_sec:,.0f} ticks/sec")
    print(f"Total trades generated: {results['total']}")
    
    return {
        'total_ticks': len(df),
        'processing_time': processing_time,
        'ticks_per_sec': ticks_per_sec,
        'total_trades': results['total'],
        'win_rate': results['win_rate'],
        'net_pnl': results['net_pnl'],
        'sharpe_ratio': results['sharpe_ratio']
    }

def main():
    # Create large dataset
    df = create_large_benchmark_dataset(4_000_000)  # 4 million ticks
    
    # Run benchmark
    results = benchmark_vectorized_system(df)
    
    print("\nLARGE SCALE BENCHMARK RESULTS:")
    print(f"  - Total ticks processed: {results['total_ticks']:,}")
    print(f"  - Processing time: {results['processing_time']:.4f} seconds")
    print(f"  - Processing rate: {results['ticks_per_sec']:,.0f} ticks/sec")
    print(f"  - Trades generated: {results['total_trades']:,}")
    print(f"  - Win rate: {results['win_rate']:.2%}")
    print(f"  - Net P&L: ${results['net_pnl']:.2f}")
    print(f" - Sharpe ratio: {results['sharpe_ratio']:.4f}")

if __name__ == "__main__":
    main()