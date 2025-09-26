import tempfile
import pandas as pd
import numpy as np
import os
from vectorized_backtest import run_vectorized_backtest
import time

def run_benchmark():
    # Create temporary directory and test data
    temp_dir = tempfile.mkdtemp()
    
    # Create more volatile price data to trigger more Bollinger Band signals
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    # Generate price series with higher volatility to trigger more signals
    for i in range(1, 50000):  # 50K ticks for better benchmark
        # Add higher volatility with occasional large moves
        if i % 1000 == 0:  # Every 1000 ticks, add a larger move
            change = np.random.normal(0, 0.1)  # Larger volatility
        else:
            change = np.random.normal(0, 0.05)  # Regular volatility
        new_price = max(prices[-1] * (1 + change), 50.0)
        prices.append(new_price)
    
    data = {
        'id': range(1, 50001),
        'price': prices,
        'qty': np.random.uniform(0.1, 10.0, 50000),
        'quote_qty': np.random.uniform(10, 1000, 50000),
        'time': [1758672000 + i * 10 for i in range(50000)],  # in milliseconds
        'is_buyer_maker': np.random.choice([True, False], 50000)
    }
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_dir, 'benchmark_data.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Created benchmark dataset with {len(df):,} ticks")
    
    # Run vectorized backtest
    start_time = time.time()
    results = run_vectorized_backtest(
        csv_path, 
        bb_period=20,  # Smaller period for more signals
        bb_std=1.5,    # Tighter bands for more signals
        stop_loss_pct=1.0,  # 1% stop loss
        max_ticks=50000  # Process all ticks
    )
    end_time = time.time()
    
    total_time = end_time - start_time
    total_ticks = results.get('total_ticks', 0)
    ticks_per_sec = total_ticks / total_time if total_time > 0 else 0
    
    print(f'\nBENCHMARK RESULTS:')
    print(f'  - Total ticks processed: {total_ticks:,}')
    print(f'  - Processing time: {total_time:.4f} seconds')
    print(f'  - Processing rate: {ticks_per_sec:,.0f} ticks/sec')
    print(f'  - Trades generated: {results.get("total", 0):,}')
    print(f'  - Win rate: {results.get("win_rate", 0):.2%}')
    print(f'  - Net P&L: ${results.get("net_pnl", 0):,.2f}')
    print(f'  - Performance improvement: {ticks_per_sec / 2600:.1f}x vs original (original: ~2,600 ticks/sec)')
    
    return results

if __name__ == "__main__":
    run_benchmark()