#!/usr/bin/env python3
"""
Test performance with 10+ million ticks
"""
import sys
import os
import time
import pandas as pd
from datetime import datetime

def benchmark_current_performance():
    """Benchmark current system with different tick volumes"""
    print("Benchmarking current performance...")

    from src.data.vectorized_tick_handler import VectorizedTickHandler
    from src.strategies.vectorized_bollinger_strategy import VectorizedBollingerStrategy

    # Find the largest CSV file
    csv_files = [f for f in os.listdir("upload/trades") if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found")
        return

    csv_path = os.path.join("upload/trades", csv_files[0])
    print(f"Testing with: {csv_files[0]}")

    # Load full dataset
    handler = VectorizedTickHandler()
    print("Loading full dataset...")
    start_time = time.time()
    df = handler.load_ticks(csv_path)
    load_time = time.time() - start_time

    total_ticks = len(df)
    print(f"Loaded {total_ticks:,} ticks in {load_time:.2f}s")
    print(f"Load rate: {total_ticks/load_time:,.0f} ticks/sec")

    # Test different volumes
    test_volumes = [
        (100_000, "100K ticks"),
        (500_000, "500K ticks"),
        (1_000_000, "1M ticks"),
        (2_500_000, "2.5M ticks"),
        (5_000_000, "5M ticks"),
    ]

    # Add full dataset if larger than 5M
    if total_ticks > 5_000_000:
        test_volumes.append((total_ticks, f"Full dataset ({total_ticks:,} ticks)"))

    strategy = VectorizedBollingerStrategy("TESTUSDT", period=50, std_dev=2.0)

    print("\nPerformance Benchmark:")
    print("=" * 60)
    print(f"{'Volume':<20} {'Time (s)':<10} {'Rate (ticks/s)':<15} {'Trades':<8} {'Memory':<10}")
    print("-" * 60)

    for volume, description in test_volumes:
        if volume > total_ticks:
            continue

        # Test subset
        test_df = df.head(volume)

        # Measure memory before
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run backtest
        start_time = time.time()
        results = strategy.vectorized_process_dataset(test_df)
        process_time = time.time() - start_time

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        # Results
        rate = volume / process_time if process_time > 0 else 0
        trades = results.get('total', 0)

        print(f"{description:<20} {process_time:<10.2f} {rate:<15,.0f} {trades:<8} {mem_used:<10.1f}MB")

        # Stop if too slow
        if process_time > 30:  # More than 30 seconds
            print("Stopping benchmark - too slow for larger volumes")
            break

    return results

def identify_bottlenecks():
    """Identify performance bottlenecks"""
    print("\nIdentifying bottlenecks...")

    # Check numba compilation
    try:
        from numba import config
        print(f"Numba available: {config.NUMBA_ENABLE_CUDASIM}")
    except:
        print("Numba configuration not accessible")

    # Check pandas optimizations
    print(f"Pandas version: {pd.__version__}")

    # Check system info
    import psutil
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")

def main():
    print("HFT Performance Benchmark - 10+ Million Ticks")
    print("=" * 50)

    identify_bottlenecks()
    benchmark_current_performance()

    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()