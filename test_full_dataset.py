#!/usr/bin/env python3
"""
Test full dataset performance
"""
import time
from src.data.vectorized_backtest import run_vectorized_backtest

def test_full_performance():
    csv_path = "upload/trades/BARDUSDT-trades-2025-09-24.csv"

    print("Testing FULL 4.8M dataset performance...")
    start = time.time()

    results = run_vectorized_backtest(
        csv_path=csv_path,
        symbol="BARDUSDT",
        bb_period=50,
        bb_std=2.0,
        stop_loss_pct=0.5,
        initial_capital=10000.0,
        max_ticks=None  # No limit - full dataset
    )

    elapsed = time.time() - start

    print(f"Full dataset: {elapsed:.2f}s")
    print(f"Trades: {results.get('total', 0)}")
    print(f"Rate: {4_828_933/elapsed:,.0f} ticks/sec")

if __name__ == "__main__":
    test_full_performance()