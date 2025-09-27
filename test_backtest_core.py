#!/usr/bin/env python3
"""
Test core backtesting functionality
"""
import sys
import os

def test_backtest_functionality():
    """Test core backtesting functionality"""
    print("Testing backtest core functionality...")

    try:
        # Test data loading
        from src.data.vectorized_tick_handler import VectorizedTickHandler
        handler = VectorizedTickHandler()

        # Find CSV files
        csv_files = [f for f in os.listdir("upload/trades") if f.endswith('.csv')]
        if not csv_files:
            print("FAIL: No CSV files found")
            return False

        csv_path = os.path.join("upload/trades", csv_files[0])
        print(f"Testing with: {csv_files[0]}")

        # Load data
        df = handler.load_ticks(csv_path)
        print(f"OK: Loaded {len(df)} ticks")

        # Test strategy
        from src.strategies.vectorized_bollinger_strategy import VectorizedBollingerStrategy
        strategy = VectorizedBollingerStrategy("TESTUSDT", period=20, std_dev=2.0)

        # Limit data for test
        test_df = df.head(10000)  # Use only 10K ticks for speed
        print(f"Testing with {len(test_df)} ticks...")

        # Run backtest
        results = strategy.vectorized_process_dataset(test_df)

        print(f"OK: Backtest completed")
        print(f"Total trades: {results.get('total', 0)}")
        print(f"Net P&L: {results.get('net_pnl', 0):.2f}")
        print(f"Win rate: {results.get('win_rate', 0)*100:.1f}%")

        return True

    except Exception as e:
        print(f"FAIL: Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("HFT Backtest Core Test")
    print("=" * 30)

    if test_backtest_functionality():
        print("SUCCESS: Core backtesting works")
        return 0
    else:
        print("FAIL: Core backtesting failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())