"""
Main Application for Unified Vectorized HFT Strategy Backtester

Super-vectorized high-frequency trading system
UNIFIED SYSTEM - Console Version:
- Unified BacktestManager for all backtesting operations
- Unified OptimizationManager for all optimization operations
- Batch backtesting support with parallel processing
- Performance optimizations for large datasets
- Console interface for backtesting and optimization

Author: HFT System
"""
import sys
import os

def main():
    """Main entry point for Unified Vectorized HFT application"""
    print("=" * 60)
    print("Unified Vectorized HFT Backtester - Console Mode")
    print("=" * 60)
    print("Super-vectorized Hierarchical Mean Reversion Strategy")
    print("Unified Console architecture - Maximum performance")
    print()

    # Verify required directories exist
    required_dirs = ['upload', 'upload/klines', 'src', 'src/data', 'src/strategies']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Creating required directory: {directory}")
            os.makedirs(directory, exist_ok=True)

    # Verify required files exist
    required_files = [
        'src/data/backtest_engine.py',
        'src/data/klines_handler.py',
        'src/strategies/turbo_mean_reversion_strategy.py'
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Please ensure all Unified Vectorized components are installed")
        return 1

    # Check numba availability for HFT optimization
    print("Checking numba optimization...")
    try:
        import numba
        print("+ Numba found - HFT optimization enabled")
    except ImportError:
        print("! Warning: Numba not found - install for better performance")
        print("  Install with: pip install numba")

    # Check for klines data
    klines_dir = "upload/klines"
    if os.path.exists(klines_dir):
        csv_files = [f for f in os.listdir(klines_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"+ Found {len(csv_files)} klines datasets ready for vectorized backtesting")
        else:
            print("! No klines data files found in upload/klines/")
            print("  Place your CSV klines data files in upload/klines/")
    else:
        print("! Klines directory not found")
        print("  Create upload/klines/ directory and place CSV data files there")

    print()
    print("Available interfaces:")
    print("1. CLI Optimizer - Fast command-line optimization")
    print("2. TUI Interface - Interactive terminal interface")
    print()
    print("Usage examples:")
    print("  python cli_optimizer.py --trials 50")
    print("  python tui_runner.py")
    print()
    print("For detailed help:")
    print("  python cli_optimizer.py --help")
    print("  See README_TUI.md for TUI interface guide")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())