"""
Main Application for Pure Tick HFT Bollinger Bands Strategy Backtester

High-frequency trading focused, no candle aggregation
Pure tick processing with numpy/numba optimization

Author: HFT System
"""
import sys
import os
from pure_tick_gui import main as pure_tick_gui_main

def main():
    """Main entry point for Pure Tick HFT application"""
    print("🚀 Starting Pure Tick HFT Backtester...")
    print("⚡ High-frequency Bollinger Bands Strategy")
    print("📊 No candle aggregation - Pure tick processing")

    # Verify required directories exist
    required_dirs = ['upload', 'upload/trades', 'src', 'src/data', 'src/strategies']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"📁 Creating required directory: {directory}")
            os.makedirs(directory, exist_ok=True)

    # Verify required files exist
    required_files = [
        'pure_tick_backtest.py',
        'src/data/vectorized_tick_handler.py',
        'src/strategies/vectorized_bollinger_strategy.py'
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Error: Missing required files: {missing_files}")
        print("🔧 Please ensure all Vectorized HFT components are installed")
        return 1

    # Check PyQt6 availability
    print("🔍 Checking PyQt6 installation...")
    try:
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 found - HFT GUI ready to launch")
    except ImportError:
        print("❌ Error: PyQt6 not installed!")
        print("📦 Install with: pip install PyQt6>=6.4.0")
        print("🔧 Or run: pip install -r requirements.txt")
        return 1

    # Check numba availability for HFT optimization
    print("⚡ Checking numba optimization...")
    try:
        import numba
        print("✅ Numba found - HFT optimization enabled")
    except ImportError:
        print("⚠️  Warning: Numba not found - install for better performance")
        print("📦 Install with: pip install numba")

    # Check for tick data
    trades_dir = "upload/trades"
    if os.path.exists(trades_dir):
        csv_files = [f for f in os.listdir(trades_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"📈 Found {len(csv_files)} tick datasets ready for HFT backtesting")
        else:
            print("⚠️  No tick data files found in upload/trades/")
            print("📄 Place your CSV tick data files in upload/trades/")

    # Start the Pure Tick HFT GUI application
    print("🚀 Launching Pure Tick HFT GUI...")
    pure_tick_gui_main()

if __name__ == "__main__":
    main()