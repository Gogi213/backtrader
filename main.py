"""
Main Application for Unified Vectorized HFT Strategy Backtester

Super-vectorized high-frequency trading system
Unified CLI/GUI architecture with maximum performance

Author: HFT System
"""
import sys
import os
from src.gui.main_window import main as gui_main

def main():
    """Main entry point for Unified Vectorized HFT application"""
    print("Starting Unified Vectorized HFT Backtester...")
    print("Super-vectorized Hierarchical Mean Reversion Strategy")
    print("Unified CLI/GUI architecture - Maximum performance")

    # Verify required directories exist
    required_dirs = ['upload', 'upload/klines', 'src', 'src/data', 'src/strategies']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Creating required directory: {directory}")
            os.makedirs(directory, exist_ok=True)

    # Verify required files exist
    required_files = [
        'src/data/backtest_engine.py',
        'src/gui/main_window.py',
        'src/data/klines_handler.py',
        'src/strategies/turbo_mean_reversion_strategy.py'
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Please ensure all Unified Vectorized components are installed")
        return 1

    # Check PyQt6 availability
    print("Checking PyQt6 installation...")
    try:
        from PyQt6.QtWidgets import QApplication
        print("PyQt6 found - HFT GUI ready to launch")
    except ImportError:
        print("Error: PyQt6 not installed!")
        print("Install with: pip install PyQt6>=6.4.0")
        print("Or run: pip install -r requirements.txt")
        return 1

    # Check numba availability for HFT optimization
    print("Checking numba optimization...")
    try:
        import numba
        print("Numba found - HFT optimization enabled")
    except ImportError:
        print("Warning: Numba not found - install for better performance")
        print("Install with: pip install numba")

    # Check for klines data
    klines_dir = "upload/klines"
    if os.path.exists(klines_dir):
        csv_files = [f for f in os.listdir(klines_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"Found {len(csv_files)} klines datasets ready for vectorized backtesting")
        else:
            print("No klines data files found in upload/klines/")
            print("Place your CSV klines data files in upload/klines/")

    # Start the Unified Vectorized HFT GUI application
    print("Launching Unified Vectorized HFT GUI...")
    gui_main()

if __name__ == "__main__":
    main()