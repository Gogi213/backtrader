"""
Main Application for Jesse Bollinger Bands Strategy Backtester

This module provides the main entry point for the GUI application that allows
users to run backtests with the Bollinger Bands strategy and visualize results.
"""
import sys
import os
from gui_visualizer import main as gui_main

def main():
    """Main entry point for the application"""
    print("Starting Jesse Bollinger Bands Strategy Backtester...")
    print("Checking for required components...")
    
    # Verify required directories exist
    required_dirs = ['upload', 'upload/trades']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Creating required directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    # Verify required files exist
    required_files = ['cli_backtest.py', 'jesse_strategy.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: Missing required files: {missing_files}")
        return 1
    
    # Check PyQt6 availability
    print("Checking PyQt6 installation...")
    try:
        from PyQt6.QtWidgets import QApplication
        print("PyQt6 found - GUI ready to launch")
    except ImportError:
        print("Error: PyQt6 not installed!")
        print("Install with: pip install PyQt6>=6.4.0")
        print("Or run: pip install -r requirements.txt")
        return 1

    # Start the GUI application
    print("Launching GUI application...")
    gui_main()

if __name__ == "__main__":
    main()