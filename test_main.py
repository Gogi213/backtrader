#!/usr/bin/env python3
"""
Simple test to check main.py functionality without hanging GUI
"""
import sys
import os

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")

    try:
        import pandas as pd
        print("OK pandas")
    except ImportError as e:
        print(f"FAIL pandas: {e}")
        return False

    try:
        import numpy as np
        print("OK numpy")
    except ImportError as e:
        print(f"FAIL numpy: {e}")
        return False

    try:
        from PyQt6.QtWidgets import QApplication
        print("OK PyQt6")
    except ImportError as e:
        print(f"FAIL PyQt6: {e}")
        return False

    try:
        import numba
        print("OK numba")
    except ImportError as e:
        print(f"FAIL numba (optional): {e}")

    # Test project imports
    try:
        from src.data.vectorized_backtest import run_vectorized_backtest
        print("OK vectorized_backtest")
    except ImportError as e:
        print(f"FAIL vectorized_backtest: {e}")
        return False

    try:
        from src.data.vectorized_tick_handler import VectorizedTickHandler
        print("OK vectorized_tick_handler")
    except ImportError as e:
        print(f"FAIL vectorized_tick_handler: {e}")
        return False

    try:
        from src.strategies.vectorized_bollinger_strategy import VectorizedBollingerStrategy
        print("OK vectorized_bollinger_strategy")
    except ImportError as e:
        print(f"FAIL vectorized_bollinger_strategy: {e}")
        return False

    return True

def test_data_files():
    """Test if required data files exist"""
    print("\nTesting data files...")

    # Check uploads directory
    if not os.path.exists("upload/trades"):
        print("FAIL upload/trades directory missing")
        return False

    csv_files = [f for f in os.listdir("upload/trades") if f.endswith('.csv')]
    if not csv_files:
        print("FAIL No CSV files in upload/trades")
        return False

    print(f"OK Found {len(csv_files)} CSV files")
    return True

def main():
    print("HFT Backtester - System Check")
    print("=" * 40)

    imports_ok = test_imports()
    data_ok = test_data_files()

    print("\n" + "=" * 40)
    if imports_ok and data_ok:
        print("SUCCESS All systems ready for HFT backtesting")
        return 0
    else:
        print("FAIL System not ready - fix issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())