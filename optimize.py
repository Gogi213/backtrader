#!/usr/bin/env python3
"""
Command-line interface for strategy parameter optimization using Optuna

This script provides a convenient way to optimize trading strategy parameters
using the Optuna framework.

Usage:
    python optimize.py --csv data/BTCUSDT.csv --strategy hierarchical_mean_reversion --trials 100

Author: HFT System
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.optimization.optimize_cli import main

if __name__ == "__main__":
    main()