"""
Core module for unified backtesting architecture

This module provides unified components for backtesting that can be used
by both CLI and GUI interfaces.

Author: HFT System
"""

from .backtest_config import BacktestConfig
from .backtest_results import BacktestResults
from .backtest_manager import BacktestManager
from .config_validator import ConfigValidator, ValidationResult

__all__ = [
    'BacktestConfig',
    'BacktestResults',
    'BacktestManager',
    'ConfigValidator',
    'ValidationResult'
]