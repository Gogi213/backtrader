"""
Optimization module for trading strategies using Optuna

This module provides parameter optimization capabilities for trading strategies
using the Optuna framework.

Author: HFT System
"""

from .fast_optimizer import (
    FastStrategyOptimizer,
    quick_fast_optimize,
    create_composite_objective
)
from .visualization import (
    quick_visualize
)

# Алиас для обратной совместимости
StrategyOptimizer = FastStrategyOptimizer
quick_optimize = quick_fast_optimize

__all__ = [
    'FastStrategyOptimizer',
    'quick_fast_optimize',
    'StrategyOptimizer',  # Алиас для обратной совместимости
    'quick_optimize',      # Алиас для обратной совместимости
    'create_composite_objective',
    'quick_visualize'
]