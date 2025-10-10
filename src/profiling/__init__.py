"""
Profiling Module for HFT System

This module provides a comprehensive set of tools for profiling and analyzing the performance of trading strategies and optimization processes.

It offers:
- Detailed function/method profiling (cProfile, line_profiler)
- Memory profiling
- Specialized profilers for strategies and Optuna studies
- Advanced analysis and visualization utilities

Author: HFT System
"""

# Core profiling classes
from .detailed_profiler import (
    DetailedProfiler,
    StrategyProfiler,
    OptunaProfiler
)

# Convenience functions for quick profiling
from .detailed_profiler import (
    profile_function,
    profile_strategy,
    profile_optuna_study
)

# Analysis and reporting utilities
from .profiling_utils import (
    ProfilingUtils,
    ProfileReport
)

__all__ = [
    # Core classes
    "DetailedProfiler",
    "StrategyProfiler",
    "OptunaProfiler",
    
    # Convenience functions
    "profile_function",
    "profile_strategy",
    "profile_optuna_study",
    
    # Utilities
    "ProfilingUtils",
    "ProfileReport"
]
