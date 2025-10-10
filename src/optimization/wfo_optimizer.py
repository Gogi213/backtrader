"""
Walk-Forward Optimization (WFO) using Optuna.

This module provides a framework for performing walk-forward optimization on trading strategies.
It uses Optuna to find the best hyperparameters for a given strategy on rolling in-sample windows
and then validates their performance on subsequent out-of-sample windows.

Author: Gemini
"""

import pandas as pd
import numpy as np
import optuna
from typing import Dict, List, Any, Callable

from ..core.backtest_manager import BacktestManager
from ..data.klines_handler import NumpyKlinesData
from ..strategies.strategy_registry import StrategyRegistry


def create_walk_forward_windows(
    data: NumpyKlinesData,
    in_sample_size: int,
    out_sample_size: int,
    gap: int = 0,
    step: int = None
) -> List[Dict[str, NumpyKlinesData]]:
    """
    Create walk-forward optimization windows from NumpyKlinesData.

    Args:
        data: A NumpyKlinesData object.
        in_sample_size: The number of data points in each in-sample window.
        out_sample_size: The number of data points in each out-of-sample window.
        gap: The number of data points to skip between in-sample and out-of-sample windows.
        step: The number of data points to move forward for each new window. Defaults to out_sample_size.

    Returns:
        A list of dictionaries, where each dictionary contains 'in_sample' and 'out_sample' NumpyKlinesData objects.
    """
    if step is None:
        step = out_sample_size

    windows = []
    start_idx = 0
    total_len = len(data)

    while start_idx + in_sample_size + gap + out_sample_size <= total_len:
        in_sample_end = start_idx + in_sample_size
        out_sample_end = in_sample_end + gap + out_sample_size

        in_sample_data = data[start_idx:in_sample_end]
        out_sample_data = data[in_sample_end + gap:out_sample_end]

        windows.append({
            'in_sample': in_sample_data,
            'out_sample': out_sample_data
        })
        start_idx += step

    return windows


class Objective:
    """
    Objective function evaluator for Optuna trials.
    Encapsulates the logic for running a backtest with a given set of parameters.
    """
    def __init__(self, strategy_name: str, in_sample_data: NumpyKlinesData, param_space: Dict[str, tuple]):
        self.strategy_name = strategy_name
        self.in_sample_data = in_sample_data
        self.param_space = param_space
        self.backtest_manager = BacktestManager()

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """
        Execute a single optimization trial.
        """
        params = {}
        for param_name, config in self.param_space.items():
            param_type, low, high = config
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, low, high)
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, low)

        # Run the backtest on the in-sample data with the suggested parameters
        results = self.backtest_manager.run_backtest_on_data(
            strategy_name=self.strategy_name,
            klines_data=self.in_sample_data,
            params=params
        )

        # If backtest fails or produces no trades, return a large negative value
        if results.total == 0 or results.error:
            return -1e9

        # Return the metric to be optimized (e.g., Sharpe Ratio)
        # You can customize this to be any metric from the BacktestResults object
        metric = results.sharpe_ratio
        if not np.isfinite(metric):
            return -1e9
            
        return metric


def run_wfo(
    data: NumpyKlinesData,
    strategy_name: str,
    in_sample_size: int,
    out_sample_size: int,
    n_trials: int = 100,
    direction: str = 'maximize',
    gap: int = 0,
    step: int = None
) -> pd.DataFrame:
    """
    Run a full Walk-Forward Optimization.

    Args:
        data: The entire dataset as a NumpyKlinesData object.
        strategy_name: The name of the strategy to optimize.
        in_sample_size: The size of the in-sample training window.
        out_sample_size: The size of the out-of-sample validation window.
        n_trials: The number of Optuna trials to run for each window.
        direction: The optimization direction ('maximize' or 'minimize').
        gap: The gap between in-sample and out-of-sample windows.
        step: The step size for moving the windows.

    Returns:
        A pandas DataFrame containing the results of the WFO.
    """
    strategy_class = StrategyRegistry.get_strategy(strategy_name)
    if not strategy_class:
        raise ValueError(f"Strategy '{strategy_name}' not found.")
    param_space = strategy_class.get_param_space()

    windows = create_walk_forward_windows(data, in_sample_size, out_sample_size, gap, step)
    wfo_results = []
    backtest_manager = BacktestManager()

    print(f"Starting WFO for strategy '{strategy_name}' with {len(windows)} windows.")

    for i, window in enumerate(windows):
        print(f"---\nProcessing Window {i+1}/{len(windows)}...")
        in_sample_data = window['in_sample']
        out_sample_data = window['out_sample']

        # 1. Optimize on in-sample data
        print(f"Optimizing on in-sample data ({len(in_sample_data)} bars)...")
        objective = Objective(strategy_name, in_sample_data, param_space)
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1) # Use all available CPU cores

        best_params = study.best_params
        in_sample_value = study.best_value

        print(f"Best in-sample metric: {in_sample_value:.4f}")
        print(f"Best params: {best_params}")

        # 2. Validate on out-of-sample data
        print(f"Validating on out-of-sample data ({len(out_sample_data)} bars)...")
        out_sample_results = backtest_manager.run_backtest_on_data(
            strategy_name=strategy_name,
            klines_data=out_sample_data,
            params=best_params
        )
        
        out_sample_metric = 0
        if out_sample_results.total > 0 and not out_sample_results.error:
            out_sample_metric = out_sample_results.sharpe_ratio
        
        print(f"Out-of-sample metric: {out_sample_metric:.4f}")

        wfo_results.append({
            'window': i + 1,
            'in_sample_metric': in_sample_value,
            'out_sample_metric': out_sample_metric,
            'best_params': best_params,
            'out_sample_trades': out_sample_results.total,
            'out_sample_pnl': out_sample_results.net_pnl
        })

    print("---
WFO Complete.")
    return pd.DataFrame(wfo_results)
