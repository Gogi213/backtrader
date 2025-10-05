"""
Optuna-based Parameter Optimizer for Trading Strategies

This module provides functionality to optimize trading strategy parameters
using Optuna framework with various optimization objectives.

Author: HFT System
"""
import os
import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Tuple
from datetime import datetime
import json
import warnings

from ..data.backtest_engine import run_vectorized_klines_backtest
from ..strategies.strategy_registry import StrategyRegistry


class StrategyOptimizer:
    """
    Optimizer for trading strategies using Optuna framework
    
    Supports multiple optimization objectives:
    - Sharpe Ratio
    - Net Profit
    - Profit Factor
    - Win Rate
    - Custom composite objectives
    """
    
    def __init__(self,
                 strategy_name: str,
                 data_path: str,
                 symbol: str = 'BTCUSDT',
                 study_name: Optional[str] = None,
                 direction: str = 'maximize',
                 storage: Optional[str] = None):
        """
        Initialize the optimizer
        
        Args:
            strategy_name: Name of the strategy to optimize
            data_path: Path to the CSV data file
            symbol: Trading symbol
            study_name: Name for the Optuna study (auto-generated if None)
            direction: Optimization direction ('maximize' or 'minimize')
            storage: Database URL for persistent storage (None for in-memory)
        """
        self.strategy_name = strategy_name
        self.data_path = data_path
        self.symbol = symbol
        
        # Validate strategy exists
        strategy_class = StrategyRegistry.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry")
        
        self.strategy_class = strategy_class
        self.param_space = strategy_class.get_param_space()
        
        # Generate study name if not provided
        if not study_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"{strategy_name}_{symbol}_{timestamp}"
        
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        
        # Initialize study
        self.study = None
        self.best_params = None
        self.best_value = None
        self.optimization_history = []
        
    def create_objective_function(self, 
                                 objective_metric: str = 'sharpe_ratio',
                                 custom_objective: Optional[Callable] = None,
                                 min_trades: int = 10,
                                 max_drawdown_threshold: float = 50.0) -> Callable:
        """
        Create the objective function for optimization
        
        Args:
            objective_metric: Metric to optimize ('sharpe_ratio', 'net_pnl', 'profit_factor', 'win_rate')
            custom_objective: Custom objective function (overrides objective_metric)
            min_trades: Minimum number of trades required
            max_drawdown_threshold: Maximum allowed drawdown percentage
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            try:
                # Suggest parameters based on strategy's parameter space
                params = {}
                for param_name, (param_type, *bounds) in self.param_space.items():
                    if param_type == 'float':
                        if len(bounds) == 2:
                            params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1], step=bounds[2])
                    elif param_type == 'int':
                        if len(bounds) == 2:
                            params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1])
                        else:
                            params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1], step=bounds[2])
                    elif param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, bounds)
                
                # Run backtest with suggested parameters
                results = run_vectorized_klines_backtest(
                    csv_path=self.data_path,
                    symbol=self.symbol,
                    strategy_name=self.strategy_name,
                    strategy_params=params
                )
                
                # Check if backtest failed
                if 'error' in results:
                    print(f"Trial failed: {results['error']}")
                    return -float('inf') if self.direction == 'maximize' else float('inf')
                
                # Extract metrics
                total_trades = results.get('total', 0)
                max_drawdown = abs(results.get('max_drawdown', 0))
                
                # Apply constraints
                if total_trades < min_trades:
                    penalty = min_trades - total_trades
                    return -penalty if self.direction == 'maximize' else penalty
                
                if max_drawdown > max_drawdown_threshold:
                    penalty = max_drawdown - max_drawdown_threshold
                    return -penalty if self.direction == 'maximize' else penalty
                
                # Return objective value
                if custom_objective:
                    return custom_objective(results)
                elif objective_metric in results:
                    return results[objective_metric]
                else:
                    raise ValueError(f"Metric '{objective_metric}' not found in backtest results")
                    
            except Exception as e:
                print(f"Trial error: {e}")
                return -float('inf') if self.direction == 'maximize' else float('inf')
        
        return objective
    
    def optimize(self,
                 n_trials: int = 100,
                 objective_metric: str = 'sharpe_ratio',
                 custom_objective: Optional[Callable] = None,
                 min_trades: int = 10,
                 max_drawdown_threshold: float = 50.0,
                 timeout: Optional[float] = None,
                 sampler: Optional[optuna.samplers.BaseSampler] = None,
                 pruner: Optional[optuna.pruners.BasePruner] = None) -> Dict[str, Any]:
        """
        Run the optimization
        
        Args:
            n_trials: Number of optimization trials
            objective_metric: Metric to optimize
            custom_objective: Custom objective function
            min_trades: Minimum number of trades required
            max_drawdown_threshold: Maximum allowed drawdown percentage
            timeout: Time limit in seconds
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: MedianPruner)
            
        Returns:
            Dictionary with optimization results
        """
        print(f"Starting optimization for {self.strategy_name} on {self.symbol}")
        print(f"Data: {self.data_path}")
        print(f"Objective: {objective_metric}")
        print(f"Trials: {n_trials}")
        
        # Create objective function
        objective = self.create_objective_function(
            objective_metric=objective_metric,
            custom_objective=custom_objective,
            min_trades=min_trades,
            max_drawdown_threshold=max_drawdown_threshold
        )
        
        # Set default sampler and pruner with advanced settings
        if sampler is None:
            # Use TPESampler with multivariate mode for better parameter relationships
            sampler = optuna.samplers.TPESampler(
                seed=42,
                multivariate=True,  # Consider parameter relationships
                group=True,  # Group parameters for better sampling
                n_startup_trials=10  # More trials before model kicks in
            )
        if pruner is None:
            # Use HyperbandPruner for more aggressive pruning
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,  # Minimum resource allocated to a trial
                max_resource='auto',  # Automatically determined based on n_trials
                reduction_factor=3  # Aggressive pruning
            )
        
        # Create or load study
        if self.storage:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage,
                sampler=sampler,
                pruner=pruner
            )
        else:
            study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                sampler=sampler,
                pruner=pruner
            )
        
        self.study = study
        
        # Run optimization
        start_time = datetime.now()
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        end_time = datetime.now()
        
        # Store results
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        # Calculate optimization statistics
        trials_df = study.trials_dataframe()
        successful_trials = trials_df[trials_df['state'] == 'COMPLETE']
        
        optimization_time = (end_time - start_time).total_seconds()
        
        results = {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'study_name': self.study_name,
            'objective_metric': objective_metric,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(study.trials),
            'successful_trials': len(successful_trials),
            'optimization_time_seconds': optimization_time,
            'optimization_completed_at': end_time.isoformat()
        }
        
        # Run final backtest with best parameters
        print("Running final backtest with best parameters...")
        final_results = run_vectorized_klines_backtest(
            csv_path=self.data_path,
            symbol=self.symbol,
            strategy_name=self.strategy_name,
            strategy_params=self.best_params
        )
        
        if 'error' not in final_results:
            results['final_backtest'] = final_results
            print(f"Final backtest completed with {final_results.get('total', 0)} trades")
            print(f"Sharpe Ratio: {final_results.get('sharpe_ratio', 0):.2f}")
            print(f"Net P&L: ${final_results.get('net_pnl', 0):,.2f}")
            print(f"Return: {final_results.get('net_pnl_percentage', 0):.2f}%")
        else:
            results['final_backtest_error'] = final_results['error']
            print(f"Final backtest failed: {final_results['error']}")
        
        self.optimization_history.append(results)
        
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best {objective_metric}: {self.best_value:.4f}")
        
        return results
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all optimizations run"""
        return self.optimization_history
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Get parameter importance based on the optimization study
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if not self.study:
            raise ValueError("No optimization study available. Run optimize() first.")
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")
            return {}
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to a JSON file
        
        Args:
            filepath: Path to save the results
        """
        if not self.optimization_history:
            raise ValueError("No optimization results to save")
        
        results = {
            'optimization_history': self.optimization_history,
            'parameter_importance': self.get_parameter_importance(),
            'saved_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load optimization results from a JSON file
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Loaded results dictionary
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.optimization_history = results.get('optimization_history', [])
        
        # Restore latest optimization state
        if self.optimization_history:
            latest = self.optimization_history[-1]
            self.best_params = latest.get('best_params')
            self.best_value = latest.get('best_value')
        
        print(f"Results loaded from {filepath}")
        return results


def create_composite_objective(weights: Dict[str, float]) -> Callable:
    """
    Create a composite objective function that combines multiple metrics
    
    Args:
        weights: Dictionary mapping metric names to weights
        
    Returns:
        Composite objective function
    """
    def composite_objective(results: Dict[str, Any]) -> float:
        score = 0.0
        for metric, weight in weights.items():
            if metric in results:
                score += results[metric] * weight
        return score
    
    return composite_objective


# Convenience function for quick optimization
def quick_optimize(strategy_name: str,
                  data_path: str,
                  symbol: str = 'BTCUSDT',
                  n_trials: int = 50,
                  objective_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
    """
    Quick optimization with default parameters
    
    Args:
        strategy_name: Name of the strategy to optimize
        data_path: Path to the CSV data file
        symbol: Trading symbol
        n_trials: Number of trials
        objective_metric: Metric to optimize
        
    Returns:
        Optimization results
    """
    optimizer = StrategyOptimizer(
        strategy_name=strategy_name,
        data_path=data_path,
        symbol=symbol
    )
    
    return optimizer.optimize(
        n_trials=n_trials,
        objective_metric=objective_metric
    )