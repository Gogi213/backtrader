"""
Backtest Manager - Unified Interface for Running Backtests
Provides a clean interface for executing backtests with various strategies

Author: HFT System
"""
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from .backtest_config import BacktestConfig
from .backtest_results import BacktestResults
from ..data.backtest_engine import run_vectorized_klines_backtest
from ..strategies.strategy_registry import StrategyRegistry


class BacktestManager:
    """
    Unified manager for running backtests with different strategies
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.results_cache = {}
    
    def run_backtest(self, strategy_name: str, data_path: str, 
                    params: Optional[Dict[str, Any]] = None,
                    sample_size: Optional[int] = None) -> BacktestResults:
        """
        Run a backtest for the specified strategy
        
        Args:
            strategy_name: Name of the strategy to test
            data_path: Path to the data file
            params: Strategy parameters (optional)
            sample_size: Sample size for testing (optional)
            
        Returns:
            BacktestResults object with test results
        """
        # Get strategy class
        strategy_class = StrategyRegistry.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry")
        
        # Get default parameters if none provided
        if params is None:
            params = strategy_class.get_default_params()
        
        # Create strategy instance
        strategy = strategy_class(symbol=self.config.symbol, **params)
        
        # Run backtest
        raw_results = run_vectorized_klines_backtest(
            strategy, data_path, sample_size
        )
        
        # Create results object
        results = BacktestResults(raw_results)
        results.config = self.config
        results.strategy_name = strategy_name
        results.data_path = data_path
        results.parameters = params
        results.timestamp = datetime.now()
        
        return results
    
    def run_multiple_backtests(self, strategy_name: str, data_path: str,
                             param_sets: List[Dict[str, Any]]) -> List[BacktestResults]:
        """
        Run multiple backtests with different parameter sets
        
        Args:
            strategy_name: Name of the strategy to test
            data_path: Path to the data file
            param_sets: List of parameter dictionaries
            
        Returns:
            List of BacktestResults objects
        """
        results = []
        for params in param_sets:
            try:
                result = self.run_backtest(strategy_name, data_path, params)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = BacktestResults({
                    'error': str(e),
                    'strategy_name': strategy_name,
                    'parameters': params
                })
                results.append(error_result)
        
        return results
    
    def compare_strategies(self, strategy_names: List[str], data_path: str,
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, BacktestResults]:
        """
        Compare multiple strategies on the same data
        
        Args:
            strategy_names: List of strategy names to compare
            data_path: Path to the data file
            params: Parameters to use for all strategies (optional)
            
        Returns:
            Dictionary mapping strategy names to their results
        """
        results = {}
        for strategy_name in strategy_names:
            try:
                result = self.run_backtest(strategy_name, data_path, params)
                results[strategy_name] = result
            except Exception as e:
                # Create error result
                error_result = BacktestResults({
                    'error': str(e),
                    'strategy_name': strategy_name
                })
                results[strategy_name] = error_result
        
        return results