"""
Backtest Manager for Unified Backtesting

This module provides a unified manager for running backtests with different
strategies and configurations.

Author: HFT System
"""
import os
import time
from typing import Dict, List, Optional, Any, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .backtest_config import BacktestConfig
from .backtest_results import BacktestResults
from .config_validator import ConfigValidator, ValidationResult


class BacktestManager:
    """
    Unified manager for running backtests
    
    This class provides a single interface for running backtests with
    different strategies and configurations, handling validation,
    execution, and result formatting.
    """
    
    def __init__(self, validator: Optional[ConfigValidator] = None):
        """
        Initialize the backtest manager
        
        Args:
            validator: Optional validator for configurations
        """
        self.validator = validator or ConfigValidator()
        self._execution_stats = {
            'total_backtests': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'total_execution_time': 0.0
        }
    
    def validate_config(self, config: BacktestConfig) -> ValidationResult:
        """
        Validate a backtest configuration
        
        Args:
            config: BacktestConfig to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        return self.validator.validate(config)
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResults:
        """
        Run a single backtest with the given configuration
        
        Args:
            config: BacktestConfig with parameters
            
        Returns:
            BacktestResults with the backtest output
        """
        print("UNIFIED BACKTEST MANAGER: Running backtest with new unified system")
        
        # Validate configuration
        print("UNIFIED BACKTEST MANAGER: Validating configuration with ConfigValidator")
        validation_result = self.validate_config(config)
        if not validation_result.is_valid:
            return BacktestResults.from_error(
                f"Configuration validation failed: {'; '.join(validation_result.errors)}",
                strategy_name=config.strategy_name,
                symbol=config.symbol
            )
        
        # Log warnings if any
        if validation_result.has_warnings():
            print(f"Backtest warnings: {'; '.join(validation_result.warnings)}")
        
        # Update execution stats
        self._execution_stats['total_backtests'] += 1
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the actual backtest
            print("UNIFIED BACKTEST MANAGER: Executing backtest with vectorized engine")
            raw_results = self._execute_backtest(config)
            
            # Create unified results
            results = BacktestResults(raw_results)
            
            # Update execution stats
            self._execution_stats['successful_backtests'] += 1
            
            print("UNIFIED BACKTEST MANAGER: Backtest completed successfully")
            return results
            
        except Exception as e:
            # Update execution stats
            self._execution_stats['failed_backtests'] += 1
            
            print(f"UNIFIED BACKTEST MANAGER: Backtest failed with error: {str(e)}")
            # Return error results
            return BacktestResults.from_error(
                f"Backtest execution failed: {str(e)}",
                strategy_name=config.strategy_name,
                symbol=config.symbol
            )
        
        finally:
            # Update total execution time
            execution_time = time.time() - start_time
            self._execution_stats['total_execution_time'] += execution_time
    
    def run_batch_backtest(self, configs: List[BacktestConfig], 
                          parallel: bool = True, 
                          max_workers: Optional[int] = None) -> List[BacktestResults]:
        """
        Run multiple backtests with the given configurations
        
        Args:
            configs: List of BacktestConfig objects
            parallel: Whether to run backtests in parallel
            max_workers: Maximum number of parallel workers (default: CPU count)
            
        Returns:
            List of BacktestResults objects
        """
        if not configs:
            return []
        
        # Validate all configurations first
        batch_validation = self.validator.validate_batch(configs)
        if not batch_validation.is_valid:
            return [BacktestResults.from_error(
                f"Batch validation failed: {'; '.join(batch_validation.errors)}",
                strategy_name="batch",
                symbol="BATCH"
            )]
        
        # Log warnings if any
        if batch_validation.has_warnings():
            print(f"Batch backtest warnings: {'; '.join(batch_validation.warnings)}")
        
        # Run backtests
        if parallel and len(configs) > 1:
            return self._run_parallel_backtests(configs, max_workers)
        else:
            return [self.run_backtest(config) for config in configs]
    
    def _execute_backtest(self, config: BacktestConfig) -> Dict[str, Any]:
        """
        Execute the actual backtest using the existing backtest engine
        
        Args:
            config: BacktestConfig with parameters
            
        Returns:
            Raw results dictionary from backtest engine
        """
        # Import here to avoid circular imports
        from ..data.backtest_engine import run_vectorized_klines_backtest
        
        # Prepare parameters for the existing backtest engine
        strategy_params = config.strategy_params.copy()
        
        # Remove conflicting parameters
        strategy_params.pop('initial_capital', None)
        strategy_params.pop('commission_pct', None)
        
        # Run the backtest
        results = run_vectorized_klines_backtest(
            csv_path=config.data_path,
            symbol=config.symbol,
            strategy_name=config.strategy_name,
            strategy_params=strategy_params,
            initial_capital=config.initial_capital,
            commission_pct=config.commission_pct,
            max_klines=config.max_klines
        )
        
        return results
    
    def _run_parallel_backtests(self, configs: List[BacktestConfig], 
                               max_workers: Optional[int] = None) -> List[BacktestResults]:
        """
        Run backtests in parallel using ThreadPoolExecutor
        
        Args:
            configs: List of BacktestConfig objects
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of BacktestResults objects
        """
        results = [None] * len(configs)
        
        # Determine number of workers
        if max_workers is None:
            import multiprocessing
            max_workers = multiprocessing.cpu_count()
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all backtest tasks
            future_to_index = {
                executor.submit(self.run_backtest, config): i 
                for i, config in enumerate(configs)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = BacktestResults.from_error(
                        f"Parallel backtest failed: {str(e)}",
                        strategy_name=configs[index].strategy_name,
                        symbol=configs[index].symbol
                    )
        
        return results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics
        
        Returns:
            Dictionary with execution statistics
        """
        stats = self._execution_stats.copy()
        
        # Calculate derived statistics
        if stats['total_backtests'] > 0:
            stats['success_rate'] = stats['successful_backtests'] / stats['total_backtests']
            stats['failure_rate'] = stats['failed_backtests'] / stats['total_backtests']
            stats['average_execution_time'] = stats['total_execution_time'] / stats['total_backtests']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['average_execution_time'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset execution statistics"""
        self._execution_stats = {
            'total_backtests': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'total_execution_time': 0.0
        }
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategies
        
        Returns:
            List of strategy names
        """
        try:
            from ..strategies.strategy_registry import StrategyRegistry
            return StrategyRegistry.list_strategies()
        except Exception:
            return []
    
    def get_strategy_params(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with default parameters
        """
        try:
            from ..strategies.strategy_registry import StrategyRegistry
            strategy_class = StrategyRegistry.get(strategy_name)
            if strategy_class and hasattr(strategy_class, 'get_default_params'):
                return strategy_class.get_default_params()
        except Exception:
            pass
        return {}
    
    def get_strategy_param_space(self, strategy_name: str) -> Dict[str, tuple]:
        """
        Get parameter space for a strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with parameter specifications
        """
        try:
            from ..strategies.strategy_registry import StrategyRegistry
            strategy_class = StrategyRegistry.get(strategy_name)
            if strategy_class and hasattr(strategy_class, 'get_param_space'):
                return strategy_class.get_param_space()
        except Exception:
            pass
        return {}
    
    def create_config_from_template(self, strategy_name: str, 
                                   symbol: str = 'BTCUSDT',
                                   data_path: Optional[str] = None) -> BacktestConfig:
        """
        Create a configuration from a strategy template
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            data_path: Path to data file
            
        Returns:
            BacktestConfig with default parameters
        """
        config = BacktestConfig(
            strategy_name=strategy_name,
            symbol=symbol,
            data_path=data_path
        )
        
        return config