"""
Optimization Manager for Unified Strategy Optimization

This module provides a unified manager for running strategy parameter optimization
with different strategies and configurations.

Author: HFT System
"""
import os
import time
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from datetime import datetime

from .backtest_config import BacktestConfig
from .backtest_results import BacktestResults
from .config_validator import ConfigValidator, ValidationResult


class OptimizationConfig:
    """Configuration for strategy optimization"""
    
    def __init__(self,
                 strategy_name: str,
                 data_path: str,
                 symbol: str = 'BTCUSDT',
                 n_trials: int = 100,
                 objective_metric: str = 'sharpe_ratio',
                 min_trades: int = 10,
                 max_drawdown_threshold: float = 50.0,
                 timeout: Optional[float] = 600,  # 10 minutes default from GUI
                 n_jobs: int = -1,
                 use_adaptive: bool = True,
                 study_name: Optional[str] = None,
                 direction: str = 'maximize',
                 storage: Optional[str] = None,
                 cache_dir: str = "optimization_cache",
                 backtest_config: Optional[BacktestConfig] = None):
        """
        Initialize optimization configuration
        
        Args:
            strategy_name: Name of the strategy to optimize
            data_path: Path to the CSV data file
            symbol: Trading symbol
            n_trials: Number of optimization trials
            objective_metric: Metric to optimize
            min_trades: Minimum number of trades required
            max_drawdown_threshold: Maximum allowed drawdown percentage
            timeout: Time limit in seconds
            n_jobs: Number of parallel jobs (-1 for all cores)
            use_adaptive: Use adaptive evaluation with reduced data for early trials
            study_name: Name for the Optuna study (auto-generated if None)
            direction: Optimization direction ('maximize' or 'minimize')
            storage: Database URL for persistent storage (None for in-memory)
            cache_dir: Directory for caching data and results
        """
        self.strategy_name = strategy_name
        self.data_path = data_path
        self.symbol = symbol
        self.n_trials = n_trials
        self.objective_metric = objective_metric
        self.min_trades = min_trades
        self.max_drawdown_threshold = max_drawdown_threshold
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.use_adaptive = use_adaptive
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        self.cache_dir = cache_dir
        
        # Use provided backtest config or create a default one
        if backtest_config:
            self.backtest_config = backtest_config
        else:
            self.backtest_config = BacktestConfig(
                strategy_name=strategy_name,
                symbol=symbol,
                data_path=data_path,
                initial_capital=10000.0,
                commission_pct=0.05,
                position_size_dollars=1000.0
            )
        
        # Generate study name if not provided
        if not self.study_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.study_name = f"{strategy_name}_{symbol}_{timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'data_path': self.data_path,
            'symbol': self.symbol,
            'n_trials': self.n_trials,
            'objective_metric': self.objective_metric,
            'min_trades': self.min_trades,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'timeout': self.timeout,
            'n_jobs': self.n_jobs,
            'use_adaptive': self.use_adaptive,
            'study_name': self.study_name,
            'direction': self.direction,
            'storage': self.storage,
            'cache_dir': self.cache_dir,
            'backtest_config': self.backtest_config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizationConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def with_gui_defaults(cls, strategy_name: str, data_path: str, symbol: str = 'BTCUSDT') -> 'OptimizationConfig':
        """
        Create configuration with GUI default settings
        
        Args:
            strategy_name: Name of the strategy to optimize
            data_path: Path to the CSV data file
            symbol: Trading symbol
            
        Returns:
            OptimizationConfig with GUI default parameters
        """
        try:
            from ..gui.tabs.tab_optimization import OptimizationTab
            defaults = OptimizationTab.get_default_optimization_config()
            
            return cls(
                strategy_name=strategy_name,
                data_path=data_path,
                symbol=symbol,
                n_trials=defaults.get('n_trials', 100),
                objective_metric=defaults.get('objective_metric', 'sharpe_ratio'),
                min_trades=defaults.get('min_trades', 10),
                max_drawdown_threshold=defaults.get('max_drawdown_threshold', 50.0),
                timeout=defaults.get('timeout', 600),
                direction=defaults.get('direction', 'maximize'),
                n_jobs=defaults.get('n_jobs', -1),
                use_adaptive=defaults.get('use_adaptive', True),
                initial_capital=defaults.get('initial_capital', 10000.0),
                position_size=defaults.get('position_size', 1000.0),
                commission_pct=defaults.get('commission_pct', 0.05)
            )
        except ImportError:
            # Fallback to hardcoded defaults if GUI not available
            return cls(
                strategy_name=strategy_name,
                data_path=data_path,
                symbol=symbol,
                n_trials=100,
                objective_metric='sharpe_ratio',
                min_trades=10,
                max_drawdown_threshold=50.0,
                timeout=600,
                direction='maximize',
                n_jobs=-1,
                use_adaptive=True,
                initial_capital=10000.0,
                position_size=1000.0,
                commission_pct=0.05
            )


class OptimizationResults:
    """Results of strategy optimization"""
    
    def __init__(self, raw_results: Dict[str, Any]):
        """
        Initialize optimization results
        
        Args:
            raw_results: Raw results from optimizer
        """
        self.data = raw_results.copy()
        self.success = 'error' not in raw_results
        self.error_message = raw_results.get('error') if not self.success else None
    
    def is_successful(self) -> bool:
        """Check if optimization was successful"""
        return self.success
    
    def get_error(self) -> Optional[str]:
        """Get error message if optimization failed"""
        return self.error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.data.copy()
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters"""
        return self.data.get('best_params', {})
    
    def get_best_value(self) -> float:
        """Get best objective value"""
        return self.data.get('best_value', 0.0)
    
    def get_final_backtest(self) -> Optional[Dict[str, Any]]:
        """Get final backtest results"""
        return self.data.get('final_backtest')
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance"""
        return self.data.get('param_importance', {})
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'n_trials': self.data.get('n_trials', 0),
            'successful_trials': self.data.get('successful_trials', 0),
            'pruned_trials': self.data.get('pruned_trials', 0),
            'optimization_time_seconds': self.data.get('optimization_time_seconds', 0.0),
            'parallel_jobs': self.data.get('parallel_jobs', 1),
            'adaptive_evaluation': self.data.get('adaptive_evaluation', False)
        }


class OptimizationManager:
    """
    Unified manager for running strategy optimization
    
    This class provides a single interface for running optimization with
    different strategies and configurations, handling validation,
    execution, and result formatting.
    """
    
    def __init__(self, validator: Optional[ConfigValidator] = None):
        """
        Initialize the optimization manager
        
        Args:
            validator: Optional validator for configurations
        """
        self.validator = validator or ConfigValidator()
        self._execution_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'total_execution_time': 0.0
        }
    
    def validate_config(self, config: OptimizationConfig) -> ValidationResult:
        """
        Validate an optimization configuration
        
        Args:
            config: OptimizationConfig to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        # Create a backtest config for validation
        backtest_config = BacktestConfig(
            strategy_name=config.strategy_name,
            symbol=config.symbol,
            data_path=config.data_path
        )
        
        # Validate the backtest config
        validation_result = self.validator.validate(backtest_config)
        
        # Add optimization-specific validation
        if config.n_trials <= 0:
            validation_result.add_error("n_trials must be positive")
        
        if config.min_trades < 0:
            validation_result.add_error("min_trades cannot be negative")
        
        if config.max_drawdown_threshold <= 0:
            validation_result.add_error("max_drawdown_threshold must be positive")
        
        if config.objective_metric not in ['sharpe_ratio', 'net_pnl', 'profit_factor', 'win_rate', 'net_pnl_percentage']:
            validation_result.add_error(f"Unknown objective_metric: {config.objective_metric}")
        
        return validation_result
    
    def run_optimization(self, config: OptimizationConfig,
                        progress_callback: Optional[Callable[[str], None]] = None) -> OptimizationResults:
        """
        Run strategy optimization with the given configuration
        
        Args:
            config: OptimizationConfig with parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            OptimizationResults with the optimization output
        """
        print("UNIFIED OPTIMIZATION MANAGER: Running optimization with new unified system")
        
        # Validate configuration
        print("UNIFIED OPTIMIZATION MANAGER: Validating configuration with ConfigValidator")
        validation_result = self.validate_config(config)
        if not validation_result.is_valid:
            return OptimizationResults({
                'error': f"Configuration validation failed: {'; '.join(validation_result.errors)}",
                'strategy_name': config.strategy_name,
                'symbol': config.symbol
            })
        
        # Log warnings if any
        if validation_result.has_warnings():
            print(f"Optimization warnings: {'; '.join(validation_result.warnings)}")
        
        # Update execution stats
        self._execution_stats['total_optimizations'] += 1
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the actual optimization
            print("UNIFIED OPTIMIZATION MANAGER: Executing optimization with FastStrategyOptimizer")
            raw_results = self._execute_optimization(config, progress_callback)
            
            # Create unified results
            results = OptimizationResults(raw_results)
            
            # Update execution stats
            if results.is_successful():
                self._execution_stats['successful_optimizations'] += 1
                print("UNIFIED OPTIMIZATION MANAGER: Optimization completed successfully")
            else:
                self._execution_stats['failed_optimizations'] += 1
                print(f"UNIFIED OPTIMIZATION MANAGER: Optimization failed: {results.get_error()}")
            
            return results
            
        except Exception as e:
            # Update execution stats
            self._execution_stats['failed_optimizations'] += 1
            
            print(f"UNIFIED OPTIMIZATION MANAGER: Optimization failed with error: {str(e)}")
            # Return error results
            return OptimizationResults({
                'error': f"Optimization execution failed: {str(e)}",
                'strategy_name': config.strategy_name,
                'symbol': config.symbol
            })
        
        finally:
            # Update total execution time
            execution_time = time.time() - start_time
            self._execution_stats['total_execution_time'] += execution_time
    
    def _execute_optimization(self, config: OptimizationConfig,
                            progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Execute the actual optimization using the existing optimizer
        
        Args:
            config: OptimizationConfig with parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            Raw results dictionary from optimizer
        """
        # Import here to avoid circular imports
        from ..optimization.fast_optimizer import FastStrategyOptimizer
        
        # Create optimizer
        optimizer = FastStrategyOptimizer(
            strategy_name=config.strategy_name,
            data_path=config.data_path,
            symbol=config.symbol,
            study_name=config.study_name,
            direction=config.direction,
            storage=config.storage,
            cache_dir=config.cache_dir,
            backtest_config=config.backtest_config
        )
        
        # Custom progress callback wrapper
        def progress_wrapper(message: str):
            print(f"OPTIMIZATION: {message}")
            if progress_callback:
                progress_callback(message)
        
        # Run optimization
        results = optimizer.optimize(
            n_trials=config.n_trials,
            objective_metric=config.objective_metric,
            min_trades=config.min_trades,
            max_drawdown_threshold=config.max_drawdown_threshold,
            timeout=config.timeout,
            n_jobs=config.n_jobs,
            use_adaptive=config.use_adaptive
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
        if stats['total_optimizations'] > 0:
            stats['success_rate'] = stats['successful_optimizations'] / stats['total_optimizations']
            stats['failure_rate'] = stats['failed_optimizations'] / stats['total_optimizations']
            stats['average_execution_time'] = stats['total_execution_time'] / stats['total_optimizations']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['average_execution_time'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset execution statistics"""
        self._execution_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
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
                                   data_path: str,
                                   symbol: str = 'BTCUSDT') -> OptimizationConfig:
        """
        Create a configuration from a strategy template
        
        Args:
            strategy_name: Name of the strategy
            data_path: Path to data file
            symbol: Trading symbol
            
        Returns:
            OptimizationConfig with default parameters
        """
        config = OptimizationConfig(
            strategy_name=strategy_name,
            data_path=data_path,
            symbol=symbol
        )
        
        return config