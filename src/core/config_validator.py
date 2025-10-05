"""
Configuration Validator for Backtesting

This module provides validation functionality for backtest configurations.

Author: HFT System
"""
from typing import Dict, List, Tuple, Any, Optional
import os
from dataclasses import dataclass

from .backtest_config import BacktestConfig


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, error: str) -> None:
        """Add an error message"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message"""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0


class ConfigValidator:
    """
    Validator for backtest configurations
    
    This class provides comprehensive validation of backtest configurations
    with detailed error and warning messages.
    """
    
    def __init__(self):
        """Initialize the validator"""
        self.required_fields = ['strategy_name']
        self.numeric_fields = [
            'initial_capital', 'commission_pct', 'position_size_dollars'
        ]
        self.positive_numeric_fields = [
            'initial_capital', 'commission_pct', 'position_size_dollars'
        ]
    
    def validate(self, config: BacktestConfig) -> ValidationResult:
        """
        Validate a backtest configuration
        
        Args:
            config: BacktestConfig to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Validate required fields
        self._validate_required_fields(config, result)
        
        # Validate strategy
        self._validate_strategy(config, result)
        
        # Validate data source
        self._validate_data_source(config, result)
        
        # Validate trading parameters
        self._validate_trading_params(config, result)
        
        # Validate performance parameters
        self._validate_performance_params(config, result)
        
        # Check for potential issues
        self._check_potential_issues(config, result)
        
        return result
    
    def _validate_required_fields(self, config: BacktestConfig, result: ValidationResult) -> None:
        """Validate required fields"""
        for field in self.required_fields:
            if not getattr(config, field, None):
                result.add_error(f"{field} is required")
    
    def _validate_strategy(self, config: BacktestConfig, result: ValidationResult) -> None:
        """Validate strategy configuration"""
        # Check if strategy exists
        try:
            from ..strategies.strategy_registry import StrategyRegistry
            strategy_class = StrategyRegistry.get(config.strategy_name)
            if not strategy_class:
                result.add_error(f"Strategy '{config.strategy_name}' not found in registry")
                return
            
            # Get default parameters if none provided
            if not config.strategy_params and hasattr(strategy_class, 'get_default_params'):
                config.strategy_params = strategy_class.get_default_params()
            
            # Validate strategy parameters
            if hasattr(strategy_class, 'get_param_space'):
                param_space = strategy_class.get_param_space()
                for param_name, param_spec in param_space.items():
                    if param_name in config.strategy_params:
                        self._validate_strategy_param(
                            param_name,
                            config.strategy_params[param_name],
                            param_spec,
                            result
                        )
        except Exception as e:
            result.add_error(f"Error validating strategy: {str(e)}")
    
    def _validate_strategy_param(self, param_name: str, value: Any, param_spec: tuple, result: ValidationResult) -> None:
        """Validate a single strategy parameter"""
        if not param_spec:
            return
        
        param_type = param_spec[0]
        
        # Check type
        if param_type == 'float':
            if not isinstance(value, (int, float)):
                result.add_error(f"Strategy parameter '{param_name}' must be a number")
                return
        elif param_type == 'int':
            if not isinstance(value, int):
                result.add_error(f"Strategy parameter '{param_name}' must be an integer")
                return
        elif param_type == 'categorical':
            valid_values = param_spec[1] if len(param_spec) > 1 else []
            if value not in valid_values:
                result.add_error(f"Strategy parameter '{param_name}' must be one of {valid_values}")
                return
        
        # Check bounds if specified - use warnings instead of errors
        if len(param_spec) > 1 and param_type in ['float', 'int']:
            min_val, max_val = param_spec[1], param_spec[2] if len(param_spec) > 2 else None
            
            if min_val is not None and value < min_val:
                result.add_warning(f"Strategy parameter '{param_name}' ({value}) is below recommended minimum ({min_val})")
            
            if max_val is not None and value > max_val:
                result.add_warning(f"Strategy parameter '{param_name}' ({value}) is above recommended maximum ({max_val})")
    
    def _validate_data_source(self, config: BacktestConfig, result: ValidationResult) -> None:
        """Validate data source configuration"""
        if config.data_source == "csv":
            if not config.data_path:
                result.add_error("Data path is required for CSV data source")
            elif not os.path.exists(config.data_path):
                result.add_error(f"Data file not found: {config.data_path}")
            elif not config.data_path.endswith('.csv'):
                result.add_warning("Data file should have .csv extension")
        elif config.data_source not in ["csv", "numpy", "dataframe"]:
            result.add_error(f"Invalid data source: {config.data_source}")
    
    def _validate_trading_params(self, config: BacktestConfig, result: ValidationResult) -> None:
        """Validate trading parameters"""
        # Initial capital
        if config.initial_capital <= 0:
            result.add_error("Initial capital must be positive")
        elif config.initial_capital < 1000:
            result.add_warning("Initial capital is less than $1000")
        
        # Commission
        if config.commission_pct < 0:
            result.add_error("Commission percentage cannot be negative")
        elif config.commission_pct > 1:
            result.add_warning("Commission percentage is very high (>1%)")
        
        # Position size
        if config.position_size_dollars <= 0:
            result.add_error("Position size must be positive")
        elif config.position_size_dollars > config.initial_capital:
            result.add_error("Position size cannot exceed initial capital")
        elif config.position_size_dollars > config.initial_capital * 0.5:
            result.add_warning("Position size is more than 50% of initial capital")
    
    def _validate_performance_params(self, config: BacktestConfig, result: ValidationResult) -> None:
        """Validate performance parameters"""
        # Max klines
        if config.max_klines is not None:
            if config.max_klines <= 0:
                result.add_error("Max klines must be positive")
            elif config.max_klines < 1000:
                result.add_warning("Max klines is less than 1000")
        
        # Parallel jobs
        if config.parallel_jobs < -1:
            result.add_error("Parallel jobs must be -1 (all cores) or a positive number")
        elif config.parallel_jobs == 0:
            result.add_warning("Parallel jobs is 0 (sequential execution)")
    
    def _check_potential_issues(self, config: BacktestConfig, result: ValidationResult) -> None:
        """Check for potential issues and add warnings"""
        # Check for very large datasets
        if config.max_klines is None:
            result.add_warning("No limit on klines - may cause memory issues with large datasets")
        
        # Check for very small datasets
        if config.max_klines is not None and config.max_klines < 100:
            result.add_warning("Very small dataset (< 100 klines) - results may not be reliable")
        
        # Check for high commission
        if config.commission_pct > 0.2:
            result.add_warning("High commission (>0.2%) - may significantly impact results")
        
        # Check for turbo mode with small datasets
        if config.enable_turbo_mode and config.max_klines is not None and config.max_klines < 10000:
            result.add_warning("Turbo mode enabled for small dataset - may not provide benefits")
    
    def validate_batch(self, configs: List[BacktestConfig]) -> ValidationResult:
        """
        Validate a batch of configurations
        
        Args:
            configs: List of BacktestConfig objects to validate
            
        Returns:
            ValidationResult with combined errors and warnings
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        for i, config in enumerate(configs):
            config_result = self.validate(config)
            
            # Add errors with index
            for error in config_result.errors:
                result.add_error(f"Config {i+1}: {error}")
            
            # Add warnings with index
            for warning in config_result.warnings:
                result.add_warning(f"Config {i+1}: {warning}")
        
        return result