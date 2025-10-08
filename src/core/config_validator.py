"""
Configuration Validator - Validation for Backtest and Optimization Configurations
Provides validation utilities for configuration parameters

Author: HFT System
"""
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    def add_error(self, error: str):
        """Add an error to the validation result"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the validation result"""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False


class ConfigValidator:
    """Validator for backtest and optimization configurations"""
    
    @staticmethod
    def validate_data_path(data_path: str) -> ValidationResult:
        """
        Validate data file path
        
        Args:
            data_path: Path to data file
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(True, [], [])
        
        if not data_path:
            result.add_error("Data path is required")
            return result
        
        if not os.path.exists(data_path):
            result.add_error(f"Data file not found: {data_path}")
        
        # Check file extension
        valid_extensions = ['.csv', '.pkl', '.parquet']
        file_ext = os.path.splitext(data_path)[1].lower()
        if file_ext not in valid_extensions:
            result.add_warning(f"Data file extension '{file_ext}' may not be supported")
        
        return result
    
    @staticmethod
    def validate_strategy_name(strategy_name: str, strategy_registry) -> ValidationResult:
        """
        Validate strategy name against registry
        
        Args:
            strategy_name: Name of the strategy
            strategy_registry: Strategy registry object
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(True, [], [])
        
        if not strategy_name:
            result.add_error("Strategy name is required")
            return result
        
        available_strategies = strategy_registry.list_strategies()
        if strategy_name not in available_strategies:
            result.add_error(f"Strategy '{strategy_name}' not found. Available strategies: {', '.join(available_strategies)}")
        
        return result
    
    @staticmethod
    def validate_parameters(params: Dict[str, Any], param_space: Dict[str, Any]) -> ValidationResult:
        """
        Validate strategy parameters against parameter space
        
        Args:
            params: Parameters to validate
            param_space: Parameter space definition
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(True, [], [])
        
        if not params:
            result.add_warning("No parameters provided, using defaults")
            return result
        
        # Check for unknown parameters
        for param_name in params:
            if param_name not in param_space:
                result.add_warning(f"Unknown parameter: {param_name}")
        
        # Check parameter types and ranges
        for param_name, (param_type, *bounds) in param_space.items():
            if param_name in params:
                value = params[param_name]
                
                # Type validation
                if param_type == 'float':
                    if not isinstance(value, (int, float)):
                        result.add_error(f"Parameter '{param_name}' must be a number, got {type(value).__name__}")
                    elif len(bounds) >= 2 and not (bounds[0] <= value <= bounds[1]):
                        result.add_error(f"Parameter '{param_name}' value {value} out of range [{bounds[0]}, {bounds[1]}]")
                
                elif param_type == 'int':
                    if not isinstance(value, int):
                        result.add_error(f"Parameter '{param_name}' must be an integer, got {type(value).__name__}")
                    elif len(bounds) >= 2 and not (bounds[0] <= value <= bounds[1]):
                        result.add_error(f"Parameter '{param_name}' value {value} out of range [{bounds[0]}, {bounds[1]}]")
                
                elif param_type == 'categorical':
                    if value not in bounds:
                        result.add_error(f"Parameter '{param_name}' value '{value}' not in allowed values: {bounds}")
        
        return result
    
    @staticmethod
    def validate_optimization_params(n_trials: int, timeout: Optional[float], n_jobs: int) -> ValidationResult:
        """
        Validate optimization parameters
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(True, [], [])
        
        if n_trials <= 0:
            result.add_error("Number of trials must be positive")
        elif n_trials > 10000:
            result.add_warning(f"Large number of trials ({n_trials}) may take a very long time")
        
        if timeout is not None and timeout <= 0:
            result.add_error("Timeout must be positive")
        
        if n_jobs < -1:
            result.add_error("Number of jobs must be -1 or positive")
        elif n_jobs > 8:
            result.add_warning(f"High number of parallel jobs ({n_jobs}) may cause system instability")
        
        return result
    
    @staticmethod
    def validate_backtest_config(config) -> ValidationResult:
        """
        Validate complete backtest configuration
        
        Args:
            config: BacktestConfig object
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(True, [], [])
        
        # Validate data path
        data_validation = ConfigValidator.validate_data_path(config.data_path)
        result.merge(data_validation)
        
        # Validate strategy name
        from ..strategies.strategy_registry import StrategyRegistry
        strategy_validation = ConfigValidator.validate_strategy_name(
            config.strategy_name, StrategyRegistry
        )
        result.merge(strategy_validation)
        
        # Validate financial parameters
        if config.initial_capital <= 0:
            result.add_error("Initial capital must be positive")
        
        if config.commission_pct < 0:
            result.add_error("Commission percentage cannot be negative")
        elif config.commission_pct > 0.1:  # 10%
            result.add_warning("Very high commission percentage")
        
        if config.position_size_dollars <= 0:
            result.add_error("Position size must be positive")
        elif config.position_size_dollars > config.initial_capital:
            result.add_error("Position size cannot exceed initial capital")
        
        return result