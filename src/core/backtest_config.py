"""
Unified Backtest Configuration

This module provides a unified configuration class for backtesting
that can be used by both CLI and GUI interfaces.

Author: HFT System
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import os
import argparse


@dataclass
class BacktestConfig:
    """
    Unified configuration for backtesting
    
    This class provides a single source of truth for all backtesting
    configuration parameters, used by both CLI and GUI interfaces.
    """
    # Strategy configuration
    strategy_name: str = 'hierarchical_mean_reversion'
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    symbol: str = 'BTCUSDT'
    data_source: str = "csv"  # csv, numpy, dataframe
    data_path: Optional[str] = None
    max_klines: Optional[int] = None  # None means no limit
    
    # Trading configuration
    initial_capital: float = 10000.0
    commission_pct: float = 0.05
    position_size_dollars: float = 1000.0
    
    # Performance configuration
    enable_turbo_mode: bool = True
    parallel_jobs: int = -1  # -1 means use all cores
    
    # Output configuration
    output_file: Optional[str] = None
    verbose: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Load default strategy parameters if not provided
        if not self.strategy_params:
            self.strategy_params = self._load_default_strategy_params()
    
    def _load_default_strategy_params(self) -> Dict[str, Any]:
        """Load default parameters for the selected strategy"""
        try:
            from ..strategies.strategy_registry import StrategyRegistry
            strategy_class = StrategyRegistry.get(self.strategy_name)
            if strategy_class and hasattr(strategy_class, 'get_default_params'):
                return strategy_class.get_default_params()
        except Exception:
            pass
        return {}
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'BacktestConfig':
        """
        Create BacktestConfig from CLI arguments
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            BacktestConfig instance
        """
        # Extract strategy parameters from args
        strategy_params = {}
        if hasattr(args, 'strategy') and args.strategy:
            try:
                from ..strategies.strategy_registry import StrategyRegistry
                strategy_class = StrategyRegistry.get(args.strategy)
                if strategy_class:
                    default_params = strategy_class.get_default_params()
                    for param_name in default_params.keys():
                        cli_name = param_name.replace('-', '_')
                        if hasattr(args, cli_name):
                            strategy_params[param_name] = getattr(args, cli_name)
            except Exception:
                pass
        
        return cls(
            strategy_name=getattr(args, 'strategy', 'hierarchical_mean_reversion'),
            strategy_params=strategy_params,
            symbol=getattr(args, 'symbol', 'BTCUSDT'),
            data_path=getattr(args, 'csv', None),
            max_klines=getattr(args, 'max_klines', None),
            initial_capital=getattr(args, 'initial_capital', 10000.0),
            commission_pct=getattr(args, 'commission_pct', 0.05),
            output_file=getattr(args, 'output', None),
            verbose=getattr(args, 'verbose', False)
        )
    
    @classmethod
    def from_gui_config(cls, config) -> 'BacktestConfig':
        """
        Create BacktestConfig from GUI configuration
        
        Args:
            config: GUI configuration object (StrategyConfig)
            
        Returns:
            BacktestConfig instance
        """
        return cls(
            strategy_name=config.strategy_name,
            strategy_params=config.strategy_params.copy(),
            symbol='BTCUSDT',  # Will be updated from dataset
            initial_capital=config.initial_capital,
            commission_pct=config.commission_pct,
            position_size_dollars=getattr(config, 'position_size_dollars', 1000.0),
            max_klines=getattr(config, 'max_ticks_gui', None)
        )
    
    def update_from_dataset(self, dataset_path: str, symbol: str) -> None:
        """
        Update configuration from dataset information
        
        Args:
            dataset_path: Path to the dataset file
            symbol: Trading symbol extracted from dataset
        """
        self.data_path = dataset_path
        self.symbol = symbol
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration parameters
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate strategy
        if not self.strategy_name:
            errors.append("Strategy name is required")
        
        # Validate data source
        if self.data_source == "csv" and not self.data_path:
            errors.append("Data path is required for CSV data source")
        elif self.data_path and not os.path.exists(self.data_path):
            errors.append(f"Data file not found: {self.data_path}")
        
        # Validate trading parameters
        if self.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if self.commission_pct < 0:
            errors.append("Commission percentage cannot be negative")
        
        if self.max_klines is not None and self.max_klines <= 0:
            errors.append("Max klines must be positive")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'strategy_name': self.strategy_name,
            'strategy_params': self.strategy_params.copy(),
            'symbol': self.symbol,
            'data_source': self.data_source,
            'data_path': self.data_path,
            'max_klines': self.max_klines,
            'initial_capital': self.initial_capital,
            'commission_pct': self.commission_pct,
            'position_size_dollars': self.position_size_dollars,
            'enable_turbo_mode': self.enable_turbo_mode,
            'parallel_jobs': self.parallel_jobs,
            'output_file': self.output_file,
            'verbose': self.verbose
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """
        Create BacktestConfig from dictionary
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            BacktestConfig instance
        """
        return cls(**config_dict)
    
    def copy(self) -> 'BacktestConfig':
        """
        Create a copy of the configuration
        
        Returns:
            New BacktestConfig instance with same parameters
        """
        return self.from_dict(self.to_dict())
    
    def update_strategy(self, strategy_name: str) -> None:
        """
        Update strategy and reset parameters to defaults
        
        Args:
            strategy_name: New strategy name
        """
        self.strategy_name = strategy_name
        self.strategy_params = self._load_default_strategy_params()
    
    def update_strategy_param(self, param_name: str, value: Any) -> None:
        """
        Update a single strategy parameter
        
        Args:
            param_name: Parameter name
            value: New parameter value
        """
        self.strategy_params[param_name] = value
    
    def get_strategy_param(self, param_name: str, default: Any = None) -> Any:
        """
        Get a strategy parameter value
        
        Args:
            param_name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        return self.strategy_params.get(param_name, default)