"""
Strategy Factory
Factory pattern for creating strategy instances with flexible configuration

Author: HFT System
"""
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry


class StrategyFactory:
    """
    Factory for creating strategy instances

    Provides multiple ways to create strategies:
    - With custom parameters
    - With default parameters
    - With parameter validation
    """

    @staticmethod
    def create(strategy_name: str, symbol: str, **params) -> BaseStrategy:
        """
        Create strategy instance with custom parameters

        Args:
            strategy_name: Name of registered strategy
            symbol: Trading symbol (e.g., 'BTCUSDT')
            **params: Strategy-specific parameters

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy not found

        Example:
            strategy = StrategyFactory.create(
                'bollinger',
                'BTCUSDT',
                period=50,
                std_dev=2.0,
                stop_loss_pct=0.5
            )
        """
        strategy_class = StrategyRegistry.get(strategy_name)

        if strategy_class is None:
            available = StrategyRegistry.list_strategies()
            raise ValueError(
                f"Strategy '{strategy_name}' not found. "
                f"Available strategies: {available}"
            )

        return strategy_class(symbol=symbol, **params)

    @staticmethod
    def create_with_defaults(strategy_name: str, symbol: str) -> BaseStrategy:
        """
        Create strategy instance with default parameters

        Args:
            strategy_name: Name of registered strategy
            symbol: Trading symbol

        Returns:
            Strategy instance with default parameters

        Raises:
            ValueError: If strategy not found

        Example:
            strategy = StrategyFactory.create_with_defaults('bollinger', 'BTCUSDT')
        """
        strategy_class = StrategyRegistry.get(strategy_name)

        if strategy_class is None:
            available = StrategyRegistry.list_strategies()
            raise ValueError(
                f"Strategy '{strategy_name}' not found. "
                f"Available strategies: {available}"
            )

        default_params = strategy_class.get_default_params()
        return strategy_class(symbol=symbol, **default_params)

    @staticmethod
    def list_available_strategies() -> list:
        """
        List all available strategy names

        Returns:
            List of strategy names
        """
        return StrategyRegistry.list_strategies()

    @staticmethod
    def get_strategy_info(strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a strategy

        Args:
            strategy_name: Strategy name

        Returns:
            Dictionary with strategy information or None if not found
            {
                'name': str,
                'class': Type[BaseStrategy],
                'default_params': Dict,
                'param_space': Dict
            }
        """
        strategy_class = StrategyRegistry.get(strategy_name)

        if strategy_class is None:
            return None

        return {
            'name': strategy_name,
            'class': strategy_class,
            'default_params': strategy_class.get_default_params(),
            'param_space': strategy_class.get_param_space()
        }

    @staticmethod
    def validate_params(strategy_name: str, params: Dict[str, Any]) -> bool:
        """
        Validate parameters against strategy's parameter space (future use)

        Args:
            strategy_name: Strategy name
            params: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        strategy_class = StrategyRegistry.get(strategy_name)

        if strategy_class is None:
            return False

        # Basic validation: check if all params exist in param_space
        param_space = strategy_class.get_param_space()
        return all(key in param_space for key in params.keys())
