"""
Strategy Factory
Simplified factory for creating strategy instances

Author: HFT System
"""
from typing import Dict, Any
from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry


class StrategyFactory:
    """
    Factory for creating strategy instances

    Simplified design following YAGNI principle - one method does it all
    """

    @staticmethod
    def create(strategy_name: str, symbol: str, **params) -> BaseStrategy:
        """
        Create strategy instance with parameters

        Args:
            strategy_name: Name of registered strategy
            symbol: Trading symbol (e.g., 'BTCUSDT')
            **params: Strategy-specific parameters (optional, uses defaults if not provided)

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy not found

        Example:
            # With custom parameters
            strategy = StrategyFactory.create('bollinger', 'BTCUSDT', period=50, std_dev=2.0)

            # With default parameters
            strategy_class = StrategyRegistry.get('bollinger')
            strategy = StrategyFactory.create('bollinger', 'BTCUSDT', **strategy_class.get_default_params())
        """
        strategy_class = StrategyRegistry.get(strategy_name)

        if strategy_class is None:
            available = StrategyRegistry.list_strategies()
            raise ValueError(
                f"Strategy '{strategy_name}' not found. "
                f"Available strategies: {available}"
            )

        return strategy_class(symbol=symbol, **params)
