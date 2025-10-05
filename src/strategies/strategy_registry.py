"""
Strategy Registry
Simple decorator-based registry for strategy discovery and management

Author: HFT System
"""
from typing import Dict, List, Type, Optional
from .base_strategy import BaseStrategy


class StrategyRegistry:
    """
    Central registry for all trading strategies

    Usage:
        @StrategyRegistry.register('my_strategy')
        class MyStrategy(BaseStrategy):
            ...

        # Later retrieve:
        strategy_class = StrategyRegistry.get('my_strategy')
    """

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a strategy class

        Args:
            name: Unique name for the strategy

        Returns:
            Decorator function

        Example:
            @StrategyRegistry.register('bollinger')
            class BollingerStrategy(BaseStrategy):
                pass
        """
        def decorator(strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
            if name in cls._strategies:
                print(f"Warning: Strategy '{name}' already registered, overwriting")
            cls._strategies[name] = strategy_class
            print(f"Registered strategy: '{name}' -> {strategy_class.__name__}")
            return strategy_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Get strategy class by name

        Args:
            name: Strategy name

        Returns:
            Strategy class or None if not found
        """
        return cls._strategies.get(name)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all registered strategy names

        Returns:
            List of strategy names
        """
        strategies = list(cls._strategies.keys())
        print(f"Available strategies: {strategies}")
        return strategies
