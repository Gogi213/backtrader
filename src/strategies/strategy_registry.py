"""
Strategy Registry
Provides a centralized registry for strategy discovery and management.
Author: HFT System
"""
from typing import Dict, List, Type, Optional

# Forward declaration for type hinting
class BaseStrategy:
    pass

class StrategyRegistry:
    """
    A centralized registry for trading strategies.
    This class uses class methods to manage a dictionary of all available
    strategies, mapping unique string names to strategy classes.
    """
    _strategies: Dict[str, Type['BaseStrategy']] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a strategy class with a unique name.
        Usage:
            @StrategyRegistry.register('my_strategy')
            class MyStrategy(BaseStrategy):
                ...
        """
        def decorator(strategy_class: Type['BaseStrategy']) -> Type['BaseStrategy']:
            if name in cls._strategies:
                print(f"Warning: Strategy '{name}' is already registered. Overwriting.")
            cls._strategies[name] = strategy_class
            # print(f"Registered strategy: '{name}' -> {strategy_class.__name__}")
            return strategy_class
        return decorator

    @classmethod
    def get_strategy(cls, name: str) -> Optional[Type['BaseStrategy']]:
        """
        Retrieves a strategy class by its registered name.
        """
        return cls._strategies.get(name)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Returns a list of names of all registered strategies.
        """
        return list(cls._strategies.keys())

    @classmethod
    def create_strategy(cls, name: str, symbol: str, **params) -> 'BaseStrategy':
        """
        Factory method to create an instance of a registered strategy.
        """
        strategy_class = cls.get_strategy(name)
        if strategy_class is None:
            available = ", ".join(cls.list_strategies())
            raise ValueError(
                f"Strategy '{name}' not found. "
                f"Available strategies: [{available}]"
            )
        return strategy_class(symbol=symbol, **params)