# Strategies module exports
from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry
from .strategy_factory import StrategyFactory
from .bollinger_strategy import VectorizedBollingerStrategy

# Optional ML-based strategy (requires pykalman, scikit-learn, scipy)
try:
    from .hierarchical_mean_reversion_strategy import HierarchicalMeanReversionStrategy
    __all__ = [
        'BaseStrategy',
        'StrategyRegistry',
        'StrategyFactory',
        'VectorizedBollingerStrategy',
        'HierarchicalMeanReversionStrategy'
    ]
except ImportError as e:
    import warnings
    warnings.warn(
        f"HierarchicalMeanReversionStrategy not available: {e}\n"
        "Install dependencies: pip install pykalman scikit-learn scipy"
    )
    __all__ = [
        'BaseStrategy',
        'StrategyRegistry',
        'StrategyFactory',
        'VectorizedBollingerStrategy'
    ]