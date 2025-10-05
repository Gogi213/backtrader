# Strategies module exports
from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry

# Алиас для обратной совместимости
StrategyFactory = StrategyRegistry

# Optional ML-based strategy (requires pykalman, scikit-learn, scipy)
try:
    # Force import to trigger decorator registration
    from . import turbo_mean_reversion_strategy
    from .turbo_mean_reversion_strategy import HierarchicalMeanReversionStrategy
    __all__ = [
        'BaseStrategy',
        'StrategyRegistry',
        'StrategyFactory',  # Алиас для обратной совместимости
        'HierarchicalMeanReversionStrategy'
    ]
except ImportError as e:
    import warnings
    warnings.warn(
        f"HierarchicalMeanReversionStrategy not available: {e}\n"
        "Install dependencies: pip install pykalman scikit-learn scipy numba"
    )
    __all__ = [
        'BaseStrategy',
        'StrategyRegistry',
        'StrategyFactory'  # Алиас для обратной совместимости
    ]