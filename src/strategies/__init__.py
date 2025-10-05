# Strategies module exports
from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry
from .strategy_factory import StrategyFactory
from .bollinger_strategy import VectorizedBollingerStrategy

# Optional ML-based strategy (requires pykalman, scikit-learn, scipy)
try:
    # Force import to trigger decorator registration
    from . import turbo_mean_reversion_strategy
    from .turbo_mean_reversion_strategy import TurboMeanReversionStrategy
    __all__ = [
        'BaseStrategy',
        'StrategyRegistry',
        'StrategyFactory',
        'VectorizedBollingerStrategy',
        'TurboMeanReversionStrategy'
    ]
except ImportError as e:
    import warnings
    warnings.warn(
        f"TurboMeanReversionStrategy not available: {e}\n"
        "Install dependencies: pip install pykalman scikit-learn scipy numba"
    )
    __all__ = [
        'BaseStrategy',
        'StrategyRegistry',
        'StrategyFactory',
        'VectorizedBollingerStrategy'
    ]