# Adding New Strategy - Quick Guide

## Step 1: Create Your Strategy File

Create a new file in `src/strategies/`, e.g., `my_strategy.py`:

```python
"""
My Custom Strategy
Description of what this strategy does

Author: Your Name
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry


@StrategyRegistry.register('my_strategy')  # Register with unique name
class MyCustomStrategy(BaseStrategy):
    """
    Your strategy description
    """

    def __init__(self, symbol: str, **params):
        """
        Initialize strategy

        Args:
            symbol: Trading symbol
            **params: Strategy parameters (e.g., period, threshold, etc.)
        """
        super().__init__(symbol, **params)

        # Extract parameters with defaults
        self.period = params.get('period', 20)
        self.threshold = params.get('threshold', 0.5)
        self.initial_capital = params.get('initial_capital', 10000.0)
        self.commission_pct = params.get('commission_pct', 0.0005)

    def vectorized_process_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main strategy logic - process entire dataset at once

        Args:
            df: DataFrame with columns: time, price, open, high, low, close, Volume

        Returns:
            Dictionary with results:
            {
                'trades': List[Dict],  # Trade records
                'total': int,
                'win_rate': float,
                'net_pnl': float,
                'sharpe_ratio': float,
                'bb_data': Dict,  # Chart data (optional)
                ...
            }
        """
        # Your strategy logic here
        # Example: Calculate indicators, generate signals, create trades

        times = df['time'].values
        prices = df['price'].values

        # Your calculations...
        trades = []  # Generate your trades

        # Calculate performance metrics
        metrics = self._calculate_metrics(trades)

        return {
            'trades': trades,
            'symbol': self.symbol,
            'total': len(trades),
            **metrics
        }

    def _calculate_metrics(self, trades):
        """Calculate performance metrics"""
        # Basic metrics calculation
        if not trades:
            return {
                'win_rate': 0, 'net_pnl': 0, 'sharpe_ratio': 0,
                'max_drawdown': 0, 'profit_factor': 0
            }

        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        net_pnl = sum(t['pnl'] for t in trades)

        return {
            'win_rate': win_rate,
            'net_pnl': net_pnl,
            'sharpe_ratio': 0,  # Calculate properly
            'max_drawdown': 0,  # Calculate properly
            'profit_factor': 0  # Calculate properly
        }

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Default parameters for this strategy

        Returns:
            Dictionary with default values
        """
        return {
            'period': 20,
            'threshold': 0.5,
            'initial_capital': 10000.0,
            'commission_pct': 0.0005
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, tuple]:
        """
        Parameter space for Optuna optimization

        Returns:
            Dictionary mapping parameter names to (type, bounds)

            Format:
            - ('int', min, max) for integers
            - ('float', min, max) for floats
            - ('categorical', [options]) for categorical
        """
        return {
            'period': ('int', 10, 100),
            'threshold': ('float', 0.1, 2.0),
            'initial_capital': ('float', 1000.0, 100000.0),
            'commission_pct': ('float', 0.0001, 0.001)
        }
```

## Step 2: Add to Module Exports (Optional)

Edit `src/strategies/__init__.py`:

```python
from .my_strategy import MyCustomStrategy

__all__ = [
    'BaseStrategy',
    'StrategyRegistry',
    'StrategyFactory',
    'VectorizedBollingerStrategy',
    'MyCustomStrategy'  # Add your strategy
]
```

## Step 3: Use Your Strategy

### In Code:

```python
from src.strategies import StrategyFactory, StrategyRegistry

# Create with custom parameters
strategy = StrategyFactory.create(
    'my_strategy',
    'BTCUSDT',
    period=30,
    threshold=0.8
)

# Or with defaults
strategy_class = StrategyRegistry.get('my_strategy')
strategy = StrategyFactory.create(
    'my_strategy',
    'BTCUSDT',
    **strategy_class.get_default_params()
)

# Run backtest
results = strategy.vectorized_process_dataset(data_df)
```

### From CLI:

```bash
python -m src.data.vectorized_klines_backtest \
    --csv data.csv \
    --strategy my_strategy \
    --symbol BTCUSDT
```

### List Available Strategies:

```python
from src.strategies import StrategyRegistry

# List all registered strategies
strategies = StrategyRegistry.list_strategies()
print(f"Available strategies: {strategies}")
```

## Step 4: Test Your Strategy

```python
from src.strategies import StrategyFactory, StrategyRegistry

# Get strategy class and info
strategy_class = StrategyRegistry.get('my_strategy')
print(f"Default params: {strategy_class.get_default_params()}")
print(f"Param space: {strategy_class.get_param_space()}")

# Test basic functionality
strategy = StrategyFactory.create('my_strategy', 'TESTUSDT', **strategy_class.get_default_params())
```

## That's It!

Your strategy is now:
- ✅ Automatically registered via decorator
- ✅ Discoverable through StrategyRegistry
- ✅ Creatable via StrategyFactory
- ✅ Available in GUI automatically
- ✅ Ready for Optuna optimization (via `get_param_space()`)

## Tips

1. **Use vectorized operations** (numpy) for performance
2. **Return consistent format** from `vectorized_process_dataset()`
3. **Include chart data** in results for visualization (optional)
4. **Follow naming conventions** (use descriptive parameter names)
5. **Test thoroughly** with different datasets

## Example: See Existing Strategies

Check `vectorized_bollinger_strategy.py` for a complete working example.
