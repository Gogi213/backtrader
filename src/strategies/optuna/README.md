# Optuna Integration (Future)

This directory is reserved for future Optuna integration for strategy optimization.

## Planned Features

### 1. **strategy_optimizer.py**
Core optimization engine that will:
- Use `StrategyFactory` to create strategy instances
- Use `BaseStrategy.get_param_space()` to define search spaces
- Run Optuna trials with different parameter combinations
- Optimize for metrics like Sharpe ratio, profit factor, etc.

### 2. **param_spaces.py**
Reusable parameter space definitions:
- Common parameter ranges for different strategy types
- Custom space builders for complex optimization scenarios

### 3. **objective_functions.py**
Custom objective functions for optimization:
- Multi-objective optimization (Sharpe + drawdown)
- Risk-adjusted returns
- Custom fitness functions

## Example Usage (Future)

```python
from src.strategies.optuna.strategy_optimizer import StrategyOptimizer

# Create optimizer
optimizer = StrategyOptimizer(
    strategy_name='bollinger',
    data=klines_df,
    optimize_metric='sharpe_ratio'
)

# Run optimization
study = optimizer.optimize(n_trials=100)
best_params = study.best_params

# Create optimized strategy
from src.strategies import StrategyFactory
optimized_strategy = StrategyFactory.create('bollinger', 'BTCUSDT', **best_params)
```

## Architecture Ready

The current architecture is **Optuna-ready**:
- ✅ `BaseStrategy.get_param_space()` defines parameter bounds
- ✅ `StrategyFactory` can create strategies with any parameters
- ✅ `BaseStrategy.get_default_params()` provides baseline comparison
- ✅ All strategies return consistent results format

When Optuna is needed, simply implement the files above without changing existing code.
