# Backlog

## Refactoring

- **2025-10-08**:
  - **Refactor `BaseStrategy` and `StrategyRegistry`**:
    - **Reason**: The `BaseStrategy` class violated the Single Responsibility Principle by combining the base strategy interface with a strategy registry.
    - **Change**:
      - Created a new `StrategyRegistry` class in `src/strategies/strategy_registry.py` to handle the registration and management of strategies.
      - Removed registry-related methods (`register`, `get_strategy`, `list_strategies`, `create_strategy`) from `BaseStrategy`.
      - Updated `cli_optimizer.py`, `src/core/backtest_manager.py`, and all strategy files to use the new `StrategyRegistry`.
    - **Benefit**: Improved code clarity, modularity, and maintainability by separating concerns.
  - **Dynamic Strategy Loading**:
    - **Reason**: The strategy loading mechanism was manual and required explicit imports in `src/strategies/__init__.py`. The default strategy in `cli_optimizer.py` was hardcoded.
    - **Change**:
      - Implemented automatic strategy discovery and registration in `src/strategies/__init__.py` using `pkgutil` and `importlib`.
      - Modified `cli_optimizer.py` to dynamically fetch the list of available strategies from `StrategyRegistry` and set the first one as the default.
    - **Benefit**: Simplifies adding new strategies (no more manual `__init__.py` edits) and makes the CLI more robust.
  - **Unify TUI and CLI Strategy Loading**:
    - **Reason**: The TUI (`src/tui/optimization_app.py`) had a hardcoded fallback strategy, making it inconsistent with the dynamic CLI.
    - **Change**:
      - Refactored `src/tui/optimization_app.py` to use the same dynamic strategy loading mechanism as the CLI.
      - The TUI now populates its strategy selection dropdown from `StrategyRegistry` and uses the first available strategy as the default.
    - **Benefit**: Consistent behavior between TUI and CLI. The TUI is now as robust and easy to extend with new strategies as the CLI.
  - **Modernize `main.py` Startup Script**:
    - **Reason**: The main entrypoint script (`main.py`) contained outdated, hardcoded checks for specific strategy files, which became obsolete after implementing dynamic strategy loading.
    - **Change**:
      - Removed the hardcoded check for `turbo_mean_reversion_strategy.py`.
      - Integrated dynamic strategy loading to provide relevant and up-to-date usage examples to the user.
    - **Benefit**: The startup script is now aligned with the dynamic architecture, providing a cleaner and more accurate introduction for the user.
  - **Eliminate Redundant Code in Strategy**:
    - **Reason**: The `HierarchicalMeanReversionStrategy` contained a redundant `_calculate_performance_metrics` method that duplicated functionality already present in `BaseStrategy`.
    - **Change**:
      - Removed the overriding method from `turbo_mean_reversion_strategy.py`.
      - Switched to a direct call to the static method `BaseStrategy.calculate_performance_metrics`.
    - **Benefit**: Reduced code duplication, simplified the strategy's implementation, and slightly decreased the overall codebase size.