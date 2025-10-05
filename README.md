
# Backtrader - High-Performance Backtesting System

## Key Features

1. **Vectorized Processing**: Full vectorization using NumPy and Numba for optimal performance
2. **Klines Data Processing**: Processes OHLCV klines data efficiently
3. **HFT Optimized**: Engineered for high-frequency trading performance
4. **Automatic Dataset Scanning**: Automatically scans the `upload/klines` folder for available datasets
5. **Symbol Detection**: Automatically detects the trading symbol from the dataset filename
6. **Backtest Execution**: Run vectorized backtests with intuitive GUI controls
7. **Parameter Optimization**: **NEW** - Optimize strategy parameters using Optuna framework
8. **GUI Optimization Tab**: **NEW** - Optimize parameters directly in the GUI with real-time visualization
9. **Fast Optimization**: **NEW** - 10x+ speedup with caching, parallel processing, and adaptive evaluation
10. **Real-Time Results**: Display results as the backtest runs
11. **Visualization**: Interactive charts with Bollinger Bands, price action, and signals
12. **Multi-Tab Interface**: Organized interface with separate tabs for different types of information
13. **Professional Dark Theme**: Dark mode interface optimized for trading applications
14. **Interactive Charts**: Zoom and pan functionality using PyQtGraph for detailed analysis
15. **Performance Metrics**: Comprehensive performance analysis and statistics
16. **Trade Details**: Detailed trade-by-trade execution information

## Requirements

- Python 3.7+
- PyQt6 >= 6.4.0
- Pandas >= 1.3.0
- NumPy >= 1.21.0
- Numba >= 0.56.0 (for optimization)
- PyQtGraph (for high-performance charts)
- **Optuna >= 3.0.0** (for parameter optimization)
- **Plotly** (for optimization visualization, optional)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For optimization visualization (optional):
```bash
pip install optuna[visualization] plotly
```

3. Make sure your datasets are in the `upload/klines` folder in CSV format
4. Run the application:
```bash
python main.py
```

## Usage

### GUI Mode
1. Place your klines CSV files in the `upload/klines` folder
2. The application will automatically detect these files
3. Select a dataset from the dropdown menu
4. Configure strategy parameters (period, standard deviation, etc.)
5. Click "Start Backtest" to begin the vectorized backtest
6. View results in the various tabs:
   - Chart & Signals: Interactive price chart with Bollinger Bands and trade signals
   - Performance: Equity curve and performance metrics
   - Trade Details: Detailed list of all trades executed
   - Optimization: **NEW** - Optimize strategy parameters using Optuna directly in GUI

### Parameter Optimization (NEW)

#### GUI Optimization (Recommended)
1. Open the application with `python main.py`
2. Navigate to the **Optimization** tab
3. Select your dataset and strategy
4. Configure optimization parameters (trials, objective, etc.)
5. Click **Start Optimization** and watch real-time results
6. View best parameters, parameter importance, and final backtest results
7. Export results for further analysis

#### CLI Quick Start
```bash
# List available strategies
python optimize.py --list-strategies

# Optimize hierarchical_mean_reversion strategy (TURBO version)
python optimize.py --csv upload/klines/MASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv --strategy hierarchical_mean_reversion --trials 50
```

#### Advanced CLI Optimization
```bash
python optimize.py \
    --csv upload/klines/MASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv \
    --strategy hierarchical_mean_reversion \
    --trials 100 \
    --objective sharpe_ratio \
    --min-trades 20 \
    --max-drawdown 30.0 \
    --output results.json \
    --verbose \
    --plot
```

#### Fast Optimization (10x+ Speedup)
```bash
# Basic fast optimization with caching and parallel processing
python optimize.py \
    --csv upload/klines/MASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv \
    --strategy hierarchical_mean_reversion \
    --trials 100 \
    --fast

# Maximum speed optimization (20-50x faster)
python optimize.py \
    --csv upload/klines/MASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv \
    --strategy hierarchical_mean_reversion \
    --trials 200 \
    --fast \
    --jobs 8 \
    --sampler tpe \
    --multivariate \
    --aggressive-pruning
```

#### Maximum Speed Optimization (250x+ Speedup)
```bash
# MAXIMUM SPEED with TURBO strategy + Fast optimization
python optimize.py \
    --csv upload/klines/MASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv \
    --strategy hierarchical_mean_reversion \
    --trials 200 \
    --fast \
    --jobs -1 \
    --sampler tpe \
    --multivariate \
    --aggressive-pruning
```

#### Programmatic Usage
```python
from src.optimization import StrategyOptimizer

# Create optimizer (uses TURBO version with Numba JIT)
optimizer = StrategyOptimizer(
    strategy_name='hierarchical_mean_reversion',
    data_path='upload/klines/MASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv',
    symbol='MASTERUSDT'
)

# Run optimization
results = optimizer.optimize(
    n_trials=100,
    objective_metric='sharpe_ratio'
)

print(f"Best parameters: {results['best_params']}")
print(f"Best Sharpe ratio: {results['best_value']}")
```

## File Format

The application expects OHLCV klines data in CSV format:
- Columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Timestamp can be in milliseconds since epoch or datetime format

## Available Strategies

### Bollinger Bands Strategy
- Classic mean reversion strategy using Bollinger Bands
- Parameters: period, standard deviation, stop loss, etc.

### Hierarchical Mean Reversion Strategy
- Advanced strategy using Kalman Filter, HMM, and Ornstein-Uhlenbeck process
- **TURBO version with Numba JIT compilation for 50x+ speed**
- Parameters: measurement noise, process noise, entry thresholds, etc.

## Directory Structure

```
backtrader/
├── main.py                         # Main application entry point
├── optimize.py                     # NEW: Parameter optimization CLI
├── src/
│   ├── data/
│   │   ├── backtest_engine.py      # Backtesting engine
│   │   ├── klines_handler.py       # Klines data processing
│   │   └── technical_indicators.py # Vectorized technical indicators
│   ├── strategies/
│   │   ├── base_strategy.py        # Base strategy class
│   │   ├── bollinger_strategy.py   # Bollinger Bands strategy
│   │   ├── hierarchical_mean_reversion_strategy.py # Mean reversion strategy
│   │   ├── strategy_factory.py     # Strategy factory
│   │   └── strategy_registry.py    # Strategy registry
│   ├── optimization/               # NEW: Optimization module
│   │   ├── optuna_optimizer.py     # Core optimization logic
│   │   ├── optimize_cli.py         # CLI interface
│   │   └── visualization.py        # Results visualization
│   └── gui/
│       ├── main_window.py          # Main GUI application
│       ├── charts/
│       │   └── pyqtgraph_chart.py  # High-performance charts
│       ├── tabs/                   # GUI tabs for different views
│       └── utils/                  # GUI utilities
├── upload/
│   └── klines/                     # Place your CSV klines datasets here
├── docs/
│   └── OPTIMIZATION_GUIDE.md       # NEW: Detailed optimization guide
└── requirements.txt                # Python dependencies
```

## Optimization Features

### Supported Objectives
- Sharpe Ratio (recommended)
- Net P&L
- Profit Factor
- Win Rate
- Custom composite objectives

### Constraints
- Minimum trade count
- Maximum drawdown limits
- Custom filtering

### Visualization
- Optimization history
- Parameter importance
- Parallel coordinate plots
- Interactive dashboards

## Troubleshooting

If you encounter issues:
1. Ensure all required dependencies are installed with `pip install -r requirements.txt`
2. Verify that your CSV files are in the correct OHLCV format
3. Check that files are placed in the `upload/klines` folder
4. Monitor the GUI log output for error messages
5. For optimization issues, check the [Optimization Guide](docs/OPTIMIZATION_GUIDE.md)

## Documentation

- [Optimization Guide](docs/OPTIMIZATION_GUIDE.md) - Detailed parameter optimization guide
- [Strategy Development](src/strategies/ADDING_NEW_STRATEGY.md) - How to add new strategies