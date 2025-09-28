# Vectorized HFT Bollinger Bands Strategy Backtester

A high-performance, fully vectorized backtesting engine for Bollinger Bands strategies using klines data, optimized with numpy/numba for maximum performance.

## Key Features

1. **Vectorized Processing**: Full vectorization using NumPy and Numba for optimal performance
2. **Klines Data Processing**: Processes OHLCV klines data efficiently
3. **HFT Optimized**: Engineered for high-frequency trading performance
4. **Automatic Dataset Scanning**: Automatically scans the `upload/klines` folder for available datasets
5. **Symbol Detection**: Automatically detects the trading symbol from the dataset filename
6. **Backtest Execution**: Run vectorized backtests with intuitive GUI controls
7. **Real-Time Results**: Display results as the backtest runs
8. **Visualization**: Interactive charts with Bollinger Bands, price action, and signals
9. **Multi-Tab Interface**: Organized interface with separate tabs for different types of information
10. **Professional Dark Theme**: Dark mode interface optimized for trading applications
11. **Interactive Charts**: Zoom and pan functionality using PyQtGraph for detailed analysis
12. **Performance Metrics**: Comprehensive performance analysis and statistics
13. **Trade Details**: Detailed trade-by-trade execution information

## Requirements

- Python 3.7+
- PyQt6 >= 6.4.0
- Pandas >= 1.3.0
- NumPy >= 1.21.0
- Numba >= 0.56.0 (for optimization)
- PyQtGraph (for high-performance charts)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your datasets are in the `upload/klines` folder in CSV format
3. Run the application:
```bash
python main.py
```

## Usage

1. Place your klines CSV files in the `upload/klines` folder
2. The application will automatically detect these files
3. Select a dataset from the dropdown menu
4. Configure strategy parameters (period, standard deviation, etc.)
5. Click "Start Backtest" to begin the vectorized backtest
6. View results in the various tabs:
   - Chart & Signals: Interactive price chart with Bollinger Bands and trade signals
   - Performance: Equity curve and performance metrics
   - Trade Details: Detailed list of all trades executed

## File Format

The application expects OHLCV klines data in CSV format:
- Columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Timestamp can be in milliseconds since epoch or datetime format

## Strategy Configuration

The Vectorized Bollinger Bands strategy uses these configurable parameters:
- Period: 20-200 (default: 50)
- Standard deviation: 1.0-4.0 (default: 2.0)
- Initial capital: Configurable (default: $10,000)

## Directory Structure

```
backtrader/
├── main.py                         # Main application entry point
├── run_app.py                      # Alternative entry point
├── install_deps.py                 # Dependency installer
├── src/
│   ├── data/
│   │   ├── vectorized_klines_handler.py     # Klines data processing
│   │   ├── vectorized_klines_backtest.py    # Backtesting engine
│   │   └── technical_indicators.py          # Vectorized technical indicators
│   ├── strategies/
│   │   └── vectorized_bollinger_strategy.py # Bollinger Bands strategy
│   └── gui/
│       ├── gui_visualizer.py               # Main GUI application
│       ├── charts/
│       │   └── pyqtgraph_chart.py          # High-performance charts
│       ├── tabs/                           # GUI tabs for different views
│       ├── panels/                         # Control panels
│       ├── data/                           # Data management
│       ├── config/                         # Configuration models
│       └── utilities/                      # GUI utilities
├── upload/
│   └── klines/         # Place your CSV klines datasets here
└── requirements.txt    # Python dependencies
```

## Troubleshooting

If you encounter issues:
1. Ensure all required dependencies are installed with `pip install -r requirements.txt`
2. Verify that your CSV files are in the correct OHLCV format
3. Check that files are placed in the `upload/klines` folder
4. Monitor the GUI log output for error messages