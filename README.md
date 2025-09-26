# Pure Tick HFT Bollinger Bands Strategy Backtester (Vectorized)

A high-performance, fully vectorized backtesting engine for High-Frequency Trading (HFT) Bollinger Bands strategies, processing 4+ million ticks with numpy/numba optimization.

## Key Features

1. **Vectorized Processing**: Full vectorization using NumPy and Numba for HFT performance
2. **Pure Tick Processing**: Direct processing of tick-by-tick data without candle aggregation
3. **HFT Optimized**: Engineered for 200,000+ ticks/second processing (achieved 344,474 ticks/sec)
4. **Automatic Dataset Scanning**: Automatically scans the `upload/trades` folder for available datasets
5. **Symbol Detection**: Automatically detects the trading symbol from the dataset filename
6. **Backtest Execution**: Run vectorized backtests with the START БЕКТЕСТА button
7. **Real-Time Results**: Display results as the backtest runs
8. **Visualization**: Shows equity curves, trade details, and performance metrics
9. **Multi-Tab Interface**: Organized interface with separate tabs for different types of information
10. **Professional Dark Theme**: Dark mode interface optimized for trading applications
11. **Interactive Charts**: Zoom and pan functionality for detailed chart analysis
12. **Real Trade Data Processing**: Load and visualize actual trade data from CSV files
13. **Real Equity Curve**: Equity curve calculated from actual trade P&L
14. **Accurate Time Axis**: Proper time formatting based on real trade timestamps
15. **Trading Signals Visualization**: Visual indicators for trade entries/exits
16. **Optimized Panel Layout**: 15%/85% split for controls and visualization
17. **Performance Metrics Tab**: Dedicated tab for equity curve and performance metrics

## Requirements

- Python 3.7+
- PyQt6
- Pandas
- NumPy
- Numba (for HFT optimization)
- Matplotlib

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Make sure your datasets are in the `upload/trades` folder in CSV format
3. Run the application:
```
python main.py
```

## Usage

1. Place your trading data CSV files in the `upload/trades` folder
2. The application will automatically detect these files
3. Select a dataset from the dropdown menu
4. The symbol will be automatically detected from the filename
5. Click "START БЕКТЕСТА" to begin the vectorized HFT backtest
6. View results in the various tabs:
   - Equity Curve: Shows the performance over time
   - Trade Details: Lists all trades executed during the backtest
   - Performance Metrics: Key statistics and ratios
   - Execution Log: Detailed log of the backtest process

## File Format

The application supports two formats for input data:

### Tick Data Format
- Columns: `timestamp`, `price`, `size` (or `volume`)
- Timestamp should be in a recognized datetime format or milliseconds since epoch

### OHLCV Format
- Columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Timestamp should be in milliseconds since epoch

## Strategy Configuration

The HFT Vectorized Bollinger Bands Mean Reversion strategy uses these default parameters:
- Period: 50 (HFT optimized)
- Standard deviation: 2.0 (HFT optimized)
- Stop loss: 0.5%
- Initial capital: $10,000

## Directory Structure

```
backtrader/
├── main.py                         # Main application entry point
├── cli_backtest.py                 # Command-line backtesting interface
├── pure_tick_gui.py                # Pure tick GUI interface code
├── pure_tick_backtest.py           # Pure tick backtesting engine
├── jesse_strategy.py               # Jesse framework strategy (DEPRECATED)
├── src/
│   ├── data/
│   │   ├── vectorized_tick_handler.py      # Vectorized data processing utilities
│   │   └── pure_tick_handler.py            # Legacy tick data handler (DEPRECATED)
│   └── strategies/
│       ├── vectorized_bollinger_strategy.py    # Vectorized HFT strategy implementation
│       └── pure_tick_bollinger_strategy.py     # Legacy pure tick strategy (DEPRECATED)
├── upload/
│   └── trades/         # Place your CSV datasets here
└── requirements.txt    # Python dependencies
```

## Troubleshooting

If you encounter issues:
1. Ensure all required dependencies are installed
2. Verify that your CSV files are in the correct format
3. Check the execution log tab for error messages
4. Make sure you have the correct permissions to access the files