# Jesse Bollinger Bands Strategy Backtester

A comprehensive GUI application for backtesting Bollinger Bands mean reversion strategies using the Jesse framework.

## Features

1. **Automatic Dataset Scanning**: Automatically scans the `upload/trades` folder for available datasets
2. **Symbol Detection**: Automatically detects the trading symbol from the dataset filename
3. **Backtest Execution**: Run backtests with the Bollinger Bands strategy using the START БЕКТЕСТА button
4. **Real-Time Results**: Display results as the backtest runs
5. **Visualization**: Shows equity curves, trade details, and performance metrics
6. **Multi-Tab Interface**: Organized interface with separate tabs for different types of information

## Requirements

- Python 3.7+
- PyQt5
- Jesse Framework
- Pandas
- NumPy
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
5. Click "START БЕКТЕСТА" to begin the backtest
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

The Bollinger Bands Mean Reversion strategy uses these default parameters:
- Period: 200
- Standard deviation: 3
- Take profit: SMA
- Stop loss: 1%

## Directory Structure

```
backtrader/
├── main.py             # Main application entry point
├── cli_backtest.py     # Command-line backtesting interface
├── gui_visualizer.py   # GUI interface code
├── jesse_strategy.py   # Bollinger Bands strategy implementation
├── src/
│   └── data/
│       └── tick_data_handler.py  # Data processing utilities
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