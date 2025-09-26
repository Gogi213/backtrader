# Jesse Bollinger Bands Mean Reversion Strategy Backtester

This is a comprehensive backtesting framework for a Bollinger Bands mean reversion strategy built specifically for Jesse, with both CLI and GUI interfaces.

## Strategy Details

- **Period**: 200
- **Standard Deviation**: 3
- **Take Profit**: SMA-based (shorter period)
- **Stop Loss**: 1%
- **Data Type**: Tick data (converted to candles for processing)

This is a sophisticated implementation designed for testing and determining profitable trading opportunities.

## Project Structure

```
backtrader/
├── run_app.py                    # Main application entry point
├── requirements.txt             # Project dependencies
├── BACKTESTER_README.md         # This file
├── src/
│   ├── strategies/
│   │   └── bollinger_bands_strategy.py    # Core strategy implementation
│   ├── data/
│   │   ├── csv_loader.py        # CSV data loading
│   │   └── tick_data_handler.py # Tick data processing
│   ├── engine/
│   │   └── backtesting_engine.py # Backtesting engine
│   ├── cli/
│   │   └── cli.py              # Command-line interface
│   ├── gui/
│   │   └── gui.py              # Graphical interface
│   └── utils/                   # Utility modules
├── test_implementation.py       # Test suite
└── upload/
    └── trades/
        └── 1INCHUSDT-trades-2025-09-24.csv  # Sample data file
```

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have Jesse installed (if not included in requirements):
   ```bash
   pip install jesse
   ```

## Usage

### CLI Mode (Backtesting)
```bash
python run_app.py cli --data-file path/to/your/trade_data.csv --symbol 1INCHUSDT --period 200 --std-dev 3.0 --capital 10000.0
```

### GUI Mode (Visualization)
```bash
python run_app.py gui
```

## Data Format

The application expects trade data in CSV format with the following columns:
- `id`: Trade ID
- `price`: Price at which the trade occurred
- `qty`: Quantity of the trade
- `quote_qty`: Quote quantity
- `time`: Timestamp in milliseconds
- `is_buyer_maker`: Boolean indicating if buyer was maker

Example dataset is provided in `upload/trades/1INCHUSDT-trades-2025-09-24.csv`

## Strategy Logic

The Bollinger Bands Mean Reversion strategy works as follows:

- **Long Entry**: When price moves below the lower Bollinger Band with additional filters
- **Short Entry**: When price moves above the upper Bollinger Band with additional filters
- **Stop Loss**: 1% from entry price
- **Take Profit**: Based on SMA of shorter period (default 20-period), or risk-to-reward ratio
- **Risk Management**: Position sizing based on risk percentage

## Features

1. **CLI Backtesting**:
   - Load and process tick data from CSV
   - Run backtest with specified parameters
   - Generate performance metrics
   - Export results to CSV files

2. **GUI Visualization**:
   - Intuitive graphical interface
   - Parameter configuration
   - Interactive charting
   - Trade markers on price charts
   - Export functionality

3. **Advanced Data Processing**:
   - Tick data aggregation to candles
   - Multiple timeframe support
   - Data validation
   - Error handling

4. **Performance Metrics**:
   - Total trades and win rate
   - PnL and return percentage
   - Sharpe ratio and max drawdown
   - Detailed trade history

## Key Components

### BollingerBandsMeanReversionStrategy
- Implements the core trading strategy
- Calculates Bollinger Bands with period 200 and std dev 3
- Uses SMA for take profit levels
- Implements proper stop loss (1%)
- Handles entry/exit logic

### TickDataHandler
- Processes tick data from CSV files
- Converts to Jesse-compatible OHLCV format
- Supports multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- Includes data validation and error handling

### BacktestingEngine
- Core backtesting engine
- Processes tick data through strategy
- Tracks performance metrics
- Exports results

### BacktestingGUI
- Complete graphical interface
- Live charting with trade markers
- Parameter configuration
- Results display
- Export functionality

## Parameters

- `--period`: Bollinger Bands period (default: 200)
- `--std-dev`: Standard deviation multiplier (default: 3.0)
- `--capital`: Initial capital for backtest (default: 10000.0)
- `--symbol`: Trading symbol (default: BTCUSDT)

## Output Files

When running backtests, the application creates results in the `results/` directory:
- `trades.csv`: Complete trade history
- `metrics.csv`: Performance metrics
- `price_history.csv`: Price history for analysis
- `equity_curve.csv`: Equity curve data

## Development Notes

- The implementation is based on the specific CSV format with tick data
- The strategy prioritizes mean reversion signals with volatility filters
- Risk management is implemented with position sizing
- The application can handle large datasets efficiently
- Both GUI and CLI provide consistent functionality