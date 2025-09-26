# Jesse Bollinger Bands Mean Reversion Strategy Backtester

This project implements a Bollinger Bands mean reversion strategy for the Jesse trading framework with both CLI and GUI interfaces.

## Strategy Details

- **Period**: 200
- **Standard Deviation**: 3
- **Take Profit**: SMA-based
- **Stop Loss**: 1%
- **Data Type**: Tick data only (no fills or slippage for MVP)

This is a naive implementation designed as a stub for testing and determining where the money is, rather than for production strategies.

## Files Structure

```
backtrader/
├── jesse_strategy.py          # Bollinger Bands Mean Reversion Strategy
├── cli_backtest.py           # CLI backtesting interface
├── gui_visualizer.py         # GUI for visualization
├── main.py                   # Main application entry point
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have Jesse installed:
   ```bash
   pip install jesse
   ```

## Usage

### CLI Mode (Backtesting)
```bash
python main.py cli --csv path/to/your/trade_data.csv --symbol 1INCHUSDT --exchange Binance
```

### GUI Mode (Visualization)
```bash
python main.py gui
```

## Features

1. **CLI Backtesting**:
   - Load trade data from CSV
   - Run backtest with specified parameters
   - Generate performance metrics
   - Save results to file

2. **GUI Visualization**:
   - Equity curve visualization
   - Trade details table
   - Performance metrics display
   - Load results from JSON files

## Strategy Logic

The Bollinger Bands Mean Reversion strategy works as follows:

- **Long Entry**: When price moves below the lower Bollinger Band
- **Short Entry**: When price moves above the upper Bollinger Band
- **Stop Loss**: 1% from entry price
- **Take Profit**: Multiple of stop loss distance (configurable)

## Data Format

The application expects trade data in CSV format with the following columns:
- `id`: Trade ID
- `price`: Price at which the trade occurred
- `qty`: Quantity of the trade
- `quote_qty`: Quote quantity
- `time`: Timestamp in milliseconds
- `is_buyer_maker`: Boolean indicating if buyer was maker

## Notes

- This is an MVP implementation with basic functionality
- CSV format is temporary and will be replaced with more efficient formats in later versions
- Microstructure is not implemented in this MVP
- Risk management is basic in this MVP
- Real-time execution is not planned for this implementation
- API is not planned for this implementation