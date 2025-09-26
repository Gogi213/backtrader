# Jesse Bollinger Bands Backtester Project Documentation

## Project Structure

```
backtrader/
├── PRODUCTION/
│   ├── DOCUMENTS/
│   │   └── jesse_backtester/
│   │       ├── BACKTESTER_README.md
│   │       ├── README.md
│   │       └── project_summary.md
│   ├── TESTS/
│   │   └── jesse_backtester/
│   │       ├── test_app.py
│   │       ├── validate_app.py
│   │       ├── test_implementation.py
│   │       └── comprehensive_test_suite.py
│   └── SPRINTS/
│       └── jesse_backtester/
│           └── sprint_planning.md
├── src/
│   ├── strategies/
│   │   └── bollinger_bands_strategy.py
│   ├── data/
│   │   ├── csv_loader.py
│   │   └── tick_data_handler.py
│   ├── engine/
│   │   └── backtesting_engine.py
│   ├── cli/
│   │   └── cli.py
│   └── gui/
│       └── gui.py
├── upload/
│   └── trades/
│       ├── 1INCHUSDT-trades-2025-09-24.csv
│       └── trades_exampls.csv
├── run_app.py
├── requirements.txt
└── main.py
```

## Project Components

### 1. Strategy Component
- **File**: `src/strategies/bollinger_bands_strategy.py`
- **Purpose**: Implementation of Bollinger Bands mean reversion strategy
- **Parameters**:
  - Period: 200
  - Standard Deviation: 3
  - Take Profit: SMA-based
  - Stop Loss: 1%

### 2. Data Processing Component
- **Files**: `src/data/csv_loader.py`, `src/data/tick_data_handler.py`
- **Purpose**: Loading and processing tick data from CSV files
- **Format**: id,price,qty,quote_qty,time,is_buyer_maker

### 3. Backtesting Engine
- **File**: `src/engine/backtesting_engine.py`
- **Purpose**: Core backtesting functionality
- **Features**: Performance metrics, trade execution, result export

### 4. CLI Interface
- **File**: `src/cli/cli.py`
- **Purpose**: Command-line interface for backtesting
- **Features**: Parameter configuration, result display

### 5. GUI Interface
- **File**: `src/gui/gui.py`
- **Purpose**: Graphical interface for backtesting
- **Features**: Interactive charts, parameter adjustment, visualization

## Usage

### Running Backtests
```bash
# CLI mode
python run_app.py cli --data-file path/to/data.csv --symbol 1INCHUSDT --period 200 --std-dev 3.0 --capital 10000.0

# GUI mode
python run_app.py gui
```

## Development Status

- [x] Strategy implementation
- [x] Data processing
- [x] Backtesting engine
- [x] CLI interface
- [x] GUI interface
- [x] Testing framework
- [x] Documentation

The project is complete and ready for use in identifying profitable mean reversion opportunities.