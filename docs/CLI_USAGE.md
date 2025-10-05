# CLI Usage Guide

## Overview

The unified CLI interface provides a powerful and flexible way to run backtests using the new BacktestManager. This guide covers all available options and features.

## Basic Usage

### Single Backtest

Run a single backtest with default parameters:

```bash
python -m src.data.backtest_engine --csv data.csv --strategy hierarchical_mean_reversion
```

### Custom Parameters

Override strategy parameters:

```bash
python -m src.data.backtest_engine \
  --csv data.csv \
  --strategy hierarchical_mean_reversion \
  --bb-period 15 \
  --bb-std 1.8 \
  --rsi-period 12 \
  --max-klines 5000
```

### Output Options

Save results to file:

```bash
python -m src.data.backtest_engine \
  --csv data.csv \
  --strategy hierarchical_mean_reversion \
  --output results.json
```

## Batch Backtests

### Running Batch Backtests

Execute multiple backtests from a configuration file:

```bash
python -m src.data.backtest_engine \
  --batch examples/batch_backtest_config.json \
  --parallel \
  --max-workers 4 \
  --output batch_results.json
```

### Batch Configuration Format

Create a JSON configuration file with multiple backtest configurations:

```json
{
  "description": "Example batch backtest configuration",
  "backtests": [
    {
      "strategy_name": "hierarchical_mean_reversion",
      "strategy_params": {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30
      },
      "symbol": "BTCUSDT",
      "data_source": "csv",
      "data_path": "data/btcusdt_1h.csv",
      "initial_capital": 10000.0,
      "commission_pct": 0.05,
      "max_klines": 10000,
      "verbose": false
    },
    {
      "strategy_name": "bollinger_strategy",
      "strategy_params": {
        "bb_period": 15,
        "bb_std": 1.8
      },
      "symbol": "ETHUSDT",
      "data_source": "csv",
      "data_path": "data/ethusdt_1h.csv",
      "initial_capital": 10000.0,
      "commission_pct": 0.05,
      "max_klines": 10000,
      "verbose": false
    }
  ]
}
```

## Command Line Options

### Basic Options

- `--csv <path>`: CSV file path with klines data
- `--symbol <symbol>`: Trading symbol (default: BTCUSDT)
- `--strategy <name>`: Strategy name (default: hierarchical_mean_reversion)
- `--max-klines <number>`: Limit klines for testing
- `--output <path>`: Output file for results
- `--verbose`: Enable verbose output

### Batch Processing Options

- `--batch <path>`: Run batch backtests from JSON configuration file
- `--parallel`: Run backtests in parallel (for batch mode)
- `--max-workers <number>`: Maximum number of parallel workers

### Utility Options

- `--list-strategies`: List available strategies
- `--legacy`: Use legacy implementation (deprecated)

## Strategy Parameters

### Available Strategies

List all available strategies:

```bash
python -m src.data.backtest_engine --list-strategies
```

### Strategy-Specific Parameters

Each strategy has its own set of parameters. Use the strategy name to see available parameters:

```bash
# Get default parameters for hierarchical_mean_reversion
python -c "
from src.strategies import StrategyRegistry
strategy = StrategyRegistry.get('hierarchical_mean_reversion')
print(strategy.get_default_params())
"
```

### Parameter Overrides

Override strategy parameters using CLI arguments:

```bash
python -m src.data.backtest_engine \
  --csv data.csv \
  --strategy hierarchical_mean_reversion \
  --bb-period 15 \
  --bb-std 1.8 \
  --rsi-period 12 \
  --rsi-overbought 75 \
  --rsi-oversold 25
```

## Performance Optimization

### Parallel Processing

Enable parallel processing for batch backtests:

```bash
python -m src.data.backtest_engine \
  --batch batch_config.json \
  --parallel \
  --max-workers 4
```

### Performance Tuning

Use the CLI optimizer to tune performance:

```bash
# Optimize batch configuration
python -m src.cli.cli_optimizer \
  --optimize-batch examples/batch_backtest_config.json \
  --output optimized_batch.json

# Benchmark performance
python -m src.cli.cli_optimizer \
  --benchmark examples/batch_backtest_config.json
```

### Memory Optimization

Limit klines for large datasets:

```bash
python -m src.data.backtest_engine \
  --csv large_dataset.csv \
  --strategy hierarchical_mean_reversion \
  --max-klines 10000
```

## Output Formats

### Console Output

Results are displayed in a formatted table:

```
============================================================
UNIFIED BACKTEST RESULTS
============================================================
Symbol: BTCUSDT
Strategy: hierarchical_mean_reversion
Total Trades: 42
Win Rate: 64.3%
Net P&L: $1,234.56
Return: 12.35%
Max Drawdown: 3.45%
Sharpe Ratio: 1.23
Profit Factor: 1.67
============================================================
```

### JSON Output

Save results in JSON format:

```bash
python -m src.data.backtest_engine \
  --csv data.csv \
  --strategy hierarchical_mean_reversion \
  --output results.json
```

### Text Output

Save results in text format:

```bash
python -m src.data.backtest_engine \
  --csv data.csv \
  --strategy hierarchical_mean_reversion \
  --output results.txt
```

## Migration from Legacy CLI

### Automatic Migration

Use the CLI optimizer to migrate legacy scripts:

```bash
python -m src.cli.cli_optimizer \
  --migrate-script old_script.py \
  --output new_script.py
```

### Manual Migration

Replace old function calls:

```python
# Old way
from src.data.backtest_engine import run_vectorized_klines_backtest
results = run_vectorized_klines_backtest(csv_path='data.csv', ...)

# New way
from src.core import BacktestManager, BacktestConfig
config = BacktestConfig(strategy_name='hierarchical_mean_reversion', data_path='data.csv')
manager = BacktestManager()
results = manager.run_backtest(config)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're running from the project root directory
2. **Data File Not Found**: Check that the CSV file path is correct
3. **Strategy Not Found**: Use `--list-strategies` to see available strategies
4. **Memory Issues**: Use `--max-klines` to limit data size

### Debug Mode

Enable verbose output for debugging:

```bash
python -m src.data.backtest_engine \
  --csv data.csv \
  --strategy hierarchical_mean_reversion \
  --verbose
```

### Legacy Mode

If you encounter issues with the unified system, you can fall back to legacy mode:

```bash
python -m src.data.backtest_engine \
  --csv data.csv \
  --strategy hierarchical_mean_reversion \
  --legacy
```

## Examples

### Basic Backtest

```bash
python -m src.data.backtest_engine \
  --csv data/btcusdt_1h.csv \
  --strategy hierarchical_mean_reversion \
  --max-klines 5000 \
  --output results.json
```

### Batch Backtest with Parallel Processing

```bash
python -m src.data.backtest_engine \
  --batch examples/batch_backtest_config.json \
  --parallel \
  --max-workers 4 \
  --verbose \
  --output batch_results.json
```

### Strategy Parameter Optimization

```bash
python -m src.data.backtest_engine \
  --csv data/btcusdt_1h.csv \
  --strategy hierarchical_mean_reversion \
  --bb-period 15 \
  --bb-std 1.8 \
  --rsi-period 12 \
  --rsi-overbought 75 \
  --rsi-oversold 25 \
  --max-klines 10000
```

### Performance Benchmarking

```bash
python -m src.cli.cli_optimizer \
  --benchmark examples/batch_backtest_config.json \
  --workers 4
```

## Advanced Usage

### Custom Configuration Files

Create custom batch configurations for different scenarios:

1. **Parameter Optimization**: Test different parameter combinations
2. **Multi-Strategy**: Compare different strategies on the same data
3. **Multi-Symbol**: Test the same strategy on different symbols
4. **Time Periods**: Test different time periods with max_klines

### Integration with Scripts

Use the unified system in Python scripts:

```python
from src.core import BacktestManager, BacktestConfig

# Create configuration
config = BacktestConfig(
    strategy_name='hierarchical_mean_reversion',
    symbol='BTCUSDT',
    data_path='data.csv',
    max_klines=10000
)

# Run backtest
manager = BacktestManager()
results = manager.run_backtest(config)

# Process results
if results.is_successful():
    print(f"Net P&L: ${results.get('net_pnl', 0):.2f}")
    print(f"Win Rate: {results.get('win_rate', 0):.1%}")
else:
    print(f"Backtest failed: {results.get_error()}")
```

## Best Practices

1. **Use Batch Mode**: For multiple backtests, use batch mode with parallel processing
2. **Limit Data Size**: Use `--max-klines` for testing and development
3. **Save Results**: Always save results to file for later analysis
4. **Monitor Performance**: Use the CLI optimizer to tune performance
5. **Validate Configurations**: Use the ConfigValidator to check configurations before running
6. **Use Verbose Mode**: Enable verbose output for debugging complex issues

## Performance Tips

1. **Parallel Processing**: Use `--parallel` for batch backtests
2. **Turbo Mode**: Ensure turbo mode is enabled for large datasets
3. **Optimal Workers**: Use the CLI optimizer to find optimal worker count
4. **Memory Management**: Limit klines for large datasets to avoid memory issues
5. **Batch Size**: Process smaller batches for better memory management