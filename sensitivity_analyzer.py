"""
Parameter Sensitivity Analyzer

This script performs a sensitivity analysis on strategy parameters using Optuna's
parameter importance features. It helps identify which parameters are most
influential on the optimization objective, fulfilling Task 1 and 2 from docs/tasks/tasks.txt.

Usage:
    python sensitivity_analyzer.py --strategy ported_from_example --trials 100

The script will:
1. Run an Optuna optimization study for the specified number of trials.
2. Calculate parameter importances using Optuna's built-in methods.
3. Print the ranked parameter importances to the console.
4. Save an interactive HTML plot of the importances to a file.
"""
import argparse
import sys
import os
import time
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.optimization.fast_optimizer import FastStrategyOptimizer
    from src.strategies import StrategyRegistry
    import optuna
    
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install optuna numpy pandas plotly")
    sys.exit(1)


def main():
    """Main CLI function for sensitivity analysis"""
    available_strategies = StrategyRegistry.list_strategies()
    if not available_strategies:
        print("Error: No strategies found. Ensure strategies are defined in 'src/strategies'.")
        sys.exit(1)

    default_strategy = 'ported_from_example' if 'ported_from_example' in available_strategies else available_strategies[0]
    
    parser = argparse.ArgumentParser(
        description="Analyze the sensitivity of strategy parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Example usage:
  %(prog)s --strategy {default_strategy} --trials 100 --jobs 4
  %(prog)s --strategy {default_strategy} --data upload/klines/ETHUSDT-1m.csv --trials 200
"""
    )
    
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default=default_strategy,
        choices=available_strategies,
        help=f"Strategy to analyze (default: {default_strategy})"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to the CSV data file (default: finds first file in upload/klines)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol (default: inferred from filename)"
    )
    
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=100,
        help="Number of optimization trials to run for analysis (default: 100)"
    )
    
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores, default: -1)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output HTML file name for the plot (default: sensitivity_plot_STRATEGY_TIMESTAMP.html)"
    )
    
    args = parser.parse_args()
    
    # --- Data and Symbol Handling (copied from cli_optimizer) ---
    if args.data is None:
        klines_dir = "upload/klines"
        if os.path.exists(klines_dir):
            data_files = [f for f in os.listdir(klines_dir) if f.endswith(('.csv', '.parquet'))]
            if data_files:
                args.data = os.path.join(klines_dir, data_files[0])
                print(f"Using default dataset: {data_files[0]}")
            else:
                print(f"Error: No data files (.csv or .parquet) found in {klines_dir}")
                sys.exit(1)
        else:
            print(f"Error: Directory {klines_dir} not found")
            sys.exit(1)
    
    if args.symbol is None:
        filename = os.path.basename(args.data)
        args.symbol = filename.split('-')[0] if '-' in filename else filename.split('.')[0]

    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    # --- Output file handling ---
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"sensitivity_plot_{args.strategy}_{timestamp}.html"

    # --- Print configuration ---
    print("=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"Strategy:      {args.strategy}")
    print(f"Data File:     {args.data}")
    print(f"Symbol:        {args.symbol}")
    print(f"Trials:        {args.trials}")
    print(f"Output Plot:   {args.output}")
    print("=" * 60)

    try:
        # --- Run Optimization ---
        print("Initializing optimizer...")
        from src.core.backtest_config import BacktestConfig
        backtest_config = BacktestConfig(
            strategy_name=args.strategy,
            symbol=args.symbol,
            data_path=args.data,
        )

        optimizer = FastStrategyOptimizer(
            strategy_name=args.strategy,
            data_path=args.data,
            symbol=args.symbol,
            backtest_config=backtest_config
        )
        
        print(f"Running optimization for {args.trials} trials to gather data...")
        optimizer.optimize(
            n_trials=args.trials,
            n_jobs=args.jobs
        )
        
        # --- Analyze and Display Results ---
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        if not optimizer.study:
            print("Error: The optimization study did not run correctly.")
            sys.exit(1)

        print("Calculating parameter importances...")
        
        # Get importances
        importances = optimizer.get_parameter_importance()
        
        if not importances:
            print("Could not calculate parameter importances. The study may have had no successful trials.")
            sys.exit(1)

        # Print ranked importances
        print("\nParameter Importances (most to least sensitive):")
        sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        for param, importance in sorted_importances:
            print(f"  - {param:<25}: {importance:.6f}")

        # Generate and save plot
        try:
            fig = optuna.visualization.plot_param_importances(optimizer.study)
            fig.write_html(args.output)
            print(f"\nSuccessfully saved interactive plot to: {args.output}")
        except Exception as e:
            print(f"\nError generating plot: {e}")
            print("You may need to install plotly: pip install plotly")

        print("\n" + "=" * 60)
        print("This analysis helps with Tasks 1 and 2:")
        print("1. You can see the most and least sensitive parameters.")
        print("2. For future optimizations, you can fix the least sensitive parameters to their default values to speed up the process.")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
