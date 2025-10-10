import argparse
import os
import sys
from datetime import datetime

def add_src_to_path():
    """
    Adds the 'src' directory to the Python path to allow for absolute imports
    from 'src'. Assumes 'utils.py' is in the project root alongside 'src'.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

def get_available_strategies():
    """Dynamically retrieves the list of available strategies."""
    # This import is done here to avoid circular dependencies and ensure src path is set
    from src.strategies import StrategyRegistry
    return StrategyRegistry.list_strategies()

def setup_cli_parser(description, epilog):
    """
    Creates and configures a base ArgumentParser for CLI tools.

    Args:
        description (str): The description for the parser.
        epilog (str): The epilog for the parser.

    Returns:
        argparse.ArgumentParser: A configured parser instance.
    """
    available_strategies = get_available_strategies()
    if not available_strategies:
        print("Error: No strategies found. Ensure strategies are defined in 'src/strategies'.")
        sys.exit(1)
    
    default_strategy = 'ported_from_example' if 'ported_from_example' in available_strategies else available_strategies[0]

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )

    # --- Common Arguments ---
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default=default_strategy,
        choices=available_strategies,
        help=f"Strategy to use (default: {default_strategy})"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to the CSV/Parquet data file (default: finds first file in upload/klines)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol (default: inferred from filename)"
    )
    
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=-1,
        help="Number of parallel jobs for optimization (-1 for all cores, default: -1)"
    )
    
    return parser

def handle_data_and_symbol(args):
    """
    Handles the logic for default data file and symbol inference.
    Modifies the 'args' namespace object in place and returns it.
    """
    if args.data is None:
        klines_dir = "upload/klines"
        if os.path.exists(klines_dir):
            # Sort to get a predictable file
            data_files = sorted([f for f in os.listdir(klines_dir) if f.endswith(('.csv', '.parquet'))])
            if data_files:
                args.data = os.path.join(klines_dir, data_files[0])
                print(f"Using default dataset: {args.data}")
            else:
                print(f"Error: No data files (.csv or .parquet) found in {klines_dir}")
                sys.exit(1)
        else:
            print(f"Error: Directory {klines_dir} not found")
            sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    if args.symbol is None:
        filename = os.path.basename(args.data)
        # More robust symbol extraction, handles formats like 'ETHUSDT-1m.csv' or 'BTC_USDT.parquet'
        args.symbol = filename.split('-')[0].split('_')[0].split('.')[0]

    return args