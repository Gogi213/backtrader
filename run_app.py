"""
Main Entry Point for Jesse Bollinger Bands Mean Reversion Strategy Backtester

This module provides the main entry point for both CLI and GUI applications.
"""
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Jesse Bollinger Bands Mean Reversion Strategy Backtester")
    parser.add_argument("mode", choices=["cli", "gui"], help="Run mode: cli for command-line interface, gui for graphical interface")
    
    # CLI arguments
    parser.add_argument("--data-file", help="Path to the CSV data file (required for CLI mode)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--period", type=int, default=200, help="Bollinger Bands period (default: 200)")
    parser.add_argument("--std-dev", type=float, default=3.0, help="Standard deviation multiplier (default: 3.0)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital (default: 10000.0)")
    
    args = parser.parse_args()
    
    if args.mode == "cli":
        if not args.data_file:
            print("Error: --data-file is required for CLI mode")
            sys.exit(1)
        
        # Import and run CLI
        from src.cli.cli import main as cli_main
        # Override sys.argv to pass arguments to CLI
        sys.argv = [
            "cli.py",
            args.data_file,
            "--symbol", args.symbol,
            "--period", str(args.period),
            "--std-dev", str(args.std_dev),
            "--capital", str(args.capital),
            "--verbose"
        ]
        cli_main()
        
    elif args.mode == "gui":
        # Import and run GUI
        from src.gui.gui_visualizer import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()