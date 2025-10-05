"""
CLI Interface for Strategy Parameter Optimization using Optuna

This module provides command-line interface for optimizing trading strategy
parameters using the Optuna framework.

Author: HFT System
"""
import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

from .optuna_optimizer import StrategyOptimizer, create_composite_objective
from ..strategies.strategy_registry import StrategyRegistry


def parse_optimization_args():
    """Parse command line arguments for optimization"""
    parser = argparse.ArgumentParser(
        description='Optimize trading strategy parameters using Optuna',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--csv', required=True, help='Path to CSV file with klines data')
    parser.add_argument('--strategy', required=True, help='Strategy name to optimize')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    
    # Optimization parameters
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--objective', default='sharpe_ratio',
                       choices=['sharpe_ratio', 'net_pnl', 'profit_factor', 'win_rate', 'net_pnl_percentage'],
                       help='Optimization objective metric')
    parser.add_argument('--min-trades', type=int, default=10, help='Minimum number of trades required')
    parser.add_argument('--max-drawdown', type=float, default=50.0, help='Maximum allowed drawdown percentage')
    parser.add_argument('--timeout', type=float, help='Optimization timeout in seconds')
    
    # Performance optimization options
    parser.add_argument('--fast', action='store_true',
                       help='Use fast optimizer with caching and parallel processing')
    parser.add_argument('--jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Disable adaptive evaluation (use full data for all trials)')
    
    # Advanced optimization options
    parser.add_argument('--sampler', default='tpe',
                       choices=['tpe', 'random', 'cmaes', 'nsgaii', 'motpe'],
                       help='Optimization sampler algorithm')
    parser.add_argument('--pruner', default='hyperband',
                       choices=['median', 'hyperband', 'successive_halving', 'none'],
                       help='Optimization pruner algorithm')
    parser.add_argument('--multivariate', action='store_true',
                       help='Enable multivariate TPE sampling (considers parameter relationships)')
    parser.add_argument('--aggressive-pruning', action='store_true',
                       help='Enable more aggressive pruning for faster optimization')
    
    # Study configuration
    parser.add_argument('--study-name', help='Study name (auto-generated if not provided)')
    parser.add_argument('--direction', default='maximize', choices=['maximize', 'minimize'],
                       help='Optimization direction')
    parser.add_argument('--storage', help='Database URL for persistent storage')
    
    # Output options
    parser.add_argument('--output', help='Output file for results (JSON format)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--plot', action='store_true', help='Show optimization plots')
    
    # List strategies
    parser.add_argument('--list-strategies', action='store_true', help='List available strategies')
    
    return parser.parse_args()


def list_available_strategies():
    """List all available strategies with their parameter spaces"""
    print("Available strategies for optimization:")
    print("=" * 60)
    
    for strategy_name in StrategyRegistry.list_strategies():
        strategy_class = StrategyRegistry.get(strategy_name)
        if strategy_class:
            print(f"\n{strategy_name}:")
            print(f"  Class: {strategy_class.__name__}")
            
            # Get parameter space
            try:
                param_space = strategy_class.get_param_space()
                print(f"  Parameters ({len(param_space)}):")
                for param_name, (param_type, *bounds) in param_space.items():
                    if param_type == 'float':
                        if len(bounds) == 2:
                            print(f"    {param_name}: float [{bounds[0]}, {bounds[1]}]")
                        else:
                            print(f"    {param_name}: float [{bounds[0]}, {bounds[1]}] step={bounds[2]}")
                    elif param_type == 'int':
                        if len(bounds) == 2:
                            print(f"    {param_name}: int [{bounds[0]}, {bounds[1]}]")
                        else:
                            print(f"    {param_name}: int [{bounds[0]}, {bounds[1]}] step={bounds[2]}")
                    elif param_type == 'categorical':
                        print(f"    {param_name}: categorical {bounds}")
            except Exception as e:
                print(f"  Error getting parameter space: {e}")
    
    print("\n" + "=" * 60)


def run_optimization(args) -> Dict[str, Any]:
    """Run optimization with specified arguments"""
    
    # Validate strategy exists
    strategy_class = StrategyRegistry.get(args.strategy)
    if not strategy_class:
        print(f"Error: Strategy '{args.strategy}' not found")
        print("Available strategies:", StrategyRegistry.list_strategies())
        sys.exit(1)
    
    # Validate data file exists
    if not os.path.exists(args.csv):
        print(f"Error: Data file '{args.csv}' not found")
        sys.exit(1)
    
    print(f"Starting optimization for strategy: {args.strategy}")
    print(f"Data file: {args.csv}")
    print(f"Symbol: {args.symbol}")
    print(f"Objective: {args.objective}")
    print(f"Trials: {args.trials}")
    print(f"Direction: {args.direction}")
    print(f"Fast mode: {args.fast}")
    if args.fast:
        print(f"Parallel jobs: {args.jobs}")
        print(f"Adaptive evaluation: {not args.no_adaptive}")
    print(f"Sampler: {args.sampler}")
    print(f"Pruner: {args.pruner}")
    if args.multivariate:
        print("Multivariate sampling: ENABLED")
    if args.aggressive_pruning:
        print("Aggressive pruning: ENABLED")
    print("-" * 60)
    
    # Create sampler based on arguments
    import optuna
    sampler = None
    if args.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(
            seed=42,
            multivariate=args.multivariate,
            group=args.multivariate,
            n_startup_trials=10 if args.multivariate else 5
        )
    elif args.sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif args.sampler == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    elif args.sampler == 'nsgaii':
        sampler = optuna.samplers.NSGAIISampler(seed=42)
    elif args.sampler == 'motpe':
        sampler = optuna.samplers.MOTPESampler(seed=42)
    
    # Create pruner based on arguments
    pruner = None
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    elif args.pruner == 'hyperband':
        if args.aggressive_pruning:
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource='auto',
                reduction_factor=2  # More aggressive
            )
        else:
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource='auto',
                reduction_factor=3
            )
    elif args.pruner == 'successive_halving':
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif args.pruner == 'none':
        pruner = optuna.pruners.NopPruner()
    
    # Choose optimizer based on fast mode
    if args.fast:
        # Use fast optimizer
        from .fast_optimizer import FastStrategyOptimizer
        
        optimizer = FastStrategyOptimizer(
            strategy_name=args.strategy,
            data_path=args.csv,
            symbol=args.symbol,
            study_name=args.study_name,
            direction=args.direction,
            storage=args.storage
        )
        
        # Run optimization with fast optimizer
        results = optimizer.optimize(
            n_trials=args.trials,
            objective_metric=args.objective,
            min_trades=args.min_trades,
            max_drawdown_threshold=args.max_drawdown,
            timeout=args.timeout,
            n_jobs=args.jobs,
            use_adaptive=not args.no_adaptive,
            sampler=sampler,
            pruner=pruner
        )
    else:
        # Use standard optimizer
        optimizer = StrategyOptimizer(
            strategy_name=args.strategy,
            data_path=args.csv,
            symbol=args.symbol,
            study_name=args.study_name,
            direction=args.direction,
            storage=args.storage
        )
        
        # Run optimization
        results = optimizer.optimize(
            n_trials=args.trials,
            objective_metric=args.objective,
            min_trades=args.min_trades,
            max_drawdown_threshold=args.max_drawdown,
            timeout=args.timeout,
            sampler=sampler,
            pruner=pruner
        )
    
    # Display results
    display_optimization_results(results, args.verbose)
    
    # Show parameter importance
    if args.verbose:
        print("\nParameter Importance:")
        print("-" * 30)
        importance = optimizer.get_parameter_importance()
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{param}: {score:.4f}")
    
    # Show plots if requested
    if args.plot:
        try:
            import optuna.visualization as vis
            import plotly
            
            # Plot optimization history
            fig1 = vis.plot_optimization_history(optimizer.study)
            fig1.show()
            
            # Plot parameter importance
            if importance:
                fig2 = vis.plot_param_importances(optimizer.study)
                fig2.show()
                
            # Plot parallel coordinate
            fig3 = vis.plot_parallel_coordinate(optimizer.study)
            fig3.show()
            
        except ImportError:
            print("Warning: Plotting requires 'optuna[visualization]' and 'plotly'")
            print("Install with: pip install optuna[visualization] plotly")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    # Save results if requested
    if args.output:
        optimizer.save_results(args.output)
        print(f"\nResults saved to: {args.output}")
    
    return results


def display_optimization_results(results: Dict[str, Any], verbose: bool = False):
    """Display optimization results"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"Strategy: {results.get('strategy_name', 'N/A')}")
    print(f"Symbol: {results.get('symbol', 'N/A')}")
    print(f"Objective: {results.get('objective_metric', 'N/A')}")
    print(f"Best Value: {results.get('best_value', 'N/A'):.4f}")
    print(f"Total Trials: {results.get('n_trials', 0)}")
    print(f"Successful Trials: {results.get('successful_trials', 0)}")
    print(f"Optimization Time: {results.get('optimization_time_seconds', 0):.2f} seconds")
    
    print("\nBest Parameters:")
    print("-" * 30)
    best_params = results.get('best_params', {})
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Show final backtest results
    final_backtest = results.get('final_backtest')
    if final_backtest:
        print("\nFinal Backtest Results:")
        print("-" * 30)
        print(f"Total Trades: {final_backtest.get('total', 0)}")
        print(f"Win Rate: {final_backtest.get('win_rate', 0):.1%}")
        print(f"Net P&L: ${final_backtest.get('net_pnl', 0):,.2f}")
        print(f"Return: {final_backtest.get('net_pnl_percentage', 0):.2f}%")
        print(f"Sharpe Ratio: {final_backtest.get('sharpe_ratio', 0):.2f}")
        print(f"Profit Factor: {final_backtest.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {final_backtest.get('max_drawdown', 0):.2f}%")
        
        if verbose:
            print(f"\nWinners: {final_backtest.get('total_winning_trades', 0)}")
            print(f"Losers: {final_backtest.get('total_losing_trades', 0)}")
            print(f"Average Win: ${final_backtest.get('average_win', 0):.2f}")
            print(f"Average Loss: ${final_backtest.get('average_loss', 0):.2f}")
            print(f"Best Trade: ${final_backtest.get('largest_win', 0):.2f}")
            print(f"Worst Trade: ${final_backtest.get('largest_loss', 0):.2f}")
    
    print("=" * 60)


def create_composite_optimization_config() -> Dict[str, float]:
    """Create a composite objective configuration"""
    # Example: Weighted combination of multiple metrics
    return {
        'sharpe_ratio': 0.4,
        'net_pnl_percentage': 0.3,
        'profit_factor': 0.2,
        'win_rate': 0.1
    }


def main():
    """Main entry point for optimization CLI"""
    args = parse_optimization_args()
    
    # List strategies if requested
    if args.list_strategies:
        list_available_strategies()
        sys.exit(0)
    
    # Run optimization
    try:
        results = run_optimization(args)
        
        # Exit with appropriate code
        if results.get('successful_trials', 0) > 0:
            print("\nOptimization completed successfully!")
            sys.exit(0)
        else:
            print("\nOptimization failed - no successful trials!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nOptimization failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()