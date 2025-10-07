#!/usr/bin/env python3
"""
Detailed Performance Profiling Script for HFT System

This script provides comprehensive profiling that shows EXACTLY what functions are slow
and where the bottlenecks are located. No more useless summaries!

Usage:
    python detailed_profile_performance.py --strategy turbo_mean_reversion --data upload/klines/BTCUSDT_1m.csv
    python detailed_profile_performance.py --optimization --trials 10 --strategy turbo_mean_reversion

Author: HFT System
"""
import argparse
import os
import sys
import time
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.profiling.detailed_profiler import StrategyProfiler, OptunaProfiler, profile_strategy_function, profile_optuna_function
from src.data.klines_handler import UltraFastKlinesHandler
from src.strategies.strategy_registry import StrategyRegistry
from src.optimization.fast_optimizer import FastStrategyOptimizer


def profile_strategy_performance(strategy_name: str, data_path: str, 
                                sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Profile strategy execution performance with detailed analysis
    
    Args:
        strategy_name: Name of strategy to profile
        data_path: Path to data file
        sample_size: Optional sample size for testing
        
    Returns:
        Dictionary with detailed profiling results
    """
    print(f"DETAILED PROFILING: {strategy_name}")
    print(f"Data file: {data_path}")
    if sample_size:
        print(f"Sample size: {sample_size}")
    print("=" * 60)
    
    # Load data
    handler = UltraFastKlinesHandler()
    klines_data = handler.load_klines(data_path)
    
    # Create strategy instance
    strategy_class = StrategyRegistry.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"Strategy '{strategy_name}' not found")
    
    default_params = strategy_class.get_default_params()
    strategy = strategy_class(symbol='BTCUSDT', **default_params)
    
    # Initialize detailed profiler
    profiler = StrategyProfiler(
        enable_line_profiling=True,
        enable_memory_profiling=True
    )
    
    # Profile strategy execution
    print("Profiling strategy execution...")
    start_time = time.time()
    
    profile_result = profiler.profile_strategy_execution(
        strategy, klines_data, sample_size
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nStrategy profiling completed in {total_time:.2f} seconds")
    
    # Get detailed analysis
    slow_functions = profiler.get_slow_functions(top_n=10)
    hotspots = profiler.get_function_hotspots()
    
    # Print results
    print("\n" + "=" * 60)
    print("DETAILED PROFILING RESULTS")
    print("=" * 60)
    
    print(f"\nEXECUTION TIME: {profile_result['execution_time']:.4f} seconds")
    
    if profile_result.get('memory_profile_stats'):
        memory_diff = profile_result['memory_profile_stats'].get('memory_diff_mb', 0)
        print(f"MEMORY USAGE: {memory_diff:.2f} MB")
    
    # Strategy analysis
    strategy_analysis = profile_result.get('strategy_analysis', {})
    if strategy_analysis:
        print(f"\nSTRATEGY ANALYSIS:")
        print(f"  Total trades: {strategy_analysis.get('total_trades', 0)}")
        print(f"  Total bars: {strategy_analysis.get('total_bars', 0)}")
        print(f"  Processing rate: {strategy_analysis.get('processing_rate', 0):.0f} bars/sec")
        print(f"  Trade rate: {strategy_analysis.get('trade_rate', 0):.6f}")
        print(f"  Performance grade: {strategy_analysis.get('performance_grade', 'Unknown')}")
    
    # Print slowest functions
    if slow_functions:
        print(f"\nTOP {len(slow_functions)} SLOWEST FUNCTIONS:")
        for i, func in enumerate(slow_functions, 1):
            print(f"  {i}. {func['function']}")
            print(f"     Calls: {func['calls']:,}")
            print(f"     Total time: {func['cumulative_time']:.4f}s")
            print(f"     Time per call: {func['per_call']:.6f}s")
    
    # Print hotspots
    if hotspots:
        print(f"\nHOTSPOTS ANALYSIS:")
        print(f"  Total functions: {hotspots['total_functions']}")
        print(f"  Total calls: {hotspots['total_calls']:,}")
        print(f"  Total time: {hotspots['total_time']:.4f}s")
        
        # Most expensive functions
        expensive = hotspots.get('expensive_functions', [])[:5]
        if expensive:
            print(f"\n  MOST EXPENSIVE FUNCTIONS (time per call):")
            for i, func in enumerate(expensive, 1):
                print(f"    {i}. {func['function']}: {func['per_call']:.6f}s")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"profiling_reports/detailed_strategy_profile_{strategy_name}_{timestamp}.md"
    profiler.save_detailed_report(report_path)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return {
        'strategy_name': strategy_name,
        'execution_time': profile_result['execution_time'],
        'memory_usage': profile_result.get('memory_profile_stats', {}),
        'slow_functions': slow_functions,
        'hotspots': hotspots,
        'strategy_analysis': strategy_analysis,
        'report_path': report_path
    }


def profile_optimization_performance(strategy_name: str, data_path: str, 
                                   n_trials: int = 10, n_jobs: int = 1) -> Dict[str, Any]:
    """
    Profile Optuna optimization performance with detailed analysis
    
    Args:
        strategy_name: Name of strategy to optimize
        data_path: Path to data file
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with detailed profiling results
    """
    print(f"DETAILED OPTIMIZATION PROFILING: {strategy_name}")
    print(f"Data file: {data_path}")
    print(f"Trials: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print("=" * 60)
    
    # Initialize detailed profiler
    profiler = OptunaProfiler(enable_resource_monitoring=True)
    
    # Create optimizer
    optimizer = FastStrategyOptimizer(
        strategy_name=strategy_name,
        data_path=data_path,
        symbol='BTCUSDT'
    )
    
    # Create objective function
    objective_func = optimizer.create_objective_function()
    
    # Profile each trial
    print("Profiling optimization trials...")
    profiler.optimization_start_time = time.time()
    
    import optuna
    study = optuna.create_study(direction='maximize')
    
    for i in range(n_trials):
        print(f"  Profiling trial {i+1}/{n_trials}...")
        
        # Create a trial manually for profiling
        trial = study.ask()
        
        # Profile the objective function
        try:
            result = profiler.profile_optimization_trial(
                i, objective_func, trial
            )
            study.tell(trial, result)
        except optuna.exceptions.TrialPruned:
            study.tell(trial, float('-inf'), skip_if_finished=True)
        except Exception as e:
            print(f"    Trial failed: {e}")
            study.tell(trial, float('-inf'), skip_if_finished=True)
    
    profiler.optimization_end_time = time.time()
    
    # Get detailed analysis
    analysis = profiler.get_optimization_analysis()
    slow_functions = profiler.get_slow_functions(top_n=10)
    hotspots = profiler.get_function_hotspots()
    
    # Print results
    print("\n" + "=" * 60)
    print("DETAILED OPTIMIZATION PROFILING RESULTS")
    print("=" * 60)
    
    if analysis:
        print(f"\nOPTIMIZATION ANALYSIS:")
        print(f"  Total trials: {analysis['total_trials']}")
        print(f"  Total time: {analysis['total_time']:.4f}s")
        print(f"  Average trial time: {analysis['avg_trial_time']:.4f}s")
        print(f"  Median trial time: {analysis['median_trial_time']:.4f}s")
        print(f"  Min trial time: {analysis['min_trial_time']:.4f}s")
        print(f"  Max trial time: {analysis['max_trial_time']:.4f}s")
        print(f"  Std trial time: {analysis['std_trial_time']:.4f}s")
        
        if 'optimization_efficiency' in analysis:
            print(f"  Optimization efficiency: {analysis['optimization_efficiency']:.2%}")
        
        # Slowest trials
        slowest_trials = analysis.get('slowest_trials', [])
        if slowest_trials:
            print(f"\n  SLOWEST TRIALS:")
            for i, (trial_num, trial_time) in enumerate(slowest_trials, 1):
                print(f"    {i}. Trial {trial_num}: {trial_time:.4f}s")
    
    # Print slowest functions
    if slow_functions:
        print(f"\nTOP {len(slow_functions)} SLOWEST FUNCTIONS:")
        for i, func in enumerate(slow_functions, 1):
            print(f"  {i}. {func['function']}")
            print(f"     Calls: {func['calls']:,}")
            print(f"     Total time: {func['cumulative_time']:.4f}s")
            print(f"     Time per call: {func['per_call']:.6f}s")
    
    # Print hotspots
    if hotspots:
        print(f"\nFUNCTION HOTSPOTS:")
        print(f"  Total functions: {hotspots['total_functions']}")
        print(f"  Total calls: {hotspots['total_calls']:,}")
        print(f"  Total time: {hotspots['total_time']:.4f}s")
        
        # Most expensive functions
        expensive = hotspots.get('expensive_functions', [])[:5]
        if expensive:
            print(f"\n  MOST EXPENSIVE FUNCTIONS (time per call):")
            for i, func in enumerate(expensive, 1):
                print(f"    {i}. {func['function']}: {func['per_call']:.6f}s")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"profiling_reports/detailed_optimization_profile_{strategy_name}_{timestamp}.md"
    profiler.save_optimization_report(report_path)
    
    print(f"\nDetailed optimization report saved to: {report_path}")
    
    return {
        'strategy_name': strategy_name,
        'optimization_analysis': analysis,
        'slow_functions': slow_functions,
        'hotspots': hotspots,
        'report_path': report_path
    }


def main():
    """Main profiling script"""
    parser = argparse.ArgumentParser(description='Detailed performance profiling for HFT system')
    
    # Common arguments
    parser.add_argument('--strategy', type=str, default='hierarchical_mean_reversion',
                       help='Strategy name to profile')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data file')
    
    # Strategy profiling arguments
    parser.add_argument('--profile-strategy', action='store_true',
                       help='Profile strategy execution')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for strategy profiling')
    
    # Optimization profiling arguments
    parser.add_argument('--profile-optimization', action='store_true',
                       help='Profile optimization process')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of optimization trials')
    parser.add_argument('--jobs', type=int, default=1,
                       help='Number of parallel jobs')
    
    # Report arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.profile_strategy and not args.profile_optimization:
        print("Error: Must specify either --profile-strategy or --profile-optimization")
        return 1
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return 1
    
    # Store results
    results = {}
    
    try:
        # Profile strategy execution
        if args.profile_strategy:
            print("\n" + "="*60)
            print("STRATEGY PROFILING")
            print("="*60)
            
            strategy_result = profile_strategy_performance(
                args.strategy, args.data, args.sample_size
            )
            results['strategy'] = strategy_result
        
        # Profile optimization process
        if args.profile_optimization:
            print("\n" + "="*60)
            print("OPTIMIZATION PROFILING")
            print("="*60)
            
            optimization_result = profile_optimization_performance(
                args.strategy, args.data, args.trials, args.jobs
            )
            results['optimization'] = optimization_result
        
        print("\n" + "="*60)
        print("DETAILED PROFILING COMPLETED")
        print("="*60)
        
        # Summary
        if 'strategy' in results:
            strategy = results['strategy']
            print(f"Strategy '{strategy['strategy_name']}': {strategy['execution_time']:.4f}s")
            if strategy['hotspots']:
                print(f"  Total functions profiled: {strategy['hotspots']['total_functions']}")
                print(f"  Slowest function: {strategy['slow_functions'][0]['function'] if strategy['slow_functions'] else 'N/A'}")
        
        if 'optimization' in results:
            optimization = results['optimization']
            analysis = optimization['optimization_analysis']
            if analysis:
                print(f"Optimization '{optimization['strategy_name']}': {analysis['total_time']:.4f}s")
                print(f"  Average trial time: {analysis['avg_trial_time']:.4f}s")
                if optimization['slow_functions']:
                    print(f"  Slowest function: {optimization['slow_functions'][0]['function']}")
        
        return 0
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())