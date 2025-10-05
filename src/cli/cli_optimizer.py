"""
CLI Performance Optimizer

This module provides utilities for optimizing CLI performance
and migrating existing CLI scripts to use the new unified system.

Author: HFT System
"""
import os
import sys
import time
import json
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import BacktestManager, BacktestConfig


class CLIOptimizer:
    """
    Optimizer for CLI performance and migration utilities
    """
    
    def __init__(self):
        """Initialize the CLI optimizer"""
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_workers = max(1, self.cpu_count - 1)  # Leave one core free
    
    def get_optimal_workers(self, batch_size: int) -> int:
        """
        Get optimal number of workers for batch processing
        
        Args:
            batch_size: Number of backtests in batch
            
        Returns:
            Optimal number of workers
        """
        # Don't use more workers than backtests
        return min(self.optimal_workers, batch_size)
    
    def estimate_execution_time(self, configs: List[BacktestConfig], 
                               sample_size: int = 1) -> float:
        """
        Estimate execution time for batch backtests
        
        Args:
            configs: List of backtest configurations
            sample_size: Number of configs to sample for estimation
            
        Returns:
            Estimated execution time in seconds
        """
        if not configs:
            return 0.0
        
        # Sample configs for estimation
        sample_configs = configs[:min(sample_size, len(configs))]
        
        # Run sample backtests
        manager = BacktestManager()
        start_time = time.time()
        
        for config in sample_configs:
            # Use small max_klines for faster estimation
            config.max_klines = min(config.max_klines or 1000, 100)
            result = manager.run_backtest(config)
            if not result.is_successful():
                print(f"Warning: Sample backtest failed: {result.get_error()}")
        
        sample_time = time.time() - start_time
        
        if sample_time == 0:
            return 0.0
        
        # Estimate total time
        avg_time_per_backtest = sample_time / len(sample_configs)
        estimated_total = avg_time_per_backtest * len(configs)
        
        # Apply parallelization factor
        if len(configs) > 1:
            parallel_factor = min(self.optimal_workers, len(configs))
            estimated_total /= parallel_factor
        
        return estimated_total
    
    def optimize_batch_config(self, batch_config_path: str, 
                             output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize batch configuration for better performance
        
        Args:
            batch_config_path: Path to batch configuration file
            output_path: Path to save optimized configuration
            
        Returns:
            Optimization results
        """
        with open(batch_config_path, 'r') as f:
            batch_config = json.load(f)
        
        # Create BacktestConfig objects
        configs = []
        for config_data in batch_config.get('backtests', []):
            config = BacktestConfig.from_dict(config_data)
            configs.append(config)
        
        # Get optimal workers
        optimal_workers = self.get_optimal_workers(len(configs))
        
        # Estimate execution time
        estimated_time = self.estimate_execution_time(configs)
        
        # Optimization suggestions
        suggestions = []
        
        # Check for large datasets
        large_datasets = sum(1 for config in configs 
                           if config.max_klines and config.max_klines > 50000)
        if large_datasets > 0:
            suggestions.append(f"Consider reducing max_klines for {large_datasets} configurations with large datasets")
        
        # Check for turbo mode
        turbo_disabled = sum(1 for config in configs if not config.enable_turbo_mode)
        if turbo_disabled > 0:
            suggestions.append(f"Enable turbo mode for {turbo_disabled} configurations to improve performance")
        
        # Check for parallel execution
        if len(configs) > 1 and optimal_workers == 1:
            suggestions.append("Consider running on a machine with more CPU cores for better parallel performance")
        
        # Create optimized configuration
        optimized_config = batch_config.copy()
        
        # Enable turbo mode for all configs
        for config in optimized_config.get('backtests', []):
            if 'enable_turbo_mode' not in config:
                config['enable_turbo_mode'] = True
        
        # Save optimized configuration
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(optimized_config, f, indent=2)
        
        return {
            'original_configs': len(configs),
            'optimal_workers': optimal_workers,
            'estimated_time_seconds': estimated_time,
            'estimated_time_minutes': estimated_time / 60,
            'suggestions': suggestions,
            'optimized_config_saved': output_path is not None
        }
    
    def migrate_legacy_script(self, script_path: str, 
                            output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Migrate legacy CLI script to use unified system
        
        Args:
            script_path: Path to legacy script
            output_path: Path to save migrated script
            
        Returns:
            Migration results
        """
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Migration patterns
        migrations = []
        migrated_content = script_content
        
        # Check for old import patterns
        if 'from src.data.backtest_engine import run_vectorized_klines_backtest' in script_content:
            migrated_content = migrated_content.replace(
                'from src.data.backtest_engine import run_vectorized_klines_backtest',
                'from src.core import BacktestManager, BacktestConfig'
            )
            migrations.append("Updated import to use unified system")
        
        # Check for old function calls
        if 'run_vectorized_klines_backtest(' in script_content:
            # This is a simplified migration - in practice, you'd need more complex parsing
            migrations.append("Detected legacy function call - manual migration may be required")
        
        # Check for argparse patterns
        if '--use-unified' in script_content:
            migrated_content = migrated_content.replace('--use-unified', '--legacy')
            migrations.append("Updated CLI argument from --use-unified to --legacy")
        
        # Add unified system usage example
        if 'from src.core import BacktestManager, BacktestConfig' in migrated_content:
            example_comment = """
# Example usage with unified system:
# config = BacktestConfig(
#     strategy_name='hierarchical_mean_reversion',
#     symbol='BTCUSDT',
#     data_path='data.csv'
# )
# manager = BacktestManager()
# results = manager.run_backtest(config)
"""
            if '# Example usage with unified system:' not in migrated_content:
                migrated_content += example_comment
                migrations.append("Added usage example for unified system")
        
        # Save migrated script
        if output_path:
            with open(output_path, 'w') as f:
                f.write(migrated_content)
        
        return {
            'migrations_detected': len(migrations),
            'migrations': migrations,
            'migrated_script_saved': output_path is not None,
            'manual_migration_required': 'Detected legacy function call - manual migration may be required' in migrations
        }
    
    def benchmark_performance(self, configs: List[BacktestConfig], 
                            workers_list: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Benchmark performance with different worker counts
        
        Args:
            configs: List of backtest configurations
            workers_list: List of worker counts to test
            
        Returns:
            Benchmark results
        """
        if workers_list is None:
            workers_list = [1, 2, 4, self.optimal_workers]
        
        # Remove duplicates and ensure valid values
        workers_list = list(set(w for w in workers_list if w > 0 and w <= len(configs)))
        workers_list.sort()
        
        results = {}
        
        for workers in workers_list:
            print(f"Benchmarking with {workers} workers...")
            
            manager = BacktestManager()
            start_time = time.time()
            
            batch_results = manager.run_batch_backtest(configs, parallel=(workers > 1), max_workers=workers)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            successful = sum(1 for result in batch_results if result.is_successful())
            
            results[workers] = {
                'execution_time': execution_time,
                'successful_backtests': successful,
                'failed_backtests': len(batch_results) - successful,
                'throughput': successful / execution_time if execution_time > 0 else 0
            }
        
        # Find optimal workers
        optimal_workers = max(results.keys(), key=lambda w: results[w]['throughput'])
        
        return {
            'results': results,
            'optimal_workers': optimal_workers,
            'optimal_throughput': results[optimal_workers]['throughput'],
            'recommendation': f"Use {optimal_workers} workers for best performance"
        }


def main():
    """CLI entry point for CLI optimizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CLI Performance Optimizer')
    parser.add_argument('--optimize-batch', help='Optimize batch configuration file')
    parser.add_argument('--migrate-script', help='Migrate legacy CLI script')
    parser.add_argument('--benchmark', help='Benchmark batch configuration file')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--workers', type=int, help='Number of workers to test')
    
    args = parser.parse_args()
    
    optimizer = CLIOptimizer()
    
    if args.optimize_batch:
        if not os.path.exists(args.optimize_batch):
            print(f"Error: Batch configuration file {args.optimize_batch} not found")
            sys.exit(1)
        
        results = optimizer.optimize_batch_config(args.optimize_batch, args.output)
        
        print("\n" + "="*60)
        print("BATCH CONFIGURATION OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Original configurations: {results['original_configs']}")
        print(f"Optimal workers: {results['optimal_workers']}")
        print(f"Estimated time: {results['estimated_time_minutes']:.2f} minutes")
        
        if results['suggestions']:
            print("\nOptimization suggestions:")
            for suggestion in results['suggestions']:
                print(f"  - {suggestion}")
        
        if results['optimized_config_saved']:
            print(f"\nOptimized configuration saved to: {args.output}")
    
    elif args.migrate_script:
        if not os.path.exists(args.migrate_script):
            print(f"Error: Script file {args.migrate_script} not found")
            sys.exit(1)
        
        results = optimizer.migrate_legacy_script(args.migrate_script, args.output)
        
        print("\n" + "="*60)
        print("SCRIPT MIGRATION RESULTS")
        print("="*60)
        print(f"Migrations detected: {results['migrations_detected']}")
        
        if results['migrations']:
            print("\nMigrations applied:")
            for migration in results['migrations']:
                print(f"  - {migration}")
        
        if results['manual_migration_required']:
            print("\n⚠️  Manual migration may be required for some function calls")
        
        if results['migrated_script_saved']:
            print(f"\nMigrated script saved to: {args.output}")
    
    elif args.benchmark:
        if not os.path.exists(args.benchmark):
            print(f"Error: Batch configuration file {args.benchmark} not found")
            sys.exit(1)
        
        # Load configurations
        with open(args.benchmark, 'r') as f:
            batch_config = json.load(f)
        
        configs = []
        for config_data in batch_config.get('backtests', []):
            config = BacktestConfig.from_dict(config_data)
            configs.append(config)
        
        # Run benchmark
        workers_list = [args.workers] if args.workers else None
        results = optimizer.benchmark_performance(configs, workers_list)
        
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        for workers, result in results['results'].items():
            print(f"\nWorkers: {workers}")
            print(f"  Execution time: {result['execution_time']:.2f}s")
            print(f"  Successful: {result['successful_backtests']}")
            print(f"  Failed: {result['failed_backtests']}")
            print(f"  Throughput: {result['throughput']:.2f} backtests/sec")
        
        print(f"\n{results['recommendation']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()