"""
Strategy Profiler for HFT System

Comprehensive profiling of trading strategies to identify performance bottlenecks.
Uses cProfile and line_profiler for detailed analysis.

Author: HFT System
"""
import cProfile
import pstats
import io
import time
import functools
from typing import Dict, Any, Callable, Optional, List
from contextlib import contextmanager
import numpy as np


try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False
    print("Warning: line_profiler not available. Install with: pip install line_profiler")

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not available. Install with: pip install memory_profiler")


class StrategyProfiler:
    """
    Comprehensive strategy profiler for identifying performance bottlenecks
    
    Features:
    - Function-level profiling with cProfile
    - Line-level profiling with line_profiler
    - Memory usage profiling
    - Execution time measurement
    - Performance comparison
    """
    
    def __init__(self, enable_line_profiling: bool = True, enable_memory_profiling: bool = True):
        """
        Initialize strategy profiler
        
        Args:
            enable_line_profiling: Enable line-level profiling
            enable_memory_profiling: Enable memory usage profiling
        """
        self.enable_line_profiling = enable_line_profiling and LINE_PROFILER_AVAILABLE
        self.enable_memory_profiling = enable_memory_profiling and MEMORY_PROFILER_AVAILABLE
        
        self.cprofile_stats = None
        self.line_profile_stats = None
        self.memory_profile_stats = None
        self.execution_times = {}
        
    @contextmanager
    def profile_strategy(self, strategy_name: str, profile_memory: bool = None):
        """
        Context manager for profiling strategy execution
        
        Args:
            strategy_name: Name of the strategy being profiled
            profile_memory: Override memory profiling setting
        """
        profile_memory = profile_memory if profile_memory is not None else self.enable_memory_profiling
        
        # Initialize profilers
        cprofiler = cProfile.Profile()
        line_profiler = LineProfiler() if self.enable_line_profiling else None
        memory_profiler_obj = None
        
        # Memory profiling setup
        if profile_memory and MEMORY_PROFILER_AVAILABLE:
            memory_profiler_obj = memory_profiler
            memory_before = memory_profiler_obj.memory_usage()[0]
        else:
            memory_before = None
            
        start_time = time.time()
        
        # Start profiling
        cprofiler.enable()
        if line_profiler:
            line_profiler.enable_by_count()
            
        try:
            yield self
        finally:
            # Stop profiling
            cprofiler.disable()
            if line_profiler:
                line_profiler.disable()
                
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Memory profiling results
            if profile_memory and MEMORY_PROFILER_AVAILABLE and memory_profiler_obj:
                memory_after = memory_profiler_obj.memory_usage()[0]
                memory_diff = memory_after - memory_before
            else:
                memory_diff = None
                
            # Store results
            self.execution_times[strategy_name] = execution_time
            
            # Process cProfile results
            s = io.StringIO()
            ps = pstats.Stats(cprofiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            self.cprofile_stats = s.getvalue()
            
            # Process line profiler results
            if line_profiler:
                line_s = io.StringIO()
                line_profiler.print_stats(stream=line_s)
                self.line_profile_stats = line_s.getvalue()
                
            # Store memory stats
            if memory_diff is not None:
                self.memory_profile_stats = {
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after,
                    'memory_diff_mb': memory_diff
                }
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator for profiling individual functions
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling enabled
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = StrategyProfiler(
                enable_line_profiling=self.enable_line_profiling,
                enable_memory_profiling=self.enable_memory_profiling
            )
            
            with profiler.profile_strategy(func.__name__):
                return func(*args, **kwargs)
                
            # Store results in function attribute
            wrapper._profile_results = {
                'cprofile_stats': profiler.cprofile_stats,
                'line_profile_stats': profiler.line_profile_stats,
                'memory_profile_stats': profiler.memory_profile_stats,
                'execution_time': profiler.execution_times.get(func.__name__)
            }
            
            return wrapper(*args, **kwargs)
            
        return wrapper
    
    def profile_strategy_method(self, strategy_instance, method_name: str, 
                               *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a specific method of a strategy instance
        
        Args:
            strategy_instance: Strategy instance
            method_name: Name of method to profile
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Dictionary with profiling results
        """
        method = getattr(strategy_instance, method_name)
        
        with self.profile_strategy(f"{strategy_instance.__class__.__name__}.{method_name}"):
            result = method(*args, **kwargs)
            
        return {
            'result': result,
            'cprofile_stats': self.cprofile_stats,
            'line_profile_stats': self.line_profile_stats,
            'memory_profile_stats': self.memory_profile_stats,
            'execution_time': self.execution_times.get(f"{strategy_instance.__class__.__name__}.{method_name}")
        }
    
    def profile_vectorized_process(self, strategy_instance, data, 
                                  sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Profile the vectorized_process_dataset method with optional sampling
        
        Args:
            strategy_instance: Strategy instance
            data: Data to process (DataFrame or dict)
            sample_size: Optional sample size for testing
            
        Returns:
            Dictionary with profiling results
        """
        # Sample data if requested
        if sample_size and hasattr(data, '__len__'):
            if hasattr(data, 'head'):  # DataFrame-like
                test_data = data.head(sample_size)
            elif hasattr(data, 'data'):  # NumpyKlinesData-like
                test_data = type(data)({
                    key: value[:sample_size] if value is not None else None
                    for key, value in data.data.items()
                })
            else:  # Dict-like
                test_data = {
                    key: value[:sample_size] if value is not None else None
                    for key, value in data.items()
                }
        else:
            test_data = data
            
        return self.profile_strategy_method(
            strategy_instance, 'vectorized_process_dataset', test_data
        )
    
    def profile_turbo_process(self, strategy_instance, times: np.ndarray, 
                             prices: np.ndarray, sample_size: Optional[int] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Profile the turbo_process_dataset method with optional sampling
        
        Args:
            strategy_instance: Strategy instance
            times: Time array
            prices: Price array
            sample_size: Optional sample size for testing
            **kwargs: Additional arguments for turbo_process_dataset
            
        Returns:
            Dictionary with profiling results
        """
        # Sample data if requested
        if sample_size:
            test_times = times[:sample_size]
            test_prices = prices[:sample_size]
            
            # Sample other arrays if provided
            other_arrays = {}
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray) and len(value) == len(times):
                    other_arrays[key] = value[:sample_size]
                else:
                    other_arrays[key] = value
        else:
            test_times = times
            test_prices = prices
            other_arrays = kwargs
            
        return self.profile_strategy_method(
            strategy_instance, 'turbo_process_dataset', 
            test_times, test_prices, **other_arrays
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all profiling results
        
        Returns:
            Dictionary with performance summary
        """
        summary = {
            'execution_times': self.execution_times,
            'total_execution_time': sum(self.execution_times.values()),
            'average_execution_time': np.mean(list(self.execution_times.values())) if self.execution_times else 0,
            'slowest_function': max(self.execution_times.items(), key=lambda x: x[1]) if self.execution_times else None,
            'fastest_function': min(self.execution_times.items(), key=lambda x: x[1]) if self.execution_times else None
        }
        
        if self.memory_profile_stats:
            summary['memory_usage'] = self.memory_profile_stats
            
        return summary
    
    def get_bottleneck_functions(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Extract bottleneck functions from cProfile results
        
        Args:
            top_n: Number of top functions to return
            
        Returns:
            List of bottleneck functions with stats
        """
        if not self.cprofile_stats:
            return []
            
        bottlenecks = []
        lines = self.cprofile_stats.split('\n')
        
        for line in lines[5:]:  # Skip header lines
            if line.strip() and not line.startswith(' '):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        calls = int(parts[0])
                        total_time = float(parts[2])
                        per_call = float(parts[3])
                        cum_time = float(parts[4])
                        func_name = ' '.join(parts[6:])
                        
                        bottlenecks.append({
                            'function': func_name,
                            'calls': calls,
                            'total_time': total_time,
                            'per_call': per_call,
                            'cumulative_time': cum_time
                        })
                    except (ValueError, IndexError):
                        continue
                        
        return sorted(bottlenecks, key=lambda x: x['cumulative_time'], reverse=True)[:top_n]
    
    def save_profile_report(self, filepath: str, include_details: bool = True) -> None:
        """
        Save profiling report to file
        
        Args:
            filepath: Path to save report
            include_details: Include detailed profiling stats
        """
        with open(filepath, 'w') as f:
            f.write("STRATEGY PROFILING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Performance summary
            summary = self.get_performance_summary()
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total execution time: {summary['total_execution_time']:.4f}s\n")
            f.write(f"Average execution time: {summary['average_execution_time']:.4f}s\n")
            
            if summary['slowest_function']:
                f.write(f"Slowest function: {summary['slowest_function'][0]} ({summary['slowest_function'][1]:.4f}s)\n")
            if summary['fastest_function']:
                f.write(f"Fastest function: {summary['fastest_function'][0]} ({summary['fastest_function'][1]:.4f}s)\n")
                
            if 'memory_usage' in summary:
                mem = summary['memory_usage']
                f.write(f"Memory usage: {mem['memory_diff_mb']:.2f}MB\n")
                
            f.write("\n")
            
            # Execution times
            f.write("EXECUTION TIMES BY FUNCTION\n")
            f.write("-" * 30 + "\n")
            for func, time_taken in sorted(summary['execution_times'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"{func}: {time_taken:.4f}s\n")
                
            f.write("\n")
            
            # Bottlenecks
            bottlenecks = self.get_bottleneck_functions()
            if bottlenecks:
                f.write("TOP BOTTLENECKS\n")
                f.write("-" * 15 + "\n")
                for i, bottleneck in enumerate(bottlenecks, 1):
                    f.write(f"{i}. {bottleneck['function']}\n")
                    f.write(f"   Calls: {bottleneck['calls']}\n")
                    f.write(f"   Total time: {bottleneck['total_time']:.4f}s\n")
                    f.write(f"   Per call: {bottleneck['per_call']:.6f}s\n")
                    f.write(f"   Cumulative: {bottleneck['cumulative_time']:.4f}s\n\n")
            
            # Detailed stats if requested
            if include_details:
                if self.cprofile_stats:
                    f.write("DETAILED CPROFILE STATS\n")
                    f.write("-" * 25 + "\n")
                    f.write(self.cprofile_stats)
                    f.write("\n")
                    
                if self.line_profile_stats:
                    f.write("DETAILED LINE PROFILE STATS\n")
                    f.write("-" * 30 + "\n")
                    f.write(self.line_profile_stats)
                    f.write("\n")
                    
        print(f"Profile report saved to {filepath}")


# Convenience decorator for quick profiling
def profile_strategy_function(func: Callable) -> Callable:
    """
    Quick decorator for profiling strategy functions
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    profiler = StrategyProfiler()
    return profiler.profile_function(func)