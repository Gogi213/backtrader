"""
Detailed Performance Profiler for HFT System

This module provides comprehensive profiling capabilities that show EXACTLY what functions
are slow and where the bottlenecks are located. No more useless summaries!

Author: HFT System
"""
import cProfile
import pstats
import io
import time
import functools
import linecache
import os
from typing import Dict, Any, Callable, Optional, List, Tuple
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


class DetailedProfiler:
    """
    Detailed profiler that shows EXACTLY what functions are slow
    """
    
    def __init__(self, enable_line_profiling: bool = True, enable_memory_profiling: bool = True):
        self.enable_line_profiling = enable_line_profiling and LINE_PROFILER_AVAILABLE
        self.enable_memory_profiling = enable_memory_profiling and MEMORY_PROFILER_AVAILABLE
        
        self.cprofile_stats = None
        self.line_profile_stats = None
        self.memory_profile_stats = None
        self.execution_times = {}
        self.function_calls = {}
        
    @contextmanager
    def profile_function(self, function_name: str, profile_memory: bool = None):
        """
        Context manager for profiling a specific function
        
        Args:
            function_name: Name of the function being profiled
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
            self.execution_times[function_name] = execution_time
            
            # Process cProfile results
            s = io.StringIO()
            ps = pstats.Stats(cprofiler, stream=s).sort_stats('cumulative')
            ps.print_stats(50)  # Top 50 functions
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
    
    def profile_method(self, obj: Any, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a specific method of an object
        
        Args:
            obj: Object instance
            method_name: Name of method to profile
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Dictionary with profiling results and method return value
        """
        method = getattr(obj, method_name)
        
        with self.profile_function(f"{obj.__class__.__name__}.{method_name}"):
            result = method(*args, **kwargs)
            
        return {
            'result': result,
            'execution_time': self.execution_times.get(f"{obj.__class__.__name__}.{method_name}"),
            'cprofile_stats': self.cprofile_stats,
            'line_profile_stats': self.line_profile_stats,
            'memory_profile_stats': self.memory_profile_stats
        }
    
    def get_slow_functions(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the slowest functions from cProfile results
        
        Args:
            top_n: Number of top functions to return
            
        Returns:
            List of slow functions with detailed stats
        """
        if not self.cprofile_stats:
            return []
            
        slow_functions = []
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
                        
                        slow_functions.append({
                            'function': func_name,
                            'calls': calls,
                            'total_time': total_time,
                            'per_call': per_call,
                            'cumulative_time': cum_time
                        })
                    except (ValueError, IndexError):
                        continue
                        
        return sorted(slow_functions, key=lambda x: x['cumulative_time'], reverse=True)[:top_n]
    
    def get_function_hotspots(self) -> Dict[str, Any]:
        """
        Analyze function hotspots from cProfile results
        
        Returns:
            Dictionary with hotspot analysis
        """
        if not self.cprofile_stats:
            return {}
            
        hotspots = {
            'total_functions': 0,
            'total_calls': 0,
            'total_time': 0,
            'slowest_functions': [],
            'most_called_functions': [],
            'expensive_functions': []
        }
        
        functions = []
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
                        
                        functions.append({
                            'function': func_name,
                            'calls': calls,
                            'total_time': total_time,
                            'per_call': per_call,
                            'cumulative_time': cum_time
                        })
                        
                        hotspots['total_functions'] += 1
                        hotspots['total_calls'] += calls
                        hotspots['total_time'] += cum_time
                    except (ValueError, IndexError):
                        continue
        
        if functions:
            # Sort by different criteria
            slowest = sorted(functions, key=lambda x: x['cumulative_time'], reverse=True)
            most_called = sorted(functions, key=lambda x: x['calls'], reverse=True)
            expensive = sorted(functions, key=lambda x: x['per_call'], reverse=True)
            
            hotspots['slowest_functions'] = slowest[:10]
            hotspots['most_called_functions'] = most_called[:10]
            hotspots['expensive_functions'] = expensive[:10]
        
        return hotspots
    
    def save_detailed_report(self, filepath: str) -> None:
        """
        Save detailed profiling report to file
        
        Args:
            filepath: Path to save report
        """
        # Ensure profiling_reports directory exists
        import os
        if not filepath.startswith('profiling_reports/'):
            os.makedirs('profiling_reports', exist_ok=True)
            filename = os.path.basename(filepath)
            filepath = os.path.join('profiling_reports', filename)
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("DETAILED PERFORMANCE PROFILING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Execution times
            f.write("FUNCTION EXECUTION TIMES\n")
            f.write("-" * 30 + "\n")
            for func, time_taken in sorted(self.execution_times.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{func}: {time_taken:.4f}s\n")
            f.write("\n")
            
            # Hotspots analysis
            hotspots = self.get_function_hotspots()
            if hotspots:
                f.write("HOTSPOTS ANALYSIS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total functions: {hotspots['total_functions']}\n")
                f.write(f"Total calls: {hotspots['total_calls']:,}\n")
                f.write(f"Total time: {hotspots['total_time']:.4f}s\n\n")
                
                # Slowest functions
                f.write("SLOWEST FUNCTIONS (by cumulative time)\n")
                f.write("-" * 40 + "\n")
                for i, func in enumerate(hotspots['slowest_functions'][:10], 1):
                    f.write(f"{i:2d}. {func['function']}\n")
                    f.write(f"    Calls: {func['calls']:,}\n")
                    f.write(f"    Total time: {func['cumulative_time']:.4f}s\n")
                    f.write(f"    Time per call: {func['per_call']:.6f}s\n\n")
                
                # Most called functions
                f.write("MOST CALLED FUNCTIONS\n")
                f.write("-" * 25 + "\n")
                for i, func in enumerate(hotspots['most_called_functions'][:10], 1):
                    f.write(f"{i:2d}. {func['function']}\n")
                    f.write(f"    Calls: {func['calls']:,}\n")
                    f.write(f"    Total time: {func['cumulative_time']:.4f}s\n\n")
                
                # Most expensive functions
                f.write("MOST EXPENSIVE FUNCTIONS (by time per call)\n")
                f.write("-" * 50 + "\n")
                for i, func in enumerate(hotspots['expensive_functions'][:10], 1):
                    f.write(f"{i:2d}. {func['function']}\n")
                    f.write(f"    Time per call: {func['per_call']:.6f}s\n")
                    f.write(f"    Calls: {func['calls']:,}\n")
                    f.write(f"    Total time: {func['cumulative_time']:.4f}s\n\n")
            
            # Memory usage
            if self.memory_profile_stats:
                f.write("MEMORY USAGE\n")
                f.write("-" * 15 + "\n")
                f.write(f"Memory before: {self.memory_profile_stats['memory_before_mb']:.2f}MB\n")
                f.write(f"Memory after: {self.memory_profile_stats['memory_after_mb']:.2f}MB\n")
                f.write(f"Memory diff: {self.memory_profile_stats['memory_diff_mb']:.2f}MB\n\n")
            
            # Detailed cProfile stats
            if self.cprofile_stats:
                f.write("DETAILED CPROFILE STATS\n")
                f.write("-" * 25 + "\n")
                f.write(self.cprofile_stats)
                f.write("\n")
            
            # Line profiler stats
            if self.line_profile_stats:
                f.write("DETAILED LINE PROFILE STATS\n")
                f.write("-" * 30 + "\n")
                f.write(self.line_profile_stats)
        
        print(f"Detailed profiling report saved to {filepath}")


class StrategyProfiler(DetailedProfiler):
    """
    Specialized profiler for trading strategies
    """
    
    def profile_strategy_execution(self, strategy, data, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Profile strategy execution with detailed analysis
        
        Args:
            strategy: Strategy instance
            data: Data to process
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
        
        # Profile the strategy execution
        if hasattr(strategy, 'turbo_process_dataset'):
            # For turbo strategies, extract numpy arrays
            if hasattr(test_data, 'data'):
                times = test_data['time']
                closes = test_data['close']
                opens = test_data['open']
                highs = test_data['high']
                lows = test_data['low']
            else:
                times = test_data['times']
                closes = test_data['closes']
                opens = test_data.get('opens')
                highs = test_data.get('highs')
                lows = test_data.get('lows')
            
            result = self.profile_method(
                strategy, 'turbo_process_dataset',
                times=times, prices=closes, opens=opens, highs=highs, lows=lows, closes=closes
            )
        else:
            # For regular strategies
            result = self.profile_method(strategy, 'vectorized_process_dataset', test_data)
        
        # Add strategy-specific analysis
        result['strategy_analysis'] = self._analyze_strategy_performance(result['result'])
        
        return result
    
    def _analyze_strategy_performance(self, strategy_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze strategy performance results
        
        Args:
            strategy_result: Results from strategy execution
            
        Returns:
            Dictionary with performance analysis
        """
        analysis = {
            'total_trades': strategy_result.get('total', 0),
            'total_bars': strategy_result.get('total_bars', 0),
            'processing_rate': 0,
            'trade_rate': 0,
            'performance_grade': 'Unknown'
        }
        
        # Calculate rates
        if analysis['total_bars'] > 0:
            execution_time = self.execution_times.get('strategy_execution', 0)
            if execution_time > 0:
                analysis['processing_rate'] = analysis['total_bars'] / execution_time
            
            if analysis['total_trades'] > 0:
                analysis['trade_rate'] = analysis['total_trades'] / analysis['total_bars']
        
        # Grade performance
        execution_time = self.execution_times.get('strategy_execution', 0)
        if execution_time < 1:
            analysis['performance_grade'] = 'Excellent'
        elif execution_time < 5:
            analysis['performance_grade'] = 'Good'
        elif execution_time < 10:
            analysis['performance_grade'] = 'Moderate'
        else:
            analysis['performance_grade'] = 'Slow'
        
        return analysis


class OptunaProfiler(DetailedProfiler):
    """
    Specialized profiler for Optuna optimization
    """
    
    def __init__(self, enable_resource_monitoring: bool = True):
        super().__init__(enable_line_profiling=False, enable_memory_profiling=enable_resource_monitoring)
        self.trial_times = {}
        self.trial_results = {}
        self.optimization_start_time = None
        self.optimization_end_time = None
        
    def profile_optimization_trial(self, trial_number: int, objective_func: Callable, *args, **kwargs) -> Any:
        """
        Profile a single Optuna trial
        
        Args:
            trial_number: Trial number
            objective_func: Objective function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result of objective function
        """
        with self.profile_function(f"optuna_trial_{trial_number}"):
            start_time = time.time()
            result = objective_func(*args, **kwargs)
            end_time = time.time()
            
            # Store trial data
            self.trial_times[trial_number] = end_time - start_time
            self.trial_results[trial_number] = result
            
            return result
    
    def get_optimization_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization analysis
        
        Returns:
            Dictionary with optimization analysis
        """
        if not self.trial_times:
            return {}
        
        times = list(self.trial_times.values())
        
        analysis = {
            'total_trials': len(times),
            'total_time': sum(times),
            'avg_trial_time': np.mean(times),
            'median_trial_time': np.median(times),
            'min_trial_time': min(times),
            'max_trial_time': max(times),
            'std_trial_time': np.std(times),
            'slowest_trials': [],
            'fastest_trials': []
        }
        
        # Sort trials by time
        sorted_trials = sorted(self.trial_times.items(), key=lambda x: x[1], reverse=True)
        analysis['slowest_trials'] = sorted_trials[:5]
        analysis['fastest_trials'] = sorted_trials[-5:]
        
        # Calculate optimization efficiency
        if self.optimization_start_time and self.optimization_end_time:
            total_optimization_time = self.optimization_end_time - self.optimization_start_time
            if total_optimization_time > 0:
                analysis['optimization_efficiency'] = sum(times) / total_optimization_time
            else:
                analysis['optimization_efficiency'] = 0
        
        return analysis
    
    def save_optimization_report(self, filepath: str) -> None:
        """
        Save detailed optimization report
        
        Args:
            filepath: Path to save report
        """
        # Ensure profiling_reports directory exists
        import os
        if not filepath.startswith('profiling_reports/'):
            os.makedirs('profiling_reports', exist_ok=True)
            filename = os.path.basename(filepath)
            filepath = os.path.join('profiling_reports', filename)
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("DETAILED OPTUNA PROFILING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Optimization analysis
            analysis = self.get_optimization_analysis()
            if analysis:
                f.write("OPTIMIZATION ANALYSIS\n")
                f.write("-" * 25 + "\n")
                f.write(f"Total trials: {analysis['total_trials']}\n")
                f.write(f"Total time: {analysis['total_time']:.4f}s\n")
                f.write(f"Average trial time: {analysis['avg_trial_time']:.4f}s\n")
                f.write(f"Median trial time: {analysis['median_trial_time']:.4f}s\n")
                f.write(f"Min trial time: {analysis['min_trial_time']:.4f}s\n")
                f.write(f"Max trial time: {analysis['max_trial_time']:.4f}s\n")
                f.write(f"Std trial time: {analysis['std_trial_time']:.4f}s\n")
                
                if 'optimization_efficiency' in analysis:
                    f.write(f"Optimization efficiency: {analysis['optimization_efficiency']:.2%}\n")
                f.write("\n")
                
                # Slowest trials
                f.write("SLOWEST TRIALS\n")
                f.write("-" * 20 + "\n")
                for i, (trial_num, trial_time) in enumerate(analysis['slowest_trials'], 1):
                    f.write(f"{i}. Trial {trial_num}: {trial_time:.4f}s\n")
                f.write("\n")
                
                # Fastest trials
                f.write("FASTEST TRIALS\n")
                f.write("-" * 20 + "\n")
                for i, (trial_num, trial_time) in enumerate(analysis['fastest_trials'], 1):
                    f.write(f"{i}. Trial {trial_num}: {trial_time:.4f}s\n")
                f.write("\n")
            
            # Function hotspots
            hotspots = self.get_function_hotspots()
            if hotspots:
                f.write("FUNCTION HOTSPOTS\n")
                f.write("-" * 20 + "\n")
                
                f.write("SLOWEST FUNCTIONS\n")
                f.write("-" * 20 + "\n")
                for i, func in enumerate(hotspots['slowest_functions'][:10], 1):
                    f.write(f"{i:2d}. {func['function']}\n")
                    f.write(f"    Calls: {func['calls']:,}\n")
                    f.write(f"    Total time: {func['cumulative_time']:.4f}s\n")
                    f.write(f"    Time per call: {func['per_call']:.6f}s\n\n")
            
            # Detailed cProfile stats
            if self.cprofile_stats:
                f.write("DETAILED CPROFILE STATS\n")
                f.write("-" * 25 + "\n")
                f.write(self.cprofile_stats)
        
        print(f"Detailed optimization profiling report saved to {filepath}")


# Convenience functions for quick profiling
def profile_strategy_function(strategy, data, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Quick function to profile a strategy
    
    Args:
        strategy: Strategy instance
        data: Data to process
        sample_size: Optional sample size
        
    Returns:
        Profiling results
    """
    profiler = StrategyProfiler()
    return profiler.profile_strategy_execution(strategy, data, sample_size)


def profile_optuna_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Quick function to profile any function
    
    Args:
        func: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Profiling results
    """
    profiler = DetailedProfiler()
    
    with profiler.profile_function(func.__name__):
        result = func(*args, **kwargs)
    
    return {
        'result': result,
        'execution_time': profiler.execution_times.get(func.__name__),
        'slow_functions': profiler.get_slow_functions(),
        'hotspots': profiler.get_function_hotspots(),
        'cprofile_stats': profiler.cprofile_stats
    }