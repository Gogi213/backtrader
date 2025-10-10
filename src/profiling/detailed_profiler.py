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
import threading
import json

# Optional dependencies
try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


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
        
        self.memory_usage_data = []
        self.memory_monitoring_active = False

    def _monitor_memory(self, interval: float = 0.1):
        """
        Background thread for monitoring memory usage.
        """
        process = psutil.Process(os.getpid())
        while self.memory_monitoring_active:
            try:
                self.memory_usage_data.append(
                    (time.time(), process.memory_info().rss / 1024 / 1024)
                )
                time.sleep(interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

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
        
        # Memory profiling setup
        memory_before = None
        if profile_memory and MEMORY_PROFILER_AVAILABLE:
            memory_before = memory_profiler.memory_usage()[0]
            self.memory_monitoring_active = True
            memory_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            memory_thread.start()
            
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

            if profile_memory and MEMORY_PROFILER_AVAILABLE:
                self.memory_monitoring_active = False
                memory_thread.join(timeout=1.0)
                
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Memory profiling results
            memory_diff = None
            if profile_memory and MEMORY_PROFILER_AVAILABLE and memory_before is not None:
                memory_after = memory_profiler.memory_usage()[0]
                memory_diff = memory_after - memory_before
                
            # Store results
            self.execution_times[function_name] = execution_time
            
            # Process cProfile results
            s = io.StringIO()
            ps = pstats.Stats(cprofiler, stream=s).sort_stats('cumulative')
            ps.print_stats(50)
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
    
    def profile(self, func: Callable) -> Callable:
        """
        Decorator for profiling a specific function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_function(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def profile_method(self, obj: Any, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a specific method of an object
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
        """
        if not self.cprofile_stats:
            return []
            
        slow_functions = []
        lines = self.cprofile_stats.split('\n')
        
        for line in lines[5:]:
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
        
        functions = self.get_slow_functions(top_n=-1) # Get all functions
        
        if functions:
            hotspots['total_functions'] = len(functions)
            hotspots['total_calls'] = sum(f['calls'] for f in functions)
            hotspots['total_time'] = sum(f['cumulative_time'] for f in functions)

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
        """
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
                
                # Slowest functions, most called, and expensive functions
                for category in ['slowest_functions', 'most_called_functions', 'expensive_functions']:
                    f.write(f"{category.replace('_', ' ').upper()}\n")
                    f.write("-" * 40 + "\n")
                    for i, func in enumerate(hotspots[category], 1):
                        f.write(f"{i:2d}. {func['function']}\n")
                        f.write(f"    Calls: {func['calls']:,}\n")
                        f.write(f"    Total time: {func['cumulative_time']:.4f}s\n")
                        f.write(f"    Time per call: {func['per_call']:.6f}s\n\n")

            # Memory usage
            if self.memory_profile_stats:
                f.write("MEMORY USAGE\n")
                f.write("-" * 15 + "\n")
                f.write(f"Memory before: {self.memory_profile_stats['memory_before_mb']:.2f}MB\n")
                f.write(f"Memory after: {self.memory_profile_stats['memory_after_mb']:.2f}MB\n")
                f.write(f"Memory diff: {self.memory_profile_stats['memory_diff_mb']:.2f}MB\n\n")

            # Memory usage over time
            if self.memory_usage_data:
                f.write("MEMORY USAGE OVER TIME\n")
                f.write("-" * 25 + "\n")
                start_time = self.memory_usage_data[0][0]
                for timestamp, memory in self.memory_usage_data:
                    f.write(f"  - At {timestamp - start_time:.2f}s: {memory:.2f}MB\n")
                f.write("\n")

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

    def to_json(self, filepath: str):
        """
        Export profiling data to a JSON file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'execution_times': self.execution_times,
            'function_hotspots': self.get_function_hotspots(),
            'memory_profile_stats': self.memory_profile_stats,
            'memory_usage_data': self.memory_usage_data,
            'cprofile_stats': self.cprofile_stats,
            'line_profile_stats': self.line_profile_stats,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Profiling data exported to {filepath}")


class StrategyProfiler(DetailedProfiler):
    """
    Specialized profiler for trading strategies
    """
    
    def profile_strategy_execution(self, strategy, data, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Profile strategy execution with detailed analysis
        """
        # Sample data if requested
        if sample_size and hasattr(data, '__len__'):
            test_data = data.head(sample_size) if hasattr(data, 'head') else data[:sample_size]
        else:
            test_data = data
        
        # Profile the strategy execution
        method_name = 'turbo_process_dataset' if hasattr(strategy, 'turbo_process_dataset') else 'vectorized_process_dataset'
        
        with self.profile_function(f"strategy_execution: {strategy.__class__.__name__}"):
            if method_name == 'turbo_process_dataset':
                # Adapt for turbo strategies
                result = strategy.turbo_process_dataset(
                    times=test_data['time'], 
                    prices=test_data['close'], 
                    opens=test_data.get('open'), 
                    highs=test_data.get('high'), 
                    lows=test_data.get('low'), 
                    closes=test_data['close']
                )
            else:
                result = strategy.vectorized_process_dataset(test_data)

        analysis = self._analyze_strategy_performance(result)
        analysis.update({
            'execution_time': self.execution_times.get(f"strategy_execution: {strategy.__class__.__name__}"),
            'cprofile_stats': self.cprofile_stats,
            'line_profile_stats': self.line_profile_stats,
            'memory_profile_stats': self.memory_profile_stats,
            'bottlenecks': self.get_slow_functions()
        })
        return analysis

    def _analyze_strategy_performance(self, strategy_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze strategy performance results
        """
        analysis = {
            'total_trades': strategy_result.get('total', 0),
            'total_bars': strategy_result.get('total_bars', 0),
            'processing_rate': 0,
            'trade_rate': 0,
            'performance_grade': 'Unknown'
        }
        
        execution_time = self.execution_times.get(next(iter(self.execution_times)), 0)
        if analysis['total_bars'] > 0 and execution_time > 0:
            analysis['processing_rate'] = analysis['total_bars'] / execution_time
            if analysis['total_trades'] > 0:
                analysis['trade_rate'] = analysis['total_trades'] / analysis['total_bars']
        
        # Grade performance
        if execution_time < 1:
            analysis['performance_grade'] = 'Excellent'
        elif execution_time < 5:
            analysis['performance_grade'] = 'Good'
        elif execution_time < 10:
            analysis['performance_grade'] = 'Moderate'
        else:
            analysis['performance_grade'] = 'Slow'
        
        return analysis


class OptunaProfiler:
    """
    Comprehensive Optuna optimization profiler.
    Profiles each trial individually to identify slow runs.
    """
    
    def __init__(self, enable_line_profiling: bool = True, enable_memory_profiling: bool = True, **kwargs):
        self.enable_line_profiling = enable_line_profiling and LINE_PROFILER_AVAILABLE
        self.enable_memory_profiling = enable_memory_profiling and MEMORY_PROFILER_AVAILABLE
        self.trial_profiles: Dict[int, Dict[str, Any]] = {}
        self.study: Optional['optuna.Study'] = None
        self.objective_func: Optional[Callable] = None

    def profile_study(self, study: 'optuna.Study', objective_func: Callable, n_trials: int, **kwargs) -> Dict[str, Any]:
        """
        Profile an entire Optuna study using the standard `study.optimize` method.
        """
        self.study = study
        self.objective_func = objective_func
        
        @functools.wraps(objective_func)
        def profiled_objective(trial: optuna.Trial) -> float:
            trial_number = trial.number
            
            trial_profiler = DetailedProfiler(
                enable_line_profiling=self.enable_line_profiling,
                enable_memory_profiling=self.enable_memory_profiling
            )
            
            result = None
            try:
                with trial_profiler.profile_function(f"trial_{trial_number}"):
                    result = self.objective_func(trial)
                
                self.trial_profiles[trial_number] = {
                    'cprofile_stats': trial_profiler.cprofile_stats,
                    'line_profile_stats': trial_profiler.line_profile_stats,
                    'memory_profile_stats': trial_profiler.memory_profile_stats,
                    'execution_time': trial_profiler.execution_times.get(f"trial_{trial_number}", 0),
                    'result': result,
                    'params': trial.params,
                    'state': 'COMPLETE'
                }
                return result
            
            except optuna.exceptions.TrialPruned as e:
                self.trial_profiles[trial_number] = {
                    'state': 'PRUNED',
                    'params': trial.params,
                    'execution_time': trial_profiler.execution_times.get(f"trial_{trial_number}", 0),
                }
                raise e
            
            except Exception as e:
                self.trial_profiles[trial_number] = {
                    'state': 'FAIL',
                    'error': str(e),
                    'params': trial.params,
                    'execution_time': trial_profiler.execution_times.get(f"trial_{trial_number}", 0),
                }
                raise e

        study.optimize(profiled_objective, n_trials=n_trials, **kwargs)
        return self.get_study_analysis()

    def get_study_analysis(self) -> Dict[str, Any]:
        """
        Analyze the results of the profiled study.
        """
        if not self.trial_profiles:
            return {}

        all_times = [p['execution_time'] for p in self.trial_profiles.values() if 'execution_time' in p]
        complete_times = [p['execution_time'] for p in self.trial_profiles.values() if p.get('state') == 'COMPLETE']
        
        slowest_trials = sorted(
            [(n, p['execution_time']) for n, p in self.trial_profiles.items() if 'execution_time' in p],
            key=lambda x: x[1],
            reverse=True
        )

        analysis = {
            'total_trials': len(self.trial_profiles),
            'total_time': sum(all_times),
            'n_complete': len([p for p in self.trial_profiles.values() if p.get('state') == 'COMPLETE']),
            'n_pruned': len([p for p in self.trial_profiles.values() if p.get('state') == 'PRUNED']),
            'n_failed': len([p for p in self.trial_profiles.values() if p.get('state') == 'FAIL']),
            'slowest_trials': slowest_trials[:5],
        }

        if complete_times:
            analysis.update({
                'avg_trial_time': np.mean(complete_times),
                'median_trial_time': np.median(complete_times),
                'min_trial_time': np.min(complete_times),
                'max_trial_time': np.max(complete_times),
                'std_trial_time': np.std(complete_times),
            })
        
        return analysis

    def get_trial_report(self, trial_number: int) -> Optional[str]:
        """
        Get a detailed performance report for a single trial.
        """
        if trial_number not in self.trial_profiles:
            return f"No profile data found for trial {trial_number}."
        
        profile = self.trial_profiles[trial_number]
        
        report = io.StringIO()
        report.write(f"DETAILED REPORT FOR TRIAL {trial_number}\n")
        report.write("=" * 40 + "\n")
        report.write(f"Status: {profile.get('state', 'UNKNOWN')}\n")
        report.write(f"Execution Time: {profile.get('execution_time', 0):.4f}s\n")
        if 'result' in profile:
            report.write(f"Result: {profile['result']}\n")
        if 'error' in profile:
            report.write(f"Error: {profile['error']}\n")
        
        report.write("\n--- Parameters ---\n")
        report.write(json.dumps(profile.get('params', {}), indent=2))
        report.write("\n\n")

        if profile.get('cprofile_stats'):
            report.write("--- cProfile Stats ---\n")
            report.write(profile['cprofile_stats'])
            report.write("\n")

        if profile.get('line_profile_stats'):
            report.write("--- Line-by-Line Stats ---\n")
            report.write(profile['line_profile_stats'])
            report.write("\n")
            
        return report.getvalue()

    def save_optimization_report(self, filepath: str) -> None:
        """
        Save comprehensive optimization report.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        analysis = self.get_study_analysis()
        
        with open(filepath, 'w') as f:
            f.write(f"OPTUNA OPTIMIZATION PROFILING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("## Study Summary\n\n")
            f.write(f"- Total Trials: {analysis.get('total_trials', 0)}\n")
            f.write(f"- Successful: {analysis.get('n_complete', 0)}\n")
            f.write(f"- Pruned: {analysis.get('n_pruned', 0)}\n")
            f.write(f"- Failed: {analysis.get('n_failed', 0)}\n")
            f.write(f"- Total Profiling Time: {analysis.get('total_time', 0):.2f}s\n\n")

            f.write("## Timing Analysis (for completed trials)\n\n")
            f.write(f"- Average Trial Time: {analysis.get('avg_trial_time', 0):.4f}s\n")
            f.write(f"- Median Trial Time: {analysis.get('median_trial_time', 0):.4f}s\n")
            f.write(f"- Min/Max Trial Time: {analysis.get('min_trial_time', 0):.4f}s / {analysis.get('max_trial_time', 0):.4f}s\n\n")

            f.write("## Slowest Trials\n\n")
            f.write("Use `profiler.get_trial_report(trial_number)` for a detailed breakdown.\n\n")
            for i, (trial_num, trial_time) in enumerate(analysis.get('slowest_trials', []), 1):
                f.write(f"  {i}. Trial {trial_num}: {trial_time:.4f}s\n")
            f.write("\n")

            f.write("## Full Data (Summary)\n\n")
            f.write("```json\n")
            terse_profiles = {}
            for num, p in self.trial_profiles.items():
                terse_profiles[num] = {k: v for k, v in p.items() if 'stats' not in k}
            json.dump(terse_profiles, f, indent=2, default=str)
            f.write("\n```\n")
            
        print(f"Optimization profiling report saved to {filepath}")


# Convenience functions
def profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Quick function to profile any function"""
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

def profile_strategy(strategy, data, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """Quick function to profile a strategy"""
    profiler = StrategyProfiler()
    return profiler.profile_strategy_execution(strategy, data, sample_size)

def profile_optuna_study(study: 'optuna.Study', objective_func: Callable, n_trials: int = 100, **kwargs) -> Dict[str, Any]:
    """Quick function to profile an Optuna study"""
    profiler = OptunaProfiler()
    analysis = profiler.profile_study(study, objective_func, n_trials=n_trials, **kwargs)
    
    return {
        'study': study,
        'profiler': profiler,
        'analysis': analysis
    }
