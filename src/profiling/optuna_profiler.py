"""
Optuna Profiler for HFT System

Comprehensive profiling of Optuna optimization process to identify bottlenecks
in hyperparameter optimization, trial execution, and parallel processing.

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
import threading
import json

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with: pip install psutil")


class OptunaProfiler:
    """
    Comprehensive Optuna optimization profiler
    
    Features:
    - Trial execution time profiling
    - Resource usage monitoring
    - Parallel processing analysis
    - Parameter space exploration tracking
    - Pruning effectiveness analysis
    """
    
    def __init__(self, enable_resource_monitoring: bool = True):
        """
        Initialize Optuna profiler
        
        Args:
            enable_resource_monitoring: Enable CPU/memory monitoring
        """
        self.enable_resource_monitoring = enable_resource_monitoring and PSUTIL_AVAILABLE
        
        # Profiling data
        self.cprofile_stats = None
        self.trial_times = {}
        self.resource_usage = []
        self.optimization_stats = {}
        self.parameter_importance = {}
        self.pruning_stats = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.start_time = None
        
    def _resource_monitor(self, interval: float = 0.5) -> None:
        """
        Background thread for monitoring resource usage
        
        Args:
            interval: Monitoring interval in seconds
        """
        process = psutil.Process()
        
        while self.monitoring_active:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self.resource_usage.append({
                    'timestamp': time.time() - self.start_time,
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'thread_count': process.num_threads()
                })
                
                time.sleep(interval)
            except Exception:
                break
    
    @contextmanager
    def profile_optimization(self, study_name: str = "optimization"):
        """
        Context manager for profiling Optuna optimization
        
        Args:
            study_name: Name of the study being profiled
        """
        self.start_time = time.time()
        self.optimization_stats = {
            'study_name': study_name,
            'start_time': self.start_time,
            'trials_total': 0,
            'trials_successful': 0,
            'trials_pruned': 0,
            'trials_failed': 0
        }
        
        # Initialize profilers
        cprofiler = cProfile.Profile()
        
        # Start resource monitoring
        if self.enable_resource_monitoring:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._resource_monitor, 
                daemon=True
            )
            self.monitoring_thread.start()
        
        # Start profiling
        cprofiler.enable()
        
        try:
            yield self
        finally:
            # Stop profiling
            cprofiler.disable()
            end_time = time.time()
            
            # Stop resource monitoring
            if self.enable_resource_monitoring:
                self.monitoring_active = False
                if self.monitoring_thread:
                    self.monitoring_thread.join(timeout=1.0)
            
            # Update optimization stats
            self.optimization_stats['end_time'] = end_time
            self.optimization_stats['total_time'] = end_time - self.start_time
            
            # Process cProfile results
            s = io.StringIO()
            ps = pstats.Stats(cprofiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Top 30 functions
            self.cprofile_stats = s.getvalue()
    
    def wrap_objective_function(self, objective_func: Callable, 
                               track_params: bool = True) -> Callable:
        """
        Wrap Optuna objective function for profiling
        
        Args:
            objective_func: Original objective function
            track_params: Whether to track parameter values
            
        Returns:
            Wrapped objective function with profiling
        """
        @functools.wraps(objective_func)
        def wrapped_objective(trial):
            trial_start = time.time()
            trial_number = trial.number
            
            try:
                # Track parameters if requested
                if track_params:
                    trial_params = trial.params.copy()
                
                # Execute original objective
                result = objective_func(trial)
                
                trial_end = time.time()
                trial_time = trial_end - trial_start
                
                # Record trial data
                self.trial_times[trial_number] = {
                    'execution_time': trial_time,
                    'result': result,
                    'state': 'COMPLETE',
                    'params': trial_params if track_params else None
                }
                
                self.optimization_stats['trials_successful'] += 1
                
                return result
                
            except optuna.exceptions.TrialPruned as e:
                trial_end = time.time()
                trial_time = trial_end - trial_start
                
                self.trial_times[trial_number] = {
                    'execution_time': trial_time,
                    'result': None,
                    'state': 'PRUNED',
                    'params': trial_params if track_params else None,
                    'pruned_reason': str(e)
                }
                
                self.optimization_stats['trials_pruned'] += 1
                raise
                
            except Exception as e:
                trial_end = time.time()
                trial_time = trial_end - trial_start
                
                self.trial_times[trial_number] = {
                    'execution_time': trial_time,
                    'result': None,
                    'state': 'FAILED',
                    'params': trial_params if track_params else None,
                    'error': str(e)
                }
                
                self.optimization_stats['trials_failed'] += 1
                raise
        
        return wrapped_objective
    
    def analyze_study_results(self, study: 'optuna.Study') -> Dict[str, Any]:
        """
        Analyze Optuna study results for performance insights
        
        Args:
            study: Completed Optuna study
            
        Returns:
            Dictionary with analysis results
        """
        if not OPTUNA_AVAILABLE or not study:
            return {}
        
        analysis = {}
        
        # Basic study statistics
        analysis['study_name'] = study.study_name
        analysis['direction'] = study.direction.name
        analysis['n_trials'] = len(study.trials)
        
        # Trial states
        complete_trials = [t for t in study.trials if t.state.name == 'COMPLETE']
        pruned_trials = [t for t in study.trials if t.state.name == 'PRUNED']
        failed_trials = [t for t in study.trials if t.state.name == 'FAILED']
        
        analysis['n_complete'] = len(complete_trials)
        analysis['n_pruned'] = len(pruned_trials)
        analysis['n_failed'] = len(failed_trials)
        
        # Timing analysis
        if complete_trials:
            complete_times = [self.trial_times.get(t.number, {}).get('execution_time', 0) 
                            for t in complete_trials]
            analysis['avg_trial_time'] = np.mean(complete_times) if complete_times else 0
            analysis['median_trial_time'] = np.median(complete_times) if complete_times else 0
            analysis['std_trial_time'] = np.std(complete_times) if complete_times else 0
        
        # Pruning effectiveness
        if pruned_trials:
            pruned_times = [self.trial_times.get(t.number, {}).get('execution_time', 0) 
                           for t in pruned_trials]
            analysis['avg_pruned_time'] = np.mean(pruned_times) if pruned_times else 0
            
            if complete_times and pruned_times:
                time_saved = np.mean(complete_times) - np.mean(pruned_times)
                analysis['pruning_time_saved'] = time_saved
                analysis['pruning_efficiency'] = time_saved / np.mean(complete_times) * 100
        
        # Parameter importance
        try:
            if complete_trials:
                importance = optuna.importance.get_param_importances(study)
                analysis['parameter_importance'] = importance
                self.parameter_importance = importance
        except Exception:
            analysis['parameter_importance'] = {}
        
        # Best trial analysis
        if study.best_trial:
            best_trial = study.best_trial
            analysis['best_trial'] = {
                'number': best_trial.number,
                'value': best_trial.value,
                'params': best_trial.params,
                'execution_time': self.trial_times.get(best_trial.number, {}).get('execution_time', 0)
            }
        
        # Convergence analysis
        if complete_trials:
            values = [t.value for t in complete_trials]
            analysis['convergence_data'] = {
                'values': values,
                'best_values': np.minimum.accumulate(values).tolist() if study.direction.name == 'MAXIMIZE' 
                              else np.maximum.accumulate(values).tolist()
            }
        
        return analysis
    
    def get_parallel_efficiency(self, n_jobs: int) -> Dict[str, Any]:
        """
        Analyze parallel processing efficiency
        
        Args:
            n_jobs: Number of parallel jobs used
            
        Returns:
            Dictionary with parallel efficiency analysis
        """
        if n_jobs <= 1:
            return {'parallel_efficiency': 0, 'speedup': 1.0}
        
        # Calculate theoretical vs actual speedup
        total_time = self.optimization_stats.get('total_time', 0)
        trial_times = [t['execution_time'] for t in self.trial_times.values() 
                      if t['execution_time'] > 0]
        
        if not trial_times:
            return {'parallel_efficiency': 0, 'speedup': 1.0}
        
        sequential_time = sum(trial_times)
        theoretical_speedup = n_jobs
        actual_speedup = sequential_time / total_time if total_time > 0 else 1.0
        parallel_efficiency = (actual_speedup / theoretical_speedup) * 100
        
        return {
            'n_jobs': n_jobs,
            'sequential_time': sequential_time,
            'parallel_time': total_time,
            'theoretical_speedup': theoretical_speedup,
            'actual_speedup': actual_speedup,
            'parallel_efficiency': parallel_efficiency
        }
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get summary of resource usage during optimization
        
        Returns:
            Dictionary with resource usage summary
        """
        if not self.resource_usage:
            return {}
        
        cpu_values = [r['cpu_percent'] for r in self.resource_usage]
        memory_values = [r['memory_mb'] for r in self.resource_usage]
        thread_values = [r['thread_count'] for r in self.resource_usage]
        
        return {
            'cpu': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory': {
                'avg_mb': np.mean(memory_values),
                'max_mb': np.max(memory_values),
                'min_mb': np.min(memory_values),
                'std_mb': np.std(memory_values)
            },
            'threads': {
                'avg': np.mean(thread_values),
                'max': np.max(thread_values),
                'min': np.min(thread_values)
            },
            'samples': len(self.resource_usage)
        }
    
    def get_optimization_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify bottlenecks in optimization process
        
        Args:
            top_n: Number of top bottlenecks to return
            
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Slowest trials
        slow_trials = sorted(
            [(num, data) for num, data in self.trial_times.items() 
             if data['execution_time'] > 0],
            key=lambda x: x[1]['execution_time'],
            reverse=True
        )[:top_n]
        
        for trial_num, trial_data in slow_trials:
            bottlenecks.append({
                'type': 'slow_trial',
                'trial_number': trial_num,
                'execution_time': trial_data['execution_time'],
                'state': trial_data['state'],
                'params': trial_data.get('params', {})
            })
        
        # Failed trials
        failed_trials = [(num, data) for num, data in self.trial_times.items() 
                        if data['state'] == 'FAILED']
        
        for trial_num, trial_data in failed_trials[:top_n]:
            bottlenecks.append({
                'type': 'failed_trial',
                'trial_number': trial_num,
                'error': trial_data.get('error', 'Unknown error'),
                'params': trial_data.get('params', {})
            })
        
        return bottlenecks[:top_n]
    
    def save_optimization_report(self, filepath: str, study: 'optuna.Study' = None) -> None:
        """
        Save comprehensive optimization report
        
        Args:
            filepath: Path to save report
            study: Optuna study object for additional analysis
        """
        with open(filepath, 'w') as f:
            f.write("OPTUNA OPTIMIZATION PROFILING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Optimization statistics
            f.write("OPTIMIZATION STATISTICS\n")
            f.write("-" * 25 + "\n")
            for key, value in self.optimization_stats.items():
                if key not in ['start_time', 'end_time']:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Trial statistics
            f.write("TRIAL EXECUTION STATISTICS\n")
            f.write("-" * 30 + "\n")
            if self.trial_times:
                times = [t['execution_time'] for t in self.trial_times.values() if t['execution_time'] > 0]
                f.write(f"Total trials: {len(self.trial_times)}\n")
                f.write(f"Average trial time: {np.mean(times):.4f}s\n")
                f.write(f"Median trial time: {np.median(times):.4f}s\n")
                f.write(f"Std trial time: {np.std(times):.4f}s\n")
            f.write("\n")
            
            # Resource usage
            if self.resource_usage:
                resource_summary = self.get_resource_summary()
                f.write("RESOURCE USAGE SUMMARY\n")
                f.write("-" * 25 + "\n")
                f.write(f"Average CPU: {resource_summary['cpu']['avg']:.2f}%\n")
                f.write(f"Max CPU: {resource_summary['cpu']['max']:.2f}%\n")
                f.write(f"Average Memory: {resource_summary['memory']['avg_mb']:.2f}MB\n")
                f.write(f"Max Memory: {resource_summary['memory']['max_mb']:.2f}MB\n")
                f.write("\n")
            
            # Study analysis
            if study:
                study_analysis = self.analyze_study_results(study)
                f.write("STUDY ANALYSIS\n")
                f.write("-" * 15 + "\n")
                f.write(f"Complete trials: {study_analysis.get('n_complete', 0)}\n")
                f.write(f"Pruned trials: {study_analysis.get('n_pruned', 0)}\n")
                f.write(f"Failed trials: {study_analysis.get('n_failed', 0)}\n")
                
                if 'pruning_efficiency' in study_analysis:
                    f.write(f"Pruning efficiency: {study_analysis['pruning_efficiency']:.2f}%\n")
                
                if 'best_trial' in study_analysis:
                    best = study_analysis['best_trial']
                    f.write(f"Best value: {best['value']:.6f}\n")
                    f.write(f"Best trial time: {best['execution_time']:.4f}s\n")
                f.write("\n")
            
            # Bottlenecks
            bottlenecks = self.get_optimization_bottlenecks()
            if bottlenecks:
                f.write("OPTIMIZATION BOTTLENECKS\n")
                f.write("-" * 25 + "\n")
                for i, bottleneck in enumerate(bottlenecks, 1):
                    f.write(f"{i}. {bottleneck['type'].replace('_', ' ').title()}\n")
                    if 'trial_number' in bottleneck:
                        f.write(f"   Trial: {bottleneck['trial_number']}\n")
                    if 'execution_time' in bottleneck:
                        f.write(f"   Time: {bottleneck['execution_time']:.4f}s\n")
                    if 'error' in bottleneck:
                        f.write(f"   Error: {bottleneck['error']}\n")
                    f.write("\n")
            
            # Detailed cProfile stats
            if self.cprofile_stats:
                f.write("DETAILED CPROFILE STATS\n")
                f.write("-" * 25 + "\n")
                f.write(self.cprofile_stats)
        
        print(f"Optimization profile report saved to {filepath}")
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export profiling metrics to JSON for further analysis
        
        Args:
            filepath: Path to save JSON metrics
        """
        metrics = {
            'optimization_stats': self.optimization_stats,
            'trial_times': self.trial_times,
            'resource_usage': self.resource_usage,
            'parameter_importance': self.parameter_importance
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Profiling metrics exported to {filepath}")


# Convenience function for quick Optuna profiling
def profile_optuna_study(study: 'optuna.Study', objective_func: Callable, 
                         n_trials: int = 100, **kwargs) -> Dict[str, Any]:
    """
    Quick function to profile an Optuna study
    
    Args:
        study: Optuna study object
        objective_func: Objective function
        n_trials: Number of trials
        **kwargs: Additional optimization parameters
        
    Returns:
        Dictionary with profiling results
    """
    profiler = OptunaProfiler()
    
    with profiler.profile_optimization(study.study_name):
        # Wrap objective function
        wrapped_objective = profiler.wrap_objective_function(objective_func)
        
        # Run optimization
        study.optimize(wrapped_objective, n_trials=n_trials, **kwargs)
    
    return {
        'study': study,
        'profiler': profiler,
        'analysis': profiler.analyze_study_results(study)
    }