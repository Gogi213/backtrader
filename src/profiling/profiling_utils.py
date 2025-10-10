"""
Profiling Utilities for HFT System

Utilities for analyzing, visualizing, and comparing profiling results.
Provides comprehensive reporting and bottleneck identification capabilities.

Author: HFT System
"""
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

class ProfilingUtils:
    """
    Advanced utility class for analyzing and visualizing profiling results
    """

    def __init__(self, output_dir: str = "profiling_reports"):
        """
        Initialize profiling utilities

        Args:
            output_dir: Directory for saving reports and plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._cached_analyses = {}  # Cache for expensive analyses
    
    def compare_strategy_performance(self, profiles: List[Dict[str, Any]], 
                                   metric: str = 'execution_time') -> Dict[str, Any]:
        """
        Compare performance across multiple strategy profiles
        
        Args:
            profiles: List of strategy profiling results
            metric: Metric to compare (execution_time, memory_usage, etc.)
            
        Returns:
            Dictionary with comparison results
        """
        if not profiles:
            return {}
        
        comparison = {
            'strategies': [],
            'values': [],
            'metric': metric,
            'best_strategy': None,
            'worst_strategy': None,
            'performance_ratio': 0
        }
        
        for profile in profiles:
            strategy_name = profile.get('strategy_name', 'Unknown')
            
            if metric == 'execution_time':
                value = profile.get('execution_time', 0)
            elif metric == 'memory_usage':
                memory_stats = profile.get('memory_profile_stats', {})
                value = memory_stats.get('memory_diff_mb', 0)
            else:
                value = profile.get(metric, 0)
            
            comparison['strategies'].append(strategy_name)
            comparison['values'].append(value)
        
        if comparison['values']:
            # Find best and worst performers
            min_idx = np.argmin(comparison['values'])
            max_idx = np.argmax(comparison['values'])
            
            comparison['best_strategy'] = comparison['strategies'][min_idx]
            comparison['worst_strategy'] = comparison['strategies'][max_idx]
            
            if comparison['values'][max_idx] > 0:
                comparison['performance_ratio'] = comparison['values'][max_idx] / comparison['values'][min_idx]
        
        return comparison
    
    def analyze_optimization_convergence(self, study_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze convergence patterns in optimization data
        
        Args:
            study_data: Study data from Optuna profiler
            
        Returns:
            Dictionary with convergence analysis
        """
        convergence = {
            'convergence_rate': 0,
            'stabilization_point': None,
            'improvement_rate': 0,
            'early_stopping_suggestion': False
        }
        
        if 'convergence_data' not in study_data:
            return convergence
        
        best_values = study_data['convergence_data']['best_values']
        if len(best_values) < 10:
            return convergence
        
        # Calculate convergence rate (improvement in last 20% of trials)
        n_trials = len(best_values)
        early_trials = best_values[:int(n_trials * 0.2)]
        late_trials = best_values[-int(n_trials * 0.2):]
        
        if len(early_trials) > 0 and len(late_trials) > 0:
            early_improvement = early_trials[-1] - early_trials[0]
            late_improvement = late_trials[-1] - late_trials[0]
            
            convergence['convergence_rate'] = abs(late_improvement) / abs(early_improvement) if early_improvement != 0 else 0
            convergence['improvement_rate'] = abs(late_improvement)
            
            # Find stabilization point (where improvement becomes minimal)
            window_size = max(10, n_trials // 10)
            for i in range(window_size, n_trials):
                window_improvement = abs(best_values[i] - best_values[i - window_size])
                if window_improvement < 0.01 * abs(best_values[i]):  # Less than 1% improvement
                    convergence['stabilization_point'] = i
                    break
            
            # Suggest early stopping if convergence is slow
            if convergence['convergence_rate'] < 0.1:
                convergence['early_stopping_suggestion'] = True
        
        return convergence
    
    def identify_performance_patterns(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify common performance patterns across profiles
        
        Args:
            profiles: List of profiling results
            
        Returns:
            Dictionary with identified patterns
        """
        patterns = {
            'common_bottlenecks': [],
            'resource_intensive_functions': [],
            'optimization_opportunities': []
        }
        
        # Collect all bottleneck functions
        all_bottlenecks = []
        for profile in profiles:
            bottlenecks = profile.get('bottlenecks', [])
            all_bottlenecks.extend(bottlenecks)
        
        if all_bottlenecks:
            # Count frequency of bottleneck functions
            bottleneck_counts = {}
            for bottleneck in all_bottlenecks:
                func_name = bottleneck.get('function', 'Unknown')
                bottleneck_counts[func_name] = bottleneck_counts.get(func_name, 0) + 1
            
            # Find most common bottlenecks
            sorted_bottlenecks = sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True)
            patterns['common_bottlenecks'] = sorted_bottlenecks[:5]
        
        # Analyze resource usage patterns
        memory_usage = []
        execution_times = []
        
        for profile in profiles:
            memory_stats = profile.get('memory_profile_stats', {})
            if memory_stats:
                memory_usage.append(memory_stats.get('memory_diff_mb', 0))
            
            execution_times.append(profile.get('execution_time', 0))
        
        if memory_usage and execution_times:
            correlation = np.corrcoef(memory_usage, execution_times)[0, 1]
            if not np.isnan(correlation) and abs(correlation) > 0.7:
                patterns['optimization_opportunities'].append(
                    "Strong correlation between memory usage and execution time - consider memory optimization"
                )
        
        return patterns

    def create_performance_dashboard(self, profiles: List[Dict[str, Any]],
                                   save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive performance dashboard (TEXT ONLY - no PDF)
        
        Args:
            profiles: List of profiling results
            save_path: Path to save dashboard (auto-generated if None)
            
        Returns:
            Path to saved dashboard
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"performance_dashboard_{timestamp}.txt")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("PERFORMANCE DASHBOARD\n")
            f.write("=" * 60 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            strategy_names = [p.get('strategy_name', 'Unknown') for p in profiles]
            exec_times = [p.get('execution_time', 0) for p in profiles]
            
            # Execution times comparison
            f.write("Execution Times by Strategy:\n")
            for i, (name, time) in enumerate(zip(strategy_names, exec_times)):
                f.write(f"  {i+1}. {name}: {time:.4f}s\n")
            f.write("\n")
            
            # Memory usage comparison
            memory_usage = []
            for p in profiles:
                mem_stats = p.get('memory_profile_stats', {})
                memory_usage.append(mem_stats.get('memory_diff_mb', 0))
            
            f.write("Memory Usage by Strategy:\n")
            for i, (name, mem) in enumerate(zip(strategy_names, memory_usage)):
                f.write(f"  {i+1}. {name}: {mem:.2f}MB\n")
            f.write("\n")
            
            # Performance correlation
            if memory_usage and exec_times:
                f.write("Memory vs Time Correlation:\n")
                for i, (name, mem, time) in enumerate(zip(strategy_names, memory_usage, exec_times)):
                    f.write(f"  {i+1}. {name}: {mem:.2f}MB, {time:.4f}s\n")
                f.write("\n")
            
            # Performance ranking
            performance_scores = []
            for i, p in enumerate(profiles):
                # Normalize metrics (lower is better)
                norm_time = exec_times[i] / max(exec_times) if max(exec_times) > 0 else 0
                norm_memory = memory_usage[i] / max(memory_usage) if max(memory_usage) > 0 else 0
                performance_scores.append((norm_time + norm_memory) / 2)
            
            sorted_indices = np.argsort(performance_scores)
            f.write("Strategy Performance Ranking (lower is better):\n")
            for rank, idx in enumerate(sorted_indices, 1):
                f.write(f"  {rank}. {strategy_names[idx]}: {performance_scores[idx]:.4f}\n")
            f.write("\n")
            
            # Detailed Bottleneck Analysis
            if profiles:
                f.write("BOTTLENECK ANALYSIS\n")
                f.write("-" * 25 + "\n")
                
                # Collect all bottlenecks
                all_bottlenecks = []
                for profile in profiles:
                    bottlenecks = profile.get('bottlenecks', [])
                    for bottleneck in bottlenecks[:5]:  # Top 5 per strategy
                        all_bottlenecks.append({
                            'function': bottleneck.get('function', 'Unknown'),
                            'time': bottleneck.get('cumulative_time', 0),
                            'strategy': profile.get('strategy_name', 'Unknown')
                        })
                
                if all_bottlenecks:
                    # Sort by time
                    all_bottlenecks.sort(key=lambda x: x['time'], reverse=True)
                    
                    f.write("Top Bottleneck Functions:\n")
                    for i, bottleneck in enumerate(all_bottlenecks[:10], 1):
                        f.write(f"  {i}. {bottleneck['function']} ({bottleneck['strategy']}): {bottleneck['time']:.4f}s\n")
        
        print(f"Performance dashboard saved to {save_path}")
        return save_path

    def analyze_trial_timeline(self, trial_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trial execution timeline to identify patterns.

        Args:
            trial_data: List of trial dictionaries with 'trial_number' and 'execution_time'

        Returns:
            Dictionary with timeline analysis including trends and anomalies
        """
        if not trial_data or len(trial_data) < 5:
            return {'error': 'Insufficient data for timeline analysis'}

        # Sort by trial number
        sorted_trials = sorted(trial_data, key=lambda x: x.get('trial_number', 0))

        trial_numbers = [t.get('trial_number', i) for i, t in enumerate(sorted_trials)]
        exec_times = [t.get('execution_time', 0) for t in sorted_trials]

        # Calculate rolling average
        window_size = min(10, len(exec_times) // 3)
        rolling_avg = []
        for i in range(len(exec_times)):
            start = max(0, i - window_size + 1)
            window = exec_times[start:i + 1]
            rolling_avg.append(np.mean(window))

        # Detect anomalies (trials significantly slower than average)
        mean_time = np.mean(exec_times)
        std_time = np.std(exec_times)
        anomalies = []
        for i, time in enumerate(exec_times):
            if time > mean_time + 2 * std_time:
                anomalies.append({
                    'trial_number': trial_numbers[i],
                    'execution_time': time,
                    'deviation': (time - mean_time) / std_time
                })

        # Detect trend (is optimization getting faster or slower over time?)
        if len(exec_times) > 10:
            # Simple linear regression
            x = np.arange(len(exec_times))
            slope, _ = np.polyfit(x, exec_times, 1)
            trend = 'accelerating' if slope < -0.01 else 'decelerating' if slope > 0.01 else 'stable'
        else:
            slope = 0
            trend = 'insufficient_data'

        return {
            'total_trials': len(exec_times),
            'mean_time': mean_time,
            'std_time': std_time,
            'trend': trend,
            'trend_slope': slope,
            'anomalies': anomalies,
            'rolling_average': rolling_avg,
            'speedup_ratio': exec_times[0] / exec_times[-1] if exec_times[0] > 0 and exec_times[-1] > 0 else 1.0
        }

    def detect_memory_leaks(self, memory_data: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Detect potential memory leaks from memory usage over time.

        Args:
            memory_data: List of (timestamp, memory_mb) tuples

        Returns:
            Dictionary with leak detection results
        """
        if not memory_data or len(memory_data) < 10:
            return {'leak_detected': False, 'confidence': 0, 'reason': 'Insufficient data'}

        timestamps = [t[0] for t in memory_data]
        memory_usage = [t[1] for t in memory_data]

        # Normalize timestamps
        start_time = timestamps[0]
        norm_times = [t - start_time for t in timestamps]

        # Linear regression to detect trend
        slope, intercept = np.polyfit(norm_times, memory_usage, 1)

        # Calculate R² to measure fit quality
        predicted = [slope * t + intercept for t in norm_times]
        ss_res = sum((m - p) ** 2 for m, p in zip(memory_usage, predicted))
        ss_tot = sum((m - np.mean(memory_usage)) ** 2 for m in memory_usage)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Detect leak: positive slope + high R²
        leak_detected = slope > 0.1 and r_squared > 0.7

        return {
            'leak_detected': leak_detected,
            'confidence': r_squared * 100,
            'memory_growth_rate': slope,  # MB per second
            'initial_memory': memory_usage[0],
            'final_memory': memory_usage[-1],
            'total_growth': memory_usage[-1] - memory_usage[0],
            'r_squared': r_squared,
            'recommendation': (
                f"[!] Potential memory leak detected! Memory growing at {slope:.2f} MB/s. "
                "Check for unclosed file handles, accumulating caches, or circular references."
                if leak_detected else
                "[OK] No significant memory leak detected."
            )
        }

    def compare_optimizations(self, opt_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple optimization runs to identify best practices.

        Args:
            opt_results: List of optimization result dictionaries

        Returns:
            Comparative analysis
        """
        if len(opt_results) < 2:
            return {'error': 'Need at least 2 optimization runs to compare'}

        comparison = {
            'runs': [],
            'best_run': None,
            'worst_run': None,
            'insights': []
        }

        for i, result in enumerate(opt_results):
            run_name = result.get('study_name', f'Run {i + 1}')
            avg_trial_time = result.get('avg_trial_time', 0)
            success_rate = (
                result.get('successful_trials', 0) / result.get('n_trials', 1) * 100
                if result.get('n_trials', 0) > 0 else 0
            )

            comparison['runs'].append({
                'name': run_name,
                'avg_trial_time': avg_trial_time,
                'success_rate': success_rate,
                'total_time': result.get('optimization_time_seconds', 0)
            })

        # Find best/worst by avg trial time
        comparison['runs'].sort(key=lambda x: x['avg_trial_time'])
        comparison['best_run'] = comparison['runs'][0]
        comparison['worst_run'] = comparison['runs'][-1]

        # Generate insights
        best_time = comparison['best_run']['avg_trial_time']
        worst_time = comparison['worst_run']['avg_trial_time']

        if worst_time > 0:
            speedup = worst_time / best_time
            if speedup > 2:
                comparison['insights'].append(
                    f"[*] Best configuration is {speedup:.1f}x faster than worst. "
                    f"Analyze '{comparison['best_run']['name']}' for optimization techniques."
                )

        return comparison

    def generate_actionable_recommendations(self, profile_data: Dict[str, Any]) -> List[str]:
        """
        Generate highly specific, actionable optimization recommendations.

        Args:
            profile_data: Complete profiling data from OptunaProfiler

        Returns:
            List of specific recommendations with priority levels
        """
        recommendations = []

        # Check bottlenecks
        bottlenecks = profile_data.get('bottlenecks', [])
        if bottlenecks:
            top_3 = bottlenecks[:3]
            total_time = sum(b.get('total_time', 0) for b in bottlenecks)

            for i, bottleneck in enumerate(top_3, 1):
                func = bottleneck.get('function', 'Unknown')
                func_time = bottleneck.get('total_time', 0)
                percentage = (func_time / total_time * 100) if total_time > 0 else 0

                if percentage > 30:
                    recommendations.append(
                        f"[P0 CRITICAL]: '{func}' consumes {percentage:.1f}% of total time. "
                        "This is the #1 optimization target."
                    )
                elif percentage > 15:
                    recommendations.append(
                        f"[P1 HIGH]: '{func}' consumes {percentage:.1f}% of total time. "
                        "Consider optimizing after addressing critical issues."
                    )

        # Check parameter correlations
        param_corr = profile_data.get('parameter_speed_correlation', {})
        for param_name, corr_data in list(param_corr.items())[:3]:
            corr = corr_data.get('correlation', 0)

            if abs(corr) > 0.7:
                if corr > 0:
                    recommendations.append(
                        f"[P2 MEDIUM]: Parameter '{param_name}' has strong positive correlation (r={corr:.2f}) with execution time. "
                        f"Reduce max value or add penalty for high values."
                    )
                else:
                    recommendations.append(
                        f"[INFO]: Parameter '{param_name}' correlates with faster execution (r={corr:.2f}). "
                        "This is good - consider favoring higher values."
                    )

        # Check memory usage
        memory_stats = profile_data.get('memory_profile_stats', {})
        if memory_stats:
            memory_diff = memory_stats.get('memory_diff_mb', 0)
            if memory_diff > 500:
                recommendations.append(
                    f"[P1 HIGH]: High memory usage ({memory_diff:.0f} MB). "
                    "Consider: 1) Using generators instead of lists, 2) Clearing caches, 3) Processing data in chunks."
                )

        # Check trial variance
        study_analysis = profile_data.get('study_analysis', {})
        if study_analysis:
            avg_time = study_analysis.get('avg_trial_time', 0)
            std_time = study_analysis.get('std_trial_time', 0)

            if std_time > avg_time * 0.8:
                recommendations.append(
                    f"[P1 HIGH]: Very high variance in trial times (sigma={std_time:.2f}s vs mu={avg_time:.2f}s). "
                    "Some parameter combinations are orders of magnitude slower. "
                    "Use parameter-speed correlation to identify and constrain slow parameters."
                )

        if not recommendations:
            recommendations.append("[OK] Performance looks excellent! No major optimizations needed.")

        return recommendations


class ProfileReport:
    """
    Comprehensive profile report generator
    """
    
    def __init__(self, output_dir: str = "profiling_reports"):
        """
        Initialize profile report generator
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.utils = ProfilingUtils(output_dir)
    
    def generate_strategy_report(self, profile_data: Dict[str, Any], 
                               strategy_name: str) -> str:
        """
        Generate comprehensive strategy performance report
        
        Args:
            profile_data: Strategy profiling data
            strategy_name: Name of the strategy
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"{strategy_name}_profile_{timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Strategy Performance Report: {strategy_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            exec_time = profile_data.get('execution_time', 0)
            memory_stats = profile_data.get('memory_profile_stats', {})
            memory_usage = memory_stats.get('memory_diff_mb', 0)
            
            f.write(f"- **Execution Time**: {exec_time:.4f} seconds\n")
            f.write(f"- **Memory Usage**: {memory_usage:.2f} MB\n")
            
            # Performance Assessment
            if exec_time < 1.0:
                f.write("- **Performance**: [EXCELLENT] (< 1 second)\n")
            elif exec_time < 5.0:
                f.write("- **Performance**: [GOOD] (1-5 seconds)\n")
            elif exec_time < 10.0:
                f.write("- **Performance**: [MODERATE] (5-10 seconds)\n")
            else:
                f.write("- **Performance**: [SLOW] (> 10 seconds)\n")
            
            f.write("\n")
            
            # Bottleneck Analysis
            bottlenecks = profile_data.get('bottlenecks', [])
            if bottlenecks:
                f.write("## Top Performance Bottlenecks\n\n")
                for i, bottleneck in enumerate(bottlenecks[:5], 1):
                    f.write(f"### {i}. {bottleneck.get('function', 'Unknown')}\n")
                    f.write(f"- **Calls**: {bottleneck.get('calls', 0)}\n")
                    f.write(f"- **Total Time**: {bottleneck.get('total_time', 0):.4f}s\n")
                    f.write(f"- **Time per Call**: {bottleneck.get('per_call', 0):.6f}s\n")
                    f.write(f"- **Cumulative Time**: {bottleneck.get('cumulative_time', 0):.4f}s\n\n")
            
            # Optimization Recommendations
            f.write("## Optimization Recommendations\n\n")
            recommendations = self._generate_recommendations(profile_data)
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            # Detailed Statistics
            f.write("## Detailed Statistics\n\n")
            f.write("```json\n")
            json.dump(profile_data, f, indent=2, default=str)
            f.write("\n```\n")
        
        print(f"Strategy report saved to {report_path}")
        return report_path
    
    def generate_optimization_report(self, optuna_data: Dict[str, Any], 
                                   study_name: str) -> str:
        """
        Generate comprehensive optimization report
        
        Args:
            optuna_data: Optuna profiling data
            study_name: Name of the optimization study
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"optimization_{study_name}_{timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Optimization Performance Report: {study_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Optimization Summary
            f.write("## Optimization Summary\n\n")
            opt_stats = optuna_data.get('optimization_stats', {})
            f.write(f"- **Total Trials**: {opt_stats.get('trials_total', 0)}\n")
            f.write(f"- **Successful Trials**: {opt_stats.get('trials_successful', 0)}\n")
            f.write(f"- **Pruned Trials**: {opt_stats.get('trials_pruned', 0)}\n")
            f.write(f"- **Failed Trials**: {opt_stats.get('trials_failed', 0)}\n")
            f.write(f"- **Total Time**: {opt_stats.get('total_time', 0):.2f} seconds\n")
            
            # Calculate success rate
            total = opt_stats.get('trials_total', 0)
            successful = opt_stats.get('trials_successful', 0)
            if total > 0:
                success_rate = (successful / total) * 100
                f.write(f"- **Success Rate**: {success_rate:.1f}%\n")
            
            f.write("\n")
            
            # Performance Analysis
            f.write("## Performance Analysis\n\n")
            
            # Trial timing analysis
            trial_times = optuna_data.get('trial_times', {})
            if trial_times:
                times = [t.get('execution_time', 0) for t in trial_times.values() if t and t.get('execution_time', 0) > 0]
                if times:
                    f.write(f"- **Average Trial Time**: {np.mean(times):.4f}s\n")
                    f.write(f"- **Median Trial Time**: {np.median(times):.4f}s\n")
                    f.write(f"- **Fastest Trial**: {min(times):.4f}s\n")
                    f.write(f"- **Slowest Trial**: {max(times):.4f}s\n")
            
            f.write("\n")
            
            # Resource Usage
            resource_usage = optuna_data.get('resource_usage', [])
            if resource_usage:
                cpu_values = [r['cpu_percent'] for r in resource_usage]
                memory_values = [r['memory_mb'] for r in resource_usage]
                
                f.write("### Resource Usage\n\n")
                f.write(f"- **Average CPU**: {np.mean(cpu_values):.2f}%\n")
                f.write(f"- **Peak CPU**: {np.max(cpu_values):.2f}%\n")
                f.write(f"- **Average Memory**: {np.mean(memory_values):.2f} MB\n")
                f.write(f"- **Peak Memory**: {np.max(memory_values):.2f} MB\n\n")
            
            # Optimization Recommendations
            f.write("## Optimization Recommendations\n\n")
            recommendations = self._generate_optimization_recommendations(optuna_data)
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

            # Bottleneck Analysis
            bottlenecks = optuna_data.get('bottlenecks', [])
            if bottlenecks:
                f.write("## Optimization Bottlenecks\n\n")
                for i, bottleneck in enumerate(bottlenecks[:5], 1):
                    f.write(f"### {i}. {bottleneck.get('type', 'Unknown').replace('_', ' ').title()}\n")
                    if 'trial_number' in bottleneck:
                        f.write(f"- **Trial**: {bottleneck['trial_number']}\n")
                    if 'execution_time' in bottleneck:
                        f.write(f"- **Execution Time**: {bottleneck['execution_time']:.4f}s\n")
                    if 'error' in bottleneck:
                        f.write(f"- **Error**: {bottleneck['error']}\n")
                    f.write("\n")
        
        print(f"Optimization report saved to {report_path}")
        return report_path
    
    def _generate_recommendations(self, profile_data: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on profile data"""
        recommendations = []
        
        exec_time = profile_data.get('execution_time', 0)
        bottlenecks = profile_data.get('bottlenecks', [])
        memory_stats = profile_data.get('memory_profile_stats', {})
        
        # Time-based recommendations
        if exec_time > 10.0:
            recommendations.append("Consider implementing caching for expensive calculations")
            recommendations.append("Profile and optimize the slowest functions (see bottlenecks below)")
        
        if exec_time > 5.0:
            recommendations.append("Consider using numba JIT compilation for numerical operations")
        
        # Memory-based recommendations
        memory_usage = memory_stats.get('memory_diff_mb', 0)
        if memory_usage > 100:
            recommendations.append("High memory usage detected - consider memory optimization techniques")
            recommendations.append("Use generators instead of lists where possible")
        
        # Bottleneck-specific recommendations
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            func_name = top_bottleneck.get('function', '')
            
            if 'numpy' in func_name.lower() or 'array' in func_name.lower():
                recommendations.append("Consider vectorizing operations with numpy")
            
            if 'loop' in func_name.lower():
                recommendations.append("Consider replacing loops with vectorized operations")
            
            if 'calculate' in func_name.lower():
                recommendations.append("Consider caching calculation results")
        
        if not recommendations:
            recommendations.append("Performance looks good - no major optimizations needed")
        
        return recommendations
    
    def _generate_optimization_recommendations(self, optuna_data: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on Optuna profile data"""
        recommendations = []
        
        opt_stats = optuna_data.get('optimization_stats', {})
        trial_times = optuna_data.get('trial_times', {})
        
        # Trial success rate
        total = opt_stats.get('trials_total', 0)
        successful = opt_stats.get('trials_successful', 0)
        pruned = opt_stats.get('trials_pruned', 0)
        failed = opt_stats.get('trials_failed', 0)
        
        if total > 0:
            success_rate = (successful / total) * 100
            if success_rate < 80:
                recommendations.append("Low success rate detected - check objective function for errors")
            
            prune_rate = (pruned / total) * 100
            if prune_rate > 50:
                recommendations.append("High pruning rate - consider adjusting pruning parameters")
            
            fail_rate = (failed / total) * 100
            if fail_rate > 10:
                recommendations.append("High failure rate - improve error handling in objective function")
        
        # Trial time analysis
        if trial_times:
            times = [t.get('execution_time', 0) for t in trial_times.values() if t and t.get('execution_time', 0) > 0]
            if times:
                avg_time = np.mean(times)
                if avg_time > 5.0:
                    recommendations.append("Slow trial execution - consider optimizing objective function")
                
                if np.std(times) > avg_time:
                    recommendations.append("High variance in trial times - check for inconsistent performance")
        
        # Resource usage
        resource_usage = optuna_data.get('resource_usage', [])
        if resource_usage:
            cpu_values = [r['cpu_percent'] for r in resource_usage]
            memory_values = [r['memory_mb'] for r in resource_usage]
            
            if cpu_values and np.mean(cpu_values) > 80:
                recommendations.append("High CPU usage - consider reducing computational complexity")
            
            if memory_values and np.max(memory_values) > 1000:
                recommendations.append("High memory usage - implement memory optimization techniques")
        
        # Parallel efficiency
        parallel_eff = optuna_data.get('parallel_efficiency', {})
        if parallel_eff:
            efficiency = parallel_eff.get('parallel_efficiency', 0)
            if efficiency < 50:
                recommendations.append("Low parallel efficiency - check for bottlenecks in parallel processing")
        
        if not recommendations:
            recommendations.append("Optimization performance looks good - no major issues detected")
        
        return recommendations