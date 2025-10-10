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
    Utility class for analyzing and visualizing profiling results
    """
    
    def __init__(self, output_dir: str = "profiling_reports"):
        """
        Initialize profiling utilities
        
        Args:
            output_dir: Directory for saving reports and plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
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
        
        with open(save_path, 'w') as f:
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
        
        with open(report_path, 'w') as f:
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
                f.write("- **Performance**: ⚡ Excellent (< 1 second)\n")
            elif exec_time < 5.0:
                f.write("- **Performance**: ✅ Good (1-5 seconds)\n")
            elif exec_time < 10.0:
                f.write("- **Performance**: ⚠️ Moderate (5-10 seconds)\n")
            else:
                f.write("- **Performance**: 🐌 Slow (> 10 seconds)\n")
            
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
        
        with open(report_path, 'w') as f:
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