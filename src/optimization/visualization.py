"""
Visualization module for optimization results

This module provides visualization capabilities for Optuna optimization results
using plotly for interactive charts.

Author: HFT System
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import warnings

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

try:
    import optuna
    import optuna.visualization as vis
    OPTUNA_VIS_AVAILABLE = True
except ImportError:
    OPTUNA_VIS_AVAILABLE = False
    warnings.warn("Optuna visualization not available. Install with: pip install optuna[visualization]")


class OptimizationVisualizer:
    """
    Visualizer for optimization results using plotly
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization. Install with: pip install plotly")
    
    def plot_optimization_history(self, study: 'optuna.Study', save_path: Optional[str] = None) -> go.Figure:
        """
        Plot optimization history
        
        Args:
            study: Optuna study object
            save_path: Path to save the plot (optional)
            
        Returns:
            Plotly figure object
        """
        if not OPTUNA_VIS_AVAILABLE:
            raise ImportError("Optuna visualization is required. Install with: pip install optuna[visualization]")
        
        fig = vis.plot_optimization_history(study)
        fig.update_layout(
            title="Optimization History",
            xaxis_title="Trial",
            yaxis_title="Objective Value",
            template="plotly_dark"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_parameter_importance(self, study: 'optuna.Study', save_path: Optional[str] = None) -> go.Figure:
        """
        Plot parameter importance
        
        Args:
            study: Optuna study object
            save_path: Path to save the plot (optional)
            
        Returns:
            Plotly figure object
        """
        if not OPTUNA_VIS_AVAILABLE:
            raise ImportError("Optuna visualization is required. Install with: pip install optuna[visualization]")
        
        try:
            fig = vis.plot_param_importances(study)
            fig.update_layout(
                title="Parameter Importance",
                xaxis_title="Importance",
                yaxis_title="Parameter",
                template="plotly_dark"
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        except Exception as e:
            # Create fallback figure if parameter importance fails
            fig = go.Figure()
            fig.add_annotation(
                text=f"Parameter importance plot failed: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Parameter Importance (Error)",
                template="plotly_dark"
            )
            return fig
    
    def plot_parallel_coordinate(self, study: 'optuna.Study', save_path: Optional[str] = None) -> go.Figure:
        """
        Plot parallel coordinate plot
        
        Args:
            study: Optuna study object
            save_path: Path to save the plot (optional)
            
        Returns:
            Plotly figure object
        """
        if not OPTUNA_VIS_AVAILABLE:
            raise ImportError("Optuna visualization is required. Install with: pip install optuna[visualization]")
        
        fig = vis.plot_parallel_coordinate(study)
        fig.update_layout(
            title="Parallel Coordinate Plot",
            template="plotly_dark"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_contour(self, study: 'optuna.Study', params: List[str], save_path: Optional[str] = None) -> go.Figure:
        """
        Plot contour plot for specified parameters
        
        Args:
            study: Optuna study object
            params: List of parameter names to plot
            save_path: Path to save the plot (optional)
            
        Returns:
            Plotly figure object
        """
        if not OPTUNA_VIS_AVAILABLE:
            raise ImportError("Optuna visualization is required. Install with: pip install optuna[visualization]")
        
        fig = vis.plot_contour(study, params=params)
        fig.update_layout(
            title=f"Contour Plot: {', '.join(params)}",
            template="plotly_dark"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_optimization_dashboard(self, 
                                   study: 'optuna.Study', 
                                   results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple plots
        
        Args:
            study: Optuna study object
            results: Optimization results dictionary
            save_path: Path to save the plot (optional)
            
        Returns:
            Plotly figure object with subplots
        """
        if not OPTUNA_VIS_AVAILABLE:
            raise ImportError("Optuna visualization is required. Install with: pip install optuna[visualization]")
        
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=("Optimization History", "Parameter Importance", 
                          "Best Parameters", "Final Backtest Results"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "bar"}]]
        )
        
        # 1. Optimization history
        history_fig = vis.plot_optimization_history(study)
        for trace in history_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # 2. Parameter importance
        try:
            importance_fig = vis.plot_param_importances(study)
            for trace in importance_fig.data:
                fig.add_trace(trace, row=1, col=2)
        except:
            # Add placeholder if importance fails
            fig.add_annotation(
                text="Parameter importance not available",
                xref="x2 domain", yref="y2 domain",
                x=0.5, y=0.5, showarrow=False
            )
        
        # 3. Best parameters table
        best_params = results.get('best_params', {})
        if best_params:
            param_data = []
            for param, value in best_params.items():
                param_data.append([param, f"{value:.4f}" if isinstance(value, float) else str(value)])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=["Parameter", "Value"]),
                    cells=dict(values=list(zip(*param_data)) if param_data else [[], []])
                ),
                row=2, col=1
            )
        
        # 4. Final backtest results
        final_backtest = results.get('final_backtest', {})
        if final_backtest:
            metrics = ['Total Trades', 'Win Rate', 'Net P&L', 'Sharpe Ratio', 'Profit Factor', 'Adjusted Score']
            values = [
                final_backtest.get('total', 0),
                final_backtest.get('win_rate', 0) * 100,
                final_backtest.get('net_pnl', 0),
                final_backtest.get('sharpe_ratio', 0),
                final_backtest.get('profit_factor', 0),
                final_backtest.get('adjusted_score', 0)
            ]
            
            fig.add_trace(
                go.Bar(x=metrics, y=values, name="Metrics"),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Optimization Dashboard: {results.get('strategy_name', 'Unknown')}",
            height=800,
            template="plotly_dark",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_parameter_evolution(self, 
                                study: 'optuna.Study', 
                                param_name: str,
                                save_path: Optional[str] = None) -> go.Figure:
        """
        Plot parameter evolution over trials
        
        Args:
            study: Optuna study object
            param_name: Parameter name to plot
            save_path: Path to save the plot (optional)
            
        Returns:
            Plotly figure object
        """
        if not OPTUNA_VIS_AVAILABLE:
            raise ImportError("Optuna visualization is required. Install with: pip install optuna[visualization]")
        
        # Extract parameter values and objective values
        trials = study.trials
        trial_numbers = []
        param_values = []
        objective_values = []
        
        for trial in trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_numbers.append(trial.number)
                param_values.append(trial.params.get(param_name))
                objective_values.append(trial.value)
        
        # Create scatter plot with color based on objective value
        fig = go.Figure()
        
        if param_values:
            fig.add_trace(go.Scatter(
                x=trial_numbers,
                y=param_values,
                mode='markers',
                marker=dict(
                    color=objective_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Objective Value")
                ),
                text=[f"Trial: {t}<br>Value: {v:.4f}<br>Objective: {o:.4f}" 
                      for t, v, o in zip(trial_numbers, param_values, objective_values)],
                hovertemplate="%{text}<extra></extra>"
            ))
        
        fig.update_layout(
            title=f"Parameter Evolution: {param_name}",
            xaxis_title="Trial Number",
            yaxis_title=param_name,
            template="plotly_dark"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_optimization_report(self, 
                                  study: 'optuna.Study',
                                  results: Dict[str, Any],
                                  output_dir: str) -> str:
        """
        Create a comprehensive HTML report with all visualizations
        
        Args:
            study: Optuna study object
            results: Optimization results dictionary
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated HTML report
        """
        import os
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate individual plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = results.get('strategy_name', 'unknown')
        
        plots = {}
        
        # Optimization history
        try:
            plots['history'] = self.plot_optimization_history(
                study, 
                save_path=os.path.join(output_dir, f"{strategy_name}_history_{timestamp}.html")
            )
        except Exception as e:
            print(f"Failed to create history plot: {e}")
        
        # Parameter importance
        try:
            plots['importance'] = self.plot_parameter_importance(
                study,
                save_path=os.path.join(output_dir, f"{strategy_name}_importance_{timestamp}.html")
            )
        except Exception as e:
            print(f"Failed to create importance plot: {e}")
        
        # Parallel coordinate
        try:
            plots['parallel'] = self.plot_parallel_coordinate(
                study,
                save_path=os.path.join(output_dir, f"{strategy_name}_parallel_{timestamp}.html")
            )
        except Exception as e:
            print(f"Failed to create parallel plot: {e}")
        
        # Dashboard
        try:
            plots['dashboard'] = self.plot_optimization_dashboard(
                study, results,
                save_path=os.path.join(output_dir, f"{strategy_name}_dashboard_{timestamp}.html")
            )
        except Exception as e:
            print(f"Failed to create dashboard: {e}")
        
        # Create HTML report
        report_path = os.path.join(output_dir, f"{strategy_name}_report_{timestamp}.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optimization Report: {strategy_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #1e1e1e; color: white; }}
                h1 {{ color: #4CAF50; }}
                h2 {{ color: #2196F3; }}
                .summary {{ background-color: #2d2d2d; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #3d3d3d; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #555; padding: 8px; text-align: left; }}
                th {{ background-color: #3d3d3d; }}
                .plot-container {{ margin: 20px 0; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
            </style>
        </head>
        <body>
            <h1>Optimization Report: {strategy_name}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Optimization Summary</h2>
                <div class="metric">Best Value: {results.get('best_value', 'N/A'):.4f}</div>
                <div class="metric">Total Trials: {results.get('n_trials', 0)}</div>
                <div class="metric">Successful Trials: {results.get('successful_trials', 0)}</div>
                <div class="metric">Optimization Time: {results.get('optimization_time_seconds', 0):.2f}s</div>
            </div>
            
            <div class="summary">
                <h2>Best Parameters</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        # Add best parameters to report
        best_params = results.get('best_params', {})
        for param, value in best_params.items():
            html_content += f"<tr><td>{param}</td><td>{value}</td></tr>"
        
        html_content += """
                </table>
            </div>
        """
        
        # Add final backtest results
        final_backtest = results.get('final_backtest', {})
        if final_backtest:
            html_content += f"""
            <div class="summary">
                <h2>Final Backtest Results</h2>
                <div class="metric">Total Trades: {final_backtest.get('total', 0)}</div>
                <div class="metric">Win Rate: {final_backtest.get('win_rate', 0):.1%}</div>
                <div class="metric">Net P&L: ${final_backtest.get('net_pnl', 0):,.2f}</div>
                <div class="metric">Return: {final_backtest.get('net_pnl_percentage', 0):.2f}%</div>
                <div class="metric">Sharpe Ratio: {final_backtest.get('sharpe_ratio', 0):.2f}</div>
                <div class="metric">Profit Factor: {final_backtest.get('profit_factor', 0):.2f}</div>
                <div class="metric">Adjusted Score: {final_backtest.get('adjusted_score', 0):.2f}</div>
                <div class="metric">Max Drawdown: {final_backtest.get('max_drawdown', 0):.2f}%</div>
            </div>
            """
        
        # Add plots to report
        for plot_name, plot_fig in plots.items():
            plot_file = f"{strategy_name}_{plot_name}_{timestamp}.html"
            html_content += f"""
            <div class="plot-container">
                <h2>{plot_name.replace('_', ' ').title()}</h2>
                <iframe src="{plot_file}"></iframe>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Optimization report saved to: {report_path}")
        return report_path


# Convenience function for quick visualization
def quick_visualize(study: 'optuna.Study', 
                   results: Dict[str, Any], 
                   output_dir: str = "optimization_plots") -> str:
    """
    Quick visualization with default settings
    
    Args:
        study: Optuna study object
        results: Optimization results dictionary
        output_dir: Directory to save plots
        
    Returns:
        Path to the generated HTML report
    """
    visualizer = OptimizationVisualizer()
    return visualizer.create_optimization_report(study, results, output_dir)