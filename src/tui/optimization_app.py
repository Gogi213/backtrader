"""
Terminal User Interface for HFT Optimization

A fast, lightweight TUI for running strategy optimizations with Optuna.
Provides real-time progress tracking and results display without GUI overhead.

Author: HFT System
"""
import os
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Static, Button, ProgressBar, DataTable,
    Input, Select, Label
)
from textual.reactive import reactive
from textual.binding import Binding
from rich.text import Text

from ..optimization.fast_optimizer import FastStrategyOptimizer
from ..strategies.strategy_registry import StrategyRegistry


class OptimizationProgress(Static):
    """Widget to display optimization progress"""
    
    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.trials_completed = 0
        self.total_trials = 0
        self.best_value = None
        self.current_trial = 0
    
    def update_progress(self, trial_number: int, total_trials: int, best_value: float, current_value: float = None):
        """Update progress information"""
        self.trials_completed = trial_number + 1
        self.total_trials = total_trials
        self.best_value = best_value
        self.current_trial = trial_number
        
        progress_text = f"[bold]Прогресс:[/bold] {self.trials_completed}/{self.total_trials} испытаний\n"
        progress_text += f"[bold]Лучшее значение:[/bold] {best_value:.4f}\n"
        
        if current_value is not None:
            progress_text += f"[bold]Текущее значение:[/bold] {current_value:.4f}\n"
        
        progress_text += f"[bold]Текущее испытание:[/bold] {trial_number + 1}"
        
        self.update(progress_text)


class ResultsTable(DataTable):
    """Table to display optimization results"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_columns("Параметр", "Значение")
        self.cursor_type = "none"
    
    def update_results(self, results: Dict[str, Any]):
        """Update table with optimization results"""
        self.clear()
        
        # Basic results
        self.add_row("Стратегия", results.get('strategy_name', 'N/A'))
        self.add_row("Символ", results.get('symbol', 'N/A'))
        self.add_row("Лучшее значение", f"{results.get('best_value', 0):.4f}")
        self.add_row("Всего испытаний", str(results.get('n_trials', 0)))
        self.add_row("Успешных испытаний", str(results.get('successful_trials', 0)))
        self.add_row("Обрезанных испытаний", str(results.get('pruned_trials', 0)))
        self.add_row("Время оптимизации", f"{results.get('optimization_time_seconds', 0):.2f} сек")
        self.add_row("Параллельных заданий", str(results.get('parallel_jobs', 1)))
        
        # Best parameters
        if 'best_params' in results and results['best_params']:
            self.add_row("", "")  # Separator
            self.add_row("[bold]Лучшие параметры:[/bold]", "")
            for param, value in results['best_params'].items():
                self.add_row(f"  {param}", str(value))
        
        # Final backtest results
        if 'final_backtest' in results and results['final_backtest']:
            self.add_row("", "")  # Separator
            self.add_row("[bold]Финальный бэктест:[/bold]", "")
            backtest = results['final_backtest']
            self.add_row("  Всего сделок", str(backtest.get('total', 0)))
            self.add_row("  Sharpe Ratio", f"{backtest.get('sharpe_ratio', 0):.2f}")
            self.add_row("  P&L", f"${backtest.get('net_pnl', 0):,.2f}")
            self.add_row("  Доходность", f"{backtest.get('net_pnl_percentage', 0):.2f}%")
            self.add_row("  Макс. просадка", f"{backtest.get('max_drawdown', 0):.2f}%")


class ParameterTable(DataTable):
    """Table to display strategy parameters"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_columns("Параметр", "Тип", "Диапазон", "Значение по умолчанию")
        self.cursor_type = "row"
    
    def update_parameters(self, strategy_name: str):
        """Update table with strategy parameters"""
        self.clear()
        
        try:
            from ..strategies.strategy_registry import StrategyRegistry
            strategy_class = StrategyRegistry.get(strategy_name)
            
            if strategy_class and hasattr(strategy_class, 'get_param_space'):
                param_space = strategy_class.get_param_space()
                
                for param_name, param_spec in param_space.items():
                    param_type = param_spec[0]
                    
                    if param_type == 'float' or param_type == 'int':
                        if len(param_spec) >= 3:
                            min_val, max_val = param_spec[1], param_spec[2]
                            range_str = f"[{min_val}, {max_val}]"
                        else:
                            range_str = "N/A"
                    elif param_type == 'categorical':
                        range_str = str(param_spec[1])
                    else:
                        range_str = "N/A"
                    
                    default_value = getattr(strategy_class, f"_{param_name}", "N/A")
                    
                    self.add_row(param_name, param_type, range_str, str(default_value))
        except Exception as e:
            self.add_row("Ошибка", str(e), "", "")


class OptimizationApp(App):
    """Main TUI application for optimization"""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    Header {
        text-align: center;
        content-align: center middle;
    }
    
    .container {
        padding: 1;
    }
    
    .progress-container {
        height: 8;
        border: solid $primary;
        margin: 1 0;
    }
    
    .results-container {
        height: 1fr;
        border: solid $accent;
    }
    
    .button-container {
        height: 3;
    }
    
    Button {
        margin: 0 1;
    }
    
    DataTable {
        height: 1fr;
    }
    
    .input-container {
        height: 4;
        margin: 1 0;
    }
    
    Input, Select {
        width: 1fr;
        margin: 0 1;
    }
    
    .title {
        text-align: center;
        content-align: center middle;
        height: 3;
        margin: 1 0;
    }
    
    #left-panel {
        width: 40%;
        height: 1fr;
    }
    
    #right-panel {
        width: 60%;
        height: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Выход"),
        Binding("r", "run_optimization", "Запуск"),
        Binding("c", "clear_results", "Очистить"),
        Binding("ctrl+c", "quit", "Выход"),
    ]
    
    optimization_results: reactive[Optional[Dict[str, Any]]] = reactive(None)
    is_optimizing: reactive[bool] = reactive(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = None
        self.optimization_task = None
    
    def compose(self) -> ComposeResult:
        """Compose the UI"""
        yield Header()
        
        with Container(classes="container"):
            with Horizontal():
                with Vertical(id="left-panel"):
                    yield Label("Конфигурация оптимизации", classes="title")
                    
                    with Horizontal():
                        with Vertical():
                            yield Label("Стратегия:")
                            strategy_options = self._get_strategy_options()
                            default_strategy = strategy_options[0][1] if strategy_options else "hierarchical_mean_reversion"
                            yield Select(
                                options=strategy_options,
                                id="strategy-select",
                                value=default_strategy
                            )
                        
                        with Vertical():
                            yield Label("Датасет:")
                            dataset_options = self._get_dataset_options()
                            default_dataset = dataset_options[0][1] if dataset_options else ""
                            yield Select(
                                options=dataset_options,
                                id="dataset-select",
                                value=default_dataset
                            )
                    
                    with Horizontal():
                        with Vertical():
                            yield Label("Испытаний:")
                            yield Input(
                                placeholder="Количество испытаний",
                                id="trials-input",
                                value="100"
                            )
                        
                        with Vertical():
                            yield Label("Метрика:")
                            yield Select(
                                options=[
                                    ("Sharpe Ratio", "sharpe_ratio"),
                                    ("Net P&L", "net_pnl"),
                                    ("Profit Factor", "profit_factor"),
                                    ("Win Rate", "win_rate"),
                                    ("Return %", "net_pnl_percentage"),
                                    ("Adjusted Score", "adjusted_score")
                                ],
                                id="metric-select",
                                value="sharpe_ratio"
                            )
                    
                    with Horizontal():
                        with Vertical():
                            yield Label("Мин. сделок:")
                            yield Input(
                                placeholder="Минимальное количество сделок",
                                id="min-trades-input",
                                value="10"
                            )
                        
                        with Vertical():
                            yield Label("Макс. просадка (%):")
                            yield Input(
                                placeholder="Максимальная просадка",
                                id="max-drawdown-input",
                                value="50.0"
                            )
                    
                    with Horizontal():
                        with Vertical():
                            yield Label("Начальный капитал ($):")
                            yield Input(
                                placeholder="Начальный капитал",
                                id="initial-capital-input",
                                value="10000.0"
                            )
                        
                        with Vertical():
                            yield Label("Размер позиции ($):")
                            yield Input(
                                placeholder="Размер позиции",
                                id="position-size-input",
                                value="1000.0"
                            )
                    
                    with Horizontal():
                        with Vertical():
                            yield Label("Комиссия (%):")
                            yield Input(
                                placeholder="Комиссия",
                                id="commission-input",
                                value="0.05"
                            )
                        
                        with Vertical():
                            yield Label("Параллельных заданий:")
                            yield Input(
                                placeholder="Параллельных заданий (-1 для всех ядер)",
                                id="jobs-input",
                                value="-1"
                            )
                    
                    with Container(classes="button-container"):
                        yield Button("Запуск оптимизации", id="run-button", variant="primary")
                        yield Button("Очистить результаты", id="clear-button", variant="default")
                        yield Button("Выход", id="quit-button", variant="error")
                
                with Vertical(id="right-panel"):
                    yield Label("Прогресс оптимизации", classes="title")
                    yield OptimizationProgress(id="progress-widget", classes="progress-container")
                    
                    yield Label("Результаты", classes="title")
                    yield ResultsTable(id="results-table", classes="results-container")
        
        yield Footer()
    
    def _get_strategy_options(self) -> List[tuple]:
        """Get available strategies"""
        try:
            strategies = StrategyRegistry.list_strategies()
            return [(strategy, strategy) for strategy in strategies]
        except Exception:
            return [("hierarchical_mean_reversion", "hierarchical_mean_reversion")]
    
    def _get_dataset_options(self) -> List[tuple]:
        """Get available datasets from upload/klines directory"""
        try:
            klines_dir = "upload/klines"
            if os.path.exists(klines_dir):
                csv_files = [f for f in os.listdir(klines_dir) if f.endswith('.csv')]
                return [(file, file) for file in csv_files]
            return []
        except Exception:
            return []
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "run-button":
            await self.action_run_optimization()
        elif button_id == "clear-button":
            self.action_clear_results()
        elif button_id == "quit-button":
            self.action_quit()
    
    async def action_run_optimization(self) -> None:
        """Run optimization with current configuration"""
        if self.is_optimizing:
            self.notify("Оптимизация уже запущена!", severity="warning")
            return
        
        try:
            # Get configuration from UI
            strategy_name = self.query_one("#strategy-select", Select).value
            dataset_file = self.query_one("#dataset-select", Select).value
            n_trials = int(self.query_one("#trials-input", Input).value)
            objective_metric = self.query_one("#metric-select", Select).value
            n_jobs = int(self.query_one("#jobs-input", Input).value)
            min_trades = int(self.query_one("#min-trades-input", Input).value)
            max_drawdown = float(self.query_one("#max-drawdown-input", Input).value)
            initial_capital = float(self.query_one("#initial-capital-input", Input).value)
            position_size = float(self.query_one("#position-size-input", Input).value)
            commission = float(self.query_one("#commission-input", Input).value)
            
            # Construct data path and extract symbol
            data_path = os.path.join("upload/klines", dataset_file)
            symbol = dataset_file.split('-')[0] if '-' in dataset_file else dataset_file.split('.')[0]
            
            # Validate inputs
            if not dataset_file:
                self.notify("Выберите датасет!", severity="error")
                return
            
            if not os.path.exists(data_path):
                self.notify(f"Файл данных не найден: {data_path}", severity="error")
                return
            
            if n_trials <= 0:
                self.notify("Количество испытаний должно быть положительным!", severity="error")
                return
            
            # Start optimization
            self.is_optimizing = True
            self.query_one("#run-button", Button).disabled = True
            
            self.notify(f"Запуск оптимизации для {strategy_name}...")
            
            # Create optimizer with backtest config
            from ..core.backtest_config import BacktestConfig
            backtest_config = BacktestConfig(
                strategy_name=strategy_name,
                symbol=symbol,
                data_path=data_path,
                initial_capital=initial_capital,
                commission_pct=commission / 100.0,  # Convert from percentage
                position_size_dollars=position_size
            )
            
            self.optimizer = FastStrategyOptimizer(
                strategy_name=strategy_name,
                data_path=data_path,
                symbol=symbol,
                backtest_config=backtest_config
            )
            
            # Run optimization in background thread
            def run_optimization_thread():
                try:
                    # Create progress callback
                    def progress_callback(trial_number: int, total_trials: int, best_value: float, current_value: float = None):
                        self.call_from_thread(
                            self.query_one("#progress-widget", OptimizationProgress).update_progress,
                            trial_number, total_trials, best_value, current_value
                        )
                    
                    # Run optimization
                    results = self.optimizer.optimize(
                        n_trials=n_trials,
                        objective_metric=objective_metric,
                        min_trades=min_trades,
                        max_drawdown_threshold=max_drawdown,
                        n_jobs=n_jobs,
                        timeout=None  # No timeout for TUI
                    )
                    
                    # Update results
                    self.call_from_thread(self._update_results, results)
                    
                except Exception as e:
                    self.call_from_thread(
                        self.notify,
                        f"Ошибка оптимизации: {e}",
                        severity="error"
                    )
                finally:
                    # Reset state
                    self.call_from_thread(self._reset_optimization_state)
            
            # Start thread
            optimization_thread = threading.Thread(target=run_optimization_thread)
            optimization_thread.daemon = True
            optimization_thread.start()
            
        except ValueError as e:
            self.notify(f"Ошибка в параметрах: {e}", severity="error")
        except Exception as e:
            self.notify(f"Ошибка запуска оптимизации: {e}", severity="error")
            self.is_optimizing = False
            self.query_one("#run-button", Button).disabled = False
    
    def _update_results(self, results: Dict[str, Any]) -> None:
        """Update results display"""
        self.optimization_results = results
        self.query_one("#results-table", ResultsTable).update_results(results)
        self.notify("Оптимизация завершена!", severity="success")
    
    def _reset_optimization_state(self) -> None:
        """Reset optimization state"""
        self.is_optimizing = False
        self.query_one("#run-button", Button).disabled = False
        self.optimization_task = None
    
    def action_clear_results(self) -> None:
        """Clear results"""
        self.optimization_results = None
        self.query_one("#results-table", ResultsTable).clear()
        self.query_one("#progress-widget", OptimizationProgress).update("")
        self.notify("Результаты очищены")
    
    def action_quit(self) -> None:
        """Quit the application"""
        if self.optimization_task:
            self.optimization_task.cancel()
        self.exit()


def run_tui_app():
    """Run the TUI application"""
    app = OptimizationApp()
    app.run()


if __name__ == "__main__":
    run_tui_app()