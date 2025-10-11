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
    Header, Footer, Static, Button, DataTable,
    Input, Select, Label
)
from textual.reactive import reactive
from textual.binding import Binding
from rich.text import Text

from ..optimization.fast_optimizer import FastStrategyOptimizer
import src.strategies  # This will trigger the dynamic loading
from ..strategies import StrategyRegistry



class CombinedResultsTable(DataTable):
    """A single horizontal table to display both final backtest metrics and best parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor_type = "none"

    def update_data(self, results: Optional[Dict[str, Any]]):
        """Update the table with final backtest results and best parameters."""
        self.clear()
        
        # Clear all columns by removing them one by one, as we rebuild them dynamically
        while len(self.columns) > 0:
            key = list(self.columns.keys())[0]
            self.remove_column(key)

        if not results or 'final_backtest' not in results or not results['final_backtest']:
            return

        metrics = results['final_backtest']
        params = results.get('best_params', {})

        # Filter only optimized params to display
        param_space = {}
        strategy_name = results.get('strategy_name')
        if strategy_name:
            try:
                strategy_class = StrategyRegistry.get_strategy(strategy_name)
                if strategy_class:
                    param_space = strategy_class.get_param_space()
            except Exception:
                pass  # Silently fail if strategy is not found

        params_to_show = {k: v for k, v in params.items() if k in param_space}

        # --- Create Columns ---
        metric_headers = [
            "Trades", "PnL", "WinRate", "WRLong", "WRShort", "Sharpe", "PF", "MaxDD%",
            "AvgWin%", "AvgLoss%", "ConsSL"
        ]
        param_headers = list(params_to_show.keys())
        
        self.add_columns(*(metric_headers + param_headers))

        # --- Create Data Row ---
        metric_values = [
            str(metrics.get('total', 0)),
            f"{metrics.get('net_pnl', 0):.2f}",
            f"{metrics.get('win_rate', 0):.2%}",
            f"{metrics.get('winrate_long', 0):.2%}",
            f"{metrics.get('winrate_short', 0):.2%}",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            f"{metrics.get('profit_factor', 0):.2f}",
            f"{metrics.get('max_drawdown', 0):.2f}",
            f"{metrics.get('average_win', 0):.2f}",
            f"{abs(metrics.get('average_loss', 0)):.2f}",
            str(metrics.get('consecutive_stops', 0))
        ]
        param_values = [str(v) for v in params_to_show.values()]

        self.add_row(*(metric_values + param_values))


class PositiveTrialsTable(DataTable):
    """A table to display all optimization trials with positive PnL, including parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor_type = "row"

    def update_data(self, results: Optional[Dict[str, Any]]):
        """Update the table with a list of positive trials and their parameters."""
        self.clear()

        # Clear all columns dynamically as they depend on the results
        while len(self.columns) > 0:
            key = list(self.columns.keys())[0]
            self.remove_column(key)

        positive_trials = results.get('positive_trials') if results else None
        if not positive_trials:
            return

        # Check if this is a multi-dataset run
        is_multi_dataset = results.get('datasets_count', 1) > 1

        # --- Determine Headers ---
        metric_headers = [
            "Trial", "Trades", "PnL", "WinRate", "WRLong", "WRShort", "Sharpe", "PF", "MaxDD%",
            "AvgWin%", "AvgLoss%", "ConsSL"
        ]

        # Add Dataset column for multi-dataset runs
        if is_multi_dataset:
            metric_headers.insert(0, "Dataset")

        # Get the set of all parameter names from all positive trials
        param_space = {}
        strategy_name = results.get('strategy_name')
        if strategy_name:
            try:
                strategy_class = StrategyRegistry.get_strategy(strategy_name)
                if strategy_class:
                    param_space = strategy_class.get_param_space()
            except Exception:
                pass

        param_keys = set()
        for trial in positive_trials:
            params = trial.get('params', {})
            for key in params:
                if key in param_space: # Only show params that are part of the optimization space
                    param_keys.add(key)

        param_headers = sorted(list(param_keys))

        self.add_columns(*(metric_headers + param_headers))

        # --- Add Data Rows ---
        sorted_trials = sorted(positive_trials, key=lambda x: x.get('pnl', 0), reverse=True)

        for trial in sorted_trials:
            metric_values = [
                str(trial.get('trial', '')),
                str(trial.get('trades', 0)),
                f"{trial.get('pnl', 0):.2f}",
                f"{trial.get('winrate', 0):.2%}",
                f"{trial.get('winrate_long', 0):.2%}",
                f"{trial.get('winrate_short', 0):.2%}",
                f"{trial.get('sharpe', 0):.2f}",
                f"{trial.get('pf', 0):.2f}",
                f"{trial.get('max_dd', 0):.2f}",
                f"{trial.get('avg_win', 0):.2f}",
                f"{abs(trial.get('avg_loss', 0)):.2f}",
                str(trial.get('consecutive_stops', 0))
            ]

            # Add dataset name for multi-dataset runs
            if is_multi_dataset:
                dataset_name = trial.get('dataset', 'unknown')
                # Shorten dataset name for display
                if len(dataset_name) > 25:
                    dataset_name = dataset_name[:22] + "..."
                metric_values.insert(0, dataset_name)

            trial_params = trial.get('params', {})
            param_values = []
            for key in param_headers:
                value = trial_params.get(key)
                if isinstance(value, float):
                    # Format floats to 2 decimal places for readability
                    param_values.append(f"{value:.2f}")
                else:
                    param_values.append(str(value) if value is not None else "N/A")

            self.add_row(*(metric_values + param_values))


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
        padding: 0 1;
    }

    #form {
        layout: grid;
        grid-size: 10;
        grid-columns: auto 1fr auto 1fr auto 1fr auto 1fr auto 1fr;
        grid-rows: auto;
        grid-gutter: 0 1;
    }

    #form > Label {
        text-align: right;
        margin-right: 1;
    }
    
    #results-table {
        height: 6; /* Fixed height for the best result table */
        margin-top: 2; /* Add vertical offset */
        border: solid $accent;
    }

    #positive-trials-table {
        height: 1fr; /* Takes the rest of the available space */
        border: solid $primary;
    }
    
    Button {
        margin: 0 1;
    }

    DataTable {
        /* Height is now controlled by ID selectors */
    }
    
    Input, Select {
        width: 1fr;
    }
    
    .title {
        text-align: center;
        content-align: center middle;
        margin-top: 1; /* Add space above the title for the second table */
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Выход"),
        Binding("r", "run_optimization", "Запуск"),
        Binding("c", "clear_results", "Очистить"),
        Binding("t", "plot_trades", "График сделок"),
        Binding("o", "open_plot", "Открыть график"),
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
            with Container(id="form"):
                yield Label("Стратегия:")
                yield Select(
                    options=[("Загрузка...", "loading")],
                    id="strategy-select"
                )
                
                yield Label("Датасет:")
                yield Select(
                    options=[("Загрузка...", "")],
                    id="dataset-select"
                )
                
                yield Label("Испытаний:")
                yield Input(
                    placeholder="Количество испытаний",
                    value="100",
                    id="trials-input"
                )
                
                yield Label("Метрика:")
                yield Select(
                    options=[
                        ("Sharpe Ratio", "sharpe_ratio"),
                        ("PnL (Profit & Loss)", "pnl"),
                        ("Sharpe * PF * Trades", "sharpe_pf_trades_score"),
                        ("Sharpe с штрафом за DD", "sharpe_with_drawdown_penalty"),
                        ("Sharpe * Profit Factor", "sharpe_x_profit_factor"),
                    ],
                    id="metric-select"
                )
                
                yield Label("Мин. сделок:")
                yield Input(
                    placeholder="Минимальное количество сделок",
                    value="10",
                    id="min-trades-input"
                )
                
                yield Label("Макс. просадка (%):")
                yield Input(
                    placeholder="Максимальная просадка",
                    value="50.0",
                    id="max-drawdown-input"
                )
                
                yield Label("Начальный капитал ($):")
                yield Input(
                    placeholder="Начальный капитал",
                    value="100.0",
                    id="initial-capital-input"
                )
                
                yield Label("Размер позиции ($):")
                yield Input(
                    placeholder="Размер позиции",
                    value="100.0",
                    id="position-size-input"
                )
                
                yield Label("Комиссия (%):")
                yield Input(
                    placeholder="Комиссия",
                    value="0.05",
                    id="commission-input"
                )
                
                yield Label("Параллельных заданий:")
                yield Input(
                    placeholder="Параллельных заданий (-1 для всех ядер)",
                    value="-1",
                    id="jobs-input"
                )
            
            # Progress Label

            # Combined results table
            yield Static("[bold]ЛУЧШИЙ РЕЗУЛЬТАТ (МЕТРИКИ И ПАРАМЕТРЫ):[/bold]")
            yield CombinedResultsTable(id="results-table")

            # Positive trials table
            yield Static("[bold]ВСЕ ПОЛОЖИТЕЛЬНЫЕ РЕЗУЛЬТАТЫ (PnL > 0):[/bold]", classes="title")
            yield PositiveTrialsTable(id="positive-trials-table")
        
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.call_later(self._set_default_values)

    def _set_default_values(self) -> None:
        """Populate selects and set all default values for the form."""
        # --- Populate and set Selects ---
        strategy_select = self.query_one("#strategy-select", Select)
        strategy_options = self._get_strategy_options()
        strategy_select.set_options(strategy_options)
        if strategy_options:
            strategy_select.value = strategy_options[0][1]
        else:
            strategy_select.set_options([("Стратегии не найдены", None)])
            strategy_select.disabled = True

        dataset_select = self.query_one("#dataset-select", Select)
        dataset_options = self._get_dataset_options()
        dataset_select.set_options(dataset_options)
        if dataset_options:
            dataset_select.value = dataset_options[0][1]
        else:
            dataset_select.set_options([("Датасеты не найдены", None)])
            dataset_select.disabled = True
        
        self.query_one("#metric-select", Select).value = "sharpe_ratio"
    
    def _get_strategy_options(self) -> List[tuple]:
        """Get available strategies"""
        strategies = StrategyRegistry.list_strategies()
        if not strategies:
            self.notify("Ни одна стратегия не найдена. Проверьте директорию 'src/strategies'.",
                        severity="error", timeout=10)
            return []
        return [(strategy, strategy) for strategy in strategies]
    
    def _get_dataset_options(self) -> List[tuple]:
        """Discover available dataset files and folders and format them for a Select widget."""
        try:
            project_root = Path(__file__).parent.parent.parent
            dataset_dir = project_root / "upload" / "klines"

            self.log.info(f"Searching for datasets in absolute path: {dataset_dir}")

            if not dataset_dir.is_dir():
                self.log.warning(f"Dataset directory not found: {dataset_dir}")
                return []

            options = []

            # Look for folders (for multi-dataset runs)
            folders = [
                f.name for f in dataset_dir.iterdir()
                if f.is_dir()
            ]
            for folder in folders:
                options.append((f"📁 {folder}", folder))

            # Look for parquet files (for single dataset runs)
            files = [
                f.name for f in dataset_dir.iterdir()
                if f.is_file() and f.name.endswith('.parquet')
            ]
            for file in files:
                options.append((f"📄 {file}", file))

            self.log.info(f"Discovered {len(folders)} folders and {len(files)} dataset files.")
            return sorted(options, key=lambda x: x[0])
        except Exception as e:
            self.log.error(f"Failed to discover datasets due to an unexpected error: {e}")
            return []
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "run-button":
            await self.action_run_optimization()
        elif button_id == "clear-button":
            self.action_clear_results()
        elif button_id == "plot-trades-button":
            await self.action_plot_trades()
        elif button_id == "open-plot-button":
            await self.action_open_plot()
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
            dataset_selection = self.query_one("#dataset-select", Select).value
            n_trials = int(self.query_one("#trials-input", Input).value)
            objective_metric = self.query_one("#metric-select", Select).value
            n_jobs = int(self.query_one("#jobs-input", Input).value)
            min_trades = int(self.query_one("#min-trades-input", Input).value)
            max_drawdown = float(self.query_one("#max-drawdown-input", Input).value)
            initial_capital = float(self.query_one("#initial-capital-input", Input).value)
            position_size = float(self.query_one("#position-size-input", Input).value)
            commission = float(self.query_one("#commission-input", Input).value)

            # Validate inputs
            if not dataset_selection:
                self.notify("Выберите датасет!", severity="error")
                return

            if n_trials <= 0:
                self.notify("Количество испытаний должно быть положительным!", severity="error")
                return

            # Check if the selection is a folder or a file
            data_path = os.path.join("upload/klines", dataset_selection)
            is_folder = os.path.isdir(data_path)

            if is_folder:
                # Multi-dataset mode: get all parquet files from the folder
                dataset_files = [
                    f for f in os.listdir(data_path)
                    if f.endswith('.parquet')
                ]
                if not dataset_files:
                    self.notify(f"Папка не содержит датасетов: {data_path}", severity="error")
                    return

                # Start multi-dataset optimization
                self.is_optimizing = True
                self.notify(f"Запуск оптимизации на {len(dataset_files)} датасетах из папки {dataset_selection}...")

                # Run multi-dataset optimization in background thread
                def run_multi_optimization_thread():
                    all_results = []
                    try:
                        for idx, dataset_file in enumerate(dataset_files, 1):
                            file_path = os.path.join(data_path, dataset_file)
                            symbol = dataset_file.split('-')[0] if '-' in dataset_file else dataset_file.split('.')[0]

                            self.call_from_thread(
                                self.notify,
                                f"[{idx}/{len(dataset_files)}] Оптимизация {dataset_file}...",
                                severity="information"
                            )

                            # Create optimizer with backtest config
                            from ..core.backtest_config import BacktestConfig
                            backtest_config = BacktestConfig(
                                strategy_name=strategy_name,
                                symbol=symbol,
                                data_path=file_path,
                                initial_capital=initial_capital,
                                commission_pct=commission / 100.0 if commission >= 1 else commission,
                                position_size_dollars=position_size
                            )

                            optimizer = FastStrategyOptimizer(
                                strategy_name=strategy_name,
                                data_path=file_path,
                                symbol=symbol,
                                backtest_config=backtest_config
                            )

                            # Run optimization for this dataset
                            results = optimizer.optimize(
                                n_trials=n_trials,
                                objective_metric=objective_metric,
                                min_trades=min_trades,
                                max_drawdown_threshold=max_drawdown,
                                n_jobs=n_jobs,
                                timeout=None,
                                callbacks=[]
                            )

                            # Add dataset info to results
                            results['dataset_file'] = dataset_file
                            all_results.append(results)

                        # Aggregate results from all datasets
                        aggregated_results = self._aggregate_multi_dataset_results(all_results)

                        # Update results
                        self.call_from_thread(self._update_results, aggregated_results)

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
                optimization_thread = threading.Thread(target=run_multi_optimization_thread)
                optimization_thread.daemon = True
                optimization_thread.start()

            else:
                # Single dataset mode (original behavior)
                symbol = dataset_selection.split('-')[0] if '-' in dataset_selection else dataset_selection.split('.')[0]

                if not os.path.exists(data_path):
                    self.notify(f"Файл данных не найден: {data_path}", severity="error")
                    return

                # Start optimization
                self.is_optimizing = True

                self.notify(f"Запуск оптимизации для {strategy_name}...")

                # Create optimizer with backtest config
                from ..core.backtest_config import BacktestConfig
                backtest_config = BacktestConfig(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    data_path=data_path,
                    initial_capital=initial_capital,
                    commission_pct=commission / 100.0 if commission >= 1 else commission,
                    position_size_dollars=position_size
                )

                self.optimizer = FastStrategyOptimizer(
                    strategy_name=strategy_name,
                    data_path=data_path,
                    symbol=symbol,
                    backtest_config=backtest_config
                )

                # Callback for progress update
                def progress_callback(study, trial):
                    total_trials = n_trials
                    completed_trials = trial.number + 1
                    progress_text = f"Прогресс: {completed_trials}|{total_trials}"
                    # Progress is disabled for a more compact view
                    pass

                # Run optimization in background thread
                def run_optimization_thread():
                    try:
                        # Run optimization
                        results = self.optimizer.optimize(
                            n_trials=n_trials,
                            objective_metric=objective_metric,
                            min_trades=min_trades,
                            max_drawdown_threshold=max_drawdown,
                            n_jobs=n_jobs,
                            timeout=None,  # No timeout for TUI
                            callbacks=[progress_callback]
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
            # self.query_one("#run-button", Button).disabled = False
    
    def _aggregate_multi_dataset_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple dataset optimizations.

        IMPORTANT: All PnL and money-related metrics are NET (after commission).
        Commission is deducted during backtest, so all trade PnLs are already net.
        """
        if not all_results:
            return {}

        # ===== SUMMED COUNTS across all datasets =====
        total_trades = sum(r['final_backtest'].get('total', 0) for r in all_results)
        total_pnl = sum(r['final_backtest'].get('net_pnl', 0) for r in all_results)
        total_winning_trades = sum(r['final_backtest'].get('total_winning_trades', 0) for r in all_results)
        total_losing_trades = sum(r['final_backtest'].get('total_losing_trades', 0) for r in all_results)
        total_long_trades = sum(r['final_backtest'].get('total_long_trades', 0) for r in all_results)
        total_short_trades = sum(r['final_backtest'].get('total_short_trades', 0) for r in all_results)

        # ===== RECALCULATED WIN RATES (total wins / total trades) =====
        # Overall win rate
        win_rate = total_winning_trades / total_trades if total_trades > 0 else 0

        # Long win rate: calculate total winning long trades, then divide by total long trades
        # We reconstruct winning_long from: winrate_long * total_long_trades for each dataset
        total_winning_long_trades = sum(
            r['final_backtest'].get('winrate_long', 0) * r['final_backtest'].get('total_long_trades', 0)
            for r in all_results
        )
        winrate_long = total_winning_long_trades / total_long_trades if total_long_trades > 0 else 0

        # Short win rate: calculate total winning short trades, then divide by total short trades
        # We reconstruct winning_short from: winrate_short * total_short_trades for each dataset
        total_winning_short_trades = sum(
            r['final_backtest'].get('winrate_short', 0) * r['final_backtest'].get('total_short_trades', 0)
            for r in all_results
        )
        winrate_short = total_winning_short_trades / total_short_trades if total_short_trades > 0 else 0

        # ===== WEIGHTED AVERAGES for average_win and average_loss =====
        # These are averages weighted by number of trades in each category
        average_win_weighted_sum = sum(
            r['final_backtest'].get('average_win', 0) * r['final_backtest'].get('total_winning_trades', 0)
            for r in all_results
        )
        average_win = average_win_weighted_sum / total_winning_trades if total_winning_trades > 0 else 0

        average_loss_weighted_sum = sum(
            r['final_backtest'].get('average_loss', 0) * r['final_backtest'].get('total_losing_trades', 0)
            for r in all_results
        )
        average_loss = average_loss_weighted_sum / total_losing_trades if total_losing_trades > 0 else 0

        # ===== PROFIT FACTOR from aggregated gross profit/loss =====
        # Reconstruct gross profit/loss from average_win/loss (which are in % of capital) and counts
        # NOTE: These are NET profits/losses (after commission) because trade PnLs are net
        total_gross_profit_pct = average_win * total_winning_trades if total_winning_trades > 0 else 0
        total_gross_loss_pct = abs(average_loss) * total_losing_trades if total_losing_trades > 0 else 0
        profit_factor = total_gross_profit_pct / total_gross_loss_pct if total_gross_loss_pct > 0 else 0

        # ===== WORST CASE (MAXIMUM) for risk metrics =====
        max_drawdown = max(abs(r['final_backtest'].get('max_drawdown', 0)) for r in all_results)
        consecutive_stops = max(r['final_backtest'].get('consecutive_stops', 0) for r in all_results)

        # ===== FROM BEST DATASET for metrics that can't be properly aggregated =====
        # Sharpe and Sortino require underlying returns data, so take from best performing dataset
        best_dataset_result = max(all_results, key=lambda r: r['final_backtest'].get('net_pnl', 0))
        sharpe_ratio = best_dataset_result['final_backtest'].get('sharpe_ratio', 0)
        sortino_ratio = best_dataset_result['final_backtest'].get('sortino_ratio', 0)

        # Aggregate final backtest metrics
        aggregated_final_backtest = {
            'total': total_trades,
            'net_pnl': total_pnl,
            'win_rate': win_rate,
            'winrate_long': winrate_long,
            'winrate_short': winrate_short,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'average_win': average_win,
            'average_loss': average_loss,
            'consecutive_stops': consecutive_stops,
            'sortino_ratio': sortino_ratio,
            'total_winning_trades': total_winning_trades,
            'total_losing_trades': total_losing_trades,
            'total_long_trades': total_long_trades,
            'total_short_trades': total_short_trades,
        }

        # Aggregate all positive trials from all datasets
        all_positive_trials = []
        for result in all_results:
            dataset_file = result.get('dataset_file', 'unknown')
            for trial in result.get('positive_trials', []):
                trial_copy = trial.copy()
                trial_copy['dataset'] = dataset_file  # Add dataset identifier
                all_positive_trials.append(trial_copy)

        # Use best parameters from the dataset with highest PnL
        best_dataset_result = max(all_results, key=lambda r: r['final_backtest'].get('net_pnl', 0))

        aggregated_results = {
            'strategy_name': all_results[0]['strategy_name'],
            'symbol': 'MULTI',  # Indicate multi-dataset run
            'study_name': f"multi_dataset_{len(all_results)}",
            'objective_metric': all_results[0]['objective_metric'],
            'best_params': best_dataset_result['best_params'],
            'best_value': best_dataset_result['best_value'],
            'n_trials': sum(r['n_trials'] for r in all_results),
            'successful_trials': sum(r['successful_trials'] for r in all_results),
            'pruned_trials': sum(r['pruned_trials'] for r in all_results),
            'optimization_time_seconds': sum(r['optimization_time_seconds'] for r in all_results),
            'optimization_completed_at': all_results[-1]['optimization_completed_at'],
            'parallel_jobs': all_results[0]['parallel_jobs'],
            'final_backtest': aggregated_final_backtest,
            'positive_trials': all_positive_trials,
            'datasets_count': len(all_results),
            'individual_results': all_results  # Keep individual results for reference
        }

        return aggregated_results

    def _update_results(self, results: Dict[str, Any]) -> None:
        """Update results display"""
        self.optimization_results = results
        self.query_one("#results-table", CombinedResultsTable).update_data(results)

        # Update the new positive trials table
        self.query_one("#positive-trials-table", PositiveTrialsTable).update_data(results)

        positive_trials_count = len(results.get('positive_trials', []))
        datasets_count = results.get('datasets_count', 1)

        if datasets_count > 1:
            self.notify(f"Оптимизация завершена на {datasets_count} датасетах! Найдено {positive_trials_count} полож. исходов.", severity="success")
        else:
            self.notify(f"Оптимизация завершена! Найдено {positive_trials_count} полож. исходов.", severity="success")
    
    def _reset_optimization_state(self) -> None:
        """Reset optimization state"""
        self.is_optimizing = False
        # self.query_one("#run-button", Button).disabled = False
        # self.query_one("#progress-label", Label).update("") # Clear progress
        self.optimization_task = None
    
    def action_clear_results(self) -> None:
        """Clear results"""
        self.optimization_results = None
        self.query_one("#results-table", CombinedResultsTable).update_data(None)
        self.query_one("#positive-trials-table", PositiveTrialsTable).update_data(None)
        self.notify("Результаты очищены")
    
    async def action_plot_trades(self) -> None:
        """Plot trades chart"""
        if not self.optimization_results:
            self.notify("Сначала запустите оптимизацию!", severity="warning")
            return
        
        try:
            self.notify("Создание графика сделок...")
            
            # Plot trades in background thread
            def plot_trades_thread():
                try:
                    from ..optimization.visualization import quick_plot_trades
                    
                    # Create output directory
                    output_dir = "optimization_plots"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate plot
                    plot_path = quick_plot_trades(
                        self.optimization_results,
                        output_dir=output_dir
                    )
                    
                    if plot_path:
                        self.call_from_thread(
                            self.notify,
                            f"График сохранен: {plot_path}",
                            severity="success"
                        )
                    else:
                        self.call_from_thread(
                            self.notify,
                            "Не удалось создать график. Проверьте данные.",
                            severity="error"
                        )
                        
                except Exception as e:
                    self.call_from_thread(
                        self.notify,
                        f"Ошибка создания графика: {e}",
                        severity="error"
                    )
            
            # Start thread
            plot_thread = threading.Thread(target=plot_trades_thread)
            plot_thread.daemon = True
            plot_thread.start()
            
        except Exception as e:
            self.notify(f"Ошибка запуска построения графика: {e}", severity="error")
    
    async def action_open_plot(self) -> None:
        """Plot trades chart and open in browser"""
        if not self.optimization_results:
            self.notify("Сначала запустите оптимизацию!", severity="warning")
            return
        
        try:
            self.notify("Создание и открытие графика...")
            
            # Plot trades in background thread
            def plot_trades_thread():
                try:
                    from ..optimization.visualization import plot_and_open
                    
                    # Create output directory
                    output_dir = "optimization_plots"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate plot and open in browser
                    plot_path = plot_and_open(
                        self.optimization_results,
                        output_dir=output_dir
                    )
                    
                    if plot_path:
                        self.call_from_thread(
                            self.notify,
                            f"График создан и открыт: {plot_path}",
                            severity="success"
                        )
                    else:
                        self.call_from_thread(
                            self.notify,
                            "Не удалось создать график. Проверьте данные.",
                            severity="error"
                        )
                        
                except Exception as e:
                    self.call_from_thread(
                        self.notify,
                        f"Ошибка создания графика: {e}",
                        severity="error"
                    )
            
            # Start thread
            plot_thread = threading.Thread(target=plot_trades_thread)
            plot_thread.daemon = True
            plot_thread.start()
            
        except Exception as e:
            self.notify(f"Ошибка запуска построения графика: {e}", severity="error")
    
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