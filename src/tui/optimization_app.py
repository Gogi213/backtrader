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
        Binding("a", "estimate_adaptation", "Адаптация"),
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
                yield Label("Режим:")
                yield Select(
                    options=[
                        ("Оптимизация", "optimize"),
                        ("Cross-Asset Validation", "validate"),
                    ],
                    id="mode-select"
                )

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
        self.query_one("#mode-select", Select).value = "optimize"
    
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
        """Run optimization or validation based on mode"""
        if self.is_optimizing:
            self.notify("Процесс уже запущен!", severity="warning")
            return

        try:
            # Get configuration from UI
            mode = self.query_one("#mode-select", Select).value
            strategy_name = self.query_one("#strategy-select", Select).value
            dataset_selection = self.query_one("#dataset-select", Select).value

            # Route to appropriate handler
            if mode == "validate":
                await self._run_validation(strategy_name, dataset_selection)
                return
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

                        # Save results to disk
                        self.call_from_thread(self._save_optimization_results, aggregated_results)

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

    async def _run_validation(self, strategy_name: str, dataset_selection: str, use_clusters: bool = True) -> None:
        """Run Cross-Asset Validation"""
        from ..optimization.cross_asset_validator import CrossAssetValidator
        from ..optimization.cluster_analyzer import DatasetClusterAnalyzer

        # Dataset folder is required for validation
        data_path = os.path.join("upload/klines", dataset_selection)

        if not os.path.isdir(data_path):
            self.notify("Validation требует выбора ПАПКИ с датасетами!", severity="error")
            return

        self.is_optimizing = True
        self.notify(f"Запуск Cross-Asset Validation для {strategy_name}...")

        def run_validation_thread():
            try:
                # Cluster analysis (if enabled)
                cluster_mapping = {}
                if use_clusters:
                    self.call_from_thread(
                        self.notify,
                        "[1/6] Анализирую кластеры датасетов...",
                        severity="information"
                    )

                    analyzer = DatasetClusterAnalyzer(datasets_dir=data_path)
                    analyzer.analyze_datasets()
                    analyzer.create_clusters(n_clusters=3)
                    analyzer.save_clusters()
                    analyzer.print_clusters()
                    cluster_mapping = analyzer.symbol_to_cluster

                    self.call_from_thread(
                        self.notify,
                        f"[1/6] Создано {len(analyzer.clusters)} кластеров",
                        severity="information"
                    )

                # Create validator with cluster mapping
                validator = CrossAssetValidator(datasets_dir=data_path, cluster_mapping=cluster_mapping)

                if not validator.datasets:
                    self.call_from_thread(
                        self.notify,
                        f"Нет датасетов в {data_path}",
                        severity="error"
                    )
                    return

                # Load positive trials from saved results
                step_offset = 1 if use_clusters else 0

                self.call_from_thread(
                    self.notify,
                    f"[{1+step_offset}/5] Загружено {len(validator.datasets)} датасетов",
                    severity="information"
                )

                self.call_from_thread(
                    self.notify,
                    f"[{2+step_offset}/5] Загружаю positive trials из БД...",
                    severity="information"
                )

                # Load from SQLite instead of JSON files
                from ..optimization.results_db import OptimizationResultsDB
                db = OptimizationResultsDB()

                # Get latest run
                run_id = db.get_latest_run(strategy_name=strategy_name)
                if not run_id:
                    self.call_from_thread(
                        self.notify,
                        "Нет результатов оптимизации в БД! Запустите сначала оптимизацию.",
                        severity="error"
                    )
                    return

                # Load positive trials from DB
                trials_data = db.load_positive_trials(
                    run_id=run_id,
                    min_pnl=0.0,
                    min_sharpe=0.0,
                    min_trades=10
                )

                # Convert to PositiveTrial objects
                from ..optimization.cross_asset_validator import PositiveTrial
                positive_trials = []
                for trial in trials_data:
                    dataset = trial.get('dataset', 'unknown')
                    symbol = dataset.split('-')[0] if '-' in dataset else dataset.split('.')[0]

                    positive_trials.append(PositiveTrial(
                        source_symbol=symbol,
                        trial_number=trial.get('trial', 0),
                        params=trial.get('params', {}),
                        source_sharpe=trial.get('sharpe', 0),
                        source_pnl=trial.get('pnl', 0),
                        source_trades=trial.get('trades', 0)
                    ))

                if not positive_trials:
                    # Fallback: load top 20 trials by PnL even if negative
                    self.call_from_thread(
                        self.notify,
                        "Нет positive trials. Загружаю топ-20 по PnL...",
                        severity="warning"
                    )

                    trials_data = db.load_positive_trials(
                        run_id=run_id,
                        min_pnl=-999999.0,  # Accept any PnL
                        min_sharpe=-999.0,  # Accept any Sharpe
                        min_trades=1,
                        limit=20  # Top 20
                    )

                    # Convert to PositiveTrial objects
                    positive_trials = []
                    for trial in trials_data:
                        dataset = trial.get('dataset', 'unknown')
                        symbol = dataset.split('-')[0] if '-' in dataset else dataset.split('.')[0]

                        positive_trials.append(PositiveTrial(
                            source_symbol=symbol,
                            trial_number=trial.get('trial', 0),
                            params=trial.get('params', {}),
                            source_sharpe=trial.get('sharpe', 0),
                            source_pnl=trial.get('pnl', 0),
                            source_trades=trial.get('trades', 0)
                        ))

                    if not positive_trials:
                        self.call_from_thread(
                            self.notify,
                            "Нет trials вообще! Запустите сначала оптимизацию.",
                            severity="error"
                        )
                        return

                    self.call_from_thread(
                        self.notify,
                        f"Загружено топ-{len(positive_trials)} trials (лучшие по PnL)",
                        severity="information"
                    )

                mode_str = "cluster-based" if use_clusters else "all datasets"

                self.call_from_thread(
                    self.notify,
                    f"[{3+step_offset}/5] Найдено {len(positive_trials)} trials. Запускаю validation ({mode_str})...",
                    severity="information"
                )

                # Validate all trials with progress callback
                def progress_callback(trial_idx, total_trials, trial_id):
                    if trial_idx % 5 == 0 or trial_idx == total_trials:
                        self.call_from_thread(
                            self.notify,
                            f"[{3+step_offset}/5] Валидация: {trial_idx}/{total_trials} trials ({trial_idx*100//total_trials}%)",
                            severity="information"
                        )

                all_results_df = validator.validate_all_trials(
                    strategy_name=strategy_name,
                    trials=positive_trials,
                    progress_callback=progress_callback,
                    cluster_mode=use_clusters
                )

                if all_results_df.empty:
                    self.call_from_thread(
                        self.notify,
                        "Validation failed!",
                        severity="error"
                    )
                    return

                self.call_from_thread(
                    self.notify,
                    f"[{4+step_offset}/5] Анализирую robust trials...",
                    severity="information"
                )

                # Find robust trials
                robust_df = validator.find_robust_trials(
                    min_positive_ratio=0.5,
                    min_avg_pnl=10.0,
                    group_by_cluster=use_clusters
                )

                self.call_from_thread(
                    self.notify,
                    f"[{5+step_offset}/5] Найдено {len(robust_df)} robust trials. Экспорт результатов...",
                    severity="information"
                )

                # Export results
                validator.export_results(all_results_df, robust_df)

                # Prepare results for display
                validation_results = {
                    'strategy_name': strategy_name,
                    'symbol': 'VALIDATION',
                    'positive_trials': [],
                    'datasets_count': len(validator.datasets),
                    'final_backtest': {
                        'total': len(positive_trials),
                        'net_pnl': 0,
                        'win_rate': 0,
                        'winrate_long': 0,
                        'winrate_short': 0,
                        'sharpe_ratio': 0,
                        'profit_factor': 0,
                        'max_drawdown': 0,
                        'average_win': 0,
                        'average_loss': 0,
                        'consecutive_stops': 0,
                    },
                    'best_params': {}
                }

                # Convert robust trials to display format
                if not robust_df.empty:
                    for _, row in robust_df.iterrows():
                        validation_results['positive_trials'].append({
                            'trial': row['trial_id'],
                            'pnl': row['avg_pnl'],
                            'sharpe': row['avg_sharpe'],
                            'trades': 0,
                            'winrate': 0,
                            'winrate_long': 0,
                            'winrate_short': 0,
                            'pf': 0,
                            'max_dd': 0,
                            'avg_win': 0,
                            'avg_loss': 0,
                            'consecutive_stops': 0,
                            'dataset': f"{row['positive_count']}/{row['total_datasets']} datasets ({row['positive_ratio']:.0%})",
                            'params': row.get('params', {})
                        })

                # Update UI
                self.call_from_thread(self._update_results, validation_results)

                self.call_from_thread(
                    self.notify,
                    f"Validation завершена! Найдено {len(robust_df)} robust trials.",
                    severity="success"
                )

            except Exception as e:
                import traceback
                error_msg = f"Ошибка validation: {e}\n{traceback.format_exc()}"
                print(error_msg)
                self.call_from_thread(
                    self.notify,
                    f"Ошибка validation: {e}",
                    severity="error"
                )
            finally:
                self.call_from_thread(self._reset_optimization_state)

        # Start thread
        validation_thread = threading.Thread(target=run_validation_thread)
        validation_thread.daemon = True
        validation_thread.start()

    async def _run_cross_asset_wfo(
        self, strategy_name, dataset_selection, n_trials, objective_metric,
        n_jobs, min_trades, max_drawdown
    ) -> None:
        """Run Cross-Asset WFO optimization"""
        from ..optimization.cross_asset_wfo import CrossAssetWFOAnalyzer, CrossAssetConfig

        # Get dataset directory
        data_path = os.path.join("upload/klines", dataset_selection)

        # Must be a folder
        if not os.path.isdir(data_path):
            self.notify("Cross-Asset WFO требует выбора ПАПКИ с датасетами!", severity="error")
            return

        self.is_optimizing = True
        self.notify(f"Запуск Cross-Asset WFO для {strategy_name}...")

        def run_cross_asset_wfo_thread():
            import logging
            from datetime import datetime

            # Настройка логирования в файл
            log_dir = "docs/logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"tui_cross_asset_wfo_{timestamp}.log")

            logger = logging.getLogger('tui_cross_asset_wfo')
            logger.setLevel(logging.DEBUG)
            logger.handlers.clear()

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"="*80)
            logger.info(f"TUI Cross-Asset WFO Thread started")
            logger.info(f"Log file: {log_file}")
            logger.info(f"="*80)

            try:
                # Create config
                logger.info(f"Creating config: strategy={strategy_name}, n_trials={n_trials}")
                config = CrossAssetConfig(
                    strategy_name=strategy_name,
                    train_ratio=0.6,
                    test_ratio=0.3,
                    holdout_ratio=0.1,
                    n_trials=n_trials,
                    objective_metric=objective_metric,
                    aggregation_method='median',
                    min_trades=min_trades,
                    max_drawdown_threshold=max_drawdown,
                    n_jobs=n_jobs
                )

                # Run Cross-Asset WFO
                logger.info("Creating analyzer...")
                analyzer = CrossAssetWFOAnalyzer(config, data_path, enable_debug=False)

                logger.info("Running cross_asset_wfo...")
                ca_results = analyzer.run_cross_asset_wfo()

                logger.info("Saving results...")
                # Save results
                analyzer.save_results(ca_results)

                # Convert to display format - with safety checks
                logger.info("Converting results to display format...")
                total_trades = 0
                try:
                    # Debug: check type of test_results
                    logger.debug(f"test_results type: {type(ca_results.test_results)}")
                    if ca_results.test_results:
                        logger.debug(f"first result type: {type(ca_results.test_results[0])}")
                        logger.debug(f"first result: {ca_results.test_results[0]}")

                    total_trades = sum(r.trades for r in ca_results.test_results)
                    logger.info(f"Total trades calculated: {total_trades}")
                except Exception as e:
                    logger.error(f"ERROR calculating total_trades: {e}")
                    logger.error(f"test_results: {ca_results.test_results}")
                    import traceback
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    total_trades = 0

                display_results = {
                    'strategy_name': strategy_name,
                    'symbol': 'CROSS-ASSET',
                    'objective_metric': objective_metric,
                    'n_trials': n_trials * len(ca_results.train_assets),
                    'final_backtest': {
                        'total': total_trades,
                        'net_pnl': ca_results.test_pnl,
                        'win_rate': 0,
                        'winrate_long': 0,
                        'winrate_short': 0,
                        'sharpe_ratio': ca_results.test_sharpe,
                        'profit_factor': 0,
                        'max_drawdown': 0,
                        'average_win': 0,
                        'average_loss': 0,
                        'consecutive_stops': 0,
                    },
                    'best_params': ca_results.best_params,
                    'positive_trials': [],
                }

                logger.info("Updating UI with results...")
                self.call_from_thread(self._update_cross_asset_results, display_results, ca_results)
                logger.info("TUI Cross-Asset WFO completed successfully")

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                logger.error(f"FULL ERROR: {e}")
                logger.error(f"Full traceback:\n{error_details}")
                print(f"FULL ERROR:\n{error_details}")
                self.call_from_thread(
                    self.notify,
                    f"Ошибка Cross-Asset WFO: {e}\nСм. {log_file}",
                    severity="error"
                )
            finally:
                logger.info("Resetting optimization state...")
                self.call_from_thread(self._reset_optimization_state)

        # Start thread
        ca_wfo_thread = threading.Thread(target=run_cross_asset_wfo_thread)
        ca_wfo_thread.daemon = True
        ca_wfo_thread.start()

    def _update_cross_asset_results(self, display_results, ca_results) -> None:
        """Update UI with Cross-Asset WFO results"""
        self.optimization_results = display_results
        self.query_one("#results-table", CombinedResultsTable).update_data(display_results)

        # Показываем детальные результаты в positive trials table
        positive_table = self.query_one("#positive-trials-table", PositiveTrialsTable)
        positive_table.clear()

        # Добавляем заголовки
        if len(positive_table.columns) == 0:
            positive_table.add_columns(
                "Asset", "Sharpe", "PnL", "Trades", "Win Rate", "Max DD", "Type"
            )

        # Train results
        for r in ca_results.train_results:
            positive_table.add_row(
                r.symbol,
                f"{r.sharpe:.3f}",
                f"{r.pnl:.2f}",
                str(r.trades),
                f"{r.win_rate:.2%}",
                f"{r.max_dd:.2f}%",
                "TRAIN"
            )

        # Test results
        for r in ca_results.test_results:
            positive_table.add_row(
                r.symbol,
                f"{r.sharpe:.3f}",
                f"{r.pnl:.2f}",
                str(r.trades),
                f"{r.win_rate:.2%}",
                f"{r.max_dd:.2f}%",
                "TEST"
            )

        # Holdout results
        if ca_results.holdout_results:
            for r in ca_results.holdout_results:
                positive_table.add_row(
                    r.symbol,
                    f"{r.sharpe:.3f}",
                    f"{r.pnl:.2f}",
                    str(r.trades),
                    f"{r.win_rate:.2%}",
                    f"{r.max_dd:.2f}%",
                    "HOLDOUT"
                )

        # Краткое уведомление
        efficiency_icon = "✅" if ca_results.cross_asset_efficiency > 0.6 else "⚠️" if ca_results.cross_asset_efficiency > 0.4 else "❌"
        msg = (
            f"Cross-Asset WFO завершен!\n"
            f"{efficiency_icon} Efficiency: {ca_results.cross_asset_efficiency:.2%} | "
            f"Consistency: {ca_results.consistency_score:.2%} ({int(ca_results.consistency_score * len(ca_results.test_assets))}/{len(ca_results.test_assets)})"
        )

        severity = "success" if ca_results.cross_asset_efficiency > 0.6 else "warning" if ca_results.cross_asset_efficiency > 0.4 else "info"
        self.notify(msg, severity=severity, timeout=10)
    
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

    def _save_optimization_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results to SQLite database"""
        try:
            from ..optimization.results_db import OptimizationResultsDB

            db = OptimizationResultsDB()
            run_id = db.save_optimization_run(results)

            strategy_name = results.get('strategy_name', 'unknown')
            positive_count = len(results.get('positive_trials', []))

            print(f"\n[SAVED] Optimization results saved to database (run_id: {run_id})")
            self.notify(
                f"Результаты сохранены в БД: {strategy_name} ({positive_count} positive trials)",
                severity="success",
                timeout=5
            )

        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to save results: {e}")
            print(traceback.format_exc())
            self.notify(f"Ошибка сохранения: {e}", severity="error")

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
    
    async def action_estimate_adaptation(self) -> None:
        """Оценка Cross-Asset WFO перед запуском"""
        mode = self.query_one("#mode-select", Select).value

        if mode != "wfo_cross_asset":
            self.notify("Адаптация доступна только для Cross-Asset WFO режима", severity="info", timeout=5)
            return

        try:
            # Get parameters
            strategy_name = self.query_one("#strategy-select", Select).value
            dataset_selection = self.query_one("#dataset-select", Select).value
            n_trials = int(self.query_one("#trials-input", Input).value)
            objective_metric = self.query_one("#metric-select", Select).value
            n_jobs = int(self.query_one("#jobs-input", Input).value)
            min_trades = int(self.query_one("#min-trades-input", Input).value)
            max_drawdown = float(self.query_one("#max-drawdown-input", Input).value)

            data_path = os.path.join("upload/klines", dataset_selection)

            if not os.path.isdir(data_path):
                self.notify("Выберите ПАПКУ с датасетами для Cross-Asset WFO", severity="error")
                return

            # Get estimation
            from ..optimization.cross_asset_wfo import estimate_cross_asset_wfo, CrossAssetConfig

            config = CrossAssetConfig(
                strategy_name=strategy_name,
                train_ratio=0.6,
                test_ratio=0.3,
                holdout_ratio=0.1,
                n_trials=n_trials,
                objective_metric=objective_metric,
                aggregation_method='median',
                min_trades=min_trades,
                max_drawdown_threshold=max_drawdown,
                n_jobs=n_jobs
            )

            estimation = estimate_cross_asset_wfo(data_path, config)
            rec = estimation.trials_recommendation

            # Check if recommendation was generated
            if rec is None:
                self.notify("Не удалось сгенерировать рекомендацию trials. Проверьте консоль для деталей.", severity="error")
                return

            # Format train/test/holdout lists (show first 5 + count)
            def format_list(items, max_show=5):
                if len(items) <= max_show:
                    return ", ".join(items)
                else:
                    shown = ", ".join(items[:max_show])
                    remaining = len(items) - max_show
                    return f"{shown} ... (+{remaining})"

            train_str = format_list(estimation.train_datasets)
            test_str = format_list(estimation.test_datasets)
            holdout_str = format_list(estimation.holdout_datasets) if estimation.holdout_datasets else "нет"

            # Sufficiency indicator
            sufficiency_icon = {
                "sufficient": "✅",
                "borderline": "⚠️",
                "insufficient": "❌"
            }.get(rec.dataset_sufficiency, "?")

            # Current trials vs recommended
            current_trials = estimation.n_trials
            if current_trials < rec.min_trials:
                trials_status = f"❌ Слишком мало (минимум {rec.min_trials})"
            elif current_trials < rec.recommended_trials:
                trials_status = f"⚠️ Ниже рекомендуемого ({rec.recommended_trials})"
            elif current_trials <= rec.optimal_trials:
                trials_status = f"✅ Хорошо"
            else:
                trials_status = f"✅ Отлично (больше оптимального)"

            # Build warnings section
            warnings_text = ""
            if rec.warnings:
                warnings_text = "\n⚠️  ПРЕДУПРЕЖДЕНИЯ:\n"
                for warning in rec.warnings:
                    warnings_text += f"   {warning}\n"

            # Build message
            msg = f"""
╔══════════════════════════════════════════════════════════════════════╗
║              АДАПТАЦИЯ CROSS-ASSET WFO                               ║
╚══════════════════════════════════════════════════════════════════════╝

📊 АНАЛИЗ СТРАТЕГИИ '{strategy_name}':
   Параметров: {rec.n_params}
   Размер пространства: ~{rec.space_size_estimate:,} комбинаций

📁 ДАТАСЕТЫ:
   Train:   {len(estimation.train_datasets)} монет
            {train_str}

   Test:    {len(estimation.test_datasets)} монет {sufficiency_icon}
            {test_str}
            (минимум {rec.min_test_datasets_needed} для статистики)

   Holdout: {len(estimation.holdout_datasets)} монет
            {holdout_str}

🎯 РЕКОМЕНДАЦИИ TRIALS:

   Минимум (быстрый тест):      {rec.min_trials} trials
   ├─ Время: {rec.estimated_times['min'][0]:.0f}-{rec.estimated_times['min'][1]:.0f} мин
   ├─ Уверенность: низкая
   └─ Для быстрой проверки идеи

   Рекомендуемое (норма):       {rec.recommended_trials} trials  ⭐
   ├─ Время: {rec.estimated_times['recommended'][0]:.0f}-{rec.estimated_times['recommended'][1]:.0f} мин
   ├─ Уверенность: средняя
   └─ Для регулярной оптимизации

   Оптимальное (макс качество): {rec.optimal_trials} trials
   ├─ Время: {rec.estimated_times['optimal'][0]:.0f}-{rec.estimated_times['optimal'][1]:.0f} мин
   ├─ Уверенность: высокая
   └─ Для финальной оптимизации перед live

📈 ТЕКУЩИЕ НАСТРОЙКИ:
   Выбрано trials: {current_trials}
   Статус: {trials_status}
{warnings_text}
💾 ПАМЯТЬ: ~{estimation.estimated_memory_gb:.1f} GB (train датасеты)

{rec.explanation}
"""

            self.notify(msg, severity="information", timeout=40)

        except ValueError as e:
            self.notify(f"Ошибка в параметрах: {e}", severity="error")
        except Exception as e:
            self.notify(f"Ошибка оценки: {e}", severity="error")

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