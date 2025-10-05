"""
Optimization Tab for GUI

This module provides a tab for strategy parameter optimization using Optuna
directly in the GUI interface.

Author: HFT System
"""
import os
import sys
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QTabWidget, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

# Добавляем путь для импорта модуля оптимизации
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from src.optimization.fast_optimizer import FastStrategyOptimizer
    from src.optimization.visualization import OptimizationVisualizer
    from src.strategies.strategy_registry import StrategyRegistry
    
    # Алиас для обратной совместимости
    StrategyOptimizer = FastStrategyOptimizer
    OPTUNA_AVAILABLE = True
except ImportError as e:
    print(f"Optimization modules not available: {e}")
    OPTUNA_AVAILABLE = False


class OptimizationWorker(QThread):
    """Worker thread for running optimization in background"""
    
    progress_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, optimizer_config: Dict[str, Any]):
        super().__init__()
        self.optimizer_config = optimizer_config
        self.optimizer = None
        
    def run(self):
        """Run optimization in background thread"""
        try:
            if not OPTUNA_AVAILABLE:
                self.error_signal.emit("Optuna not available. Install with: pip install optuna")
                return
                
            # Always use fast optimizer (now the only optimizer)
            self.progress_signal.emit("Using FAST optimizer with caching and parallel processing...")
            self.optimizer = FastStrategyOptimizer(
                strategy_name=self.optimizer_config['strategy_name'],
                data_path=self.optimizer_config['data_path'],
                symbol=self.optimizer_config['symbol'],
                study_name=self.optimizer_config.get('study_name'),
                direction=self.optimizer_config.get('direction', 'maximize')
            )
            
            # Run optimization with fast optimizer
            results = self.optimizer.optimize(
                n_trials=self.optimizer_config['n_trials'],
                objective_metric=self.optimizer_config['objective_metric'],
                min_trades=self.optimizer_config.get('min_trades', 10),
                max_drawdown_threshold=self.optimizer_config.get('max_drawdown', 50.0),
                timeout=self.optimizer_config.get('timeout'),
                n_jobs=self.optimizer_config.get('n_jobs', -1),
                use_adaptive=self.optimizer_config.get('use_adaptive', True)
            )
            
            # Get parameter importance
            try:
                param_importance = self.optimizer.get_parameter_importance()
                results['param_importance'] = param_importance
            except Exception as e:
                self.progress_signal.emit(f"Could not calculate parameter importance: {e}")
                results['param_importance'] = {}
            
            self.result_signal.emit(results)
            
        except Exception as e:
            self.error_signal.emit(f"Optimization failed: {str(e)}")


class OptimizationTab(QWidget):
    """Tab for strategy parameter optimization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.current_results = None
        self.current_optimizer = None
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Create splitter for controls and results
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = self._create_results_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([300, 700])
        
    def _create_control_panel(self) -> QWidget:
        """Create the control panel"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Parameter Optimization")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Dataset selection
        dataset_group = QGroupBox("Data Source")
        dataset_layout = QFormLayout(dataset_group)
        
        self.dataset_combo = QComboBox()
        self._load_datasets()
        dataset_layout.addRow("Dataset:", self.dataset_combo)
        
        layout.addWidget(dataset_group)
        
        # Strategy selection
        strategy_group = QGroupBox("Strategy")
        strategy_layout = QFormLayout(strategy_group)
        
        self.strategy_combo = QComboBox()
        self._load_strategies()
        strategy_layout.addRow("Strategy:", self.strategy_combo)
        
        layout.addWidget(strategy_group)
        
        # Optimization parameters
        opt_group = QGroupBox("Optimization Parameters")
        opt_layout = QFormLayout(opt_group)
        
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(10, 1000)
        self.trials_spin.setValue(100)
        opt_layout.addRow("Trials:", self.trials_spin)
        
        self.objective_combo = QComboBox()
        self.objective_combo.addItems(['sharpe_ratio', 'net_pnl', 'profit_factor', 'win_rate', 'net_pnl_percentage'])
        self.objective_combo.setCurrentText('sharpe_ratio')
        opt_layout.addRow("Objective:", self.objective_combo)
        
        self.min_trades_spin = QSpinBox()
        self.min_trades_spin.setRange(1, 100)
        self.min_trades_spin.setValue(10)
        opt_layout.addRow("Min Trades:", self.min_trades_spin)
        
        self.max_drawdown_spin = QDoubleSpinBox()
        self.max_drawdown_spin.setRange(1.0, 100.0)
        self.max_drawdown_spin.setValue(50.0)
        self.max_drawdown_spin.setSuffix("%")
        opt_layout.addRow("Max Drawdown:", self.max_drawdown_spin)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(60, 3600)
        self.timeout_spin.setValue(600)
        self.timeout_spin.setSuffix(" sec")
        self.timeout_spin.setSpecialValueText("No limit")
        opt_layout.addRow("Timeout:", self.timeout_spin)
        
        layout.addWidget(opt_group)
        
        # Performance info (read-only)
        perf_group = QGroupBox("Performance Mode")
        perf_layout = QFormLayout(perf_group)
        
        perf_info = QLabel("TURBO MODE ENABLED (250x Speedup)")
        perf_info.setStyleSheet("color: green; font-weight: bold;")
        perf_layout.addRow(perf_info)
        
        perf_details = QLabel("• Numba JIT compilation\n• Parallel processing\n• Adaptive evaluation\n• Data caching")
        perf_layout.addRow(perf_details)
        
        layout.addWidget(perf_group)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Optimization")
        self.start_btn.clicked.connect(self._start_optimization)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_optimization)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        
        layout.addStretch()
        return panel
        
    def _create_results_panel(self) -> QWidget:
        """Create the results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different results views
        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)
        
        # Summary tab
        self.summary_widget = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_widget)
        self.results_tabs.addTab(self.summary_widget, "Summary")
        
        # Best parameters tab
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        self.results_tabs.addTab(self.params_widget, "Best Parameters")
        
        # Parameter importance tab
        self.importance_widget = QWidget()
        self.importance_layout = QVBoxLayout(self.importance_widget)
        self.results_tabs.addTab(self.importance_widget, "Parameter Importance")
        
        # Log tab
        self.log_widget = QWidget()
        self.log_layout = QVBoxLayout(self.log_widget)
        self.results_tabs.addTab(self.log_widget, "Log")
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_layout.addWidget(self.log_text)
        
        return panel
        
    def _load_datasets(self):
        """Load available datasets"""
        self.dataset_combo.clear()
        
        klines_dir = "upload/klines"
        if os.path.exists(klines_dir):
            csv_files = [f for f in os.listdir(klines_dir) if f.endswith('.csv')]
            self.dataset_combo.addItems(csv_files)
            
            if csv_files:
                # Extract symbol from filename
                filename = csv_files[0]
                symbol = filename.split('-')[0] if '-' in filename else filename.split('.')[0]
                # This will be used when creating optimizer
                self.current_symbol = symbol
                
    def _load_strategies(self):
        """Load available strategies"""
        self.strategy_combo.clear()
        
        if OPTUNA_AVAILABLE:
            strategies = StrategyRegistry.list_strategies()
            self.strategy_combo.addItems(strategies)
            # Connect strategy change signal
            self.strategy_combo.currentTextChanged.connect(self._on_strategy_change)
            
    def _on_strategy_change(self):
        """Handle strategy change"""
        strategy = self.strategy_combo.currentText()
        
        # Log info about turbo strategy if hierarchical_mean_reversion is selected
        if strategy == 'hierarchical_mean_reversion':
            self._log("Using TURBO Hierarchical Mean Reversion with Numba JIT (50x+ speed)")
            
    def _start_optimization(self):
        """Start the optimization process"""
        if not OPTUNA_AVAILABLE:
            QMessageBox.critical(self, "Error", "Optuna not available. Install with: pip install optuna")
            return
            
        dataset = self.dataset_combo.currentText()
        if not dataset:
            QMessageBox.warning(self, "Warning", "Please select a dataset")
            return
            
        strategy = self.strategy_combo.currentText()
        if not strategy:
            QMessageBox.warning(self, "Warning", "Please select a strategy")
            return
            
        # Get dataset path
        dataset_path = os.path.join("upload/klines", dataset)
        if not os.path.exists(dataset_path):
            QMessageBox.critical(self, "Error", f"Dataset file not found: {dataset_path}")
            return
            
        # Extract symbol from filename
        symbol = dataset.split('-')[0] if '-' in dataset else dataset.split('.')[0]
        
        # Get timeout value (0 means no timeout)
        timeout = self.timeout_spin.value() if self.timeout_spin.value() != self.timeout_spin.minimum() else None
        
        # Create optimizer configuration with TURBO mode enabled by default
        optimizer_config = {
            'strategy_name': strategy,
            'data_path': dataset_path,
            'symbol': symbol,
            'n_trials': self.trials_spin.value(),
            'objective_metric': self.objective_combo.currentText(),
            'min_trades': self.min_trades_spin.value(),
            'max_drawdown': self.max_drawdown_spin.value(),
            'timeout': timeout,
            'direction': 'maximize',
            'use_fast': True,  # Always use fast optimization
            'n_jobs': -1,  # Use all cores
            'use_adaptive': True  # Use adaptive evaluation
        }
        
        # Clear previous results
        self._clear_results()
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.export_btn.setEnabled(False)
        
        # Log start
        self._log(f"Starting TURBO optimization for {strategy} on {symbol}")
        self._log(f"Dataset: {dataset}")
        self._log(f"Trials: {optimizer_config['n_trials']}")
        self._log(f"Objective: {optimizer_config['objective_metric']}")
        self._log(f"TURBO MODE: Numba JIT + Parallel Processing + Adaptive Evaluation")
        self._log(f"Parallel jobs: {optimizer_config['n_jobs']} (all cores)")
        
        # Create and start worker thread
        self.worker = OptimizationWorker(optimizer_config)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.result_signal.connect(self._on_complete)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()
        
    def _stop_optimization(self):
        """Stop the optimization process"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            
        self._on_worker_finished()
        self._log("Optimization stopped by user")
        
    def _on_progress(self, message: str):
        """Handle progress updates"""
        self._log(message)
        
    def _on_complete(self, results: Dict[str, Any]):
        """Handle optimization completion"""
        self.current_results = results
        
        # Display results
        self._display_results(results)
        
        self._log(f"Optimization completed successfully!")
        self._log(f"Best {results.get('objective_metric', 'value')}: {results.get('best_value', 'N/A'):.4f}")
        
        self.export_btn.setEnabled(True)
        self._on_worker_finished()
        
    def _on_error(self, error_msg: str):
        """Handle optimization errors"""
        self._log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Optimization Error", error_msg)
        self._on_worker_finished()
        
    def _on_worker_finished(self):
        """Handle worker thread completion"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
            
    def _display_results(self, results: Dict[str, Any]):
        """Display optimization results"""
        # Clear previous results
        self._clear_results()
        
        # Display summary
        self._display_summary(results)
        
        # Display best parameters
        self._display_best_params(results)
        
        # Display parameter importance
        self._display_param_importance(results)
        
    def _display_summary(self, results: Dict[str, Any]):
        """Display optimization summary"""
        # Clear summary layout
        for i in reversed(range(self.summary_layout.count())):
            child = self.summary_layout.itemAt(i).widget()
            if child is not None:
                child.setParent(None)
                
        # Create summary form
        form = QFormLayout()
        
        form.addRow("Strategy:", QLabel(results.get('strategy_name', 'N/A')))
        form.addRow("Symbol:", QLabel(results.get('symbol', 'N/A')))
        form.addRow("Objective:", QLabel(results.get('objective_metric', 'N/A')))
        form.addRow("Best Value:", QLabel(f"{results.get('best_value', 'N/A'):.4f}"))
        form.addRow("Total Trials:", QLabel(str(results.get('n_trials', 0))))
        form.addRow("Successful Trials:", QLabel(str(results.get('successful_trials', 0))))
        form.addRow("Optimization Time:", QLabel(f"{results.get('optimization_time_seconds', 0):.2f} seconds"))
        
        self.summary_layout.addLayout(form)
        
        # Display final backtest results if available
        final_backtest = results.get('final_backtest')
        if final_backtest:
            self.summary_layout.addWidget(QLabel("<h3>Final Backtest Results</h3>"))
            
            backtest_form = QFormLayout()
            backtest_form.addRow("Total Trades:", QLabel(str(final_backtest.get('total', 0))))
            backtest_form.addRow("Win Rate:", QLabel(f"{final_backtest.get('win_rate', 0):.1%}"))
            backtest_form.addRow("Net P&L:", QLabel(f"${final_backtest.get('net_pnl', 0):,.2f}"))
            backtest_form.addRow("Return:", QLabel(f"{final_backtest.get('net_pnl_percentage', 0):.2f}%"))
            backtest_form.addRow("Sharpe Ratio:", QLabel(f"{final_backtest.get('sharpe_ratio', 0):.2f}"))
            backtest_form.addRow("Profit Factor:", QLabel(f"{final_backtest.get('profit_factor', 0):.2f}"))
            backtest_form.addRow("Max Drawdown:", QLabel(f"{final_backtest.get('max_drawdown', 0):.2f}%"))
            
            self.summary_layout.addLayout(backtest_form)
            
        self.summary_layout.addStretch()
        
    def _display_best_params(self, results: Dict[str, Any]):
        """Display best parameters"""
        # Clear params layout
        for i in reversed(range(self.params_layout.count())):
            child = self.params_layout.itemAt(i).widget()
            if child is not None:
                child.setParent(None)
                
        # Create parameters table
        table = QTableWidget()
        best_params = results.get('best_params', {})
        
        table.setColumnCount(2)
        table.setRowCount(len(best_params))
        table.setHorizontalHeaderLabels(["Parameter", "Value"])
        
        for row, (param, value) in enumerate(best_params.items()):
            table.setItem(row, 0, QTableWidgetItem(param))
            table.setItem(row, 1, QTableWidgetItem(str(value)))
            
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setAlternatingRowColors(True)
        
        self.params_layout.addWidget(table)
        
    def _display_param_importance(self, results: Dict[str, Any]):
        """Display parameter importance"""
        # Clear importance layout
        for i in reversed(range(self.importance_layout.count())):
            child = self.importance_layout.itemAt(i).widget()
            if child is not None:
                child.setParent(None)
                
        # Create importance table
        table = QTableWidget()
        param_importance = results.get('param_importance', {})
        
        if param_importance:
            # Sort by importance
            sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            
            table.setColumnCount(2)
            table.setRowCount(len(sorted_params))
            table.setHorizontalHeaderLabels(["Parameter", "Importance"])
            
            for row, (param, importance) in enumerate(sorted_params):
                table.setItem(row, 0, QTableWidgetItem(param))
                table.setItem(row, 1, QTableWidgetItem(f"{importance:.4f}"))
                
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setAlternatingRowColors(True)
            
            self.importance_layout.addWidget(table)
        else:
            self.importance_layout.addWidget(QLabel("Parameter importance not available"))
            
    def _clear_results(self):
        """Clear previous results"""
        # Clear all result layouts
        for layout in [self.summary_layout, self.params_layout, self.importance_layout]:
            for i in reversed(range(layout.count())):
                child = layout.itemAt(i).widget()
                if child is not None:
                    child.setParent(None)
                    
        # Clear log
        self.log_text.clear()
        
    def _export_results(self):
        """Export optimization results"""
        if not self.current_results:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", 
            f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_results, f, indent=2)
                    
                QMessageBox.information(self, "Export Successful", f"Results exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
                
    def _log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")