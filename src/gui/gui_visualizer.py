"""
Professional GUI Application for Jesse Bollinger Bands Strategy Backtesting
Refactored with HFT principles: high performance, no duplication, YAGNI compliance
"""
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QLineEdit, QHeaderView, QSplitter, QProgressBar, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCursor
from .tabs.tab_performance import PerformanceTab
from .tabs.tab_trade_details import TradeDetailsTab
from .tabs.tab_chart_signals import ChartSignalsTab
from .utilities.gui_utilities import Logger, export_results
from .data.dataset_manager import DatasetManager
from .config.config_models import StrategyConfig, BacktestWorker






class ProfessionalBacktester(QMainWindow):
    """Professional backtesting GUI with minimal complexity"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Bollinger Bands Backtester")
        self.setGeometry(50, 50, 1800, 1200)

        # Core data
        self.config = StrategyConfig()
        self.results_data = None
        self.worker = None

        # Tab components
        self.performance_tab = PerformanceTab()
        self.trade_details_tab = TradeDetailsTab()
        self.chart_signals_tab = ChartSignalsTab()

        # Utilities (will be initialized after UI creation)
        self.logger = None
        self.dataset_manager = None

        self._init_ui()
        # Dataset loading moved to after control panel creation

    def _init_ui(self):
        """Initialize UI with minimal complexity"""
        central = QWidget()
        self.setCentralWidget(central)

        # Main splitter: left controls, right tabs
        splitter = QSplitter(Qt.Orientation.Horizontal)
        central_layout = QHBoxLayout(central)
        central_layout.addWidget(splitter)

        # Left panel: controls
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)

        # Right panel: tabs
        self.tabs = QTabWidget()
        self._create_tabs()
        splitter.addWidget(self.tabs)

        # Set splitter sizes (15% left, 85% right)
        splitter.setSizes([270, 1530])

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready to backtest")

    def _create_control_panel(self):
        """Create streamlined control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header
        title = QLabel("BB Strategy Backtester")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Dataset selection
        dataset_group = QGroupBox("Data Source")
        dataset_layout = QFormLayout(dataset_group)

        self.dataset_combo = QComboBox()
        dataset_layout.addRow("Dataset:", self.dataset_combo)

        self.symbol_label = QLabel("Auto-detected")
        dataset_layout.addRow("Symbol:", self.symbol_label)

        layout.addWidget(dataset_group)

        # Initialize dataset manager after UI elements are created
        self.dataset_manager = DatasetManager(self.dataset_combo, self.symbol_label, self.logger)
        self.dataset_combo.currentTextChanged.connect(self.dataset_manager.on_dataset_changed)

        # Load datasets
        self.dataset_manager.load_datasets()

        # Strategy parameters
        strategy_group = QGroupBox("Strategy Parameters")
        strategy_layout = QFormLayout(strategy_group)

        # BB parameters
        self.bb_period_spin = QSpinBox()
        self.bb_period_spin.setRange(100, 300)
        self.bb_period_spin.setValue(self.config.bb_period)
        strategy_layout.addRow("BB Period:", self.bb_period_spin)

        self.bb_std_spin = QDoubleSpinBox()
        self.bb_std_spin.setRange(1.5, 4.0)
        self.bb_std_spin.setSingleStep(0.1)
        self.bb_std_spin.setValue(self.config.bb_std)
        strategy_layout.addRow("BB Std Dev:", self.bb_std_spin)

        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.5, 5.0)
        self.stop_loss_spin.setSingleStep(0.1)
        self.stop_loss_spin.setValue(self.config.stop_loss_pct)
        strategy_layout.addRow("Stop Loss %:", self.stop_loss_spin)

        self.sma_tp_spin = QSpinBox()
        self.sma_tp_spin.setRange(10, 50)
        self.sma_tp_spin.setValue(self.config.sma_tp_period)
        strategy_layout.addRow("SMA TP Period:", self.sma_tp_spin)

        # Removed Data Mode selection - using optimal performance mode only

        layout.addWidget(strategy_group)

        # Risk management
        risk_group = QGroupBox("Risk Management")
        risk_layout = QFormLayout(risk_group)

        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000, 1000000)
        self.capital_spin.setValue(self.config.initial_capital)
        risk_layout.addRow("Capital ($):", self.capital_spin)

        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(100, 5000)
        self.position_size_spin.setSingleStep(100)
        self.position_size_spin.setValue(1000)  # Default $1000 per trade
        risk_layout.addRow("Position Size ($):", self.position_size_spin)

        layout.addWidget(risk_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Action buttons
        self.start_btn = QPushButton("START BACKTEST")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.start_btn.clicked.connect(self._start_backtest)
        layout.addWidget(self.start_btn)

        self.export_btn = QPushButton("Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results)
        layout.addWidget(self.export_btn)

        layout.addStretch()
        return panel

    def _create_tabs(self):
        """Create result tabs with minimal complexity"""
        # Charts & Signals tab (Plotly high-performance)
        self.tabs.addTab(self.chart_signals_tab.get_widget(), "Charts & Signals")

        # Trade Details tab (refactored module)
        self.tabs.addTab(self.trade_details_tab.get_widget(), "Trade Details")

        # Performance tab (refactored module)
        self.tabs.addTab(self.performance_tab.get_widget(), "Performance")

        # Log tab
        self.log_widget = QWidget()
        log_layout = QVBoxLayout(self.log_widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        self.tabs.addTab(self.log_widget, "Execution Log")

        # Initialize logger after log_text is created
        self.logger = Logger(self.log_text)

        # Initialize dataset manager (will be setup after control panel creation)
        # self.dataset_manager will be initialized in _create_control_panel




    def _update_config(self):
        """Update strategy configuration from UI"""
        self.config.bb_period = self.bb_period_spin.value()
        self.config.bb_std = self.bb_std_spin.value()
        self.config.stop_loss_pct = self.stop_loss_spin.value()
        self.config.sma_tp_period = self.sma_tp_spin.value()
        self.config.initial_capital = self.capital_spin.value()
        self.config.position_size_dollars = self.position_size_spin.value()

    def _start_backtest(self):
        """Start backtesting process"""
        dataset = self.dataset_combo.currentText()
        if not dataset:
            self.status_bar.showMessage("Select a dataset first", 3000)
            return

        dataset_path = os.path.join("upload/trades", dataset)
        if not os.path.exists(dataset_path):
            self.status_bar.showMessage("Dataset not found", 3000)
            return

        # Update config and start
        self._update_config()
        symbol = self.dataset_manager.extract_symbol(dataset)

        # Always use performance mode (vectorized tick processing)
        tick_mode = True
        self._log(f"Starting backtest: {symbol} with BB({self.config.bb_period}, {self.config.bb_std}) - HFT MODE")

        # UI state
        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_bar.showMessage("Running backtest...")

        # Clear previous results
        self.results_data = None
        self._clear_results()

        # Use optimal performance mode - process full dataset without limits
        max_ticks = None  # No tick limits - process full dataset for maximum performance
        self._log("Performance Mode: Processing full dataset for maximum accuracy")

        self.worker = BacktestWorker(dataset_path, symbol, self.config, tick_mode=tick_mode, max_ticks=max_ticks)
        self.worker.progress_signal.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        self.worker.result_signal.connect(self._on_complete, Qt.ConnectionType.QueuedConnection)
        self.worker.error_signal.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self._on_worker_finished, Qt.ConnectionType.QueuedConnection)
        self.worker.start()

    def _on_progress(self, message):
        """Handle progress updates"""
        self._log(message)
        # Removed QApplication.processEvents() to prevent event loop recursion

    def _on_complete(self, results):
        """Handle successful backtest completion"""
        self.results_data = results

        # Log completion
        self._log(f"Backtest completed successfully with {len(results.get('trades', []))} trades")

        # Log detailed results to console and GUI
        trades = results.get('trades', [])
        total_trades = results.get('total', len(trades))
        win_rate = results.get('win_rate', 0)
        net_pnl = results.get('net_pnl', 0)
        return_pct = results.get('net_pnl_percentage', 0)

        # Console output (what user sees)
        print("\n" + "="*60)
        print("BACKTEST COMPLETED - RESULTS")
        print("="*60)
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Net P&L: ${net_pnl:.2f}")
        print(f"Return: {return_pct:.2f}%")
        print(f"Winners: {results.get('total_winning_trades', 0)}")
        print(f"Losers: {results.get('total_losing_trades', 0)}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print("="*60)

        # GUI log
        self._log("Backtest completed successfully!")
        self._log(f"Results: {total_trades} trades, {win_rate:.1%} win rate, ${net_pnl:.2f} P&L")

        # Display results in GUI
        self._display_results()

        # Auto-switch to Trade Details tab to show results
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Trade Details":
                self.tabs.setCurrentIndex(i)
                break

        self.export_btn.setEnabled(True)
        self.status_bar.showMessage(f"Backtest completed: {total_trades} trades, ${net_pnl:.2f} P&L", 10000)

    def _on_error(self, error_msg):
        """Handle backtest errors"""
        self._log(f"Error: {error_msg}")
        self.status_bar.showMessage("Backtest failed", 5000)

        # CRITICAL FIX: Re-enable button and cleanup on error
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_worker_finished(self):
        """Handle worker thread completion"""
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # CRITICAL FIX: Properly cleanup worker thread to prevent GUI freezing
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def _clear_results(self):
        """Clear previous results efficiently"""
        self.chart_signals_tab.clear()
        self.trade_details_tab.clear()
        self.performance_tab.clear()

    def _display_results(self):
        """Display backtest results efficiently"""
        if not self.results_data:
            return

        # Display all results including charts
        self.chart_signals_tab.update_chart(self.results_data)
        self.trade_details_tab.populate_trades(self.results_data)
        self.performance_tab.update_metrics(self.results_data)











    def _export_results(self):
        """Export results to CSV"""
        export_results(self.results_data, self.status_bar, self.logger)

    def _log(self, message):
        """Add log entry efficiently"""
        self.logger.log(message)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    # Professional dark theme
    app.setStyleSheet("""
        QMainWindow { background-color: #2b2b2b; color: #ffffff; }
        QWidget { background-color: #2b2b2b; color: #ffffff; }
        QGroupBox { font-weight: bold; margin: 5px; padding: 10px; border: 1px solid #555555; color: #ffffff; }
        QTabWidget::pane { border: 1px solid #555555; background-color: #2b2b2b; }
        QTabBar::tab { padding: 8px 12px; background-color: #3c3c3c; color: #ffffff; }
        QTabBar::tab:selected { background-color: #4c4c4c; color: #ffffff; }
        QTabBar::tab:hover { background-color: #5c5c5c; }
        QPushButton { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 6px; }
        QPushButton:hover { background-color: #4c4c4c; }
        QPushButton:pressed { background-color: #5c5c5c; }
        QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 4px; }
        QTableWidget { background-color: #3c3c3c; color: #ffffff; gridline-color: #555555; }
        QTableWidget::item { border: 1px solid #555555; }
        QHeaderView::section { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 4px; }
        QTextEdit { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; }
        QLineEdit { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 4px; }
        QSpinBox, QDoubleSpinBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; }
        QProgressBar { border: 1px solid #555555; text-align: center; background-color: #3c3c3c; }
        QProgressBar::chunk { background-color: #4CAF50; }
    """)

    window = ProfessionalBacktester()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())