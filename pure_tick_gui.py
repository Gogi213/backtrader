"""
Pure Tick GUI Application for High-Frequency Bollinger Bands Strategy
Simplified interface, no candle options - only pure tick processing

Author: HFT System
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QLineEdit, QHeaderView, QSplitter, QProgressBar, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QTextCursor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.dates as mdates


class PureTickConfig:
    """Pure tick strategy configuration"""
    def __init__(self):
        self.bb_period = 50  # HFT optimized
        self.bb_std = 2.0
        self.stop_loss_pct = 0.5
        self.initial_capital = 10000.0
        self.max_ticks = None  # None = process all ticks


class PureTickBacktestWorker(QThread):
    """Pure tick backtest worker thread"""
    progress_signal = pyqtSignal(str)
    result_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, csv_path: str, symbol: str, config: PureTickConfig):
        super().__init__()
        self.csv_path = csv_path
        self.symbol = symbol
        self.config = config

    def run(self):
        try:
            from pure_tick_backtest import run_pure_tick_backtest

            self.progress_signal.emit("🔄 Loading pure tick data...")
            self.progress_signal.emit("⚡ Processing HFT Bollinger Bands with vectorized engine...")
            self.progress_signal.emit("📊 Running pure tick backtest...")

            # Run pure tick backtest
            results = run_pure_tick_backtest(
                csv_path=self.csv_path,
                symbol=self.symbol,
                bb_period=self.config.bb_period,
                bb_std=self.config.bb_std,
                stop_loss_pct=self.config.stop_loss_pct,
                initial_capital=self.config.initial_capital,
                max_ticks=self.config.max_ticks
            )

            self.result_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(f"Pure tick backtest failed: {str(e)}")


class PureTickBacktesterGUI(QMainWindow):
    """Pure Tick Backtesting GUI - HFT focused"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 Pure Tick HFT Backtester - Bollinger Bands")
        self.setGeometry(50, 50, 1800, 1200)

        # Core data
        self.config = PureTickConfig()
        self.results_data = None
        self.worker = None

        self._init_ui()
        self._load_datasets()

    def _init_ui(self):
        """Initialize simplified UI for pure tick processing"""
        central = QWidget()
        self.setCentralWidget(central)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        central_layout = QHBoxLayout(central)
        central_layout.addWidget(splitter)

        # Left panel: controls (15%)
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)

        # Right panel: tabs (85%)
        self.tabs = QTabWidget()
        self._create_tabs()
        splitter.addWidget(self.tabs)

        # Set splitter sizes
        splitter.setSizes([270, 1530])

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("🚀 Ready for HFT backtesting")

    def _create_control_panel(self):
        """Create HFT-focused control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header
        title = QLabel("🚀 HFT Tick Backtester")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Dataset selection
        dataset_group = QGroupBox("📊 Tick Data Source")
        dataset_layout = QFormLayout(dataset_group)

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        dataset_layout.addRow("Dataset:", self.dataset_combo)

        self.symbol_label = QLabel("Auto-detected")
        dataset_layout.addRow("Symbol:", self.symbol_label)

        layout.addWidget(dataset_group)

        # Strategy parameters
        strategy_group = QGroupBox("⚡ HFT Strategy Parameters")
        strategy_layout = QFormLayout(strategy_group)

        # BB parameters (HFT optimized)
        self.bb_period_spin = QSpinBox()
        self.bb_period_spin.setRange(10, 200)
        self.bb_period_spin.setValue(self.config.bb_period)
        strategy_layout.addRow("BB Period:", self.bb_period_spin)

        self.bb_std_spin = QDoubleSpinBox()
        self.bb_std_spin.setRange(1.0, 4.0)
        self.bb_std_spin.setSingleStep(0.1)
        self.bb_std_spin.setValue(self.config.bb_std)
        strategy_layout.addRow("BB Std Dev:", self.bb_std_spin)

        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.1, 2.0)
        self.stop_loss_spin.setSingleStep(0.1)
        self.stop_loss_spin.setValue(self.config.stop_loss_pct)
        strategy_layout.addRow("Stop Loss %:", self.stop_loss_spin)

        layout.addWidget(strategy_group)

        # Risk management
        risk_group = QGroupBox("💰 Risk Management")
        risk_layout = QFormLayout(risk_group)

        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000, 1000000)
        self.capital_spin.setValue(self.config.initial_capital)
        self.capital_spin.setSuffix(" USD")
        risk_layout.addRow("Capital:", self.capital_spin)

        # Max ticks for testing
        self.max_ticks_spin = QSpinBox()
        self.max_ticks_spin.setRange(1000, 10000000)
        self.max_ticks_spin.setValue(50000)
        self.max_ticks_spin.setSuffix(" ticks")
        risk_layout.addRow("Max Ticks (test):", self.max_ticks_spin)

        self.unlimited_check = QCheckBox("Process ALL ticks")
        self.unlimited_check.setChecked(False)
        self.unlimited_check.toggled.connect(self._on_unlimited_toggled)
        risk_layout.addRow("Mode:", self.unlimited_check)

        layout.addWidget(risk_group)

        # Controls
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.start_btn = QPushButton("🚀 Start HFT Backtest")
        self.start_btn.clicked.connect(self._start_backtest)
        layout.addWidget(self.start_btn)

        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results)
        layout.addWidget(self.export_btn)

        layout.addStretch()
        return panel

    def _create_tabs(self):
        """Create result tabs"""
        # Trade Details tab
        self.trades_widget = QWidget()
        trades_layout = QVBoxLayout(self.trades_widget)

        # Trades table
        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        trades_layout.addWidget(self.trades_table)

        self.tabs.addTab(self.trades_widget, "📊 Trades")

        # Charts tab
        self.chart_widget = QWidget()
        chart_layout = QVBoxLayout(self.chart_widget)

        self.chart_figure = Figure(figsize=(12, 8))
        self.chart_canvas = FigureCanvas(self.chart_figure)
        chart_layout.addWidget(self.chart_canvas)

        self.tabs.addTab(self.chart_widget, "📈 Charts")

        # Performance tab
        self.performance_widget = QWidget()
        perf_layout = QHBoxLayout(self.performance_widget)

        # Metrics text
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        perf_layout.addWidget(self.metrics_text)

        self.tabs.addTab(self.performance_widget, "🏆 Performance")

        # Execution log
        self.log_widget = QWidget()
        log_layout = QVBoxLayout(self.log_widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        self.tabs.addTab(self.log_widget, "📝 Log")

    def _load_datasets(self):
        """Load available tick datasets"""
        trades_dir = "upload/trades"
        if not os.path.exists(trades_dir):
            self._log("⚠️ No trades directory found")
            return

        csv_files = [f for f in os.listdir(trades_dir) if f.endswith('.csv')]

        self.dataset_combo.addItems(csv_files)

        if csv_files:
            self.dataset_combo.setCurrentIndex(0)
            self._log(f"📄 Found {len(csv_files)} tick datasets")
            self._on_dataset_changed(csv_files[0])

    def _extract_symbol(self, filename: str) -> str:
        """Extract symbol from filename"""
        name = os.path.splitext(filename)[0]
        if '-trades-' in name:
            return name.split('-trades-')[0]
        return name.upper()

    def _on_dataset_changed(self, filename: str):
        """Handle dataset selection"""
        if filename:
            symbol = self._extract_symbol(filename)
            self.symbol_label.setText(symbol)
            self._log(f"📊 Dataset: {filename} → Symbol: {symbol}")

    def _on_unlimited_toggled(self, checked: bool):
        """Handle unlimited ticks toggle"""
        self.max_ticks_spin.setEnabled(not checked)

    def _update_config(self):
        """Update config from UI"""
        self.config.bb_period = self.bb_period_spin.value()
        self.config.bb_std = self.bb_std_spin.value()
        self.config.stop_loss_pct = self.stop_loss_spin.value()
        self.config.initial_capital = self.capital_spin.value()

        if self.unlimited_check.isChecked():
            self.config.max_ticks = None
        else:
            self.config.max_ticks = self.max_ticks_spin.value()

    def _start_backtest(self):
        """Start pure tick backtesting"""
        dataset = self.dataset_combo.currentText()
        if not dataset:
            self.status_bar.showMessage("❌ Select a dataset first", 3000)
            return

        dataset_path = os.path.join("upload/trades", dataset)
        if not os.path.exists(dataset_path):
            self.status_bar.showMessage("❌ Dataset not found", 3000)
            return

        self._update_config()
        symbol = self._extract_symbol(dataset)

        mode_text = "ALL TICKS" if self.config.max_ticks is None else f"{self.config.max_ticks:,} ticks"
        self._log(f"🚀 Starting HFT backtest: {symbol} - {mode_text}")

        # UI state
        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_bar.showMessage("⚡ Running HFT backtest...")

        # Clear previous results
        self.results_data = None
        self._clear_results()

        # Start worker
        self.worker = PureTickBacktestWorker(dataset_path, symbol, self.config)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.result_signal.connect(self._on_complete)
        self.worker.error_signal.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_progress(self, message: str):
        """Handle progress updates"""
        self._log(message)
        QApplication.processEvents()

    def _on_complete(self, results: dict):
        """Handle successful backtest completion"""
        self.results_data = results

        if 'error' in results:
            self._log(f"❌ Error: {results['error']}")
            return

        # Log detailed results
        trades = results.get('trades', [])
        total_trades = results.get('total', len(trades))
        win_rate = results.get('win_rate', 0)
        net_pnl = results.get('net_pnl', 0)
        ticks_processed = results.get('ticks_processed', 0)

        # Console output
        print("\n" + "="*60)
        print("🏆 HFT BACKTEST COMPLETED")
        print("="*60)
        print(f"⚡ Ticks Processed: {ticks_processed:,}")
        print(f"📊 Total Trades: {total_trades}")
        print(f"🎯 Win Rate: {win_rate:.1%}")
        print(f"💰 Net P&L: ${net_pnl:.2f}")
        print(f"📈 Return: {results.get('net_pnl_percentage', 0):.2f}%")
        print("="*60)

        self._log("✅ HFT backtest completed!")
        self._log(f"📊 Results: {ticks_processed:,} ticks → {total_trades} trades, ${net_pnl:.2f} P&L")

        # Display results
        self._display_results()

        # Switch to trades tab
        self.tabs.setCurrentIndex(0)

        self.export_btn.setEnabled(True)
        self.status_bar.showMessage(f"✅ HFT Backtest completed: {total_trades} trades, ${net_pnl:.2f} P&L", 10000)

    def _on_error(self, error_msg: str):
        """Handle errors"""
        self._log(f"❌ Error: {error_msg}")
        self.status_bar.showMessage("❌ Backtest failed", 5000)

    def _on_worker_finished(self):
        """Handle worker completion"""
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _clear_results(self):
        """Clear previous results"""
        self.chart_figure.clear()
        self.chart_canvas.draw()
        self.trades_table.setRowCount(0)
        self.metrics_text.clear()

    def _display_results(self):
        """Display results in all tabs"""
        if not self.results_data:
            return
        self._populate_trades()
        self._show_metrics()

    def _populate_trades(self):
        """Populate trades table"""
        if not self.results_data or 'trades' not in self.results_data:
            return

        trades = self.results_data['trades']
        if not trades:
            return

        headers = ['ID', 'Entry Time', 'Exit Time', 'Side', 'Entry $', 'Exit $', 'P&L $', 'P&L %', 'Duration ms', 'Reason']
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        self.trades_table.setRowCount(len(trades))

        for i, trade in enumerate(trades):
            # Format times
            entry_time = pd.to_datetime(trade.get('timestamp', 0), unit='ms').strftime('%H:%M:%S.%f')[:-3]
            exit_time = pd.to_datetime(trade.get('exit_timestamp', 0), unit='ms').strftime('%H:%M:%S.%f')[:-3]

            items = [
                QTableWidgetItem(str(i + 1)),
                QTableWidgetItem(entry_time),
                QTableWidgetItem(exit_time),
                QTableWidgetItem(trade.get('side', 'N/A').upper()),
                QTableWidgetItem(f"${trade.get('entry_price', 0):.4f}"),
                QTableWidgetItem(f"${trade.get('exit_price', 0):.4f}"),
                QTableWidgetItem(f"${trade.get('pnl', 0):.2f}"),
                QTableWidgetItem(f"{trade.get('pnl_percentage', 0):.2f}%"),
                QTableWidgetItem(f"{trade.get('duration', 0):.0f}"),
                QTableWidgetItem(trade.get('exit_reason', 'N/A'))
            ]

            # Color P&L cells
            pnl = trade.get('pnl', 0)
            color = QColor(0, 150, 0) if pnl > 0 else QColor(150, 0, 0) if pnl < 0 else QColor(128, 128, 128)
            items[6].setForeground(color)  # P&L $
            items[7].setForeground(color)  # P&L %

            for j, item in enumerate(items):
                self.trades_table.setItem(i, j, item)

    def _show_metrics(self):
        """Display performance metrics"""
        if not self.results_data:
            return

        results = self.results_data
        metrics_text = f"""
🏆 HFT PERFORMANCE METRICS

📊 TRADING STATISTICS:
   • Total Trades: {results.get('total', 0)}
   • Win Rate: {results.get('win_rate', 0):.1%}
   • Winners: {results.get('total_winning_trades', 0)}
   • Losers: {results.get('total_losing_trades', 0)}

💰 P&L ANALYSIS:
   • Net P&L: ${results.get('net_pnl', 0):.2f}
   • Return: {results.get('net_pnl_percentage', 0):.2f}%
   • Average Win: ${results.get('average_win', 0):.2f}
   • Average Loss: ${results.get('average_loss', 0):.2f}
   • Best Trade: ${results.get('largest_win', 0):.2f}
   • Worst Trade: ${results.get('largest_loss', 0):.2f}

⚡ RISK METRICS:
   • Max Drawdown: {results.get('max_drawdown', 0):.2f}%
   • Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
   • Profit Factor: {results.get('profit_factor', 0):.2f}

🔥 HFT STATISTICS:
   • Ticks Processed: {results.get('ticks_processed', 0):,}
   • Current Capital: ${results.get('current_capital', 0):,.2f}
   • Has Open Position: {results.get('has_position', False)}
        """

        self.metrics_text.setText(metrics_text)

    def _export_results(self):
        """Export results to CSV"""
        if not self.results_data or 'trades' not in self.results_data:
            return

        trades = self.results_data['trades']
        if not trades:
            return

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hft_backtest_results_{timestamp}.csv"

        # Convert to DataFrame and save
        df = pd.DataFrame(trades)
        df.to_csv(filename, index=False)

        self._log(f"💾 Results exported to {filename}")
        self.status_bar.showMessage(f"💾 Exported to {filename}", 3000)

    def _log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)

        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # HFT Dark theme
    app.setStyleSheet("""
        QMainWindow { background-color: #1e1e1e; color: #ffffff; }
        QWidget { background-color: #1e1e1e; color: #ffffff; }
        QGroupBox { font-weight: bold; margin: 5px; padding: 10px; border: 1px solid #404040; color: #ffffff; }
        QTabWidget::pane { border: 1px solid #404040; background-color: #1e1e1e; }
        QTabBar::tab { padding: 8px 12px; background-color: #2d2d2d; color: #ffffff; }
        QTabBar::tab:selected { background-color: #0078d4; color: #ffffff; }
        QPushButton { background-color: #0078d4; color: #ffffff; border: none; padding: 8px 16px; border-radius: 4px; }
        QPushButton:hover { background-color: #106ebe; }
        QPushButton:disabled { background-color: #404040; color: #808080; }
        QComboBox, QSpinBox, QDoubleSpinBox { background-color: #2d2d2d; color: #ffffff; border: 1px solid #404040; padding: 4px; }
        QTableWidget { background-color: #2d2d2d; color: #ffffff; gridline-color: #404040; }
        QHeaderView::section { background-color: #2d2d2d; color: #ffffff; border: 1px solid #404040; }
        QTextEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #404040; }
        QProgressBar { border: 1px solid #404040; text-align: center; background-color: #2d2d2d; color: #ffffff; }
        QProgressBar::chunk { background-color: #0078d4; }
    """)

    window = PureTickBacktesterGUI()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())