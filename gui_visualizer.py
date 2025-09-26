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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates


class StrategyConfig:
    """Strategy configuration with validation"""
    def __init__(self):
        self.bb_period = 200
        self.bb_std = 3.0
        self.stop_loss_pct = 1.0
        self.sma_tp_period = 20
        self.initial_capital = 10000.0
        self.position_size_pct = 2.0
        self.risk_reward_ratio = 2.0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def from_dict(self, config_dict):
        for k, v in config_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)


class BacktestWorker(QThread):
    """High-performance worker thread with progress tracking"""
    progress_signal = pyqtSignal(str)
    result_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, csv_path, symbol, config, exchange='Binance', timeframe='1m'):
        super().__init__()
        self.csv_path = csv_path
        self.symbol = symbol
        self.config = config
        self.exchange = exchange
        self.timeframe = timeframe

    def run(self):
        try:
            from cli_backtest import run_backtest

            self.progress_signal.emit("Loading CSV data...")
            self.progress_signal.emit("Processing tick data to candles...")
            self.progress_signal.emit("Running Bollinger Bands strategy...")

            # Run backtest with strategy config
            results = run_backtest(
                self.csv_path, self.symbol, self.exchange, self.timeframe,
                bb_period=self.config.bb_period,
                bb_std=self.config.bb_std,
                stop_loss_pct=self.config.stop_loss_pct,
                sma_tp_period=self.config.sma_tp_period,
                initial_capital=self.config.initial_capital
            )

            self.result_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(f"Backtest failed: {str(e)}")


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

        self._init_ui()
        self._load_datasets()

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

        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([500, 1300])

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
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        dataset_layout.addRow("Dataset:", self.dataset_combo)

        self.symbol_label = QLabel("Auto-detected")
        dataset_layout.addRow("Symbol:", self.symbol_label)

        layout.addWidget(dataset_group)

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

        layout.addWidget(strategy_group)

        # Risk management
        risk_group = QGroupBox("Risk Management")
        risk_layout = QFormLayout(risk_group)

        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000, 1000000)
        self.capital_spin.setValue(self.config.initial_capital)
        risk_layout.addRow("Capital ($):", self.capital_spin)

        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(0.5, 10.0)
        self.position_size_spin.setSingleStep(0.1)
        self.position_size_spin.setValue(self.config.position_size_pct)
        risk_layout.addRow("Position Size %:", self.position_size_spin)

        self.risk_reward_spin = QDoubleSpinBox()
        self.risk_reward_spin.setRange(1.0, 5.0)
        self.risk_reward_spin.setSingleStep(0.1)
        self.risk_reward_spin.setValue(self.config.risk_reward_ratio)
        risk_layout.addRow("Risk:Reward:", self.risk_reward_spin)

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
        # Charts tab
        self.chart_widget = QWidget()
        chart_layout = QVBoxLayout(self.chart_widget)

        self.chart_figure = Figure(figsize=(14, 10))
        self.chart_canvas = FigureCanvas(self.chart_figure)
        chart_layout.addWidget(self.chart_canvas)

        self.tabs.addTab(self.chart_widget, "Charts & Signals")

        # Trades tab
        self.trades_widget = QWidget()
        trades_layout = QVBoxLayout(self.trades_widget)

        # Search/filter row
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))

        self.trades_search = QLineEdit()
        self.trades_search.setPlaceholderText("Search trades...")
        self.trades_search.textChanged.connect(self._filter_trades)
        filter_layout.addWidget(self.trades_search)

        self.profitable_only = QCheckBox("Profitable only")
        self.profitable_only.toggled.connect(self._filter_trades)
        filter_layout.addWidget(self.profitable_only)

        filter_layout.addStretch()
        trades_layout.addLayout(filter_layout)

        # Trades table
        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        trades_layout.addWidget(self.trades_table)

        self.tabs.addTab(self.trades_widget, "Trade Details")

        # Metrics tab
        self.metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_widget)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setFont(QFont("Consolas", 11))
        metrics_layout.addWidget(self.metrics_text)

        self.tabs.addTab(self.metrics_widget, "Performance")

        # Log tab
        self.log_widget = QWidget()
        log_layout = QVBoxLayout(self.log_widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        self.tabs.addTab(self.log_widget, "Execution Log")

    def _load_datasets(self):
        """Load available datasets efficiently"""
        trades_dir = "upload/trades"
        if not os.path.exists(trades_dir):
            self._log("Warning: No trades directory found")
            return

        csv_files = [f for f in os.listdir(trades_dir) if f.endswith('.csv')]
        self.dataset_combo.addItems(csv_files)

        if csv_files:
            self._log(f"Found {len(csv_files)} datasets")

    def _extract_symbol(self, filename):
        """Extract trading symbol from filename"""
        name = os.path.splitext(filename)[0]
        if '-trades-' in name:
            return name.split('-trades-')[0]
        if name.startswith('trades_'):
            return name[7:]
        return name

    def _on_dataset_changed(self, filename):
        """Handle dataset selection"""
        if filename:
            symbol = self._extract_symbol(filename)
            self.symbol_label.setText(symbol)
            self._log(f"Dataset: {filename} → Symbol: {symbol}")

    def _update_config(self):
        """Update strategy configuration from UI"""
        self.config.bb_period = self.bb_period_spin.value()
        self.config.bb_std = self.bb_std_spin.value()
        self.config.stop_loss_pct = self.stop_loss_spin.value()
        self.config.sma_tp_period = self.sma_tp_spin.value()
        self.config.initial_capital = self.capital_spin.value()
        self.config.position_size_pct = self.position_size_spin.value()
        self.config.risk_reward_ratio = self.risk_reward_spin.value()

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
        symbol = self._extract_symbol(dataset)

        self._log(f"Starting backtest: {symbol} with BB({self.config.bb_period}, {self.config.bb_std})")

        # UI state
        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_bar.showMessage("Running backtest...")

        # Clear previous results
        self.results_data = None
        self._clear_results()

        # Start worker
        self.worker = BacktestWorker(dataset_path, symbol, self.config)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.result_signal.connect(self._on_complete)
        self.worker.error_signal.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_progress(self, message):
        """Handle progress updates"""
        self._log(message)
        QApplication.processEvents()

    def _on_complete(self, results):
        """Handle successful backtest completion"""
        self.results_data = results
        self._log("Backtest completed successfully")
        self._display_results()
        self.export_btn.setEnabled(True)
        self.status_bar.showMessage("Backtest completed", 5000)

    def _on_error(self, error_msg):
        """Handle backtest errors"""
        self._log(f"Error: {error_msg}")
        self.status_bar.showMessage("Backtest failed", 5000)

    def _on_worker_finished(self):
        """Handle worker thread completion"""
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _clear_results(self):
        """Clear previous results efficiently"""
        self.chart_figure.clear()
        self.chart_canvas.draw()
        self.trades_table.setRowCount(0)
        self.metrics_text.clear()

    def _display_results(self):
        """Display backtest results efficiently"""
        if not self.results_data:
            return

        self._plot_charts()
        self._populate_trades()
        self._show_metrics()

    def _plot_charts(self):
        """Create professional charts with signals"""
        if not self.results_data:
            return

        self.chart_figure.clear()

        # Create subplots
        ax1 = self.chart_figure.add_subplot(2, 1, 1)  # Price + BB + Signals
        ax2 = self.chart_figure.add_subplot(2, 1, 2)  # Equity curve

        # Mock price data for demonstration (replace with real data)
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        price = 100 + np.cumsum(np.random.randn(200) * 0.5)

        # Bollinger Bands
        sma = pd.Series(price).rolling(20).mean()
        std = pd.Series(price).rolling(20).std()
        upper_bb = sma + 2 * std
        lower_bb = sma - 2 * std

        # Plot price and BB
        ax1.plot(dates, price, 'k-', linewidth=1.5, label='Price')
        ax1.plot(dates, sma, 'b--', alpha=0.7, label='SMA')
        ax1.fill_between(dates, upper_bb, lower_bb, alpha=0.2, color='blue', label='BB')

        # Mock trading signals
        buy_signals = np.random.choice(len(dates), 10)
        sell_signals = np.random.choice(len(dates), 8)

        ax1.scatter(dates[buy_signals], price[buy_signals],
                   color='green', marker='^', s=100, label='Buy')
        ax1.scatter(dates[sell_signals], price[sell_signals],
                   color='red', marker='v', s=100, label='Sell')

        ax1.set_title('Price Chart with Bollinger Bands & Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Equity curve
        equity = 10000 + np.cumsum(np.random.randn(200) * 50)
        ax2.plot(dates, equity, 'g-', linewidth=2, label='Equity Curve')
        ax2.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.chart_canvas.draw()

    def _populate_trades(self):
        """Populate trades table efficiently"""
        if not self.results_data or 'trades' not in self.results_data:
            return

        trades = self.results_data['trades']
        if not trades:
            return

        headers = ['ID', 'Time', 'Side', 'Entry $', 'Exit $', 'P&L $', 'P&L %', 'Size', 'Duration']
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        self.trades_table.setRowCount(len(trades))

        for i, trade in enumerate(trades):
            # Create items with proper formatting
            items = [
                QTableWidgetItem(str(i + 1)),
                QTableWidgetItem(datetime.fromtimestamp(trade.get('time', 0)/1000).strftime('%Y-%m-%d %H:%M')),
                QTableWidgetItem(trade.get('side', 'N/A')),
                QTableWidgetItem(f"{trade.get('entry_price', 0):.4f}"),
                QTableWidgetItem(f"{trade.get('exit_price', 0):.4f}"),
                QTableWidgetItem(f"{trade.get('pnl', 0):.2f}"),
                QTableWidgetItem(f"{trade.get('pnl_percentage', 0):.1f}"),
                QTableWidgetItem(f"{trade.get('qty', 0):.4f}"),
                QTableWidgetItem(f"{trade.get('duration', 0)} min")
            ]

            # Color coding for P&L
            pnl = trade.get('pnl', 0)
            color = QColor(0, 150, 0) if pnl > 0 else QColor(200, 0, 0) if pnl < 0 else QColor(100, 100, 100)
            items[5].setForeground(color)  # P&L $
            items[6].setForeground(color)  # P&L %

            for j, item in enumerate(items):
                self.trades_table.setItem(i, j, item)

    def _show_metrics(self):
        """Display performance metrics"""
        if not self.results_data:
            return

        r = self.results_data
        metrics_html = f"""
        <h2>Performance Summary</h2>
        <table border="1" cellpadding="5">
        <tr><td><b>Total Trades:</b></td><td>{r.get('total', 0)}</td></tr>
        <tr><td><b>Win Rate:</b></td><td>{r.get('win_rate', 0)*100:.1f}%</td></tr>
        <tr><td><b>Net P&L:</b></td><td>${r.get('net_pnl', 0):,.2f}</td></tr>
        <tr><td><b>Return %:</b></td><td>{r.get('net_pnl_percentage', 0):.2f}%</td></tr>
        <tr><td><b>Max Drawdown:</b></td><td>{r.get('max_drawdown', 0):.2f}%</td></tr>
        <tr><td><b>Sharpe Ratio:</b></td><td>{r.get('sharpe_ratio', 0):.2f}</td></tr>
        <tr><td><b>Profit Factor:</b></td><td>{r.get('profit_factor', 0):.2f}</td></tr>
        </table>

        <h3>Trade Statistics</h3>
        <table border="1" cellpadding="5">
        <tr><td><b>Winners:</b></td><td>{r.get('total_winning_trades', 0)}</td></tr>
        <tr><td><b>Losers:</b></td><td>{r.get('total_losing_trades', 0)}</td></tr>
        <tr><td><b>Avg Win:</b></td><td>${r.get('average_win', 0):.2f}</td></tr>
        <tr><td><b>Avg Loss:</b></td><td>${r.get('average_loss', 0):.2f}</td></tr>
        <tr><td><b>Best Trade:</b></td><td>${r.get('largest_win', 0):.2f}</td></tr>
        <tr><td><b>Worst Trade:</b></td><td>${r.get('largest_loss', 0):.2f}</td></tr>
        </table>
        """
        self.metrics_text.setHtml(metrics_html)

    def _filter_trades(self):
        """Filter trades table based on search and checkboxes"""
        search_text = self.trades_search.text().lower()
        profitable_only = self.profitable_only.isChecked()

        for row in range(self.trades_table.rowCount()):
            show_row = True

            # Search filter
            if search_text:
                row_text = " ".join([
                    self.trades_table.item(row, col).text().lower()
                    for col in range(self.trades_table.columnCount())
                    if self.trades_table.item(row, col)
                ])
                if search_text not in row_text:
                    show_row = False

            # Profitable filter
            if profitable_only and show_row:
                pnl_item = self.trades_table.item(row, 5)  # P&L column
                if pnl_item:
                    try:
                        pnl = float(pnl_item.text())
                        if pnl <= 0:
                            show_row = False
                    except ValueError:
                        show_row = False

            self.trades_table.setRowHidden(row, not show_row)

    def _export_results(self):
        """Export results to CSV"""
        if not self.results_data:
            return

        # Export trades to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.csv"

        trades = self.results_data.get('trades', [])
        if trades:
            df = pd.DataFrame(trades)
            df.to_csv(filename, index=False)
            self._log(f"Results exported to {filename}")
            self.status_bar.showMessage(f"Exported to {filename}", 5000)
        else:
            self._log("No trade data to export")

    def _log(self, message):
        """Add log entry efficiently"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    # Dark theme (optional)
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QGroupBox { font-weight: bold; margin: 5px; padding: 10px; }
        QTabWidget::pane { border: 1px solid #c0c0c0; }
        QTabBar::tab { padding: 8px 12px; }
        QTabBar::tab:selected { background-color: #ffffff; }
    """)

    window = ProfessionalBacktester()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())