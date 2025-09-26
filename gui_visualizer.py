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
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
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
        self.position_size_dollars = 1000.0

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

    def __init__(self, csv_path, symbol, config, exchange='Binance', timeframe='1m', tick_mode=False):
        super().__init__()
        self.csv_path = csv_path
        self.symbol = symbol
        self.config = config
        self.exchange = exchange
        self.timeframe = timeframe
        self.tick_mode = tick_mode

    def run(self):
        try:
            from cli_backtest import run_backtest

            if self.tick_mode:
                self.progress_signal.emit("Loading raw tick data...")
                self.progress_signal.emit("Processing individual ticks...")
                self.progress_signal.emit("Running HFT Bollinger Bands strategy on ticks...")
            else:
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
                initial_capital=self.config.initial_capital,
                use_real_trades=False,  # Run full backtest
                use_tick_mode=self.tick_mode  # Use tick mode if selected
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

        # Tick mode checkbox
        self.tick_mode_check = QCheckBox("Use Tick Mode (HFT)")
        self.tick_mode_check.setToolTip("Process raw ticks instead of converting to candles - for high-frequency trading")
        strategy_layout.addRow("Data Mode:", self.tick_mode_check)

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
        # Charts tab
        self.chart_widget = QWidget()
        chart_layout = QVBoxLayout(self.chart_widget)
        
        self.chart_figure = Figure(figsize=(14, 10))
        self.chart_canvas = FigureCanvas(self.chart_figure)
        
        # Add navigation toolbar for interactivity
        self.nav_toolbar = NavigationToolbar(self.chart_canvas, self.chart_widget)
        
        chart_layout.addWidget(self.nav_toolbar)
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

        # Force trades_exampls.csv to be the default and first choice
        final_files = []
        if 'trades_exampls.csv' in csv_files:
            final_files.append('trades_exampls.csv')
            # Add other files after trades_exampls.csv
            for f in csv_files:
                if f != 'trades_exampls.csv':
                    final_files.append(f)
        else:
            final_files = csv_files

        self.dataset_combo.addItems(final_files)

        if final_files:
            # Force select the first item (trades_exampls.csv if available)
            self.dataset_combo.setCurrentIndex(0)
            self._log(f"Found {len(final_files)} datasets, forced default: {final_files[0]}")
            # Trigger the selection change to update symbol
            self._on_dataset_changed(final_files[0])

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
        symbol = self._extract_symbol(dataset)

        tick_mode = self.tick_mode_check.isChecked()
        mode_text = "TICK MODE" if tick_mode else "CANDLE MODE"
        self._log(f"Starting backtest: {symbol} with BB({self.config.bb_period}, {self.config.bb_std}) - {mode_text}")

        # UI state
        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_bar.showMessage("Running backtest...")

        # Clear previous results
        self.results_data = None
        self._clear_results()

        # Start worker
        tick_mode = self.tick_mode_check.isChecked()
        self.worker = BacktestWorker(dataset_path, symbol, self.config, tick_mode=tick_mode)
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
        """Create professional charts with signals based on real trade data"""
        if not self.results_data:
            return

        self.chart_figure.clear()

        # Create subplot - only price chart and signals (no equity curve here)
        ax1 = self.chart_figure.add_subplot(1, 1, 1)  # Price + Signals

        # Extract trade data to create real price chart and signals
        trades = self.results_data.get('trades', [])
        if trades:
            # First, get the CSV file path to load the full price data
            # We'll use the same dataset that was used for the backtest
            dataset = self.dataset_combo.currentText()
            if dataset:
                csv_path = os.path.join("upload/trades", dataset)
                if os.path.exists(csv_path):
                    # Load the full price data from the CSV
                    df = pd.read_csv(csv_path)
                    # Convert time column to datetime for plotting
                    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
                    df = df.sort_values('datetime')  # Sort by time
                    
                    # Calculate Bollinger Bands
                    bb_period = self.config.bb_period
                    bb_std = self.config.bb_std

                    df['price_sma'] = df['price'].rolling(window=bb_period).mean()
                    df['price_std'] = df['price'].rolling(window=bb_period).std()
                    df['bb_upper'] = df['price_sma'] + (df['price_std'] * bb_std)
                    df['bb_lower'] = df['price_sma'] - (df['price_std'] * bb_std)

                    # Plot price data
                    ax1.plot(df['datetime'], df['price'], 'k-', linewidth=0.8, label='Price', alpha=0.8)

                    # Plot Bollinger Bands
                    ax1.plot(df['datetime'], df['price_sma'], 'b--', linewidth=1, label=f'SMA({bb_period})', alpha=0.7)
                    ax1.plot(df['datetime'], df['bb_upper'], 'r-', linewidth=1, label=f'BB Upper (±{bb_std}σ)', alpha=0.6)
                    ax1.plot(df['datetime'], df['bb_lower'], 'r-', linewidth=1, label='BB Lower', alpha=0.6)
                    ax1.fill_between(df['datetime'], df['bb_upper'], df['bb_lower'], alpha=0.1, color='blue', label='BB Zone')

                    # Now add trade signals on top
                    trade_times = []
                    entry_prices = []
                    sides = []
                    
                    for trade in trades:
                        if 'timestamp' in trade and 'entry_price' in trade:
                            # Use entry time and price
                            time_val = trade['timestamp']
                            if isinstance(time_val, (int, float)):
                                # Convert timestamp to datetime
                                dt = pd.to_datetime(time_val, unit='ms')
                            else:
                                dt = pd.to_datetime(time_val)
                            trade_times.append(dt)
                            entry_prices.append(trade['entry_price'])
                            sides.append(trade.get('side', 'N/A'))
                        
                    if trade_times:
                        # Sort by time
                        sorted_data = sorted(zip(trade_times, entry_prices, sides))
                        trade_times, entry_prices, sides = zip(*sorted_data)
                        
                        # Convert to numpy arrays
                        times = np.array(trade_times)
                        prices = np.array(entry_prices)
                        
                        # Separate buy and sell signals
                        buy_times = [times[i] for i, side in enumerate(sides) if side == 'long']
                        buy_prices = [prices[i] for i, side in enumerate(sides) if side == 'long']
                        sell_times = [times[i] for i, side in enumerate(sides) if side == 'short']
                        sell_prices = [prices[i] for i, side in enumerate(sides) if side == 'short']
                        
                        # Plot signals
                        if buy_times:
                            ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy', zorder=5)
                        if sell_times:
                            ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell', zorder=5)
                        
                        # Format the x-axis to show time properly
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        else:
            # No trades or price data available
            ax1.text(0.5, 0.5, 'No trading data available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Price Chart with Trading Signals - No Data')

        ax1.set_title('Price Chart with Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        self.chart_canvas.draw()

    def _populate_trades(self):
        """Populate trades table efficiently"""
        if not self.results_data or 'trades' not in self.results_data:
            return

        trades = self.results_data['trades']
        if not trades:
            return

        headers = ['ID', 'Entry Time', 'Exit Time', 'Side', 'Entry $', 'Exit $', 'P&L $', 'P&L %', 'Size $', 'Duration']
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        self.trades_table.setRowCount(len(trades))

        for i, trade in enumerate(trades):
            # Format entry time
            entry_timestamp = trade.get('timestamp', 0)
            if isinstance(entry_timestamp, pd.Timestamp):
                entry_time_str = entry_timestamp.strftime('%H:%M:%S')
            elif entry_timestamp:
                entry_time_str = datetime.fromtimestamp(entry_timestamp/1000).strftime('%H:%M:%S')
            else:
                entry_time_str = 'N/A'

            # Format exit time
            exit_timestamp = trade.get('exit_timestamp', 0)
            if isinstance(exit_timestamp, pd.Timestamp):
                exit_time_str = exit_timestamp.strftime('%H:%M:%S')
            elif exit_timestamp:
                exit_time_str = datetime.fromtimestamp(exit_timestamp/1000).strftime('%H:%M:%S')
            else:
                exit_time_str = 'N/A'

            # Calculate position size in dollars
            position_size_dollars = trade.get('entry_price', 0) * trade.get('size', 0)

            items = [
                QTableWidgetItem(str(i + 1)),
                QTableWidgetItem(entry_time_str),
                QTableWidgetItem(exit_time_str),
                QTableWidgetItem(trade.get('side', 'N/A')),
                QTableWidgetItem(f"{trade.get('entry_price', 0):.4f}"),
                QTableWidgetItem(f"{trade.get('exit_price', 0):.4f}"),
                QTableWidgetItem(f"${trade.get('pnl', 0):.2f}"),
                QTableWidgetItem(f"{trade.get('pnl_percentage', 0):.1f}%"),
                QTableWidgetItem(f"${position_size_dollars:.2f}"),
                QTableWidgetItem(f"{trade.get('duration', 0)} min")
            ]

            # Color coding for P&L
            pnl = trade.get('pnl', 0)
            color = QColor(0, 150, 0) if pnl > 0 else QColor(200, 0, 0) if pnl < 0 else QColor(100, 100, 100)
            items[6].setForeground(color)  # P&L $ (now at index 6)
            items[7].setForeground(color)  # P&L % (now at index 7)

            for j, item in enumerate(items):
                self.trades_table.setItem(i, j, item)

    def _show_metrics(self):
        """Display performance metrics with equity curve chart"""
        if not self.results_data:
            return

        # Create equity curve chart
        trades = self.results_data.get('trades', [])
        if trades:
            # Calculate equity curve from trades
            initial_capital = self.results_data.get('initial_capital', 10000.0)
            equity_values = [initial_capital]
            cumulative_pnl = 0
            
            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', 0))
            
            for trade in sorted_trades:
                if 'pnl' in trade:
                    cumulative_pnl += trade['pnl']
                    equity_values.append(initial_capital + cumulative_pnl)
            
            # Create a new figure for the equity curve
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(len(equity_values)), equity_values, 'g-', linewidth=2, label='Equity Curve')
            ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
            ax.set_title('Equity Curve')
            ax.set_ylabel('Portfolio Value ($)')
            ax.set_xlabel('Trade Number')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Embed the chart in the metrics text widget
            from io import BytesIO
            import base64
            
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#2b2b2b', edgecolor='none')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            
            # Create HTML with new layout: Summary left, Equity right, Statistics bottom
            r = self.results_data
            metrics_html = f"""
            <style>
            .container {{ display: flex; margin-bottom: 20px; }}
            .summary {{ width: 50%; padding-right: 10px; }}
            .equity {{ width: 50%; padding-left: 10px; }}
            .statistics {{ width: 100%; }}
            table {{ width: 100%; border-collapse: collapse; }}
            td, th {{ padding: 5px; border: 1px solid #555; }}
            </style>

            <div class="container">
                <div class="summary">
                    <h2>Performance Summary</h2>
                    <table>
                    <tr><td><b>Total Trades:</b></td><td>{r.get('total', 0)}</td></tr>
                    <tr><td><b>Win Rate:</b></td><td>{r.get('win_rate', 0)*100:.1f}%</td></tr>
                    <tr><td><b>Net P&L:</b></td><td>${r.get('net_pnl', 0):,.2f}</td></tr>
                    <tr><td><b>Return %:</b></td><td>{r.get('net_pnl_percentage', 0):.2f}%</td></tr>
                    <tr><td><b>Max Drawdown:</b></td><td>{r.get('max_drawdown', 0):.2f}%</td></tr>
                    <tr><td><b>Sharpe Ratio:</b></td><td>{r.get('sharpe_ratio', 0):.2f}</td></tr>
                    <tr><td><b>Profit Factor:</b></td><td>{r.get('profit_factor', 0):.2f}</td></tr>
                    <tr><td><b>Winners:</b></td><td>{r.get('total_winning_trades', 0)}</td></tr>
                    <tr><td><b>Losers:</b></td><td>{r.get('total_losing_trades', 0)}</td></tr>
                    <tr><td><b>Avg Win:</b></td><td>${r.get('average_win', 0):.2f}</td></tr>
                    <tr><td><b>Avg Loss:</b></td><td>${r.get('average_loss', 0):.2f}</td></tr>
                    <tr><td><b>Best Trade:</b></td><td>${r.get('largest_win', 0):.2f}</td></tr>
                    <tr><td><b>Worst Trade:</b></td><td>${r.get('largest_loss', 0):.2f}</td></tr>
                    </table>
                </div>
                <div class="equity">
                    <h2>Equity Curve</h2>
                    <img src="data:image/png;base64,{img_str}" alt="Equity Curve" style="max-width:100%; height:auto;">
                </div>
            </div>
            """
        else:
            # No trades case
            metrics_html = "<h2>No backtest results available</h2><p>Run a backtest to see performance metrics.</p>"
        
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