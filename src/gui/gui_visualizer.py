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
        self.max_ticks_gui = 1000000  # PERFORMANCE FIX: Default limit 1M ticks for GUI performance
        self.max_ticks_unlimited = None  # No limit for advanced users

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

    def __init__(self, csv_path, symbol, config, exchange='Binance', timeframe='1m', tick_mode=False, max_ticks=None):
        super().__init__()
        self.csv_path = csv_path
        self.symbol = symbol
        self.config = config
        self.exchange = exchange
        self.timeframe = timeframe
        self.tick_mode = tick_mode
        self.max_ticks = max_ticks  # None means no limit, otherwise apply limit

    def run(self):
        try:
            # UNIFIED SYSTEM: Use vectorized_backtest for maximum performance
            from src.data.vectorized_backtest import run_vectorized_backtest

            self.progress_signal.emit(f"VECTORIZED PURE TICK BACKTEST: {self.symbol}")
            if self.max_ticks:
                self.progress_signal.emit(f"Loading tick data (limited to {self.max_ticks:,} ticks for GUI performance)...")
            else:
                self.progress_signal.emit("Loading full tick data (no limit)...")
            self.progress_signal.emit("Running super-vectorized Bollinger Bands strategy...")

            # Run vectorized backtest - UNIFIED SYSTEM!
            results = run_vectorized_backtest(
                csv_path=self.csv_path,
                symbol=self.symbol,
                bb_period=self.config.bb_period,
                bb_std=self.config.bb_std,
                stop_loss_pct=self.config.stop_loss_pct,
                initial_capital=self.config.initial_capital,
                max_ticks=self.max_ticks  # KEY: Limit processing for GUI performance (None means no limit)
            )

            self.result_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(f"Vectorized backtest failed: {str(e)}")


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
        self.chart_figure.clear()
        # Removed chart_canvas.draw() to prevent GUI freeze during clearing
        self.trades_table.setRowCount(0)
        self.metrics_text.clear()

    def _display_results(self):
        """Display backtest results efficiently"""
        if not self.results_data:
            return

        # Display all results including charts
        self._plot_charts_simple()  # New simplified chart version
        self._populate_trades()
        self._show_metrics_simple()

    def _plot_charts_simple(self):
        """Create high-quality charts without sampling - full dataset visualization"""
        if not self.results_data:
            return

        try:
            # Clear chart figure safely
            self.chart_figure.clear()

            # Create basic subplot
            ax = self.chart_figure.add_subplot(1, 1, 1)

            # Use BB data from results if available
            bb_data = self.results_data.get('bb_data')
            trades = self.results_data.get('trades', [])

            if bb_data and 'times' in bb_data:
                # Get full data without sampling
                times = bb_data['times']  # timestamp in milliseconds
                prices = bb_data['prices']

                # Convert timestamps to datetime for proper x-axis
                import pandas as pd
                datetime_series = pd.to_datetime(times, unit='ms')

                # Plot full price data
                ax.plot(datetime_series, prices, 'b-', linewidth=0.8, label='Price', alpha=0.9)

                # Add Bollinger Bands if available
                if 'bb_upper' in bb_data and 'bb_lower' in bb_data:
                    bb_upper = bb_data['bb_upper']
                    bb_middle = bb_data['bb_middle']
                    bb_lower = bb_data['bb_lower']

                    ax.plot(datetime_series, bb_upper, 'r--', linewidth=0.5, alpha=0.7, label='BB Upper')
                    ax.plot(datetime_series, bb_middle, 'y--', linewidth=0.5, alpha=0.7, label='BB Middle')
                    ax.plot(datetime_series, bb_lower, 'g--', linewidth=0.5, alpha=0.7, label='BB Lower')

                # Add ALL trade signals on the actual price chart
                if trades:
                    buy_times = []
                    buy_prices = []
                    sell_times = []
                    sell_prices = []

                    for trade in trades:
                        entry_time = trade.get('timestamp')  # Changed from 'entry_time' to 'timestamp'
                        entry_price = trade.get('entry_price', 0)
                        side = trade.get('side', 'unknown')

                        if entry_time and entry_price:
                            # Convert trade time to datetime
                            trade_datetime = pd.to_datetime(entry_time, unit='ms')

                            if side == 'long':
                                buy_times.append(trade_datetime)
                                buy_prices.append(entry_price)
                            else:
                                sell_times.append(trade_datetime)
                                sell_prices.append(entry_price)

                    # Plot buy signals
                    if buy_times:
                        ax.scatter(buy_times, buy_prices, color='green', marker='^',
                                 s=40, label=f'Long Entry ({len(buy_times)})', alpha=0.8, zorder=5)

                    # Plot sell signals
                    if sell_times:
                        ax.scatter(sell_times, sell_prices, color='red', marker='v',
                                 s=40, label=f'Short Entry ({len(sell_times)})', alpha=0.8, zorder=5)

            # Improved chart formatting
            ax.set_title('HFT Price Chart with Bollinger Bands Strategy', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Price (USDT)', fontsize=10)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # Smart datetime axis formatting like TradingView
            time_range = datetime_series.max() - datetime_series.min()

            if time_range.total_seconds() < 3600:  # Less than 1 hour
                # Show every 5-10 minutes for short timeframes
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
            elif time_range.total_seconds() < 86400:  # Less than 1 day
                # Show every 1-2 hours for medium timeframes
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
            else:  # More than 1 day
                # Show date and time for long timeframes
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

            # Rotate x-axis labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Add grid for minor ticks (lighter)
            ax.grid(True, which='minor', alpha=0.1, linestyle='-', linewidth=0.3)

            # Add interactive crosshair cursor for precise time/price display
            from matplotlib.widgets import Cursor
            cursor = Cursor(ax, useblit=True, color='gray', linewidth=0.8, alpha=0.7)
            cursor.horizOn = True
            cursor.vertOn = True

            # Add format_coord for precise coordinates display on hover
            def format_coord(x, y):
                # Convert matplotlib date to pandas datetime
                import matplotlib.dates as mdates
                date_obj = mdates.num2date(x)
                time_str = date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
                return f'Time: {time_str}, Price: ${y:.4f}'

            ax.format_coord = format_coord

            # Tight layout for better appearance
            self.chart_figure.tight_layout()

            # Draw canvas efficiently
            self.chart_canvas.draw()

        except Exception as e:
            # If chart fails, show error in chart area
            self.chart_figure.clear()
            ax = self.chart_figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Chart Error: {str(e)}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Chart Error')
            self.chart_canvas.draw()

    def _plot_charts(self):
        """Create professional charts with signals based on real trade data - OPTIMIZED VERSION"""
        if not self.results_data:
            return

        self.chart_figure.clear()

        # Create subplot - only price chart and signals (no equity curve here)
        ax1 = self.chart_figure.add_subplot(1, 1, 1)  # Price + Signals

        # Extract trade data to create real price chart and signals
        trades = self.results_data.get('trades', [])
        if trades:
            # OPTIMIZATION 1: Use BB data from backtest results instead of recalculating
            bb_data = self.results_data.get('bb_data')
            if bb_data and 'times' in bb_data:
                print(f"Using pre-calculated BB data: {len(bb_data['prices'])} points")

                # Convert timestamps to datetime
                times = pd.to_datetime(bb_data['times'], unit='ms')
                prices = bb_data['prices']
                sma = bb_data['sma']
                upper_band = bb_data['upper_band']
                lower_band = bb_data['lower_band']

                # OPTIMIZATION 2: Sample data for large datasets (performance boost)
                max_points = 10000  # Maximum points to plot for performance
                if len(prices) > max_points:
                    step = len(prices) // max_points
                    indices = range(0, len(prices), step)
                    times = times[indices]
                    prices = prices[indices]
                    sma = sma[indices]
                    upper_band = upper_band[indices]
                    lower_band = lower_band[indices]
                    print(f"Sampled to {len(prices)} points for performance")

                # Plot price data (much faster with sampled data)
                ax1.plot(times, prices, 'k-', linewidth=0.8, label='Price', alpha=0.8)

                # Plot Bollinger Bands (using pre-calculated values)
                bb_period = self.config.bb_period
                bb_std = self.config.bb_std
                ax1.plot(times, sma, 'b--', linewidth=1, label=f'SMA({bb_period})', alpha=0.7)
                ax1.plot(times, upper_band, 'r-', linewidth=1, label=f'BB Upper (±{bb_std}σ)', alpha=0.6)
                ax1.plot(times, lower_band, 'r-', linewidth=1, label='BB Lower', alpha=0.6)
                ax1.fill_between(times, upper_band, lower_band, alpha=0.1, color='blue', label='BB Zone')

            else:
                print("BB data not available, using fallback method with optimization...")
                # FALLBACK: Load CSV but with optimizations
                dataset = self.dataset_combo.currentText()
                if dataset:
                    csv_path = os.path.join("upload/trades", dataset)
                    if os.path.exists(csv_path):
                        # OPTIMIZATION 3: Sample CSV data during loading
                        print("Loading CSV with sampling for performance...")
                        df_sample = pd.read_csv(csv_path, skiprows=lambda i: i % 100 != 0 if i > 0 else False)  # Sample every 100th row
                        print(f"Loaded {len(df_sample)} sampled rows instead of full dataset")

                        # Convert time column to datetime for plotting
                        df_sample['datetime'] = pd.to_datetime(df_sample['time'], unit='ms')
                        df_sample = df_sample.sort_values('datetime')  # Sort by time

                        # Calculate Bollinger Bands on sampled data only
                        bb_period = min(self.config.bb_period, len(df_sample) // 2)  # Adjust period for sampled data
                        bb_std = self.config.bb_std

                        df_sample['price_sma'] = df_sample['price'].rolling(window=bb_period).mean()
                        df_sample['price_std'] = df_sample['price'].rolling(window=bb_period).std()
                        df_sample['bb_upper'] = df_sample['price_sma'] + (df_sample['price_std'] * bb_std)
                        df_sample['bb_lower'] = df_sample['price_sma'] - (df_sample['price_std'] * bb_std)

                        # Plot sampled price data
                        ax1.plot(df_sample['datetime'], df_sample['price'], 'k-', linewidth=0.8, label='Price (Sampled)', alpha=0.8)

                        # Plot Bollinger Bands
                        ax1.plot(df_sample['datetime'], df_sample['price_sma'], 'b--', linewidth=1, label=f'SMA({bb_period})', alpha=0.7)
                        ax1.plot(df_sample['datetime'], df_sample['bb_upper'], 'r-', linewidth=1, label=f'BB Upper (±{bb_std}σ)', alpha=0.6)
                        ax1.plot(df_sample['datetime'], df_sample['bb_lower'], 'r-', linewidth=1, label='BB Lower', alpha=0.6)
                        ax1.fill_between(df_sample['datetime'], df_sample['bb_upper'], df_sample['bb_lower'], alpha=0.1, color='blue', label='BB Zone')

            # OPTIMIZATION 4: Optimized trade signals processing
            print(f"Processing {len(trades)} trade signals...")
            if len(trades) > 0:
                # Convert all trades to numpy arrays at once (vectorized processing)
                trade_data = []
                for trade in trades:
                    if 'timestamp' in trade and 'entry_price' in trade:
                        time_val = trade['timestamp']
                        if isinstance(time_val, (int, float)):
                            dt = pd.to_datetime(time_val, unit='ms')
                        else:
                            dt = pd.to_datetime(time_val)
                        trade_data.append((dt, trade['entry_price'], trade.get('side', 'N/A')))

                if trade_data:
                    # Sort all trade data at once
                    trade_data.sort()
                    trade_times, entry_prices, sides = zip(*trade_data)

                    # OPTIMIZATION 5: Limit number of trade signals for performance
                    max_signals = 2000  # Limit trade signals for performance
                    if len(trade_times) > max_signals:
                        step = len(trade_times) // max_signals
                        indices = range(0, len(trade_times), step)
                        trade_times = [trade_times[i] for i in indices]
                        entry_prices = [entry_prices[i] for i in indices]
                        sides = [sides[i] for i in indices]
                        print(f"Limited trade signals to {len(trade_times)} for performance")

                    # Vectorized separation of buy/sell signals
                    trade_times = np.array(trade_times)
                    entry_prices = np.array(entry_prices)
                    sides = np.array(sides)

                    buy_mask = sides == 'long'
                    sell_mask = sides == 'short'

                    buy_times = trade_times[buy_mask]
                    buy_prices = entry_prices[buy_mask]
                    sell_times = trade_times[sell_mask]
                    sell_prices = entry_prices[sell_mask]

                    # Plot signals with optimized scatter plot
                    if len(buy_times) > 0:
                        ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=50, label=f'Buy ({len(buy_times)})', zorder=5, alpha=0.8)
                    if len(sell_times) > 0:
                        ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=50, label=f'Sell ({len(sell_times)})', zorder=5, alpha=0.8)

                    print(f"Plotted {len(buy_times)} buy signals and {len(sell_times)} sell signals")

                    # OPTIMIZATION 6: Simplified time axis formatting
                    if len(trade_times) > 0:
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
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
        """Populate trades table efficiently with pagination for large datasets"""
        if not self.results_data or 'trades' not in self.results_data:
            return

        trades = self.results_data['trades']
        if not trades:
            return

        total_trades = len(trades)

        # Show all trades without limits
        displayed_trades = trades
        info_msg = f"Displaying all {total_trades:,} trades"
        self._log(info_msg)

        headers = ['ID', 'Entry Time', 'Exit Time', 'Side', 'Entry $', 'Exit $', 'P&L $', 'P&L %', 'Size $', 'Duration']
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        self.trades_table.setRowCount(len(displayed_trades))

        for i, trade in enumerate(displayed_trades):
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

    def _show_metrics_simple(self):
        """Display performance metrics WITHOUT matplotlib charts to prevent GUI freeze"""
        if not self.results_data:
            return

        r = self.results_data

        # Create simple HTML table without equity curve chart
        metrics_html = f"""
        <style>
        table {{ width: 100%; border-collapse: collapse; }}
        td, th {{ padding: 8px; border: 1px solid #555; }}
        .header {{ background-color: #3c3c3c; font-weight: bold; }}
        </style>

        <h2>Performance Summary</h2>
        <table>
        <tr class="header"><td><b>Metric</b></td><td><b>Value</b></td></tr>
        <tr><td>Total Trades</td><td>{r.get('total', 0)}</td></tr>
        <tr><td>Win Rate</td><td>{r.get('win_rate', 0)*100:.1f}%</td></tr>
        <tr><td>Net P&L</td><td>${r.get('net_pnl', 0):,.2f}</td></tr>
        <tr><td>Return %</td><td>{r.get('net_pnl_percentage', 0):.2f}%</td></tr>
        <tr><td>Max Drawdown</td><td>{r.get('max_drawdown', 0):.2f}%</td></tr>
        <tr><td>Sharpe Ratio</td><td>{r.get('sharpe_ratio', 0):.2f}</td></tr>
        <tr><td>Profit Factor</td><td>{r.get('profit_factor', 0):.2f}</td></tr>
        <tr><td>Winners</td><td>{r.get('total_winning_trades', 0)}</td></tr>
        <tr><td>Losers</td><td>{r.get('total_losing_trades', 0)}</td></tr>
        <tr><td>Avg Win</td><td>${r.get('average_win', 0):.2f}</td></tr>
        <tr><td>Avg Loss</td><td>${r.get('average_loss', 0):.2f}</td></tr>
        <tr><td>Best Trade</td><td>${r.get('largest_win', 0):.2f}</td></tr>
        <tr><td>Worst Trade</td><td>${r.get('largest_loss', 0):.2f}</td></tr>
        </table>

        <h3>Chart: Price and signals displayed in Charts & Signals tab</h3>
        """

        self.metrics_text.setHtml(metrics_html)

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