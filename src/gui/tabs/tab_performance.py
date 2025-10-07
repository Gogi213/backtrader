"""
Performance metrics tab for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
from datetime import datetime


class PerformanceTab:
    """Performance metrics display with equity curve visualization"""

    def __init__(self):
        self.widget = None
        self.metrics_text = None
        self.equity_plot = None
        self._init_ui()

    def _init_ui(self):
        """Initialize performance metrics UI with equity curve"""
        self.widget = QWidget()
        main_layout = QHBoxLayout(self.widget)

        # LEFT SIDE: Metrics table
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setFont(QFont("Consolas", 9))
        self.metrics_text.setMaximumWidth(280)  # Fixed width for metrics (30% narrower)
        left_layout.addWidget(self.metrics_text)

        # RIGHT SIDE: Equity curve
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Equity curve title
        equity_title = QLabel("Equity Curve")
        equity_title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        right_layout.addWidget(equity_title)

        # Equity curve plot
        self.equity_plot = pg.PlotWidget(
            title="Portfolio Value Over Time",
            labels={'left': 'Portfolio Value ($)', 'bottom': 'Time'}
        )
        self.equity_plot.setBackground('#2b2b2b')
        self.equity_plot.getAxis('left').setTextPen('#ffffff')
        self.equity_plot.getAxis('bottom').setTextPen('#ffffff')
        self.equity_plot.getAxis('left').setPen('#555555')
        self.equity_plot.getAxis('bottom').setPen('#555555')
        right_layout.addWidget(self.equity_plot)

        # Add widgets to main layout
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_layout.setStretch(0, 1)  # Metrics take 1/4 (30% narrower)
        main_layout.setStretch(1, 3)  # Equity curve takes 3/4

    def get_widget(self):
        """Get the widget for tab integration"""
        return self.widget

    def update_metrics(self, results_data):
        """Display performance metrics and equity curve"""
        if not results_data:
            self.metrics_text.clear()
            if self.equity_plot:
                self.equity_plot.clear()
            return

        r = results_data

        # LEFT SIDE: Create optimized HTML table (removed chart message)
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
        <tr><td>Loose Streak</td><td>{r.get('loose_streak', 0)}</td></tr>
        </table>
        """

        self.metrics_text.setHtml(metrics_html)

        # RIGHT SIDE: Build equity curve
        self._update_equity_curve(results_data)

    def _update_equity_curve(self, results_data):
        """Build and display equity curve from trades data"""
        if not self.equity_plot:
            return

        # Clear previous equity curve
        self.equity_plot.clear()

        trades = results_data.get('trades', [])
        initial_capital = 10000.0  # Default initial capital

        if not trades:
            return

        # Performance optimization: downsample for large datasets
        max_equity_points = 5000
        if len(trades) > max_equity_points:
            # Calculate downsample factor
            downsample_factor = len(trades) // max_equity_points
            if downsample_factor < 1:
                downsample_factor = 1
            
            # Downsample trades
            downsampled_trades = trades[::downsample_factor]
            print(f"Performance optimization: Downsampling equity curve from {len(trades)} to {len(downsampled_trades)} points")
            trades = downsampled_trades

        # Calculate cumulative equity over time
        times = []
        equity_values = []
        current_equity = initial_capital

        # Start with initial capital at first trade time
        first_trade = trades[0]
        start_time = first_trade.get('timestamp', 0)
        # Handle different timestamp formats
        if hasattr(start_time, 'timestamp'):
            start_time = start_time.timestamp()
        elif isinstance(start_time, (int, float)) and start_time > 1e10:  # milliseconds
            start_time = start_time / 1000.0
        
        times.append(start_time)
        equity_values.append(current_equity)

        # Process each trade to build equity curve
        for trade in trades:
            trade_time = trade.get('timestamp', 0)
            # Handle different timestamp formats
            if hasattr(trade_time, 'timestamp'):
                trade_time = trade_time.timestamp()
            elif isinstance(trade_time, (int, float)) and trade_time > 1e10:  # milliseconds
                trade_time = trade_time / 1000.0
                
            trade_pnl = trade.get('pnl', 0)

            # Add equity point after trade
            current_equity += trade_pnl
            times.append(trade_time)
            equity_values.append(current_equity)

        # Convert to numpy arrays for plotting
        times_array = np.array(times)
        equity_array = np.array(equity_values)

        print(f"EQUITY: Plotting {len(times)} equity points")
        print(f"EQUITY: Range ${equity_array.min():.2f} - ${equity_array.max():.2f}")

        # Plot equity curve
        self.equity_plot.plot(
            times_array, equity_array,
            pen=pg.mkPen(color='#00aaff', width=2),
            name='Portfolio Value'
        )

        # Add horizontal line at initial capital
        self.equity_plot.addLine(
            y=initial_capital,
            pen=pg.mkPen(color='#888888', style=pg.QtCore.Qt.PenStyle.DashLine),
            label='Initial Capital'
        )

        # Add horizontal line at final value
        final_value = equity_array[-1]
        color = '#00ff00' if final_value > initial_capital else '#ff0000'
        self.equity_plot.addLine(
            y=final_value,
            pen=pg.mkPen(color=color, style=pg.QtCore.Qt.PenStyle.DashLine),
            label=f'Final: ${final_value:.2f}'
        )

        # Set auto-range
        self.equity_plot.enableAutoRange()
        self.equity_plot.autoRange()

    def clear(self):
        """Clear metrics display and equity curve"""
        if self.metrics_text:
            self.metrics_text.clear()
        if self.equity_plot:
            self.equity_plot.clear()