"""
High-Performance PyQtGraph Chart Component
Optimized for HFT data visualization with 500k+ data points
Following HFT principles: maximum performance, minimal complexity

Author: HFT System
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from datetime import datetime


class HighPerformanceChart(QWidget):
    """
    Ultra-fast chart component optimized for 500k+ data points
    Uses PyQtGraph for maximum rendering performance
    """

    def __init__(self):
        super().__init__()
        self.plot_widget = None
        self.price_curve = None
        self.bb_upper_curve = None
        self.bb_middle_curve = None
        self.bb_lower_curve = None
        self.bb_fill = None
        self.buy_scatter = None
        self.sell_scatter = None

        # Performance optimizations
        self.max_display_points = 500000  # Maximum points to display
        self.downsample_threshold = 100000  # Start downsampling above this

        self._init_ui()
        self._setup_performance_optimizations()

    def _init_ui(self):
        """Initialize high-performance chart UI"""
        layout = QVBoxLayout(self)

        # Chart info bar
        info_layout = QHBoxLayout()
        self.info_label = QLabel("HFT Chart - Ready")
        self.info_label.setFont(QFont("Consolas", 9))
        self.info_label.setStyleSheet("color: #888888; padding: 5px;")

        self.performance_label = QLabel("Performance: -")
        self.performance_label.setFont(QFont("Consolas", 9))
        self.performance_label.setStyleSheet("color: #888888; padding: 5px;")

        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        info_layout.addWidget(self.performance_label)

        layout.addLayout(info_layout)

        # Create high-performance plot widget with time axis
        axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(
            title="HFT Price Chart - Bollinger Bands Strategy",
            labels={'left': 'Price (USDT)', 'bottom': 'Time'},
            axisItems={'bottom': axis}
        )

        # Configure plot for maximum performance
        self.plot_widget.setBackground('#2b2b2b')
        self.plot_widget.getAxis('left').setTextPen('#ffffff')
        self.plot_widget.getAxis('bottom').setTextPen('#ffffff')
        self.plot_widget.getAxis('left').setPen('#555555')
        self.plot_widget.getAxis('bottom').setPen('#555555')

        # Enable OpenGL for maximum performance
        self.plot_widget.setAntialiasing(False)  # Disable for performance

        layout.addWidget(self.plot_widget)

    def _setup_performance_optimizations(self):
        """Setup performance optimizations for large datasets"""
        # Enable OpenGL rendering for maximum speed
        try:
            pg.setConfigOptions(useOpenGL=True)
            pg.setConfigOptions(enableExperimental=True)
            pg.setConfigOptions(antialias=False)  # Disable for performance
        except:
            pass  # OpenGL might not be available

        # Configure plot widget optimizations
        self.plot_widget.setClipToView(True)  # Only render visible data
        self.plot_widget.setDownsampling(mode='peak')  # Smart downsampling

        # Disable automatic range updates for performance
        self.plot_widget.enableAutoRange(enable=False)

        # Set view limits for large datasets
        self.plot_widget.setLimits(maxXRange=self.max_display_points)

    def update_chart(self, results_data):
        """
        Update chart with new data using maximum performance optimizations

        Args:
            results_data: Dictionary with bb_data and trades
        """
        if not results_data:
            return

        start_time = datetime.now()

        try:
            # Clear previous data efficiently
            self.plot_widget.clear()

            # Get data from results
            bb_data = results_data.get('bb_data')
            trades = results_data.get('trades', [])

            if not bb_data or 'times' not in bb_data:
                self.info_label.setText("No chart data available")
                return

            # Convert and optimize data
            times_ms = np.array(bb_data['times'], dtype=np.float64)
            prices = np.array(bb_data['prices'], dtype=np.float32)

            # Convert milliseconds to seconds for PyQtGraph timestamp handling
            times_sec = times_ms / 1000.0

            data_points = len(times_sec)
            self.info_label.setText(f"Displaying {data_points:,} data points")

            # Apply downsampling if needed for performance
            if data_points > self.downsample_threshold:
                downsample_factor = max(1, data_points // self.max_display_points)
                times_sec = times_sec[::downsample_factor]
                prices = prices[::downsample_factor]

                self.info_label.setText(
                    f"Downsampled {data_points:,} → {len(times_sec):,} points (factor: {downsample_factor}x)"
                )

            # Remove NaN values
            valid_mask = ~(np.isnan(prices) | np.isnan(times_sec))
            times_clean = times_sec[valid_mask]
            prices_clean = prices[valid_mask]

            # Plot main price line with high performance
            self.price_curve = self.plot_widget.plot(
                times_clean, prices_clean,
                pen=pg.mkPen(color='#00aaff', width=1.5),
                name='Price',
                antialias=False  # Disable for performance
            )

            # Plot Bollinger Bands if available
            if all(key in bb_data for key in ['bb_upper', 'bb_lower', 'bb_middle']):
                bb_upper = np.array(bb_data['bb_upper'], dtype=np.float32)
                bb_middle = np.array(bb_data['bb_middle'], dtype=np.float32)
                bb_lower = np.array(bb_data['bb_lower'], dtype=np.float32)

                # Apply same downsampling to BB data
                if data_points > self.downsample_threshold:
                    downsample_factor = max(1, len(bb_upper) // self.max_display_points)
                    bb_upper = bb_upper[::downsample_factor]
                    bb_middle = bb_middle[::downsample_factor]
                    bb_lower = bb_lower[::downsample_factor]

                # Ensure BB data matches times length
                min_len = min(len(times_clean), len(bb_upper), len(bb_middle), len(bb_lower))
                times_bb = times_clean[:min_len]
                bb_upper = bb_upper[:min_len]
                bb_middle = bb_middle[:min_len]
                bb_lower = bb_lower[:min_len]

                # Remove NaN values from BB data
                bb_valid_mask = ~(np.isnan(bb_upper) | np.isnan(bb_middle) | np.isnan(bb_lower))
                times_bb_clean = times_bb[bb_valid_mask]
                bb_upper_clean = bb_upper[bb_valid_mask]
                bb_middle_clean = bb_middle[bb_valid_mask]
                bb_lower_clean = bb_lower[bb_valid_mask]

                # Upper band
                self.bb_upper_curve = self.plot_widget.plot(
                    times_bb_clean, bb_upper_clean,
                    pen=pg.mkPen(color='#ff4444', width=1, style=Qt.PenStyle.DashLine),
                    name='BB Upper',
                    antialias=False
                )

                # Middle band (SMA)
                self.bb_middle_curve = self.plot_widget.plot(
                    times_bb_clean, bb_middle_clean,
                    pen=pg.mkPen(color='#ffaa00', width=1),
                    name='BB Middle',
                    antialias=False
                )

                # Lower band
                self.bb_lower_curve = self.plot_widget.plot(
                    times_bb_clean, bb_lower_clean,
                    pen=pg.mkPen(color='#44ff44', width=1, style=Qt.PenStyle.DashLine),
                    name='BB Lower',
                    antialias=False
                )

                # Fill between bands for visual appeal (with transparency)
                try:
                    self.bb_fill = pg.FillBetweenItem(
                        self.bb_upper_curve,
                        self.bb_lower_curve,
                        brush=pg.mkBrush(color=(100, 149, 237, 30))  # Light blue with transparency
                    )
                    self.plot_widget.addItem(self.bb_fill)
                except:
                    pass  # Skip fill if it causes performance issues

            # Add trading signals with high performance
            if trades:
                self._add_trading_signals(trades, times_clean)

            # Auto-range to fit all data
            self.plot_widget.autoRange()

            # Update performance info
            end_time = datetime.now()
            render_time = (end_time - start_time).total_seconds()
            points_per_sec = len(times_clean) / render_time if render_time > 0 else 0

            self.performance_label.setText(
                f"Rendered in {render_time:.3f}s ({points_per_sec:,.0f} pts/sec)"
            )

        except Exception as e:
            self.info_label.setText(f"Chart error: {str(e)}")

    def _add_trading_signals(self, trades, chart_times):
        """
        Add trading signals as scatter plots using real timestamps

        Args:
            trades: List of trade dictionaries with timestamp and entry_price
            chart_times: Array of chart timestamps in seconds for mapping
        """
        if not trades or len(chart_times) == 0:
            return

        # Separate buy/sell signals
        buy_times = []
        buy_prices = []
        sell_times = []
        sell_prices = []

        # Map trades to their actual timestamps
        for trade in trades:
            entry_time = trade.get('timestamp')  # in milliseconds
            entry_price = trade.get('entry_price', 0)
            side = trade.get('side', 'unknown')

            if entry_time and entry_price:
                # Convert milliseconds to seconds to match chart_times
                trade_time_sec = entry_time / 1000.0

                # Check if trade time is within chart range
                if chart_times[0] <= trade_time_sec <= chart_times[-1]:
                    if side == 'long':
                        buy_times.append(trade_time_sec)
                        buy_prices.append(entry_price)
                    else:
                        sell_times.append(trade_time_sec)
                        sell_prices.append(entry_price)

        # Limit number of signals for performance
        max_signals = 1000
        if len(buy_times) > max_signals:
            step = len(buy_times) // max_signals
            buy_times = buy_times[::step]
            buy_prices = buy_prices[::step]

        if len(sell_times) > max_signals:
            step = len(sell_times) // max_signals
            sell_times = sell_times[::step]
            sell_prices = sell_prices[::step]

        # Add buy signals (green triangles pointing up)
        if buy_times:
            self.buy_scatter = self.plot_widget.plot(
                buy_times, buy_prices,
                pen=None,
                symbol='t',  # Triangle up
                symbolSize=10,
                symbolBrush=pg.mkBrush(color='#00ff00'),
                symbolPen=pg.mkPen(color='#008800', width=1),
                name=f'Long Entry ({len(buy_times)})'
            )

        # Add sell signals (red triangles pointing down)
        if sell_times:
            self.sell_scatter = self.plot_widget.plot(
                sell_times, sell_prices,
                pen=None,
                symbol='t1',  # Triangle down
                symbolSize=10,
                symbolBrush=pg.mkBrush(color='#ff0000'),
                symbolPen=pg.mkPen(color='#880000', width=1),
                name=f'Short Entry ({len(sell_times)})'
            )

    def clear(self):
        """Clear chart data efficiently"""
        if self.plot_widget:
            self.plot_widget.clear()
            self.info_label.setText("Chart cleared")
            self.performance_label.setText("Performance: -")

    def get_widget(self):
        """Get the widget for integration"""
        return self