"""
High-Performance PyQtGraph Chart Component with CRITICAL AUTORANGE FIX
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
        self.long_tp_scatter = None
        self.long_sl_scatter = None
        self.short_tp_scatter = None
        self.short_sl_scatter = None

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

        # CRITICAL FIX: Don't disable autoRange during initialization
        # Note: autoRange will be properly managed in update_chart

        # Set view limits for large datasets
        self.plot_widget.setLimits(maxXRange=self.max_display_points)

    def update_chart(self, results_data):
        """
        Update chart with new data using maximum performance optimizations

        Args:
            results_data: Dictionary with bb_data and trades
        """
        print("CHART: update_chart() called!")

        if not results_data:
            print("CHART: No results_data, returning")
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

            # CRITICAL FIX: Data now comes in milliseconds from backtest
            times_ms = np.array(bb_data['times'], dtype=np.float64)
            prices = np.array(bb_data['prices'], dtype=np.float32)

            # Convert milliseconds to seconds for PyQtGraph timestamp handling
            times_sec = times_ms / 1000.0

            data_points = len(times_sec)
            print(f"CHART: Received {data_points} data points")

            self.info_label.setText(f"Displaying {data_points:,} data points")

            # Validate data
            if len(times_sec) != len(prices):
                self.info_label.setText(f"Data mismatch: {len(times_sec)} times vs {len(prices)} prices")
                return

            if data_points == 0:
                self.info_label.setText("No data: BB period too large or no valid data")
                return

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

            print(f"CHART: Plotting {len(times_clean)} clean data points")

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

            # CRITICAL FIX: Proper autoRange handling
            print("CHART: Applying autoRange fix...")

            # Enable autoRange for both axes
            self.plot_widget.enableAutoRange(True, True)

            # Apply autoRange to fit all data
            self.plot_widget.autoRange()

            print("CHART: AutoRange applied successfully")

            # Update performance info
            end_time = datetime.now()
            render_time = (end_time - start_time).total_seconds()
            points_per_sec = len(times_clean) / render_time if render_time > 0 else 0

            self.performance_label.setText(
                f"Rendered in {render_time:.3f}s ({points_per_sec:,.0f} pts/sec)"
            )

        except Exception as e:
            self.info_label.setText(f"Chart error: {str(e)}")
            print(f"CHART ERROR: {e}")

    def _add_trading_signals(self, trades, chart_times):
        """
        Add trading signals with ENTRIES and EXITS using real timestamps

        Args:
            trades: List of trade dictionaries with entry/exit timestamps and prices
            chart_times: Array of chart timestamps in seconds for mapping
        """
        if not trades or len(chart_times) == 0:
            return

        # Separate signals: entries and exits by type
        long_entry_times, long_entry_prices = [], []
        long_tp_times, long_tp_prices = [], []  # Take profit exits
        long_sl_times, long_sl_prices = [], []  # Stop loss exits
        short_entry_times, short_entry_prices = [], []
        short_tp_times, short_tp_prices = [], []  # Take profit exits
        short_sl_times, short_sl_prices = [], []  # Stop loss exits

        print(f"CHART: Processing {len(trades)} trades for ENTRY and EXIT signals with types")

        # Process each trade for entry AND exit with type classification
        for trade in trades:
            # ENTRY processing
            entry_time = trade.get('timestamp')  # Entry time in milliseconds
            entry_price = trade.get('entry_price', 0)

            # EXIT processing
            exit_time = trade.get('exit_timestamp')  # Exit time in milliseconds
            exit_price = trade.get('exit_price', 0)
            exit_reason = trade.get('exit_reason', 'unknown')

            side = trade.get('side', 'unknown')

            # Process ENTRY signals
            if entry_time and entry_price:
                entry_time_sec = entry_time / 1000.0

                if chart_times[0] <= entry_time_sec <= chart_times[-1]:
                    if side == 'long':
                        long_entry_times.append(entry_time_sec)
                        long_entry_prices.append(entry_price)
                    elif side == 'short':
                        short_entry_times.append(entry_time_sec)
                        short_entry_prices.append(entry_price)

            # Process EXIT signals by TYPE
            if exit_time and exit_price:
                exit_time_sec = exit_time / 1000.0

                if chart_times[0] <= exit_time_sec <= chart_times[-1]:
                    if side == 'long':
                        if 'stop_loss' in exit_reason:
                            long_sl_times.append(exit_time_sec)
                            long_sl_prices.append(exit_price)
                        else:  # take_profit_sma
                            long_tp_times.append(exit_time_sec)
                            long_tp_prices.append(exit_price)
                    elif side == 'short':
                        if 'stop_loss' in exit_reason:
                            short_sl_times.append(exit_time_sec)
                            short_sl_prices.append(exit_price)
                        else:  # take_profit_sma
                            short_tp_times.append(exit_time_sec)
                            short_tp_prices.append(exit_price)

        print(f"CHART: Long signals - {len(long_entry_times)} entries, {len(long_tp_times)} TPs, {len(long_sl_times)} SLs")
        print(f"CHART: Short signals - {len(short_entry_times)} entries, {len(short_tp_times)} TPs, {len(short_sl_times)} SLs")

        # CRITICAL FIX: Corrected triangle directions based on user feedback
        # User observed: Green triangles DOWN on lower BB, Red triangles UP on upper BB

        # LONG ENTRIES: Green triangles DOWN (at lower BB)
        if long_entry_times:
            self.buy_scatter = self.plot_widget.plot(
                long_entry_times, long_entry_prices,
                pen=None,
                symbol='t1',  # Triangle DOWN for long entries (at lower BB)
                symbolSize=12,
                symbolBrush=pg.mkBrush(color='#00ff00'),  # Green
                symbolPen=pg.mkPen(color='#008800', width=1),
                name=f'Long Entry ({len(long_entry_times)})'
            )

        # LONG TAKE PROFITS: Green squares (at SMA center)
        if long_tp_times:
            self.long_tp_scatter = self.plot_widget.plot(
                long_tp_times, long_tp_prices,
                pen=None,
                symbol='s',  # Square for take profits
                symbolSize=10,
                symbolBrush=pg.mkBrush(color='#44ff44'),  # Light green
                symbolPen=pg.mkPen(color='#008800', width=1),
                name=f'Long TP ({len(long_tp_times)})'
            )

        # LONG STOP LOSSES: Red X marks
        if long_sl_times:
            self.long_sl_scatter = self.plot_widget.plot(
                long_sl_times, long_sl_prices,
                pen=None,
                symbol='x',  # X for stop losses
                symbolSize=12,
                symbolBrush=pg.mkBrush(color='#ff4444'),  # Light red
                symbolPen=pg.mkPen(color='#aa0000', width=2),
                name=f'Long SL ({len(long_sl_times)})'
            )

        # SHORT ENTRIES: Red triangles UP (at upper BB)
        if short_entry_times:
            self.sell_scatter = self.plot_widget.plot(
                short_entry_times, short_entry_prices,
                pen=None,
                symbol='t',  # Triangle UP for short entries (at upper BB)
                symbolSize=12,
                symbolBrush=pg.mkBrush(color='#ff0000'),  # Red
                symbolPen=pg.mkPen(color='#880000', width=1),
                name=f'Short Entry ({len(short_entry_times)})'
            )

        # SHORT TAKE PROFITS: Green squares (at SMA center)
        if short_tp_times:
            self.short_tp_scatter = self.plot_widget.plot(
                short_tp_times, short_tp_prices,
                pen=None,
                symbol='s',  # Square for take profits
                symbolSize=10,
                symbolBrush=pg.mkBrush(color='#44ff44'),  # Light green
                symbolPen=pg.mkPen(color='#008800', width=1),
                name=f'Short TP ({len(short_tp_times)})'
            )

        # SHORT STOP LOSSES: Red X marks
        if short_sl_times:
            self.short_sl_scatter = self.plot_widget.plot(
                short_sl_times, short_sl_prices,
                pen=None,
                symbol='x',  # X for stop losses
                symbolSize=12,
                symbolBrush=pg.mkBrush(color='#ff4444'),  # Light red
                symbolPen=pg.mkPen(color='#aa0000', width=2),
                name=f'Short SL ({len(short_sl_times)})'
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