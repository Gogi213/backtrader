"""
High-Performance PyQtGraph Chart Component with CRITICAL AUTORANGE FIX
Optimized for HFT data visualization with 500k+ data points
Following HFT principles: maximum performance, minimal complexity

Author: HFT System
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsRectItem
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

        # Performance optimizations - RESTORED normal limits
        self.max_display_points = 50000  # Normal limit for all data
        self.downsample_threshold = 100000  # Only downsample if absolutely necessary

        self._init_ui()
        self._setup_performance_optimizations()

    def _init_ui(self):
        """Initialize high-performance chart UI"""
        layout = QVBoxLayout(self)

        # Chart info bar
        info_layout = QHBoxLayout()
        self.info_label = QLabel("HFT Chart - Ready")
        self.info_label.setFont(QFont("Consolas", 7))
        self.info_label.setStyleSheet("color: #888888; padding: 4px;")

        self.performance_label = QLabel("Performance: -")
        self.performance_label.setFont(QFont("Consolas", 7))
        self.performance_label.setStyleSheet("color: #888888; padding: 4px;")

        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        info_layout.addWidget(self.performance_label)

        layout.addLayout(info_layout)

        # Create high-performance plot widget with time axis and right price axis
        axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(
            title="HFT Price Chart - Strategy",
            labels={'right': 'Price (USDT)', 'bottom': 'Time'},
            axisItems={'bottom': axis}
        )

        # Configure plot for maximum performance
        self.plot_widget.setBackground('#2b2b2b')
        self.plot_widget.getAxis('right').setTextPen('#ffffff')
        self.plot_widget.getAxis('bottom').setTextPen('#ffffff')
        self.plot_widget.getAxis('right').setPen('#555555')
        self.plot_widget.getAxis('bottom').setPen('#555555')

        # Enable OpenGL for maximum performance
        self.plot_widget.setAntialiasing(False)  # Disable for performance

        # Setup scroll-based scaling
        self._setup_scroll_scaling()

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

    def _setup_scroll_scaling(self):
        """Setup scroll-based scaling for horizontal and vertical zoom"""
        # Override the default wheel event
        self.plot_widget.plotItem.vb.wheelEvent = self._custom_wheel_event

        # Scaling parameters
        self.zoom_factor = 0.1  # 10% zoom per scroll

    def _custom_wheel_event(self, event):
        """
        Custom wheel event for scaling:
        - Normal scroll: horizontal zoom (time axis)
        - Shift+scroll: vertical zoom (price axis)
        """
        delta = event.delta()
        view_box = self.plot_widget.plotItem.vb

        # Get current view range
        x_range, y_range = view_box.viewRange()

        # Calculate zoom (positive delta = zoom in, negative = zoom out)
        zoom_scale = 1.0 + (self.zoom_factor if delta > 0 else -self.zoom_factor)

        # Get mouse position
        pos = event.pos()
        mouse_point = view_box.mapSceneToView(pos)
        mouse_x, mouse_y = mouse_point.x(), mouse_point.y()

        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            # Vertical scaling (price axis)
            y_center = mouse_y
            y_width = y_range[1] - y_range[0]
            new_y_width = y_width / zoom_scale

            # Center zoom around mouse position
            y_offset = (y_center - y_range[0]) / y_width
            new_y_min = y_center - new_y_width * y_offset
            new_y_max = new_y_min + new_y_width

            view_box.setYRange(new_y_min, new_y_max, padding=0)
        else:
            # Horizontal scaling (time axis) - default
            x_center = mouse_x
            x_width = x_range[1] - x_range[0]
            new_x_width = x_width / zoom_scale

            # Center zoom around mouse position
            x_offset = (x_center - x_range[0]) / x_width
            new_x_min = x_center - new_x_width * x_offset
            new_x_max = new_x_min + new_x_width

            view_box.setXRange(new_x_min, new_x_max, padding=0)

        event.accept()

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

            # Get data from results - use indicator_data
            chart_data = results_data.get('indicator_data')
            trades = results_data.get('trades', [])

            if not chart_data or 'times' not in chart_data:
                self.info_label.setText("No chart data available")
                return

            # CRITICAL FIX: Data now comes in milliseconds from backtest
            times_ms = np.array(chart_data['times'], dtype=np.float64)
            prices = np.array(chart_data['prices'], dtype=np.float32)

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

            # Plot candlestick chart with calm colors (replacing line chart)
            # Check if OHLC data is available for candlesticks
            if all(key in chart_data for key in ['open', 'high', 'low', 'close']):
                open_prices = np.array(chart_data['open'], dtype=np.float32)
                high_prices = np.array(chart_data['high'], dtype=np.float32)
                low_prices = np.array(chart_data['low'], dtype=np.float32)
                close_prices = np.array(chart_data['close'], dtype=np.float32)

                # Apply same downsampling to OHLC data if needed
                if data_points > self.downsample_threshold:
                    open_prices = open_prices[::downsample_factor]
                    high_prices = high_prices[::downsample_factor]
                    low_prices = low_prices[::downsample_factor]
                    close_prices = close_prices[::downsample_factor]

                # Ensure OHLC data matches times length
                min_len = min(len(times_clean), len(open_prices), len(high_prices), len(low_prices), len(close_prices))
                times_ohlc = times_clean[:min_len]
                open_data = open_prices[:min_len]
                high_data = high_prices[:min_len]
                low_data = low_prices[:min_len]
                close_data = close_prices[:min_len]

                # Remove NaN values from OHLC data
                ohlc_valid_mask = ~(np.isnan(open_data) | np.isnan(high_data) | np.isnan(low_data) | np.isnan(close_data))
                times_ohlc_clean = times_ohlc[ohlc_valid_mask]
                open_clean = open_data[ohlc_valid_mask]
                high_clean = high_data[ohlc_valid_mask]
                low_clean = low_data[ohlc_valid_mask]
                close_clean = close_data[ohlc_valid_mask]

                print(f"CHART: Plotting {len(times_ohlc_clean)} candlesticks")

                # Create candlestick bars manually using calm colors
                self._draw_candlesticks(times_ohlc_clean, open_clean, high_clean, low_clean, close_clean)

            else:
                # Fallback to line chart if OHLC data not available
                print("CHART: OHLC data not available, using fallback line chart")
                self.price_curve = self.plot_widget.plot(
                    times_clean, prices_clean,
                    pen=pg.mkPen(color='#6B9BD2', width=1.5),  # Calm blue color
                    name='Price',
                    antialias=False  # Disable for performance
                )

            # Plot strategy indicators if available
            if all(key in chart_data for key in ['bb_upper', 'bb_lower', 'bb_middle']):
                bb_upper = np.array(chart_data['bb_upper'], dtype=np.float32)
                bb_middle = np.array(chart_data['bb_middle'], dtype=np.float32)
                bb_lower = np.array(chart_data['bb_lower'], dtype=np.float32)

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

            # Plot Fair Value for Hierarchical Mean Reversion strategy if available
            if 'fair_values' in chart_data:
                fair_values = np.array(chart_data['fair_values'], dtype=np.float32)

                # Apply same downsampling
                if data_points > self.downsample_threshold:
                    fair_values = fair_values[::downsample_factor]

                # Align lengths
                min_len = min(len(times_clean), len(fair_values))
                fair_values = fair_values[:min_len]
                times_fv = times_clean[:min_len]

                # Remove NaN/inf
                fv_valid_mask = ~(np.isnan(fair_values) | np.isinf(fair_values))
                fair_values_clean = fair_values[fv_valid_mask]
                times_fv_clean = times_fv[fv_valid_mask]

                # Plot Fair Value line
                self.plot_widget.plot(
                    times_fv_clean, fair_values_clean,
                    pen=pg.mkPen(color='#FFA500', width=2, style=Qt.PenStyle.DashLine),  # Orange dashed
                    name='Fair Value',
                    antialias=False
                )

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
        print(f"CHART: chart_times range: {chart_times[0]:.2f} to {chart_times[-1]:.2f} seconds")

        # Process each trade for entry AND exit with type classification
        trades_in_range = 0
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
                    trades_in_range += 1
                    if side == 'long':
                        long_entry_times.append(entry_time_sec)
                        long_entry_prices.append(entry_price)
                    elif side == 'short':
                        short_entry_times.append(entry_time_sec)
                        short_entry_prices.append(entry_price)
                else:
                    if trades_in_range == 0:  # Print only first out-of-range trade
                        print(f"CHART: Trade out of range: entry_time={entry_time_sec:.2f}s, chart range=[{chart_times[0]:.2f}, {chart_times[-1]:.2f}]")

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

        print(f"CHART: Trades in chart time range: {trades_in_range}/{len(trades)}")
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

    def _draw_candlesticks(self, times, open_prices, high_prices, low_prices, close_prices):
        """
        Draw candlestick chart manually using PyQtGraph with calm colors

        Args:
            times: Array of timestamps in seconds
            open_prices: Array of open prices
            high_prices: Array of high prices
            low_prices: Array of low prices
            close_prices: Array of close prices
        """
        if len(times) == 0:
            return

        candles_count = len(times)

        # OPTIMIZED: Use BarGraphItem for efficient rendering of large datasets
        # This approach uses only 3 objects instead of 17k*2 individual objects

        # Calculate candle width based on time interval (approximately 80% of interval)
        if len(times) > 1:
            avg_interval = np.mean(np.diff(times))
            candle_width = avg_interval * 0.8
        else:
            candle_width = 60  # Default 1 minute

        print(f"CHART DEBUG: Calculated candle_width: {candle_width}")

        # Calm color scheme for candlesticks
        bullish_body_color = '#7FB069'    # Calm green
        bullish_wick_color = '#5D8A4A'    # Darker green for wicks
        bearish_body_color = '#D66853'    # Calm red/orange
        bearish_wick_color = '#B54C3A'    # Darker red for wicks

        # Prepare vectorized data arrays
        bullish_x, bullish_heights, bullish_y = [], [], []
        bearish_x, bearish_heights, bearish_y = [], [], []
        wick_x, wick_y = [], []

        print(f"CHART DEBUG: Processing candlestick data into vectorized arrays...")

        for i, (time, open_val, high_val, low_val, close_val) in enumerate(
            zip(times, open_prices, high_prices, low_prices, close_prices)
        ):
            # Progress logging every 5000 candles (less frequent for performance)
            if i % 5000 == 0:
                print(f"CHART DEBUG: Processing candle {i}/{candles_count} ({i/candles_count*100:.1f}%)")

            # Skip invalid data
            if np.isnan(open_val) or np.isnan(high_val) or np.isnan(low_val) or np.isnan(close_val):
                continue

            # Determine if bullish (green) or bearish (red)
            is_bullish = close_val >= open_val

            # Calculate body boundaries
            body_top = max(open_val, close_val)
            body_bottom = min(open_val, close_val)
            body_height = body_top - body_bottom

            # Avoid zero-height bodies for doji candles
            if body_height == 0:
                body_height = (high_val - low_val) * 0.02  # 2% of wick height
                body_bottom = min(open_val, close_val) - body_height/2

            # Add wick data (vectorized) - using np.nan for separation
            wick_x.extend([time, time, np.nan])  # np.nan separates line segments
            wick_y.extend([low_val, high_val, np.nan])

            # Collect body data for BarGraphItem
            if is_bullish:
                bullish_x.append(time)
                bullish_heights.append(body_height)
                bullish_y.append(body_bottom)
            else:
                bearish_x.append(time)
                bearish_heights.append(body_height)
                bearish_y.append(body_bottom)

        print(f"CHART DEBUG: Processed {len(bullish_x)} bullish and {len(bearish_x)} bearish bodies")

        # Draw all wicks at once using PlotCurveItem (super efficient!)
        if wick_x:
            print(f"CHART DEBUG: Drawing {len(wick_x)//3} wicks using single PlotCurveItem...")
            wick_curve = self.plot_widget.plot(
                wick_x, wick_y,
                pen=pg.mkPen(color='#8A8A8A', width=1),
                antialias=False,
                connect='finite'  # Connects finite values, skips None separators
            )
            print(f"CHART DEBUG: All wicks drawn in single operation!")

        # Draw all bullish bodies at once using BarGraphItem
        if bullish_x:
            print(f"CHART DEBUG: Drawing {len(bullish_x)} bullish bodies using BarGraphItem...")
            bullish_bars = pg.BarGraphItem(
                x=bullish_x,
                height=bullish_heights,
                y0=bullish_y,
                width=candle_width,
                brush=bullish_body_color,
                pen=pg.mkPen(color=bullish_wick_color, width=1)
            )
            self.plot_widget.addItem(bullish_bars)
            print(f"CHART DEBUG: All bullish bodies drawn in single operation!")

        # Draw all bearish bodies at once using BarGraphItem
        if bearish_x:
            print(f"CHART DEBUG: Drawing {len(bearish_x)} bearish bodies using BarGraphItem...")
            bearish_bars = pg.BarGraphItem(
                x=bearish_x,
                height=bearish_heights,
                y0=bearish_y,
                width=candle_width,
                brush=bearish_body_color,
                pen=pg.mkPen(color=bearish_wick_color, width=1)
            )
            self.plot_widget.addItem(bearish_bars)
            print(f"CHART DEBUG: All bearish bodies drawn in single operation!")

        print(f"CHART: Drew {len(bullish_x)} bullish and {len(bearish_x)} bearish candlesticks using optimized BarGraphItem")

    def clear(self):
        """Clear chart data efficiently"""
        if self.plot_widget:
            self.plot_widget.clear()
            self.info_label.setText("Chart cleared")
            self.performance_label.setText("Performance: -")

    def get_widget(self):
        """Get the widget for integration"""
        return self