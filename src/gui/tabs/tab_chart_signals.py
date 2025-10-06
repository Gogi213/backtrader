"""
Chart signals tab for Professional GUI Application
Ultra-high-performance PyQtGraph implementation optimized for 500k+ data points
Extracted from gui_visualizer following HFT principles: maximum performance, minimal complexity
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from src.gui.charts.pyqtgraph_chart import HighPerformanceChart


class ChartSignalsTab:
    """Ultra-high-performance chart display with PyQtGraph for 500k+ data points"""

    def __init__(self):
        self.widget = None
        self.chart = None
        self._init_ui()

    def _init_ui(self):
        """Initialize chart UI with high-performance PyQtGraph"""
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        # Create high-performance chart component
        self.chart = HighPerformanceChart()
        layout.addWidget(self.chart)

    def get_widget(self):
        """Get the widget for tab integration"""
        return self.widget

    def update_chart(self, results_data):
        """Update chart with ultra-high-performance rendering"""
        if not self.chart:
            return
            
        # Performance optimization: limit data points for large datasets
        if results_data and 'trades' in results_data:
            trades = results_data['trades']
            if len(trades) > 5000:  # Limit to prevent GUI freezing
                print(f"Performance optimization: Limiting chart to first 5000 of {len(trades)} trades")
                # Create a copy with limited trades
                limited_results = results_data.copy()
                limited_results['trades'] = trades[:5000]
                self.chart.update_chart(limited_results)
            else:
                self.chart.update_chart(results_data)
        else:
            self.chart.update_chart(results_data)

    def clear(self):
        """Clear chart display efficiently"""
        if self.chart:
            self.chart.clear()