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
        if self.chart:
            self.chart.update_chart(results_data)

    def clear(self):
        """Clear chart display efficiently"""
        if self.chart:
            self.chart.clear()