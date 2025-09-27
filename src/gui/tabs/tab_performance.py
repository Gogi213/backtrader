"""
Performance metrics tab for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtGui import QFont


class PerformanceTab:
    """Performance metrics display with clean separation of concerns"""

    def __init__(self):
        self.widget = None
        self.metrics_text = None
        self._init_ui()

    def _init_ui(self):
        """Initialize performance metrics UI efficiently"""
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setFont(QFont("Consolas", 11))
        layout.addWidget(self.metrics_text)

    def get_widget(self):
        """Get the widget for tab integration"""
        return self.widget

    def update_metrics(self, results_data):
        """Display performance metrics efficiently without charts to prevent GUI freeze"""
        if not results_data:
            self.metrics_text.clear()
            return

        r = results_data

        # Create optimized HTML table without equity curve chart
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

    def clear(self):
        """Clear metrics display"""
        if self.metrics_text:
            self.metrics_text.clear()