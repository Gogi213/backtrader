"""
GUI Utilities for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
"""
import pandas as pd
from datetime import datetime
from PyQt6.QtGui import QTextCursor


def export_results(results_data, status_bar=None, logger=None):
    """Export results to CSV"""
    if not results_data:
        return

    # Export trades to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_results_{timestamp}.csv"

    trades = results_data.get('trades', [])
    if trades:
        df = pd.DataFrame(trades)
        df.to_csv(filename, index=False)
        if logger:
            logger.log(f"Results exported to {filename}")
        if status_bar:
            status_bar.showMessage(f"Exported to {filename}", 5000)
    else:
        if logger:
            logger.log("No trade data to export")


class Logger:
    """Efficient logging utility for GUI"""

    def __init__(self, log_text_widget):
        self.log_text = log_text_widget

    def log(self, message):
        """Add log entry efficiently"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)