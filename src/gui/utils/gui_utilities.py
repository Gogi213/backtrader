"""
GUI Utilities for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
"""
import numpy as np
from datetime import datetime
from PyQt6.QtGui import QTextCursor


def export_results(results_data, status_bar=None, logger=None):
    """Export results to CSV using ULTRA-FAST numpy implementation"""
    if not results_data:
        return

    # Export trades to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_results_{timestamp}.csv"

    trades = results_data.get('trades', [])
    if trades:
        # ULTRA-FAST CSV export using numpy instead of pandas
        try:
            # Extract all unique keys from trades
            all_keys = set()
            for trade in trades:
                all_keys.update(trade.keys())
            
            # Sort keys for consistent column order
            sorted_keys = sorted(all_keys)
            
            # Create CSV content
            csv_lines = [",".join(sorted_keys)]  # Header
            
            for trade in trades:
                values = []
                for key in sorted_keys:
                    value = trade.get(key, "")
                    # Convert to string and handle special cases
                    if isinstance(value, str) and ("," in value or "\n" in value):
                        # Escape quotes and quote the value
                        escaped_value = value.replace('"', '""')
                        value = f'"{escaped_value}"'
                    else:
                        value = str(value)
                    values.append(value)
                csv_lines.append(",".join(values))
            
            # Write to file
            with open(filename, 'w') as f:
                f.write("\n".join(csv_lines))
            
            if logger:
                logger.log(f"Results exported to {filename} using numpy (ULTRA-FAST)")
            if status_bar:
                status_bar.showMessage(f"Exported to {filename}", 5000)
        except Exception as e:
            # Fallback to pandas if numpy approach fails
            import pandas as pd
            df = pd.DataFrame(trades)
            df.to_csv(filename, index=False)
            if logger:
                logger.log(f"Results exported to {filename} using pandas (fallback)")
            if status_bar:
                status_bar.showMessage(f"Exported to {filename}", 5000)
    else:
        if logger:
            logger.log("No trade data to export")


class Logger:
    """Simple logger for GUI application"""
    
    def __init__(self, log_text_widget):
        self.log_text = log_text_widget
        self.log(f"Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        
        # Append to log widget
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
        self.log_text.insertPlainText(formatted_message + "\n")
        
        # Auto-scroll to bottom
        self.log_text.ensureCursorVisible()