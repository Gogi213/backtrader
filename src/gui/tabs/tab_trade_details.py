"""
Trade details tab for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
"""
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QLineEdit, QCheckBox, QHeaderView
)
from PyQt6.QtGui import QColor


class TradeDetailsTab:
    """Trade details display with efficient filtering and data management"""

    def __init__(self):
        self.widget = None
        self.trades_table = None
        self.trades_search = None
        self.profitable_only = None
        self._init_ui()

    def _init_ui(self):
        """Initialize trade details UI efficiently"""
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

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
        layout.addLayout(filter_layout)

        # Trades table
        self.trades_table = QTableWidget()
        self.trades_table.setSortingEnabled(True)
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.trades_table)

    def get_widget(self):
        """Get the widget for tab integration"""
        return self.widget

    def populate_trades(self, results_data):
        """Populate trades table efficiently with pagination for large datasets"""
        if not results_data or 'trades' not in results_data:
            return

        trades = results_data['trades']
        if not trades:
            return

        total_trades = len(trades)
        
        # Performance optimization: limit display for large datasets
        max_display_trades = 10000  # Limit to prevent GUI freezing
        
        if total_trades > max_display_trades:
            displayed_trades = trades[:max_display_trades]
            print(f"Performance optimization: Displaying first {max_display_trades:,} of {total_trades:,} trades")
        else:
            displayed_trades = trades
            print(f"Displaying all {total_trades:,} trades")

        headers = ['ID', 'Entry Time', 'Exit Time', 'Side', 'Entry $', 'Exit $', 'P&L $', 'P&L %', 'Size $', 'Duration', 'МПП $', 'МПУ $']
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        self.trades_table.setRowCount(len(displayed_trades))

        # Pre-calculate time formatting for performance
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

            # МПП и МПУ - пока заглушки, в следующих версиях будет полноценный расчет
            max_floating_profit = trade.get('max_floating_profit', 0.00)  # МПП
            max_floating_loss = trade.get('max_floating_loss', 0.00)      # МПУ

            # Create items list for batch insertion
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
                QTableWidgetItem(f"{trade.get('duration', 0):.2f} min"),
                QTableWidgetItem(f"${max_floating_profit:.2f}"),  # МПП
                QTableWidgetItem(f"${max_floating_loss:.2f}")     # МПУ
            ]

            # Color coding for P&L
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                color = QColor(0, 150, 0)  # Green for profit
            elif pnl < 0:
                color = QColor(200, 0, 0)  # Red for loss
            else:
                color = QColor(100, 100, 100)  # Gray for break-even
                
            items[6].setForeground(color)  # P&L $ (now at index 6)
            items[7].setForeground(color)  # P&L % (now at index 7)

            # Color coding for МПП (green) and МПУ (red)
            items[10].setForeground(QColor(0, 150, 0))  # МПП $ - green
            items[11].setForeground(QColor(200, 0, 0))  # МПУ $ - red

            # Batch insert items
            for j, item in enumerate(items):
                self.trades_table.setItem(i, j, item)
        
        # Add info label if trades were truncated
        if total_trades > max_display_trades:
            info_label = QLabel(f"Showing first {max_display_trades:,} of {total_trades:,} trades for performance")
            info_label.setStyleSheet("color: #888888; font-style: italic; margin: 5px;")
            self.trades_table.setParent(None)
            
            # Create new layout with info label
            parent_layout = self.trades_table.parent().layout()
            if parent_layout:
                parent_layout.addWidget(info_label)
                parent_layout.addWidget(self.trades_table)

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
                pnl_item = self.trades_table.item(row, 6)  # P&L column at index 6
                if pnl_item:
                    try:
                        # Remove $ and parse float
                        pnl_text = pnl_item.text().replace('$', '')
                        pnl = float(pnl_text)
                        if pnl <= 0:
                            show_row = False
                    except ValueError:
                        show_row = False

            self.trades_table.setRowHidden(row, not show_row)

    def clear(self):
        """Clear trades table"""
        if self.trades_table:
            self.trades_table.setRowCount(0)