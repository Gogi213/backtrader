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

        # Show all trades without limits
        displayed_trades = trades
        print(f"Displaying all {total_trades:,} trades")

        headers = ['ID', 'Entry Time', 'Exit Time', 'Side', 'Entry $', 'Exit $', 'P&L $', 'P&L %', 'Size $', 'Duration']
        self.trades_table.setColumnCount(len(headers))
        self.trades_table.setHorizontalHeaderLabels(headers)
        self.trades_table.setRowCount(len(displayed_trades))

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
                QTableWidgetItem(f"{trade.get('duration', 0):.2f} min")
            ]

            # Color coding for P&L
            pnl = trade.get('pnl', 0)
            color = QColor(0, 150, 0) if pnl > 0 else QColor(200, 0, 0) if pnl < 0 else QColor(100, 100, 100)
            items[6].setForeground(color)  # P&L $ (now at index 6)
            items[7].setForeground(color)  # P&L % (now at index 7)

            for j, item in enumerate(items):
                self.trades_table.setItem(i, j, item)

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