"""
Dataset Management for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
"""
import os


class DatasetManager:
    """Manages dataset loading and symbol extraction"""

    def __init__(self, dataset_combo, symbol_label, logger):
        self.dataset_combo = dataset_combo
        self.symbol_label = symbol_label
        self.logger = logger

    def load_datasets(self):
        """Load available datasets efficiently"""
        trades_dir = "upload/trades"
        if not os.path.exists(trades_dir):
            self.logger.log("Warning: No trades directory found")
            return

        csv_files = [f for f in os.listdir(trades_dir) if f.endswith('.csv')]

        # Force trades_exampls.csv to be the default and first choice
        final_files = []
        if 'trades_exampls.csv' in csv_files:
            final_files.append('trades_exampls.csv')
            # Add other files after trades_exampls.csv
            for f in csv_files:
                if f != 'trades_exampls.csv':
                    final_files.append(f)
        else:
            final_files = csv_files

        self.dataset_combo.addItems(final_files)

        if final_files:
            # Force select the first item (trades_exampls.csv if available)
            self.dataset_combo.setCurrentIndex(0)
            self.logger.log(f"Found {len(final_files)} datasets, forced default: {final_files[0]}")
            # Trigger the selection change to update symbol
            self.on_dataset_changed(final_files[0])

    def extract_symbol(self, filename):
        """Extract trading symbol from filename"""
        name = os.path.splitext(filename)[0]
        if '-trades-' in name:
            return name.split('-trades-')[0]
        if name.startswith('trades_'):
            return name[7:]
        return name

    def on_dataset_changed(self, filename):
        """Handle dataset selection"""
        if filename:
            symbol = self.extract_symbol(filename)
            self.symbol_label.setText(symbol)
            self.logger.log(f"Dataset: {filename} → Symbol: {symbol}")