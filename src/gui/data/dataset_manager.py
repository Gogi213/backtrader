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
        """Load available klines datasets efficiently"""
        klines_dir = "upload/klines"
        if not os.path.exists(klines_dir):
            if self.logger:
                self.logger.log("Warning: No klines directory found")
            return

        csv_files = [f for f in os.listdir(klines_dir) if f.endswith('.csv')]

        if not csv_files:
            if self.logger:
                self.logger.log("Warning: No CSV files found in klines directory")
            return

        # Sort files for consistent ordering
        csv_files.sort()
        self.dataset_combo.addItems(csv_files)

        # Select first file by default
        self.dataset_combo.setCurrentIndex(0)
        if self.logger:
            self.logger.log(f"Found {len(csv_files)} klines datasets, default: {csv_files[0]}")
        # Trigger the selection change to update symbol
        self.on_dataset_changed(csv_files[0])

    def extract_symbol(self, filename):
        """Extract trading symbol from klines filename"""
        name = os.path.splitext(filename)[0]
        if '-klines-' in name:
            return name.split('-klines-')[0]
        if name.startswith('klines_'):
            return name[7:]
        # Handle generic case where symbol might be at the beginning
        parts = name.split('-')
        if parts:
            return parts[0]
        return name

    def on_dataset_changed(self, filename):
        """Handle dataset selection"""
        if filename:
            symbol = self.extract_symbol(filename)
            self.symbol_label.setText(symbol)
            if self.logger:
                self.logger.log(f"Dataset: {filename} → Symbol: {symbol}")

    def get_dataset_path(self, filename):
        """Get full path to klines dataset"""
        return os.path.join("upload/klines", filename)