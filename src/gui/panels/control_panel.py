"""
Control Panel for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QProgressBar
)
from PyQt6.QtGui import QFont
from ..data.dataset_manager import DatasetManager


class ControlPanel:
    """Control panel with dataset, strategy, and risk management controls"""

    def __init__(self, config, logger, callbacks):
        self.config = config
        self.logger = logger
        self.callbacks = callbacks  # Dict with callback functions

        self.widget = None
        self.dataset_manager = None

        # UI elements
        self.dataset_combo = None
        self.symbol_label = None
        self.bb_period_spin = None
        self.bb_std_spin = None
        self.stop_loss_spin = None
        self.sma_tp_spin = None
        self.capital_spin = None
        self.position_size_spin = None
        self.progress_bar = None
        self.start_btn = None
        self.export_btn = None

    def create_panel(self):
        """Create streamlined control panel"""
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        # Header
        title = QLabel("BB Strategy Backtester")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Dataset selection
        dataset_group = QGroupBox("Data Source")
        dataset_layout = QFormLayout(dataset_group)

        self.dataset_combo = QComboBox()
        dataset_layout.addRow("Dataset:", self.dataset_combo)

        self.symbol_label = QLabel("Auto-detected")
        dataset_layout.addRow("Symbol:", self.symbol_label)

        layout.addWidget(dataset_group)

        # Initialize dataset manager after UI elements are created
        self.dataset_manager = DatasetManager(self.dataset_combo, self.symbol_label, self.logger)
        self.dataset_combo.currentTextChanged.connect(self.dataset_manager.on_dataset_changed)

        # Load datasets
        self.dataset_manager.load_datasets()

        # Strategy parameters
        strategy_group = QGroupBox("Strategy Parameters")
        strategy_layout = QFormLayout(strategy_group)

        # BB parameters
        self.bb_period_spin = QSpinBox()
        self.bb_period_spin.setRange(100, 300)
        self.bb_period_spin.setValue(self.config.bb_period)
        strategy_layout.addRow("BB Period:", self.bb_period_spin)

        self.bb_std_spin = QDoubleSpinBox()
        self.bb_std_spin.setRange(1.5, 4.0)
        self.bb_std_spin.setSingleStep(0.1)
        self.bb_std_spin.setValue(self.config.bb_std)
        strategy_layout.addRow("BB Std Dev:", self.bb_std_spin)

        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.5, 5.0)
        self.stop_loss_spin.setSingleStep(0.1)
        self.stop_loss_spin.setValue(self.config.stop_loss_pct)
        strategy_layout.addRow("Stop Loss %:", self.stop_loss_spin)

        self.sma_tp_spin = QSpinBox()
        self.sma_tp_spin.setRange(10, 50)
        self.sma_tp_spin.setValue(self.config.sma_tp_period)
        strategy_layout.addRow("SMA TP Period:", self.sma_tp_spin)

        layout.addWidget(strategy_group)

        # Risk management
        risk_group = QGroupBox("Risk Management")
        risk_layout = QFormLayout(risk_group)

        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000, 1000000)
        self.capital_spin.setValue(self.config.initial_capital)
        risk_layout.addRow("Capital ($):", self.capital_spin)

        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(100, 5000)
        self.position_size_spin.setSingleStep(100)
        self.position_size_spin.setValue(1000)  # Default $1000 per trade
        risk_layout.addRow("Position Size ($):", self.position_size_spin)

        layout.addWidget(risk_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Action buttons
        self.start_btn = QPushButton("START BACKTEST")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.start_btn.clicked.connect(self.callbacks.get('start_backtest'))
        layout.addWidget(self.start_btn)

        self.export_btn = QPushButton("Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.callbacks.get('export_results'))
        layout.addWidget(self.export_btn)

        layout.addStretch()
        return self.widget

    def get_widget(self):
        """Get the widget for integration"""
        return self.widget

    def get_current_dataset(self):
        """Get currently selected dataset"""
        return self.dataset_combo.currentText()

    def extract_symbol_from_dataset(self, dataset):
        """Extract symbol from dataset filename"""
        return self.dataset_manager.extract_symbol(dataset)

    def update_config_from_ui(self):
        """Update strategy configuration from UI"""
        self.config.bb_period = self.bb_period_spin.value()
        self.config.bb_std = self.bb_std_spin.value()
        self.config.stop_loss_pct = self.stop_loss_spin.value()
        self.config.sma_tp_period = self.sma_tp_spin.value()
        self.config.initial_capital = self.capital_spin.value()
        self.config.position_size_dollars = self.position_size_spin.value()

    def set_backtest_running(self, running):
        """Set UI state for backtest running"""
        self.start_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setRange(0, 0)  # Indeterminate

    def enable_export(self, enabled):
        """Enable/disable export button"""
        self.export_btn.setEnabled(enabled)