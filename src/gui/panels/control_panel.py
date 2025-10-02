"""
Control Panel for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
Updated to support dynamic strategy selection via StrategyFactory
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
        self.strategy_combo = None
        self.strategy_params_group = None
        self.strategy_params_layout = None
        self.param_widgets = {}  # Dictionary to store parameter widgets
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
        title = QLabel("Strategy Backtester")
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

        # Strategy selection
        strategy_group = QGroupBox("Strategy Selection")
        strategy_layout = QFormLayout(strategy_group)

        self.strategy_combo = QComboBox()
        self._load_available_strategies()
        self.strategy_combo.setCurrentText(self.config.strategy_name)
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        strategy_layout.addRow("Strategy:", self.strategy_combo)

        layout.addWidget(strategy_group)

        # Strategy parameters (dynamic)
        self.strategy_params_group = QGroupBox("Strategy Parameters")
        self.strategy_params_layout = QFormLayout(self.strategy_params_group)
        self._create_strategy_param_widgets()
        layout.addWidget(self.strategy_params_group)

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

    def _load_available_strategies(self):
        """Load available strategies from StrategyFactory"""
        try:
            from ...strategies.strategy_factory import StrategyFactory
            strategies = StrategyFactory.list_available_strategies()
            self.strategy_combo.clear()
            self.strategy_combo.addItems(strategies)
        except Exception as e:
            self.logger.error(f"Failed to load strategies: {e}")
            # Fallback to Bollinger Bands
            self.strategy_combo.clear()
            self.strategy_combo.addItem("bollinger")

    def _on_strategy_changed(self, strategy_name):
        """Handle strategy selection change"""
        self.config.update_strategy(strategy_name)
        self._create_strategy_param_widgets()

    def _create_strategy_param_widgets(self):
        """Create parameter widgets for the selected strategy"""
        # Clear existing widgets
        for i in reversed(range(self.strategy_params_layout.count())):
            child = self.strategy_params_layout.itemAt(i).widget()
            if child is not None:
                child.setParent(None)
        self.param_widgets.clear()

        # Create new widgets based on strategy parameters
        for param_name, param_value in self.config.strategy_params.items():
            if isinstance(param_value, int):
                widget = QSpinBox()
                widget.setRange(1, 1000)
                widget.setValue(param_value)
            elif isinstance(param_value, float):
                widget = QDoubleSpinBox()
                widget.setRange(0.01, 100.0)
                widget.setSingleStep(0.01)
                widget.setValue(param_value)
            else:
                # Skip unsupported parameter types
                continue

            self.param_widgets[param_name] = widget
            self.strategy_params_layout.addRow(f"{param_name.replace('_', ' ').title()}:", widget)

    def update_config_from_ui(self):
        """Update strategy configuration from UI"""
        self.config.strategy_name = self.strategy_combo.currentText()
        
        # Update strategy parameters
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                self.config.strategy_params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                self.config.strategy_params[param_name] = widget.value()
        
        # Update common parameters
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