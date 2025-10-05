"""
Configuration Models for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
Updated to support dynamic strategy selection via StrategyRegistry
"""
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any


class StrategyConfig:
    """Strategy configuration with dynamic strategy selection support"""
    def __init__(self, strategy_name: str = 'hierarchical_mean_reversion'):
        # Strategy selection
        self.strategy_name = strategy_name
        
        # Load default parameters for the selected strategy
        self.strategy_params = self._load_default_params()
        
        # Common parameters
        self.initial_capital = 10000.0
        self.commission_pct = 0.05  # Commission percentage (default 0.05%)
        self.max_ticks_gui = 1000000  # PERFORMANCE FIX: Default limit 1M ticks for GUI performance
        self.max_ticks_unlimited = None  # No limit for advanced users

    def _load_default_params(self) -> Dict[str, Any]:
        """Load default parameters for the selected strategy"""
        from .utils.strategy_params_helper import StrategyParamsHelper
        return StrategyParamsHelper.get_strategy_params(self.strategy_name)

    def update_strategy(self, strategy_name: str):
        """Update strategy and reset parameters to defaults"""
        self.strategy_name = strategy_name
        self.strategy_params = self._load_default_params()

    def to_dict(self):
        """Convert to dictionary"""
        result = {
            'strategy_name': self.strategy_name,
            'strategy_params': self.strategy_params.copy(),
            'initial_capital': self.initial_capital,
            'commission_pct': self.commission_pct,
            'max_ticks_gui': self.max_ticks_gui,
            'max_ticks_unlimited': self.max_ticks_unlimited
        }
        return result

    def from_dict(self, config_dict):
        """Load from dictionary"""
        for k, v in config_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)


class BacktestWorker(QThread):
    """High-performance worker thread with progress tracking"""
    progress_signal = pyqtSignal(str)
    result_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, csv_path, symbol, config, exchange='Binance', timeframe='1m', tick_mode=False, max_ticks=None):
        super().__init__()
        self.csv_path = csv_path
        self.symbol = symbol
        self.config = config
        self.exchange = exchange
        self.timeframe = timeframe
        self.tick_mode = tick_mode
        self.max_ticks = max_ticks  # None means no limit, otherwise apply limit

    def run(self):
        try:
            # UNIFIED SYSTEM: Use BacktestManager for unified backtesting
            from src.core import BacktestManager, BacktestConfig

            self.progress_signal.emit(f"UNIFIED BACKTEST: {self.symbol}")
            self.progress_signal.emit("Using new BacktestManager and ConfigValidator components")
            if self.max_ticks:
                self.progress_signal.emit(f"Loading klines data (limited to {self.max_ticks:,} klines for GUI performance)...")
            else:
                self.progress_signal.emit("Loading full klines data (no limit)...")
            self.progress_signal.emit(f"Running {self.config.strategy_name} strategy...")

            # Create configuration
            config = BacktestConfig(
                strategy_name=self.config.strategy_name,
                symbol=self.symbol,
                data_path=self.csv_path,
                initial_capital=self.config.initial_capital,
                commission_pct=self.config.commission_pct,
                max_klines=self.max_ticks,
                strategy_params=self.config.strategy_params
            )

            # Run backtest using BacktestManager
            manager = BacktestManager()
            self.progress_signal.emit("Executing backtest with unified BacktestManager...")
            results = manager.run_backtest(config)

            # Convert to dictionary for compatibility
            if results.is_successful():
                self.progress_signal.emit("Backtest completed successfully with new components")
                self.result_signal.emit(results.to_dict())
            else:
                self.progress_signal.emit(f"Backtest failed: {results.get_error()}")
                self.error_signal.emit(f"Backtest failed: {results.get_error()}")
        except Exception as e:
            self.progress_signal.emit(f"Exception in BacktestWorker: {str(e)}")
            self.error_signal.emit(f"Backtest failed: {str(e)}")