"""
Configuration Models for Professional GUI Application
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
"""
from PyQt6.QtCore import QThread, pyqtSignal


class StrategyConfig:
    """Strategy configuration with validation"""
    def __init__(self):
        self.bb_period = 50  # FIXED: Более разумный период для большинства datasets
        self.bb_std = 2.0   # FIXED: Стандартное значение для BB
        self.stop_loss_pct = 1.0
        self.sma_tp_period = 20
        self.initial_capital = 10000.0
        self.position_size_dollars = 1000.0
        self.max_ticks_gui = 1000000  # PERFORMANCE FIX: Default limit 1M ticks for GUI performance
        self.max_ticks_unlimited = None  # No limit for advanced users

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def from_dict(self, config_dict):
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
            # UNIFIED SYSTEM: Use vectorized_klines_backtest for maximum performance
            from src.data.vectorized_klines_backtest import run_vectorized_klines_backtest

            self.progress_signal.emit(f"VECTORIZED KLINES BACKTEST: {self.symbol}")
            if self.max_ticks:
                self.progress_signal.emit(f"Loading klines data (limited to {self.max_ticks:,} klines for GUI performance)...")
            else:
                self.progress_signal.emit("Loading full klines data (no limit)...")
            self.progress_signal.emit("Running super-vectorized Bollinger Bands strategy...")

            # Run vectorized backtest - UNIFIED SYSTEM!
            results = run_vectorized_klines_backtest(
                csv_path=self.csv_path,
                symbol=self.symbol,
                bb_period=self.config.bb_period,
                bb_std=self.config.bb_std,
                stop_loss_pct=self.config.stop_loss_pct,
                initial_capital=self.config.initial_capital,
                max_klines=self.max_ticks  # KEY: Limit processing for GUI performance (None means no limit)
            )

            self.result_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(f"Vectorized klines backtest failed: {str(e)}")