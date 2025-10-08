"""
BacktestConfig — минимальный dataclass
Без GUI, без __post_init__, без StrategyRegistry
Author: HFT System (optimized)
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class BacktestConfig:
    strategy_name: str = 'hierarchical_mean_reversion'
    strategy_params: Optional[Dict[str, Any]] = None
    symbol: str = 'BTCUSDT'
    data_source: str = "csv"
    data_path: Optional[str] = None
    max_klines: Optional[int] = None
    initial_capital: float = 10000.0
    commission_pct: float = 0.0005
    position_size_dollars: float = 1000.0
    enable_turbo_mode: bool = True
    parallel_jobs: int = -1
    output_file: Optional[str] = None
    verbose: bool = False

    def __post_init__(self):
        if self.strategy_params is None:
            self.strategy_params = {}

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BacktestConfig':
        return cls(**d)