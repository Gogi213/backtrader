"""
BacktestResults — минимальный контейнер
Без GUI, без .to_gui_format(), без дублирования
Author: HFT System (optimized)
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional


class BacktestResults:
    def __init__(self, raw: Dict[str, Any]):
        self.data = raw
        self._ts = datetime.now()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def is_ok(self) -> bool:
        return 'error' not in self.data and self.get('total', 0) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {**self.data, 'generated_at': self._ts.isoformat()}

    def to_console(self) -> str:
        return (
            f"Strategy: {self.get('strategy_name')}\n"
            f"Trades: {self.get('total', 0)}\n"
            f"Net PnL: ${self.get('net_pnl', 0):.2f}\n"
            f"Sharpe: {self.get('sharpe_ratio', 0):.2f}\n"
            f"Max DD: {self.get('max_drawdown', 0):.2f}%"
        )

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_error(cls, msg: str, name: str = 'unknown', symbol: str = 'UNKNOWN') -> 'BacktestResults':
        return cls({'error': msg, 'strategy_name': name, 'symbol': symbol, 'total': 0})