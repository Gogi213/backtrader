import backtrader as bt
import pandas as pd

class TradeRecorderMixin:
    def start(self):
        self.open_trade = None  # Текущая открытая сделка
        self.completed_trades = []  # Список завершённых сделок

    def notify_order(self, order):
        # Вход в позицию (открытие сделки)
        if order.status == order.Completed:
            if order.isbuy():
                if self.open_trade is None:
                    self.open_trade = {
                        'entry_time': self.data.datetime.datetime(0),
                        'entry_idx': len(self.data) - 1,
                        'entry_price': order.executed.price,
                        'size': order.executed.size,
                        'direction': 'long',
                    }
            elif order.issell():
                # Закрытие позиции (выход)
                if self.open_trade is not None:
                    exit_trade = self.open_trade.copy()
                    exit_trade.update({
                        'exit_time': self.data.datetime.datetime(0),
                        'exit_idx': len(self.data) - 1,
                        'exit_price': order.executed.price,
                        # pnl будет добавлен в notify_trade
                    })
                    self._pending_exit = exit_trade

    def notify_trade(self, trade):
        # Выход из позиции (закрытие сделки)
        if trade.isclosed and hasattr(self, '_pending_exit') and self._pending_exit is not None:
            exit_trade = self._pending_exit
            exit_trade['pnl'] = trade.pnl
            self.completed_trades.append(exit_trade)
            self.open_trade = None
            self._pending_exit = None

    def get_trades_df(self):
        """Экспорт завершённых сделок в DataFrame для Plotly-графика."""
        if not hasattr(self, 'completed_trades') or not self.completed_trades:
            return pd.DataFrame()
        return pd.DataFrame(self.completed_trades)
