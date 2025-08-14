import backtrader as bt
from .trade_recorder_mixin import TradeRecorderMixin

class MomentumBreakoutStrategy(TradeRecorderMixin, bt.Strategy):
    params = (
        ("momentum_period", 14),
        ("leverage", 100),
    )

    def __init__(self):
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash * self.params.leverage / asset_price * 0.99

            # Buy when 14-period Momentum crosses above zero
            if self.momentum > 0 and self.momentum[-1] <= 0:
                self.log('Buy Create, Momentum crossed above zero')
                self.order = self.buy(size=position_size)

        else:
            # Sell when 14-period Momentum crosses below zero
            if self.momentum < 0 and self.momentum[-1] >= 0:
                self.log('Position Closed, Momentum crossed below zero')
                self.order = self.close()

    def log(self, txt):
        pass

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None
        # Вызвать миксин для фиксации buy-сделок
        TradeRecorderMixin.notify_order(self, order)
