import backtrader as bt
from .trade_recorder_mixin import TradeRecorderMixin

class MeanReversionStrategy(TradeRecorderMixin, bt.Strategy):
	params = (
		("bollinger_period", 20),
		("bollinger_dev", 2),
		("leverage", 100),
	)

	def __init__(self):
		self.bollinger = bt.indicators.BollingerBands(self.data.close, period=self.params.bollinger_period, devfactor=self.params.bollinger_dev)
		self.order = None

	def next(self):
		if self.order:
			return

		if not self.position:
			cash = self.broker.get_cash()
			asset_price = self.data.close[0]
			position_size = cash * self.params.leverage / asset_price * 0.99
			if self.data.close < self.bollinger.lines.bot:
				self.log('Buy Create, Price touches lower Bollinger Band')
				self.order = self.buy(size=position_size)

		else:
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
		# Вызвать миксин для фиксации сделок
		TradeRecorderMixin.notify_order(self, order)
