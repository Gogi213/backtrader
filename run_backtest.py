import backtrader as bt
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.default_freq_strategy import DefaultFreqStrategy
from strategies.multi_timeframe_strategy import MultiTimeframeStrategy


def run_backtest(data_path, cash=10000, strategy_name='DefaultFreqStrategy', data_path2=None):
    cerebro = bt.Cerebro()
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    if data_path2:
        df2 = pd.read_csv(data_path2, index_col='timestamp', parse_dates=True)
        data2 = bt.feeds.PandasData(dataname=df2)
        cerebro.adddata(data2)
    if strategy_name == 'MultiTimeframeStrategy':
        cerebro.addstrategy(MultiTimeframeStrategy)
    else:
        cerebro.addstrategy(DefaultFreqStrategy)
    cerebro.broker.setcash(cash)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    result = cerebro.run()
    strat = result[0]
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('Sharpe Ratio:', strat.analyzers.sharpe.get_analysis())
    print('Drawdown:', strat.analyzers.drawdown.get_analysis())
    print('Trade Analysis:', strat.analyzers.trades.get_analysis())
    cerebro.plot()


if __name__ == '__main__':
    # Пример использования
    data_path = 'data/BTCUSDT_1h_1_Jan_2022_1_Feb_2022.csv'
    data_path2 = 'data/BTCUSDT_4h_1_Jan_2022_1_Feb_2022.csv'
    run_backtest(data_path, strategy_name='MultiTimeframeStrategy', data_path2=data_path2)
