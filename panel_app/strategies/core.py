import vectorbt as vbt
from panel_app.strategies.momentum_breakout import momentum_breakout_signals

def run_vbt_strategy(ohlcv_df, strategy_name, **params):
    close = ohlcv_df['close']
    if strategy_name == 'MomentumBreakout':
        entries, exits = momentum_breakout_signals(close, period=params.get('momentum_period', 14))
    else:
        raise ValueError('Unknown strategy')
    cash = params.get('cash', 10000)
    leverage = params.get('leverage', 1)
    size = (cash * leverage) / close if leverage > 1 else None
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=cash,
        fees=params.get('commission', 0.0005),
        direction='both',
        slippage=0.0,
        size_type='amount',
        size=size
    )
    return portfolio
