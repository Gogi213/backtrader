import vectorbt as vbt

def mean_reversion_signals(close, period=20, dev=2):
    bb = vbt.BBANDS.run(close, window=period, std=dev)
    entries = close < bb.lower
    exits = close > bb.middle
    return entries, exits
