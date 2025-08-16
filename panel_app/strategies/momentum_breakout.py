import pandas as pd

def momentum_breakout_signals(close, period=14):
    momentum = close.diff(period)
    entries = (momentum > 0) & (momentum.shift(1) <= 0)
    exits = (momentum < 0) & (momentum.shift(1) >= 0)
    return entries, exits
