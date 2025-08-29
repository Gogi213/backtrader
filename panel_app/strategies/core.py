import vectorbt as vbt
from .zscore_sma import run_zscore_sma_strategy
from .kama_zscore import run_kama_zscore_strategy

def run_vbt_strategy(ohlcv_df, strategy_name, **params):
    """
    Диспетчер стратегий. Запускает нужную стратегию по имени.
    """
    if strategy_name in ['ZScoreSMA', 'Z-Score + SMA with SL/TP']:
        return run_zscore_sma_strategy(ohlcv_df, **params)
    elif strategy_name in ['KAMAZScore', 'KAMA + Adaptive Z-Score']:
        return run_kama_zscore_strategy(ohlcv_df, **params)
    else:
        raise ValueError(f'Unknown strategy: {strategy_name}')
