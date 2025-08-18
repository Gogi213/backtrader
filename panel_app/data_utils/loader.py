import os
import pandas as pd
import asyncio
import nest_asyncio
nest_asyncio.apply()
from data.fetch_binance_data_fast import fetch_binance_ohlcv_fast

def load_ohlcv(symbol, timeframe, start, end):
    save_path = f'../data/cache_binance/{symbol}_{timeframe}_{start}_{end}_fast.parquet'
    if not os.path.exists(save_path):
        asyncio.get_event_loop().run_until_complete(fetch_binance_ohlcv_fast(symbol, timeframe, start, end, save_path=save_path))
    df = pd.read_parquet(save_path)
    df = df.dropna()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    return df
