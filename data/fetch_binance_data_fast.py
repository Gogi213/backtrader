
import asyncio
import pandas as pd
from binance import AsyncClient
import datetime
import os

API_KEY = ''
API_SECRET = ''

async def fetch_ohlcv_batch(client, symbol, interval, start_str, end_str):
    # Используем futures_historical_klines для Binance Futures
    klines = await client.futures_historical_klines(symbol, interval, start_str, end_str, limit=1000)
    return klines

async def fetch_binance_ohlcv_fast(symbol, interval, start, end, save_path=None):
    # Разбиваем диапазон на батчи по 1000 свечей
    start_ts = int(pd.to_datetime(start).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end).timestamp() * 1000)
    delta_sec = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
    }[interval]
    candles_per_batch = 1000
    batch_ms = delta_sec * 1000 * candles_per_batch
    all_klines = []
    client = await AsyncClient.create(API_KEY, API_SECRET)
    curr = start_ts
    while curr < end_ts:
        batch_end = min(curr + batch_ms - 1, end_ts)
        start_str = pd.to_datetime(curr, unit='ms').strftime('%d %b, %Y %H:%M:%S')
        end_str = pd.to_datetime(batch_end, unit='ms').strftime('%d %b, %Y %H:%M:%S')
        klines = await fetch_ohlcv_batch(client, symbol, interval, start_str, end_str)
        all_klines.extend(klines)
        curr = batch_end + 1
    await client.close_connection()
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if save_path:
        save_path = save_path.replace('.csv', '.parquet')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path)
    else:
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache_binance')
        os.makedirs(cache_dir, exist_ok=True)
        default_path = os.path.join(cache_dir, f'{symbol}_{interval}_{start}_{end}_fast.parquet')
        df.to_parquet(default_path)
    return df
    all_klines = [item for sublist in results for item in sublist]
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
    return df

if __name__ == '__main__':
    # Пример: BTCUSDT, 1h, с 2022-01-01 по 2022-02-01
    symbol = 'BTCUSDT'
    interval = '1h'
    start = '2022-01-01'
    end = '2022-02-01'
    save_path = f'data/cache_binance/{symbol}_{interval}_{start}_{end}_fast.csv'
    asyncio.run(fetch_binance_ohlcv_fast(symbol, interval, start, end, save_path))
