import pandas as pd
from binance.client import Client
import datetime
import os

# Задайте свои ключи, если нужен больший лимит
API_KEY = ''
API_SECRET = ''

client = Client(API_KEY, API_SECRET)

def fetch_binance_ohlcv(symbol, interval, start_str, end_str=None, save_path=None):
    # Binance лимит: 1000 свечей за запрос
    limit = 1000
    all_klines = []
    from_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    to_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None
    curr_start = from_ts
    while True:
        curr_start_str = pd.to_datetime(curr_start, unit='ms').strftime('%d %b, %Y %H:%M:%S')
        klines = client.get_historical_klines(symbol, interval, curr_start_str, end_str, limit=limit)
        if not klines:
            break
        all_klines.extend(klines)
        last_time = klines[-1][0]
        if to_ts and last_time >= to_ts:
            break
        if len(klines) < limit:
            break
        curr_start = last_time + 1
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
    interval = Client.KLINE_INTERVAL_1HOUR
    start = '1 Jan, 2022'
    end = '1 Feb, 2022'
    save_path = f'data/{symbol}_{interval}_{start.replace(", ", "_")}_{end.replace(", ", "_")}.csv'
    fetch_binance_ohlcv(symbol, interval, start, end, save_path)
