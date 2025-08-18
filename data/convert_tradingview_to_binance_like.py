import pandas as pd
import argparse
import os
from datetime import datetime

def tradingview_to_binance_like(input_path, output_path=None):
    df = pd.read_csv(input_path)
    # Оставляем только нужные столбцы
    columns_map = {
        'time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'Volume': 'volume',
        'volume': 'volume',
    }
    # Найти правильное имя столбца для объёма
    for vcol in ['Volume', 'volume']:
        if vcol in df.columns:
            columns_map['volume'] = vcol
            break
    # Оставляем только нужные
    keep_cols = [k for k in columns_map.keys() if k in df.columns]
    df = df[keep_cols]
    # Переименовываем
    df = df.rename(columns=columns_map)
    # Преобразуем время
    if 'timestamp' in df.columns:
        # Если timestamp — это unix, переводим в строку как у бинанса
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(int(float(x))).strftime('%Y-%m-%d %H:%M:%S'))
    # Сохраняем
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = base + '_binance_like.csv'
    df.to_csv(output_path, index=False)
    print(f'Saved: {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert all TradingView CSVs in cache_tradingview to Binance-like OHLCV CSVs')
    parser.add_argument('--folder', type=str, default='cache_tradingview', help='Folder with TradingView CSVs (default: cache_tradingview)')
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isabs(folder):
        folder = os.path.join(os.path.dirname(__file__), folder)
    for fname in os.listdir(folder):
        if fname.lower().endswith('.csv'):
            in_path = os.path.join(folder, fname)
            tradingview_to_binance_like(in_path)
