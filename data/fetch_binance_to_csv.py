import argparse
import pandas as pd
import os
from fetch_binance_data_fast import fetch_binance_ohlcv_fast
import asyncio

def save_binance_ohlcv_to_csv(symbol, timeframe, start, end, out_path):
    # Скачиваем и сохраняем в Parquet (как обычно)
    parquet_path = out_path.replace('.csv', '.parquet')
    if not os.path.exists(parquet_path):
        asyncio.get_event_loop().run_until_complete(
            fetch_binance_ohlcv_fast(symbol, timeframe, start, end, save_path=parquet_path)
        )
    df = pd.read_parquet(parquet_path)
    df.to_csv(out_path, index=True)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance OHLCV and save as CSV for comparison.")
    parser.add_argument('--symbol', type=str, required=True, help='Symbol, e.g. BTCUSDT')
    parser.add_argument('--timeframe', type=str, required=True, help='Timeframe, e.g. 1m, 5m, 1h')
    parser.add_argument('--start', type=str, required=True, help='Start date, e.g. 2025-08-01')
    parser.add_argument('--end', type=str, required=True, help='End date, e.g. 2025-08-15')
    parser.add_argument('--out', type=str, required=False, help='Output CSV path')
    args = parser.parse_args()

    # Путь по умолчанию — всегда в data/cache_binance внутри текущего проекта
    base_dir = os.path.join(os.path.dirname(__file__), 'cache_binance')
    os.makedirs(base_dir, exist_ok=True)
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(base_dir, f"{args.symbol}_{args.timeframe}_{args.start}_{args.end}_fast.csv")

    save_binance_ohlcv_to_csv(args.symbol, args.timeframe, args.start, args.end, out_path)
