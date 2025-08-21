import os
import argparse
import pandas as pd
from datetime import datetime

def tv_csv_to_parquet(input_path: str, output_dir: str | None = None, overwrite: bool = True) -> str | None:
    # Пропускаем уже преобразованные промежуточные CSV
    base_name = os.path.basename(input_path)
    if base_name.lower().endswith('_binance_like.csv'):
        return None

    df = pd.read_csv(input_path)

    # Определяем колонку объёма
    vol_col = None
    for v in ['Volume', 'volume']:
        if v in df.columns:
            vol_col = v
            break

    columns_map = {
        'time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
    }
    if vol_col:
        columns_map[vol_col] = 'volume'

    # Оставляем только нужные и переименовываем
    keep = [c for c in columns_map.keys() if c in df.columns]
    df = df[keep].rename(columns=columns_map)

    # Преобразуем время в формат Binance: 'YYYY-MM-DD HH:MM:SS' (UTC)
    if 'timestamp' in df.columns:
        def fmt_ts(x):
            try:
                return datetime.utcfromtimestamp(int(float(x))).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                try:
                    return pd.to_datetime(x, utc=True).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    return str(x)
        df['timestamp'] = df['timestamp'].apply(fmt_ts)

    # Формируем путь вывода
    in_dir = os.path.dirname(input_path)
    out_dir = output_dir if output_dir else in_dir
    os.makedirs(out_dir, exist_ok=True)

    stem, _ = os.path.splitext(os.path.basename(input_path))
    out_path = os.path.join(out_dir, f"{stem}_binance_like.parquet")

    if os.path.exists(out_path) and not overwrite:
        print(f"Skip (exists): {out_path}")
        return out_path

    # Сохраняем Parquet
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Convert TradingView CSVs to Binance-like Parquet in one step")
    parser.add_argument('--folder', type=str, default='cache_tradingview',
                        help='Folder with TradingView CSVs (default: data/cache_tradingview relative to this script)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output folder for Parquet (optional; default: same as input file)')
    parser.add_argument('--no-overwrite', action='store_false', dest='overwrite', default=True,
                        help='Do not overwrite existing .parquet files')
    args = parser.parse_args()

    # Нормализуем пути относительно расположения скрипта
    here = os.path.dirname(__file__)
    folder = args.folder
    if not os.path.isabs(folder):
        folder = os.path.join(here, folder)
    out_dir = args.out
    if out_dir and not os.path.isabs(out_dir):
        out_dir = os.path.join(here, out_dir)

    if not os.path.isdir(folder):
        print(f"Input folder not found: {folder}")
        return

    count = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.csv'):
            continue
        in_path = os.path.join(folder, fname)
        tv_csv_to_parquet(in_path, output_dir=out_dir, overwrite=args.overwrite)
        count += 1
    print(f"Done. Processed CSV files: {count}")

if __name__ == '__main__':
    main()