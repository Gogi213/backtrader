import os
import argparse
import pandas as pd
from datetime import datetime
import re

SYMBOL_RE = re.compile(r"^[^_]*_([^.]*)\.")

def extract_symbol_from_filename(filename: str) -> str | None:
    """Extracts symbol from filename between first '_' and first '.'.
    Example: 'BINANCE_CUDISUSDT.P, 15S_4fc8b.csv' -> 'CUDISUSDT'
    """
    m = SYMBOL_RE.match(filename)
    if not m:
        return None
    return m.group(1)

def tv_csv_to_parquet(input_path: str, output_dir: str | None = None, overwrite: bool = True) -> str | None:
    # Пропускаем уже преобразованные промежуточные CSV
    base_name = os.path.basename(input_path)
    if base_name.lower().endswith('_binance_like.csv'):
        return None

    df = pd.read_csv(input_path)
    symbol = extract_symbol_from_filename(base_name)

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

    # Преобразуем время в единый формат: DatetimeIndex (UTC)
    if 'timestamp' in df.columns:
        ts = df['timestamp']
        # Попробуем распознать числовую эпоху (сек/мс) или строку
        def to_datetime_utc(series: pd.Series) -> pd.DatetimeIndex:
            # Если все значения числовые или строковые цифры
            if pd.api.types.is_numeric_dtype(series) or series.astype(str).str.fullmatch(r"\d+(\.\d+)?").all():
                vals = pd.to_numeric(series, errors='coerce')
                # Определим единицу: мс если медиана > 1e12
                unit = 'ms' if vals.median(skipna=True) > 1e12 else 's'
                return pd.to_datetime(vals, unit=unit, utc=True)
            # Иначе парсим как строки дат/времени
            return pd.to_datetime(series, utc=True, errors='coerce', infer_datetime_format=True)

        dt_utc = to_datetime_utc(ts)
        # Отбросим нераспознанные
        ok_mask = dt_utc.notna()
        df = df.loc[ok_mask].copy()
        dt_utc = dt_utc.loc[ok_mask]
        # Установим индекс
        df.index = dt_utc
        df.index.name = 'timestamp'
        # Сортировка по времени
        df.sort_index(inplace=True)
        # Больше не храним дубликат столбца timestamp
        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True, errors='ignore')

    # Вставляем символ первым столбцом, если удалось распарсить
    if symbol is not None:
        if 'symbol' in df.columns:
            if not df['symbol'].astype(str).eq(symbol).all():
                df.drop(columns=['symbol'], inplace=True, errors='ignore')
                df.insert(0, 'symbol', symbol)
        else:
            df.insert(0, 'symbol', symbol)

    # Формируем путь вывода
    in_dir = os.path.dirname(input_path)
    out_dir = output_dir if output_dir else in_dir
    os.makedirs(out_dir, exist_ok=True)

    stem, _ = os.path.splitext(os.path.basename(input_path))
    out_path = os.path.join(out_dir, f"{stem}_binance_like.parquet")

    if os.path.exists(out_path) and not overwrite:
        print(f"Skip (exists): {out_path}")
        return out_path

    # Приведём типы числовых OHLCV к float32 для экономии
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

    # Сохраняем Parquet с индексом времени
    df.to_parquet(out_path, index=True)
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

    # Нормализуем пути относительно ТЕКУЩЕЙ рабочей директории (CWD),
    # чтобы вызов из корня проекта с --folder data/... не добавлял лишнее 'data/'
    folder = args.folder
    if not os.path.isabs(folder):
        folder = os.path.abspath(folder)
    out_dir = args.out
    if out_dir and not os.path.isabs(out_dir):
        out_dir = os.path.abspath(out_dir)

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