import os
import re
import argparse
import pandas as pd
from typing import Optional

"""
Batch-insert first column 'symbol' into all CSV files in a folder.
Symbol is parsed from filename between first '_' and first '.'.
Example filename: 'BINANCE_CUDISUSDT.P, 15S_4fc8b.csv' -> symbol: 'CUDISUSDT'

Usage:
  python add_symbol_to_merge_csv.py --folder merge
"""

FILENAME_SYMBOL_RE = re.compile(r"^[^_]*_([^.]*)\.")


def extract_symbol_from_filename(filename: str) -> Optional[str]:
    match = FILENAME_SYMBOL_RE.match(filename)
    if not match:
        return None
    raw = match.group(1)
    # In examples, after '_' comes like 'CUDISUSDT.P, 15S_4fc8b' before first dot -> we get 'CUDISUSDT'
    # The regex already stops at first '.', so raw should be the symbol.
    return raw


def process_file(path: str, inplace: bool = True, backup_ext: Optional[str] = None) -> None:
    base = os.path.basename(path)
    symbol = extract_symbol_from_filename(base)
    if not symbol:
        print(f"[SKIP] Can't extract symbol from: {base}")
        return

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[ERROR] Reading {base}: {e}")
        return

    # If column already exists and equals symbol for all rows, skip
    if 'symbol' in df.columns:
        if df['symbol'].astype(str).eq(symbol).all():
            print(f"[OK] Already has symbol={symbol}: {base}")
            return
        else:
            # Ensure 'symbol' is first and set to correct value
            df.drop(columns=['symbol'], inplace=True, errors='ignore')

    df.insert(0, 'symbol', symbol)

    out_path = path
    if not inplace:
        root, ext = os.path.splitext(path)
        out_path = f"{root}.with_symbol{ext}"

    # Optional backup
    if inplace and backup_ext:
        try:
            os.replace(path, path + backup_ext)
            # After this, we will write to original path
            read_df = df  # noqa: F841 (for clarity)
        except Exception as e:
            print(f"[WARN] Backup failed for {base}: {e}")

    try:
        df.to_csv(out_path, index=False)
        print(f"[WROTE] {out_path} (symbol={symbol})")
    except Exception as e:
        print(f"[ERROR] Writing {out_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Insert first column 'symbol' parsed from filename into CSVs")
    parser.add_argument('--folder', type=str, default='merge', help="Folder with CSV files (default: data/merge relative to this script)")
    parser.add_argument('--pattern', type=str, default='*.csv', help="Glob-like suffix filter (default: *.csv)")
    parser.add_argument('--inplace', action='store_true', help="Overwrite files in place (default)")
    parser.add_argument('--no-inplace', dest='inplace', action='store_false')
    parser.set_defaults(inplace=True)
    parser.add_argument('--backup-ext', type=str, default=None, help="If inplace, rename original to this extension before writing (e.g., .bak)")
    args = parser.parse_args()

    folder = args.folder
    # Resolve default to data/merge next to this script
    if folder == 'merge':
        folder = os.path.join(os.path.dirname(__file__), 'merge')

    if not os.path.isdir(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return

    # Simple filter: process files ending with provided pattern's suffix
    suffix = args.pattern.replace('*', '')

    entries = sorted(os.listdir(folder))
    total = 0
    for name in entries:
        if suffix and not name.endswith(suffix):
            continue
        if not name.lower().endswith('.csv'):
            continue
        total += 1
        process_file(os.path.join(folder, name), inplace=args.inplace, backup_ext=args.backup_ext)

    print(f"[DONE] Processed files: {total}")


if __name__ == '__main__':
    main()
