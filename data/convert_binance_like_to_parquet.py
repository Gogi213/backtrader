import pandas as pd
import os
import argparse

def csv_to_parquet(input_path, output_path=None):
    df = pd.read_csv(input_path)
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + '.parquet'
    df.to_parquet(output_path, index=False)
    print(f'Saved: {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert all *_binance_like.csv in cache_tradingview to parquet')
    parser.add_argument('--folder', type=str, default='cache_tradingview', help='Folder with binance-like CSVs (default: cache_tradingview)')
    args = parser.parse_args()
    folder = args.folder
    if not os.path.isabs(folder):
        folder = os.path.join(os.path.dirname(__file__), folder)
    for fname in os.listdir(folder):
        if fname.endswith('_binance_like.csv'):
            in_path = os.path.join(folder, fname)
            csv_to_parquet(in_path)
