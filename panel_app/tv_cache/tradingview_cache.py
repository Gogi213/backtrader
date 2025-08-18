import os

def get_tradingview_parquet_files():
    folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cache_tradingview')
    try:
        files = [f for f in os.listdir(folder) if f.endswith('.parquet')]
        files.sort()
        return files
    except Exception:
        return []
