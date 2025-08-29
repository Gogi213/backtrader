"""
Утилиты для мультиассет бэктестинга.
Позволяет запускать стратегию на множестве активов и агрегировать результаты.
"""
import pandas as pd
import os
from typing import List, Dict, Any
from ..strategies.core import run_vbt_strategy


def extract_symbol_from_filename(filename: str) -> str:
    """
    Извлекает символ актива из имени файла TradingView.
    
    Примеры:
    'BINANCE_BIOUSDT.P, 15S_9c830_binance_like.parquet' -> 'BIOUSDT'
    'BINANCE_CUDISUSDT.P, 15S_4fc8b_binance_like.parquet' -> 'CUDISUSDT'
    """
    try:
        # Убираем расширение
        name = filename.replace('.parquet', '').replace('.csv', '')
        
        # Паттерн: BINANCE_SYMBOL.P, ...
        if 'BINANCE_' in name and '.P,' in name:
            # Извлекаем часть между BINANCE_ и .P,
            start = name.find('BINANCE_') + len('BINANCE_')
            end = name.find('.P,')
            if start < end:
                return name[start:end]
        
        # Fallback: используем первую часть до первого символа разделителя
        for sep in ['.P,', '_', '.', ' ']:
            if sep in name:
                parts = name.split(sep)
                if len(parts) > 1 and 'BINANCE' not in parts[1]:
                    return parts[1] if parts[0] == 'BINANCE' else parts[0]
        
        return filename  # Если не смогли распарсить, вернем как есть
    except Exception:
        return filename


def run_multiasset_backtest(
    parquet_files: List[str], 
    strategy_name: str = 'ZScoreSMA',
    base_dir: str = None,
    **strategy_params
) -> pd.DataFrame:
    """
    Запускает бэктест на множестве активов и возвращает агрегированную таблицу результатов.
    
    Args:
        parquet_files: Список файлов parquet для обработки
        strategy_name: Название стратегии
        base_dir: Базовая директория для файлов (если None, используется cache_tradingview)
        **strategy_params: Параметры стратегии
        
    Returns:
        pd.DataFrame: Таблица с результатами по каждому активу
    """
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'cache_tradingview'
        ))
    
    results = []
    
    for filename in parquet_files:
        try:
            # Загружаем данные
            file_path = os.path.join(base_dir, filename)
            if not os.path.exists(file_path):
                continue
                
            df = pd.read_parquet(file_path)
            symbol = extract_symbol_from_filename(filename)
            
            # Проверяем что есть нужные колонки
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                continue
            
            # Запускаем стратегию
            pf, signals = run_vbt_strategy(df, strategy_name, **strategy_params)
            
            # Собираем базовые метрики
            try:
                stats = pf.stats()
                start_value = stats.get('Start Value', strategy_params.get('init_cash', 10000))
                end_value = stats.get('End Value', pf.value().iloc[-1])
                total_return = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0
                
                result = {
                    'Symbol': symbol,
                    'Filename': filename,
                    'Start Value': float(start_value),
                    'End Value': float(end_value),
                    'Total Return [%]': float(total_return),
                    'Sharpe Ratio': float(stats.get('Sharpe Ratio', 0)) if stats.get('Sharpe Ratio') else None,
                    'Max Drawdown [%]': float(stats.get('Max Drawdown [%]', 0)) if stats.get('Max Drawdown [%]') else None,
                    'Trade Count': len(pf.trades.records),
                }
                
                # Дополнительные метрики из сделок
                if len(pf.trades.records) > 0:
                    trades = pf.trades.records
                    if 'pnl' in trades.columns:
                        wins = len(trades[trades['pnl'] > 0])
                        total_trades = len(trades)
                        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                        result['Win Rate [%]'] = float(win_rate)
                        
                        # Profit Factor
                        gains = trades[trades['pnl'] > 0]['pnl'].sum()
                        losses = abs(trades[trades['pnl'] < 0]['pnl'].sum())
                        profit_factor = (gains / losses) if losses > 0 else float('inf') if gains > 0 else 0
                        result['Profit Factor'] = float(profit_factor) if profit_factor != float('inf') else 999.0
                
            except Exception as e:
                # Fallback метрики если stats() не работает
                end_value = pf.value().iloc[-1]
                start_value = strategy_params.get('init_cash', 10000)
                total_return = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0
                
                result = {
                    'Symbol': symbol,
                    'Filename': filename,
                    'Start Value': float(start_value),
                    'End Value': float(end_value),
                    'Total Return [%]': float(total_return),
                    'Trade Count': len(pf.trades.records),
                    'Error': str(e)[:100]  # Первые 100 символов ошибки
                }
            
            results.append(result)
            
        except Exception as e:
            # Если файл не удалось обработать
            symbol = extract_symbol_from_filename(filename)
            results.append({
                'Symbol': symbol,
                'Filename': filename,
                'Error': f'Failed to process: {str(e)[:80]}'
            })
    
    # Создаем DataFrame и сортируем по доходности
    if results:
        df_results = pd.DataFrame(results)
        if 'Total Return [%]' in df_results.columns:
            df_results = df_results.sort_values('Total Return [%]', ascending=False)
        return df_results
    else:
        return pd.DataFrame()


def get_multiasset_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Создает сводную статистику по мультиассет результатам.
    
    Args:
        results_df: DataFrame с результатами от run_multiasset_backtest
        
    Returns:
        Dict с агрегированной статистикой
    """
    if results_df.empty:
        return {}
    
    # Фильтруем только успешные результаты (без ошибок)
    successful = results_df[~results_df.get('Error', pd.Series()).notna()].copy()
    
    if successful.empty:
        return {'Total Assets': len(results_df), 'Successful': 0, 'Failed': len(results_df)}
    
    summary = {
        'Total Assets': len(results_df),
        'Successful': len(successful),
        'Failed': len(results_df) - len(successful),
    }
    
    if 'Total Return [%]' in successful.columns:
        returns = successful['Total Return [%]'].dropna()
        if len(returns) > 0:
            summary.update({
                'Best Return [%]': float(returns.max()),
                'Worst Return [%]': float(returns.min()),
                'Average Return [%]': float(returns.mean()),
                'Median Return [%]': float(returns.median()),
                'Profitable Assets': int((returns > 0).sum()),
                'Profitable Rate [%]': float((returns > 0).sum() / len(returns) * 100),
            })
    
    if 'Trade Count' in successful.columns:
        trades = successful['Trade Count'].dropna()
        if len(trades) > 0:
            summary.update({
                'Total Trades': int(trades.sum()),
                'Avg Trades per Asset': float(trades.mean()),
            })
    
    return summary
