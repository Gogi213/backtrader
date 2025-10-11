"""
Визуализация результатов бэктеста с использованием Plotly.
"""
import os
import webbrowser
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def plot_trades_chart(results: Dict[str, Any], save_path: str) -> Optional[str]:
    """
    Создает и сохраняет линейный график цены с нанесенными сделками.
    Использует Scattergl для максимальной производительности.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly не установлен. Визуализация невозможна. Установите: pip install plotly")
        return None

    final_backtest = results.get('final_backtest')
    if not final_backtest:
        print("В результатах отсутствует 'final_backtest'. Невозможно построить график.")
        return None

    indicator_data = final_backtest.get('indicator_data')
    trades = final_backtest.get('trades')

    if not indicator_data or 'times' not in indicator_data or 'prices' not in indicator_data:
        print("Отсутствуют данные индикаторов или цен. Невозможно построить график.")
        return None

    # Преобразуем данные в DataFrame
    df = pd.DataFrame({
        'time': pd.to_datetime(indicator_data['times'], unit='ms'),
        'open': indicator_data['opens'],
        'high': indicator_data['highs'],
        'low': indicator_data['lows'],
        'close': indicator_data['prices'], # 'prices' это close
    })
    df.set_index('time', inplace=True)

    # Создание фигуры
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Линейный график цены для производительности
    fig.add_trace(go.Scattergl(
        x=df.index,
        y=df['close'], # Используем только цены закрытия
        mode='lines',
        name='Цена',
        line=dict(color='lightblue', width=1)
    ), row=1, col=1)

    # Нанесение сделок
    if trades:
        long_entries = [t for t in trades if t['side'] == 'long']
        short_entries = [t for t in trades if t['side'] == 'short']
        
        # Используем Scattergl для аппаратного ускорения рендеринга большого кол-ва сделок
        fig.add_trace(go.Scattergl(
            x=[t['timestamp'] for t in long_entries],
            y=[t['entry_price'] for t in long_entries],
            mode='markers', name='Long Entry',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))
        fig.add_trace(go.Scattergl(
            x=[t['timestamp'] for t in short_entries],
            y=[t['entry_price'] for t in short_entries],
            mode='markers', name='Short Entry',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ))
        fig.add_trace(go.Scattergl(
            x=[t['exit_timestamp'] for t in trades],
            y=[t['exit_price'] for t in trades],
            mode='markers', name='Exit',
            marker=dict(color='blue', size=8, symbol='square')
        ))

    # Настройка лейаута
    fig.update_layout(
        title=f"График сделок для {results.get('symbol', 'N/A')}",
        xaxis_title="Время",
        yaxis_title="Цена",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    # Сохранение в файл
    try:
        fig.write_html(save_path)
        return save_path
    except Exception as e:
        print(f"Ошибка при сохранении графика: {e}")
        return None

def quick_plot_trades(results: Dict[str, Any], output_dir: str = "optimization_plots") -> Optional[str]:
    """
    Быстрое создание и сохранение графика сделок.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    strategy = results.get("strategy_name", "strategy")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{strategy}_trades_{ts}.html"
    save_path = os.path.join(output_dir, filename)
    
    return plot_trades_chart(results, save_path)

def plot_and_open(results: Dict[str, Any], output_dir: str = "optimization_plots") -> Optional[str]:
    """
    Создает график, сохраняет его и открывает в браузере.
    """
    plot_path = quick_plot_trades(results, output_dir)
    if plot_path and os.path.exists(plot_path):
        try:
            webbrowser.open(f"file://{os.path.abspath(plot_path)}")
            return plot_path
        except Exception as e:
            print(f"Не удалось открыть график в браузере: {e}")
            return plot_path
    return None

# Псевдонимы для обратной совместимости
quick_visualize = quick_plot_trades