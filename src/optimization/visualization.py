"""
Минималистичная визуализация с Plotly
Только самое необходимое для свечных графиков
"""
import os
import pandas as pd
import webbrowser
from datetime import datetime
from typing import Dict, Any, Optional

# Импортируем Plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_candlestick_chart(results: Dict[str, Any], save_path: str) -> Optional[str]:
    """
    Создает свечной график с входами и выходами
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly не установлен. Установите: pip install plotly")
        return None
    
    # Получаем данные
    fb = results.get("final_backtest", {})
    ind = fb.get("indicator_data", {})
    trades = fb.get("trades", [])
    
    if not trades or not ind:
        return None
    
    # Создаем DataFrame для свечей
    df = pd.DataFrame({
        'open': ind.get("open", ind["prices"]),
        'high': ind.get("high", ind["prices"]),
        'low': ind.get("low", ind["prices"]),
        'close': ind.get("close", ind["prices"]),
    })
    df.index = pd.to_datetime(ind["times"], unit="s")
    
    # Создаем график
    fig = go.Figure()
    
    # Добавляем свечи
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    
    # Собираем данные для маркеров по типам
    long_entries = []
    short_entries = []
    take_exits = []
    stop_exits = []
    
    # Разделяем сделки по типам
    for trade in trades:
        entry_time = pd.to_datetime(trade["timestamp"], unit="s")
        entry_price = trade["entry_price"]
        
        # Входы
        if trade["side"] == "long":
            long_entries.append((entry_time, entry_price))
        else:
            short_entries.append((entry_time, entry_price))
        
        # Выходы
        if "exit_price" in trade:
            exit_time = pd.to_datetime(trade.get("exit_timestamp", trade["timestamp"]), unit="s")
            exit_price = trade["exit_price"]
            
            # Определяем тип выхода (стоп или тейк)
            pnl = trade.get("pnl", 0)
            # Прибыль для лонга: pnl > 0, для шорта: pnl > 0 (уже учтено в бэктесте)
            is_profit = pnl > 0
            
            if is_profit:
                take_exits.append((exit_time, exit_price))
            else:
                stop_exits.append((exit_time, exit_price))
    
    # Добавляем входы в лонг
    if long_entries:
        times, prices = zip(*long_entries)
        fig.add_trace(go.Scatter(
            x=times, y=prices,
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='green'),
            name='Long Entry'
        ))
    
    # Добавляем входы в шорт
    if short_entries:
        times, prices = zip(*short_entries)
        fig.add_trace(go.Scatter(
            x=times, y=prices,
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red'),
            name='Short Entry'
        ))
    
    # Добавляем выходы по тейку
    if take_exits:
        times, prices = zip(*take_exits)
        fig.add_trace(go.Scatter(
            x=times, y=prices,
            mode='markers',
            marker=dict(symbol='x', size=12, color='green'),
            name='Take Profit'
        ))
    
    # Добавляем выходы по стопу
    if stop_exits:
        times, prices = zip(*stop_exits)
        fig.add_trace(go.Scatter(
            x=times, y=prices,
            mode='markers',
            marker=dict(symbol='x', size=12, color='red'),
            name='Stop Loss'
        ))
    
    # Настраиваем внешний вид
    fig.update_layout(
        title=f"{results.get('strategy_name', 'Strategy')} – {results.get('symbol', 'SYM')}",
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        width=1728,  # 90% of 1920
        height=800   # Уменьшенная высота чтобы не было прокрутки
    )
    
    # Сохраняем
    if save_path.endswith('.html'):
        fig.write_html(save_path)
    elif save_path.endswith('.svg'):
        try:
            fig.write_image(save_path, format='svg')
        except:
            # Если SVG не поддерживается, сохраняем как HTML
            html_path = save_path.rsplit('.', 1)[0] + '.html'
            fig.write_html(html_path)
            save_path = html_path
            print(f"SVG не поддерживается, сохранен как HTML: {save_path}")
    
    return save_path


def quick_plot(results: Dict[str, Any], output_dir: str = "optimization_plots") -> str:
    """
    Быстрое создание графика
    """
    os.makedirs(output_dir, exist_ok=True)
    
    strategy = results.get("strategy_name", "strategy")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{strategy}_trades_{ts}.html"
    save_path = os.path.join(output_dir, filename)
    
    return plot_candlestick_chart(results, save_path) or save_path


def plot_and_open(results: Dict[str, Any], output_dir: str = "optimization_plots") -> str:
    """
    Создает график и открывает его в браузере
    """
    plot_path = quick_plot(results, output_dir)
    
    # Открываем в браузере
    try:
        webbrowser.open(f"file://{os.path.abspath(plot_path)}")
        print(f"График открыт в браузере: {plot_path}")
    except Exception as e:
        print(f"Не удалось открыть график в браузере: {e}")
    
    return plot_path


# Класс для совместимости с TUI
class OptimizationVisualizer:
    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def plot_trades(self, output_dir: str = "optimization_plots",
                   fname: Optional[str] = None) -> Optional[str]:
        if fname:
            save_path = os.path.join(output_dir, fname)
            return plot_candlestick_chart(self.results, save_path)
        else:
            return quick_plot(self.results, output_dir)
    
    def plot_and_open(self, output_dir: str = "optimization_plots",
                     fname: Optional[str] = None) -> Optional[str]:
        if fname:
            save_path = os.path.join(output_dir, fname)
            plot_path = plot_candlestick_chart(self.results, save_path)
        else:
            plot_path = quick_plot(self.results, output_dir)
        
        # Открываем в браузере
        if plot_path and os.path.exists(plot_path):
            try:
                webbrowser.open(f"file://{os.path.abspath(plot_path)}")
                print(f"График открыт в браузере: {plot_path}")
            except Exception as e:
                print(f"Не удалось открыть график в браузере: {e}")
        
        return plot_path


# Псевдонимы для обратной совместимости
quick_visualize = quick_plot
quick_plot_trades = quick_plot