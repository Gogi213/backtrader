import panel as pn
pn.extension('plotly', 'bokeh', 'tabulator', raw_css=[
    '.bk-panel-models-tabulator-DataTabulator {padding-left:75px !important; width:1958px !important;}',
    '.bk-input, .bk-slider-title, .bk-slider-value, .bk-checkbox, .bk-checkbox-label, .bk-select, .bk-btn, .bk-date-picker, .bk-panel-models-input-TextInput, .bk-panel-models-input-IntInput, .bk-panel-models-input-FloatInput, .bk-panel-models-input-Select, .bk-panel-models-input-DatePicker, .bk-panel-models-input-Button, .bk-panel-models-input-Checkbox {font-size: 10px !important;}',
    '.bk-input-group label {display: block; margin-bottom: 8px;}',
])
import pandas as pd
import plotly.graph_objs as go
import os
import nest_asyncio
nest_asyncio.apply()
import asyncio
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))

from fetch_binance_data_fast import fetch_binance_ohlcv_fast
from panel_app.strategies.zscore_atr_volume import run_zscore_atr_volume_strategy
from panel_app.strategies.zscore_vbt_basic import run_zscore_vbt_strategy
from panel_app.strategies.core import run_vbt_strategy

output = pn.Column(sizing_mode='stretch_width')

# --- Кеширование данных ---
def load_ohlcv(symbol, timeframe, start, end):
    save_path = f'../data/cache_binance/{symbol}_{timeframe}_{start}_{end}_fast.parquet'
    if not os.path.exists(save_path):
        asyncio.get_event_loop().run_until_complete(fetch_binance_ohlcv_fast(symbol, timeframe, start, end, save_path=save_path))
    df = pd.read_parquet(save_path)
    df = df.dropna()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    return df

# --- UI ---
disable_trades_chart = pn.widgets.Checkbox(name='Отключить график', value=True)
enable_grid_search = pn.widgets.Checkbox(name='Перебор', value=False)
z_window = pn.widgets.IntInput(name='ZScore Window', value=30, step=1)
z_thresh = pn.widgets.FloatInput(name='ZScore Threshold', value=2.0, step=0.1)
atr_len = pn.widgets.IntInput(name='ATR Length', value=30, step=1)
vol_z_thresh = pn.widgets.FloatInput(name='Volume Z Threshold', value=0.5, step=0.1)
atr_min = pn.widgets.FloatInput(name='Min ATR', value=0.01, step=0.01)

strategy_options = [
    ('MeanReversion', 'Mean Reversion'),
    ('MomentumBreakout', 'Momentum Breakout'),
    ('ZScore', 'Z-Score'),
    ('ZScoreATRVolume', 'Z-Score ATR Volume'),
]
strategy_select = pn.widgets.Select(name='Strategy', options=[x[1] for x in strategy_options], value='Z-Score')

symbol = pn.widgets.TextInput(name='Symbol', value='MYXUSDT')
timeframe = pn.widgets.Select(name='Timeframe', options=['1m', '5m', '15m', '1h', '4h', '1d'], value='1m')
start = pn.widgets.DatePicker(name='Start date', value=pd.to_datetime('2025-08-14'))
end = pn.widgets.DatePicker(name='End date', value=pd.to_datetime('2025-08-15'))
deposit = pn.widgets.FloatInput(name='Initial Deposit', value=10000, step=100)
commission = pn.widgets.FloatInput(name='Commission (%)', value=0.05, step=0.01)
leverage = pn.widgets.IntInput(name='Leverage', value=1, step=1, start=1)

run_btn = pn.widgets.Button(name='Run Backtest', button_type='primary')
download_btn = pn.widgets.Button(name='Download Cash', button_type='success')

# Параметры для всех стратегий
def make_param_row(widget):
    width = 70
    # Создаём новые виджеты того же класса, но с пустым именем (чтобы не было подписи)
    w_from = widget.__class__(name='', value=widget.value, step=getattr(widget, 'step', 1), width=width)
    w_to = widget.__class__(name='', value=widget.value, step=getattr(widget, 'step', 1), width=width)
    w_step = widget.__class__(name='', value=getattr(widget, 'step', 1), step=getattr(widget, 'step', 1), width=width)
    label = pn.pane.Markdown(f"**{widget.name}**", margin=(0,0,2,0))
    # Только три поля в одну строку, без подписей
    row = pn.Row(w_from, w_to, w_step, width=312)
    return pn.Column(label, row, margin=(0,0,8,0))

params_widgets = {
    'MeanReversion': [
        make_param_row(pn.widgets.IntInput(name='Bollinger Period', value=20, step=1)),
        make_param_row(pn.widgets.FloatInput(name='Bollinger Deviation', value=2.0, step=0.1)),
    ],
    'MomentumBreakout': [
        make_param_row(pn.widgets.IntInput(name='Momentum Period', value=14, step=1)),
    ],
    'ZScore': [
        make_param_row(pn.widgets.IntInput(name='ZScore Window', value=60, step=1)),
        make_param_row(pn.widgets.FloatInput(name='Entry Z', value=2.0, step=0.1)),
        make_param_row(pn.widgets.FloatInput(name='Exit Z', value=0.0, step=0.1)),
        make_param_row(pn.widgets.IntInput(name='Max Hold Bars', value=30, step=1)),
    ],
    'ZScoreATRVolume': [
        make_param_row(pn.widgets.IntInput(name='ZScore Window', value=30, step=1)),
        make_param_row(pn.widgets.FloatInput(name='ZScore Threshold', value=2.0, step=0.1)),
        make_param_row(pn.widgets.IntInput(name='ATR Length', value=30, step=1)),
        make_param_row(pn.widgets.FloatInput(name='Volume Z Threshold', value=0.5, step=0.1)),
        make_param_row(pn.widgets.FloatInput(name='Min ATR', value=0.01, step=0.01)),
    ],
}

def get_params_widgets(strategy_key):
    for k, v in strategy_options:
        if v == strategy_key:
            return params_widgets[k]
    return []

params_panel = pn.Column(*params_widgets['MeanReversion'])

settings_panel = pn.Column(
    strategy_select,
    symbol,
    timeframe,
    start,
    end,
    deposit,
    commission,
    leverage,
    disable_trades_chart,
    params_panel,
    run_btn,
    download_btn,
    width=350
)

def update_params_panel(event):
    for k, v in strategy_options:
        if v == strategy_select.value:
            params_panel[:] = params_widgets[k]
            break
strategy_select.param.watch(update_params_panel, 'value')

download_output = pn.Column()

# --- Callback ---
def run_backtest(event=None):
    elements = []
    try:
        df = load_ohlcv(symbol.value, timeframe.value, str(start.value), str(end.value))
        strategy_key = None
        for k, v in strategy_options:
            if v == strategy_select.value:
                strategy_key = k
                break

        # Собираем параметры
        params = {}
        widgets = params_widgets[strategy_key]

        # <<< FIX: универсальная функция для извлечения первого виджета ('от') из структуры Column/Row/widget
        def get_first_field(item):
            # Если это pn.Row — возвращаем первый элемент
            if isinstance(item, pn.Row):
                return item[0]
            # Если это pn.Column — ищем внутри Row или первый виджет с .value
            if isinstance(item, pn.Column):
                # сначала попробуем найти pn.Row внутри
                for sub in item:
                    if isinstance(sub, pn.Row) and len(sub) > 0:
                        return sub[0]
                # иначе — первый элемент с атрибутом value
                for sub in item:
                    if hasattr(sub, 'value'):
                        return sub
                # fallback — вернуть сам объект (позволит упасть понятной ошибкой дальше)
                return item
            # Если это уже виджет — вернуть как есть
            if hasattr(item, 'value'):
                return item
            # Любой другой случай — вернуть оригинал
            return item
        # <<< /FIX

        # Если перебор не включён, берём только первое поле ('от') из каждой тройки
        if not enable_grid_search.value:
            if strategy_key == 'MeanReversion':
                params['bollinger_period'] = get_first_field(widgets[0]).value
                params['bollinger_dev'] = get_first_field(widgets[1]).value
                pf = run_vbt_strategy(df, 'MeanReversion', **params)
            elif strategy_key == 'MomentumBreakout':
                params['momentum_period'] = get_first_field(widgets[0]).value
                pf = run_vbt_strategy(df, 'MomentumBreakout', **params)
            elif strategy_key == 'ZScore':
                params['window'] = get_first_field(widgets[0]).value
                params['entry_z'] = get_first_field(widgets[1]).value
                params['exit_z'] = get_first_field(widgets[2]).value
                params['max_hold'] = get_first_field(widgets[3]).value
                pf, _ = run_zscore_vbt_strategy(
                    df,
                    window=params['window'],
                    entry_z=params['entry_z'],
                    exit_z=params['exit_z'],
                    max_hold=params['max_hold'],
                    init_cash=deposit.value,
                    fee=commission.value / 100,
                    leverage=leverage.value
                )
            elif strategy_key == 'ZScoreATRVolume':
                params['window'] = get_first_field(widgets[0]).value
                params['z_thresh'] = get_first_field(widgets[1]).value
                params['atr_len'] = get_first_field(widgets[2]).value
                params['vol_z_thresh'] = get_first_field(widgets[3]).value
                params['atr_min'] = get_first_field(widgets[4]).value
                pf, _ = run_zscore_atr_volume_strategy(
                    df,
                    window=params['window'],
                    z_thresh=params['z_thresh'],
                    atr_len=params['atr_len'],
                    vol_z_thresh=params['vol_z_thresh'],
                    atr_min=params['atr_min'],
                    fee=commission.value / 100,
                    init_cash=deposit.value,
                    leverage=leverage.value
                )
            else:
                raise ValueError('Unknown strategy')
        else:
            # ...логика перебора будет добавлена позже...
            raise NotImplementedError('Grid search is not implemented yet')

        try:
            stats = pf.stats()
            # Оставляем только пользовательские метрики
            user_metrics = {
                'Start Date': str(start.value) if hasattr(start, 'value') else str(start),
                'End Date': str(end.value) if hasattr(end, 'value') else str(end),
                'Initial Deposit': stats.get('Start Value', deposit.value),
                'Net Profit': stats.get('End Value', 0) - stats.get('Start Value', 0),
                'Profit Factor': stats.get('Profit Factor', None),
                'Sortino Ratio': stats.get('Sortino Ratio', None),
                'Gross Profit': stats.get('End Value', None),
                'Fees': stats.get('Total Fees Paid', None),
                'Drawdown': stats.get('Max Drawdown [%]', None),
                'Total Trades': stats.get('Total Trades', None),
                'Winrate': stats.get('Win Rate [%]', None),
                'Avg Winning Trade': stats.get('Avg Winning Trade [%]', None),
                'Avg Losing Trade': stats.get('Avg Losing Trade [%]', None),
            }
            # Добавляем параметры стратегии в метрики (только первое поле 'от')
            for w in widgets:
                # Если это Column с [label, Row(...)] — безопасно извлекаем label и значение
                label = None
                try:
                    if isinstance(w, pn.Column) and len(w) > 0:
                        # w[0] — Markdown label
                        label_obj = w[0]
                        # стараемся получить текст метки
                        if hasattr(label_obj, 'object'):
                            label = label_obj.object
                        elif hasattr(label_obj, 'text'):
                            label = label_obj.text
                        else:
                            label = str(label_obj)
                    else:
                        # fallback
                        label = str(w)
                except Exception:
                    label = str(w)
                # извлекаем значение через get_first_field (безопасно)
                try:
                    first_field_widget = get_first_field(w)
                    value = first_field_widget.value if hasattr(first_field_widget, 'value') else None
                except Exception:
                    value = None
                # чистим форматирование Markdown (**...**)
                if isinstance(label, str):
                    label = label.replace('**', '').strip()
                user_metrics[label] = value

            stats_df = pd.DataFrame([user_metrics])
            stats_table = pn.widgets.Tabulator(stats_df, show_index=False, header_filters=True, disabled=True, width=700)
            elements.append(stats_table)
        except Exception as stats_err:
            elements.append(pn.pane.Markdown(f'❌ Ошибка при расчёте метрик: {stats_err}'))

        # --- Получаем trades ДО построения графиков ---
        trades = pf.trades.records.reset_index(drop=True)
        # --- Свечной график с отображением сделок ---
        # Сначала вычисляем entry_time/exit_time для trades
        if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns:
            trades['entry_time'] = trades['entry_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
            trades['exit_time'] = trades['exit_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
        else:
            trades['entry_time'] = pd.NaT
            trades['exit_time'] = pd.NaT

        # cumulative profit area
        y_data = None
        y_label = ''
        if 'pnl_net' in trades.columns:
            trades['cum_pnl_net'] = trades['pnl_net'].cumsum()
            y_data = trades['cum_pnl_net']
            y_label = 'Cumulative Profit ($, net)'
        elif 'pnl' in trades.columns:
            trades['cum_pnl'] = trades['pnl'].cumsum()
            y_data = trades['cum_pnl']
            y_label = 'Cumulative Profit ($, gross)'
        if y_data is not None:
            area_trades_fig = go.Figure()
            area_trades_fig.add_trace(go.Scatter(
                x=trades.index + 1,
                y=y_data,
                fill='tozeroy',
                mode='lines',
                name=y_label,
                line=dict()
            ))
            area_trades_fig.update_layout(title='Cumulative Profit (Area)', height=600, xaxis_title='Trade #', yaxis_title=y_label)
            trades_plotly_pane = pn.pane.Plotly(area_trades_fig, config={'responsive': True}, sizing_mode='stretch_width')
            elements.append(trades_plotly_pane)

        # profit per trade
        try:
            if 'pnl_net' in trades.columns:
                y_profit = trades['pnl_net']
            elif 'pnl' in trades.columns:
                y_profit = trades['pnl']
            else:
                y_profit = None
            x_time = trades['entry_time'] if 'entry_time' in trades.columns else trades.index
            if y_profit is not None and x_time is not None:
                profit_fig = go.Figure()
                profit_fig.add_trace(go.Bar(
                    x=x_time,
                    y=y_profit,
                    name='Trade Profit',
                    # marker_color left default (no explicit color)
                ))
                profit_fig.update_layout(
                    title='Прибыль по сделкам',
                    height=500,
                    xaxis_title='Время входа',
                    yaxis_title='Прибыль по сделке',
                    bargap=0.2
                )
                profit_pane = pn.pane.Plotly(profit_fig, config={'responsive': True}, sizing_mode='stretch_width')
                elements.append(profit_pane)
            else:
                elements.append(pn.pane.Markdown('⚠️ Нет данных по прибыли сделок.'))
        except Exception as eqerr:
            elements.append(pn.pane.Markdown(f'❌ Ошибка при построении графика прибыли по сделкам: {eqerr}'))

        # свечной график
        if not disable_trades_chart.value:
            try:
                candle_fig = go.Figure()
                candle_fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC',
                    showlegend=False
                ))
                if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns:
                    show_trades = trades.head(100)
                    for _, trade in show_trades.iterrows():
                        entry_time = trade.get('entry_time')
                        entry_price = trade.get('entry_price')
                        exit_time = trade.get('exit_time')
                        exit_price = trade.get('exit_price')
                        side = trade.get('side', None)
                        if side is None and all(x in trade for x in ['entry_price','exit_price','pnl']):
                            if trade['entry_price'] == trade['exit_price'] or trade['pnl'] == 0:
                                side = 'flat'
                            elif (trade['exit_price'] > trade['entry_price'] and trade['pnl'] > 0) or (trade['exit_price'] < trade['entry_price'] and trade['pnl'] < 0):
                                side = 'long'
                            elif (trade['exit_price'] < trade['entry_price'] and trade['pnl'] > 0) or (trade['exit_price'] > trade['entry_price'] and trade['pnl'] < 0):
                                side = 'short'
                            else:
                                side = 'unknown'
                        if side == 'short':
                            entry_marker = dict(symbol='triangle-down', size=14, line=dict(width=2, color='black'))
                            exit_marker = dict(symbol='triangle-up', size=14, line=dict(width=2, color='black'))
                        else:
                            entry_marker = dict(symbol='triangle-up', size=14, line=dict(width=2, color='black'))
                            exit_marker = dict(symbol='triangle-down', size=14, line=dict(width=2, color='black'))
                        candle_fig.add_trace(go.Scatter(
                            x=[entry_time],
                            y=[entry_price],
                            mode='markers',
                            marker=entry_marker,
                            name='Entry',
                            showlegend=False
                        ))
                        candle_fig.add_trace(go.Scatter(
                            x=[exit_time],
                            y=[exit_price],
                            mode='markers',
                            marker=exit_marker,
                            name='Exit',
                            showlegend=False
                        ))
                        candle_fig.add_trace(go.Scatter(
                            x=[entry_time, exit_time],
                            y=[entry_price, exit_price],
                            mode='lines',
                            line=dict(width=2, dash='dot'),
                            name='Trade',
                            showlegend=False
                        ))
                candle_fig.update_layout(title='Candlestick Chart with Trades', height=900, xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False)
                candle_plotly_pane = pn.pane.Plotly(candle_fig, config={'responsive': True}, sizing_mode='stretch_width')
                elements.append(candle_plotly_pane)
            except Exception as candle_err:
                elements.append(pn.pane.Markdown(f'❌ Ошибка при построении свечного графика: {candle_err}'))

        # Таблица сделок
        def detect_side(row):
            if 'entry_price' not in row or 'exit_price' not in row or 'pnl' not in row:
                return 'unknown'
            if row['entry_price'] == row['exit_price'] or row['pnl'] == 0:
                return 'flat'
            if (row['exit_price'] > row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] < row['entry_price'] and row['pnl'] < 0):
                return 'long'
            if (row['exit_price'] < row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] > row['entry_price'] and row['pnl'] < 0):
                return 'short'
            return 'unknown'
        trade_cols = ['entry_time', 'entry_price', 'exit_time', 'exit_price']
        if all(col in trades.columns for col in ['entry_price', 'exit_price', 'pnl']):
            trades['side'] = trades.apply(detect_side, axis=1)
            trade_cols.append('side')
        if 'pnl' in trades.columns:
            trade_cols.append('pnl')
        if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns:
            trades['entry_time'] = trades['entry_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
            trades['exit_time'] = trades['exit_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
        else:
            trades['entry_time'] = pd.NaT
            trades['exit_time'] = pd.NaT
        elements.append(pn.pane.DataFrame(trades[trade_cols], width=900, height=500, max_rows=20))
    except Exception as e:
        elements = [pn.pane.Markdown(f'**Ошибка:** {e}')]
    output.objects = elements

def download_cash(event=None):
    download_output.clear()
    symbol_val = symbol.value
    timeframe_val = timeframe.value
    start_val = str(start.value)
    end_val = str(end.value)
    save_path = f'../data/cache_binance/{symbol_val}_{timeframe_val}_{start_val}_{end_val}_fast.parquet'
    print(f'[Download Cash] Кнопка нажата для: {symbol_val} {timeframe_val} {start_val} {end_val}')
    if os.path.exists(save_path):
        print(f'[Download Cash] Файл уже есть: {save_path}')
        download_output.append(pn.pane.Markdown("Данные уже есть в кеше"))
    else:
        try:
            print(f'[Download Cash] Начинаю скачивание данных в {save_path}')
            loop = asyncio.get_event_loop()
            if loop.is_running():
                coro = fetch_binance_ohlcv_fast(symbol_val, timeframe_val, start_val, end_val, save_path=save_path)
                task = asyncio.ensure_future(coro)
                download_output.append(pn.pane.Markdown('⏳ Загрузка данных... (асинхронно)'))
            else:
                loop.run_until_complete(fetch_binance_ohlcv_fast(symbol_val, timeframe_val, start_val, end_val, save_path=save_path))
                print(f'[Download Cash] Данные успешно скачаны и сохранены: {save_path}')
                download_output.append(pn.pane.Markdown(f'✅ Данные успешно скачаны и сохранены: `{save_path}`'))
        except Exception as e:
            print(f'[Download Cash] Ошибка при загрузке: {e}')
            download_output.append(pn.pane.Markdown(f'❌ Ошибка при загрузке: {e}'))

run_btn.on_click(run_backtest)
download_btn.on_click(download_cash)

# UI
button_row = pn.Row(run_btn, download_btn)
controls = pn.Column(
    strategy_select, symbol, timeframe, start, end, deposit, commission, leverage, params_panel,
    pn.Row(disable_trades_chart, enable_grid_search),
    button_row,
    download_output
)
app = pn.Row(controls, output, sizing_mode='stretch_width')

if __name__ == '__main__':
    pn.serve(app, show=True, port=5006)
