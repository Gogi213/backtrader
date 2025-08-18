# Импорт универсальных функций для бэктеста и визуализации
from panel_app.backtest_runner import add_pf_visuals, run_backtest_unified
import panel as pn
pn.extension('plotly', 'bokeh', 'tabulator', raw_css=[
    '.bk-panel-models-tabulator-DataTabulator {padding-left:75px !important; width:1958px !important;}',
    '.bk-input, .bk-slider-title, .bk-slider-value, .bk-checkbox, .bk-checkbox-label, .bk-select, .bk-btn, .bk-date-picker, .bk-panel-models-input-TextInput, .bk-panel-models-input-IntInput, .bk-panel-models-input-FloatInput, .bk-panel-models-input-Select, .bk-panel-models-input-DatePicker, .bk-panel-models-input-Button, .bk-panel-models-input-Checkbox {font-size: 10px !important;}',
    '.bk-input-group label {display: block; margin-bottom: 8px;}',
])
from panel_app.ui.param_widgets import make_param_row, get_params_widgets
import pandas as pd
import plotly.graph_objs as go
import os
import nest_asyncio
nest_asyncio.apply()
import asyncio
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))

from data.fetch_binance_data_fast import fetch_binance_ohlcv_fast
from panel_app.strategies.zscore_atr_volume import run_zscore_atr_volume_strategy
from panel_app.strategies.core import run_vbt_strategy


# Унифицированная функция загрузки данных (binance или tradingview parquet)
def get_ohlcv_df(source, **kwargs):
    if source == 'binance':
        return load_ohlcv(kwargs['symbol'], kwargs['timeframe'], kwargs['start'], kwargs['end'])
    elif source == 'tradingview':
        return pd.read_parquet(kwargs['parquet_path'])
    else:
        raise ValueError('Unknown data source')

progress_bar = pn.indicators.Progress(name='Grid Search Progress', value=0, max=100, width=303, bar_color='primary', sizing_mode=None)
progress_text = pn.pane.Markdown('', width=303, margin=(0,0,20,0), sizing_mode=None)
output = pn.Column(sizing_mode='stretch_width')

from panel_app.data_utils.loader import load_ohlcv

# --- UI ---
disable_trades_chart = pn.widgets.Checkbox(name='Отключить график', value=True)
enable_grid_search = pn.widgets.Checkbox(name='Перебор', value=False)

# --- Синхронизация чекбокса 'Перебор' с чекбоксами параметров ---
def sync_param_checkboxes(event=None):
    # Получаем все строки параметров текущей стратегии
    widgets = get_params_widgets(strategy_select.value, strategy_options)
    for w in widgets:
        if isinstance(w, pn.Column):
            for sub in w:
                if isinstance(sub, pn.Row) and len(sub) == 4:
                    # sub[3] — это чекбокс перебора
                    sub[3].value = enable_grid_search.value

enable_grid_search.param.watch(lambda event: sync_param_checkboxes(), 'value')
z_window = pn.widgets.IntInput(name='ZScore Window', value=30, step=1)
z_thresh = pn.widgets.FloatInput(name='ZScore Threshold', value=2.0, step=0.1)
atr_len = pn.widgets.IntInput(name='ATR Length', value=30, step=1)
vol_z_thresh = pn.widgets.FloatInput(name='Volume Z Threshold', value=0.5, step=0.1)
atr_min = pn.widgets.FloatInput(name='Min ATR', value=0.01, step=0.01)


strategy_options = [
    ('ZScoreATRVolume', 'Z-Score ATR Volume'),
]
strategy_select = pn.widgets.Select(name='Strategy', options=[x[1] for x in strategy_options], value='Z-Score ATR Volume')

symbol = pn.widgets.TextInput(name='Symbol', value='MYXUSDT')
timeframe = pn.widgets.Select(name='Timeframe', options=['1m', '5m', '15m', '1h', '4h', '1d'], value='1m')
start = pn.widgets.DatePicker(name='Start date', value=pd.to_datetime('2025-08-14'))
end = pn.widgets.DatePicker(name='End date', value=pd.to_datetime('2025-08-15'))
deposit = pn.widgets.FloatInput(name='Initial Deposit', value=10000, step=100)
commission = pn.widgets.FloatInput(name='Commission (%)', value=0.05, step=0.01)
leverage = pn.widgets.IntInput(name='Leverage', value=1, step=1, start=1)

run_btn = pn.widgets.Button(name='Run Backtest', button_type='primary')

download_btn = pn.widgets.Button(name='Download Cash', button_type='success')


from panel_app.tv_cache.tradingview_cache import get_tradingview_parquet_files

# Получаем список файлов и добавляем пустой элемент "-" в начало
tv_files = get_tradingview_parquet_files()
tv_options = ['-'] + tv_files if '-' not in tv_files else tv_files

tradingview_cache_select = pn.widgets.Select(
    name='Кеш TradingView (parquet)',
    options=tv_options,
    value='-',
    width=310
)

# --- Автоматическое управление доступностью полей ---

def update_fields_state(event=None):
    # Блокируем поля только если выбран НЕ дефолтный ('-') файл кеша TradingView
    is_tv = tradingview_cache_select.value not in (None, '', '-')
    for w in [symbol, timeframe, start, end]:
        w.disabled = is_tv

tradingview_cache_select.param.watch(update_fields_state, 'value')
update_fields_state()  # инициализация состояния при запуске

# Кнопка для запуска бэктеста с кешем TradingView
run_tv_btn = pn.widgets.Button(name='Run BT with TV cash', button_type='primary', width=310)

## генерация параметров и виджетов вынесена в panel_app/ui/param_widgets.py
## используйте get_params_widgets(strategy_key, strategy_options)

params_panel = pn.Column(*get_params_widgets(strategy_select.value, strategy_options))

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
            params_panel[:] = get_params_widgets(v, strategy_options)
            break
strategy_select.param.watch(update_params_panel, 'value')

download_output = pn.Column()


# --- Callback ---
def run_backtest(event=None):
    # Только Binance (API или кеш binance)
    try:
        from panel_app.backtest_runner import run_backtest_unified
        # Передаём реальные виджеты из params_panel
        current_widgets = list(params_panel)
        run_backtest_unified(
            source='binance',
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            strategy_options=strategy_options,
            params_widgets={strategy_options[0][0]: current_widgets},
            strategy_select=strategy_select,
            deposit=deposit,
            commission=commission,
            leverage=leverage,
            enable_grid_search=enable_grid_search,
            disable_trades_chart=disable_trades_chart,
            output=output,
            progress_bar=progress_bar,
            progress_text=progress_text
        )
    except Exception as e:
        output.objects = [pn.pane.Markdown(f'**Ошибка:** {e}')]

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


# Обработчик для кнопки запуска бэктеста на TradingView parquet
def run_backtest_tv_handler(event=None):
    parquet_file = tradingview_cache_select.value
    if not parquet_file or parquet_file == '-' or parquet_file.strip() == '':
        output.objects = [pn.pane.Markdown('❌ Не выбран файл кеша TradingView!')]
        return
    parquet_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache_tradingview', parquet_file)
    from panel_app.backtest_runner import run_backtest_unified
    current_widgets = list(params_panel)
    run_backtest_unified(
        source='tradingview',
        parquet_path=parquet_path,
        strategy_options=strategy_options,
        params_widgets={strategy_options[0][0]: current_widgets},
        strategy_select=strategy_select,
        deposit=deposit,
        commission=commission,
        leverage=leverage,
        enable_grid_search=enable_grid_search,
        disable_trades_chart=disable_trades_chart,
        output=output,
        progress_bar=progress_bar,
        progress_text=progress_text
    )

run_btn.on_click(run_backtest)
download_btn.on_click(download_cash)
run_tv_btn.on_click(run_backtest_tv_handler)

# UI
button_row = pn.Row(run_btn, download_btn)
controls = pn.Column(
    progress_text,
    progress_bar,
    strategy_select, symbol, timeframe, start, end, deposit, commission, leverage, params_panel,
    pn.Row(disable_trades_chart, enable_grid_search),
    button_row,
    tradingview_cache_select,
    run_tv_btn,
    download_output
)
app = pn.Row(controls, output, sizing_mode='stretch_width')

if __name__ == '__main__':
    pn.serve(app, show=True, port=5006)
