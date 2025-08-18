# backtest_runner.py
"""
Модуль для запуска бэктеста и визуализации результатов (скопировано из panel_app.py, без изменений).
"""


# Все импорты должны быть внутри функций или передаваться через параметры, чтобы не было конфликтов с глобальными переменными panel_app.py

# --- Универсальная функция для построения графиков и таблицы сделок по pf ---
def add_pf_visuals(elements, pf, df, widgets, deposit, start, end, get_first_field, symbol=None, disable_trades_chart=None):
    try:
        import panel as pn
        import pandas as pd
        import os
        from panel_app.metrics.user_metrics import collect_user_metrics
        stats = pf.stats()
        stats_df = collect_user_metrics(stats, widgets, deposit, start, end, get_first_field)
        # --- Добавляем столбец Symbol первым ---
        symbol_val = symbol.value if symbol is not None and hasattr(symbol, 'value') else None
        if 'Symbol' not in stats_df.columns:
            stats_df.insert(0, 'Symbol', symbol_val if symbol_val else '')
        else:
            cols = ['Symbol'] + [c for c in stats_df.columns if c != 'Symbol']
            stats_df = stats_df[cols]
        stats_table = pn.widgets.Tabulator(stats_df, show_index=False, header_filters=True, disabled=True, width=700)
        def download_stats(event=None):
            save_dir = os.path.join(os.path.dirname(__file__), '..', 'downloads', 'metrics')
            os.makedirs(save_dir, exist_ok=True)
            def fmt_date(val):
                if isinstance(val, str):
                    try:
                        val = pd.to_datetime(val)
                    except Exception:
                        return ''
                return val.strftime('%Y%m%d') if hasattr(val, 'strftime') else ''
            start_str = fmt_date(start.value) if hasattr(start, 'value') else ''
            end_str = fmt_date(end.value) if hasattr(end, 'value') else ''
            if symbol_val:
                fname = f"{symbol_val}_{start_str}_{end_str}.csv"
            else:
                fname = f"_{start_str}_{end_str}.csv"
            save_path = os.path.join(save_dir, fname)
            stats_df.to_csv(save_path, index=False)
        download_btn = pn.widgets.Button(name='', button_type='default', width=30, height=30, icon='⬇', margin=(0,0,0,77))
        download_btn.on_click(download_stats)
        elements.append(pn.Row(download_btn, align='start'))
        elements.append(stats_table)
    except Exception as stats_err:
        elements.append(pn.pane.Markdown(f'❌ Ошибка при расчёте метрик: {stats_err}'))

    import pandas as pd
    from panel_app.plotting.trade_plots import plot_cumulative_profit, plot_trade_profits, plot_candlestick_with_trades
    trades = pf.trades.records.reset_index(drop=True)
    if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns:
        trades['entry_time'] = trades['entry_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
        trades['exit_time'] = trades['exit_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
    else:
        trades['entry_time'] = pd.NaT
        trades['exit_time'] = pd.NaT

    elements.append(plot_cumulative_profit(trades))
    elements.append(plot_trade_profits(trades))
    if disable_trades_chart is not None:
        candle_plot = plot_candlestick_with_trades(df, trades, disable_trades_chart.value)
        if candle_plot is not None:
            elements.append(candle_plot)

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

# Универсальный запуск бэктеста для binance и tradingview parquet
def run_backtest_unified(
    source='binance', parquet_path=None,
    symbol=None, timeframe=None, start=None, end=None,
    strategy_options=None, params_widgets=None, strategy_select=None,
    deposit=None, commission=None, leverage=None, enable_grid_search=None, disable_trades_chart=None, output=None,
    progress_bar=None, progress_text=None
):
    elements = []
    import panel as pn
    import pandas as pd
    from panel_app.data_utils.loader import load_ohlcv
    from panel_app.param_grid.grid_search import grid_search_params
    from panel_app.strategies.zscore_atr_volume import run_zscore_atr_volume_strategy
    from panel_app.strategies.zscore_atr_volume_1235 import run_zscore_atr_volume_strategy_1235
    from panel_app.strategies.core import run_vbt_strategy
    try:
        # Унифицированная функция загрузки данных (binance или tradingview parquet)
        def get_ohlcv_df(source, symbol=None, timeframe=None, start=None, end=None, parquet_path=None):
            if source == 'binance':
                return load_ohlcv(symbol.value, timeframe.value, str(start.value), str(end.value))
            elif source == 'tradingview':
                return pd.read_parquet(parquet_path)
            else:
                raise ValueError('Unknown data source')
        if source == 'binance':
            df = get_ohlcv_df('binance', symbol=symbol, timeframe=timeframe, start=start, end=end)
        elif source == 'tradingview':
            df = get_ohlcv_df('tradingview', parquet_path=parquet_path)
        else:
            raise ValueError('Unknown data source')
        from panel_app.ui.param_widgets import extract_strategy_params
        
        strategy_key = strategy_select.value
        widgets = params_widgets[list(params_widgets.keys())[0]]  # Получаем виджеты
        
        def get_first_field(item):
            if isinstance(item, pn.Row):
                return item[0]
            if isinstance(item, pn.Column):
                for sub in item:
                    if isinstance(sub, pn.Row) and len(sub) > 0:
                        return sub[0]
                for sub in item:
                    if hasattr(sub, 'value'):
                        return sub
                return item
            if hasattr(item, 'value'):
                return item
            return item
        
        pf = None
        best_stats = None

        # --- Выбор функции стратегии --- #
        strategy_func = None
        if strategy_key == 'Z-Score ATR Volume':
            strategy_func = run_zscore_atr_volume_strategy
        elif strategy_key == 'Z-Score ATR Volume 1235':
            strategy_func = run_zscore_atr_volume_strategy_1235
        else:
            raise ValueError(f'Unknown strategy: {strategy_key}')

        if not enable_grid_search.value:
            # Используем автоматическое извлечение параметров
            params = extract_strategy_params(strategy_key, strategy_options, widgets)
            
            pf, _ = strategy_func(
                df,
                window=params.get('ZScore Window', 30),
                z_thresh=params.get('ZScore Threshold', 2.0),
                atr_len=params.get('ATR Length', 30),
                vol_z_thresh=params.get('Volume Z Threshold', 0.5),
                atr_min=params.get('Min ATR', 0.01),
                fee=commission.value / 100,
                init_cash=deposit.value,
                leverage=leverage.value
            )
        else:
            def run_strategy_func(params, deposit, commission, leverage):
                pf, _ = strategy_func(
                    df,
                    window=params.get('ZScore Window'),
                    z_thresh=params.get('ZScore Threshold'),
                    atr_len=params.get('ATR Length'),
                    vol_z_thresh=params.get('Volume Z Threshold'),
                    atr_min=params.get('Min ATR'),
                    fee=commission.value / 100,
                    init_cash=deposit.value,
                    leverage=leverage.value
                )
                return pf, pf.stats()
            from panel_app.param_grid.grid_search import grid_search_params
            stats_df = grid_search_params(widgets, strategy_key, run_strategy_func, deposit, commission, leverage, start, end, progress_bar=progress_bar, progress_text=progress_text)
            symbol_val = None
            if source == 'binance':
                symbol_val = symbol.value if symbol is not None and hasattr(symbol, 'value') else None
            if 'Symbol' not in stats_df.columns:
                stats_df.insert(0, 'Symbol', symbol_val if symbol_val else '')
            else:
                cols = ['Symbol'] + [c for c in stats_df.columns if c != 'Symbol']
                stats_df = stats_df[cols]
            stats_table = pn.widgets.Tabulator(
                stats_df, show_index=False, header_filters=True, disabled=True, width=700,
                pagination='local', page_size=25
            )
            def download_stats(event=None):
                import os
                save_dir = os.path.join(os.path.dirname(__file__), '..', 'downloads', 'metrics')
                os.makedirs(save_dir, exist_ok=True)
                def fmt_date(val):
                    if isinstance(val, str):
                        try:
                            val = pd.to_datetime(val)
                        except Exception:
                            return ''
                    return val.strftime('%Y%m%d') if hasattr(val, 'strftime') else ''
                start_str = fmt_date(start.value) if hasattr(start, 'value') else ''
                end_str = fmt_date(end.value) if hasattr(end, 'value') else ''
                if symbol_val:
                    fname = f"{symbol_val}_{start_str}_{end_str}.csv"
                else:
                    fname = f"_{start_str}_{end_str}.csv"
                save_path = os.path.join(save_dir, fname)
                stats_df.to_csv(save_path, index=False)
            download_btn = pn.widgets.Button(name='', button_type='default', width=30, height=30, icon='⬇', margin=(0,0,0,77))
            download_btn.on_click(download_stats)
            elements.append(pn.Row(download_btn, align='start'))
            elements.append(stats_table)
            if not stats_df.empty:
                best_idx = stats_df['Net Profit'].idxmax() if 'Net Profit' in stats_df.columns else stats_df.index[0]
                best_row = stats_df.loc[best_idx]
                pf, _ = strategy_func(
                    df,
                    window=best_row['ZScore Window'],
                    z_thresh=best_row['ZScore Threshold'],
                    atr_len=best_row['ATR Length'],
                    vol_z_thresh=best_row['Volume Z Threshold'],
                    atr_min=best_row['Min ATR'],
                    fee=commission.value / 100,
                    init_cash=deposit.value,
                    leverage=leverage.value
                )
        if pf is not None and not enable_grid_search.value:
            add_pf_visuals(elements, pf, df, widgets, deposit, start, end, get_first_field, symbol=symbol, disable_trades_chart=disable_trades_chart)
        elif pf is not None and enable_grid_search.value:
            try:
                trades = pf.trades.records.reset_index(drop=True)
                from panel_app.plotting.trade_plots import plot_cumulative_profit, plot_trade_profits, plot_candlestick_with_trades
                if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns:
                    trades['entry_time'] = trades['entry_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
                    trades['exit_time'] = trades['exit_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
                else:
                    trades['entry_time'] = pd.NaT
                    trades['exit_time'] = pd.NaT
                elements.append(plot_cumulative_profit(trades))
                elements.append(plot_trade_profits(trades))
                if disable_trades_chart is not None:
                    candle_plot = plot_candlestick_with_trades(df, trades, disable_trades_chart.value)
                    if candle_plot is not None:
                        elements.append(candle_plot)
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
            except Exception as stats_err:
                elements.append(pn.pane.Markdown(f'❌ Ошибка при построении графиков/сделок: {stats_err}'))
    except Exception as e:
        elements = [pn.pane.Markdown(f'**Ошибка:** {e}')]
    if output is not None:
        output.objects = elements
    return elements

# Callback-функция для обычного запуска бэктеста (Binance)
def run_backtest(
    symbol,
    timeframe,
    start,
    end,
    strategy_options,
    params_widgets,
    strategy_select,
    deposit,
    commission,
    leverage,
    enable_grid_search,
    disable_trades_chart,
    output,
    event=None
):
    """
    Универсальный запуск бэктеста для Binance (без глобальных переменных).
    Все параметры должны передаваться явно.
    """
    elements = []
    try:
        import panel as pn
        import pandas as pd
        import os
        from panel_app.data_utils.loader import load_ohlcv
        from panel_app.param_grid.grid_search import grid_search_params
        from panel_app.strategies.zscore_atr_volume import run_zscore_atr_volume_strategy
        from panel_app.strategies.zscore_atr_volume_1235 import run_zscore_atr_volume_strategy_1235
        from panel_app.strategies.zscore_vbt_basic import run_zscore_vbt_strategy
        from panel_app.strategies.core import run_vbt_strategy
        from panel_app.plotting.trade_plots import plot_cumulative_profit, plot_trade_profits, plot_candlestick_with_trades

        df = load_ohlcv(symbol.value, timeframe.value, str(start.value), str(end.value))
        from panel_app.ui.param_widgets import extract_strategy_params
        
        strategy_key = strategy_select.value
        widgets = params_widgets[list(params_widgets.keys())[0]]  # Получаем виджеты
        
        def get_first_field(item):
            if isinstance(item, pn.Row):
                return item[0]
            if isinstance(item, pn.Column):
                for sub in item:
                    if isinstance(sub, pn.Row) and len(sub) > 0:
                        return sub[0]
                for sub in item:
                    if hasattr(sub, 'value'):
                        return sub
                return item
            if hasattr(item, 'value'):
                return item
            return item
        
        pf = None

        # --- Выбор функции стратегии --- #
        strategy_func = None
        if strategy_key == 'Z-Score ATR Volume':
            strategy_func = run_zscore_atr_volume_strategy
        elif strategy_key == 'Z-Score ATR Volume 1235':
            strategy_func = run_zscore_atr_volume_strategy_1235
        else:
            raise ValueError(f'Unknown strategy: {strategy_key}')

        if not enable_grid_search.value:
            # Используем автоматическое извлечение параметров
            params = extract_strategy_params(strategy_key, strategy_options, widgets)
            
            pf, _ = strategy_func(
                df,
                window=params.get('ZScore Window', 30),
                z_thresh=params.get('ZScore Threshold', 2.0),
                atr_len=params.get('ATR Length', 30),
                vol_z_thresh=params.get('Volume Z Threshold', 0.5),
                atr_min=params.get('Min ATR', 0.01),
                fee=commission.value / 100,
                init_cash=deposit.value,
                leverage=leverage.value
            )
        else:
            def run_strategy_func(params, deposit, commission, leverage):
                pf, _ = strategy_func(
                    df,
                    window=params.get('ZScore Window'),
                    z_thresh=params.get('ZScore Threshold'),
                    atr_len=params.get('ATR Length'),
                    vol_z_thresh=params.get('Volume Z Threshold'),
                    atr_min=params.get('Min ATR'),
                    fee=commission.value / 100,
                    init_cash=deposit.value,
                    leverage=leverage.value
                )
                return pf, pf.stats()
            stats_df = grid_search_params(widgets, strategy_key, run_strategy_func, deposit, commission, leverage, start, end)
            symbol_val = symbol.value if symbol is not None and hasattr(symbol, 'value') else None
            if 'Symbol' not in stats_df.columns:
                stats_df.insert(0, 'Symbol', symbol_val if symbol_val else '')
            else:
                cols = ['Symbol'] + [c for c in stats_df.columns if c != 'Symbol']
                stats_df = stats_df[cols]
            stats_table = pn.widgets.Tabulator(
                stats_df, show_index=False, header_filters=True, disabled=True, width=700,
                pagination='local', page_size=25
            )
            def download_stats(event=None):
                import os
                save_dir = os.path.join(os.path.dirname(__file__), '..', 'downloads', 'metrics')
                os.makedirs(save_dir, exist_ok=True)
                def fmt_date(val):
                    if isinstance(val, str):
                        try:
                            val = pd.to_datetime(val)
                        except Exception:
                            return ''
                    return val.strftime('%Y%m%d') if hasattr(val, 'strftime') else ''
                start_str = fmt_date(start.value) if hasattr(start, 'value') else ''
                end_str = fmt_date(end.value) if hasattr(end, 'value') else ''
                if symbol_val:
                    fname = f"{symbol_val}_{start_str}_{end_str}.csv"
                else:
                    fname = f"_{start_str}_{end_str}.csv"
                save_path = os.path.join(save_dir, fname)
                stats_df.to_csv(save_path, index=False)
            download_btn = pn.widgets.Button(name='', button_type='default', width=30, height=30, icon='⬇', margin=(0,0,0,77))
            download_btn.on_click(download_stats)
            elements.append(pn.Row(download_btn, align='start'))
            elements.append(stats_table)
            if not stats_df.empty:
                best_idx = stats_df['Net Profit'].idxmax() if 'Net Profit' in stats_df.columns else stats_df.index[0]
                best_row = stats_df.loc[best_idx]
                
                # Для Z-Score ATR Volume стратегии
                if 'Z-Score ATR Volume' in strategy_key:
                    pf, _ = run_zscore_atr_volume_strategy(
                        df,
                        window=best_row['ZScore Window'],
                        z_thresh=best_row['ZScore Threshold'],
                        atr_len=best_row['ATR Length'],
                        vol_z_thresh=best_row['Volume Z Threshold'],
                        atr_min=best_row['Min ATR'],
                        fee=commission.value / 100,
                        init_cash=deposit.value,
                        leverage=leverage.value
                    )
        if pf is not None and not enable_grid_search.value:
            add_pf_visuals(elements, pf, df, widgets, deposit, start, end, get_first_field, symbol=symbol, disable_trades_chart=disable_trades_chart)
        elif pf is not None and enable_grid_search.value:
            try:
                trades = pf.trades.records.reset_index(drop=True)
                if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns:
                    trades['entry_time'] = trades['entry_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
                    trades['exit_time'] = trades['exit_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
                else:
                    trades['entry_time'] = pd.NaT
                    trades['exit_time'] = pd.NaT
                elements.append(plot_cumulative_profit(trades))
                elements.append(plot_trade_profits(trades))
                candle_plot = plot_candlestick_with_trades(df, trades, disable_trades_chart.value)
                if candle_plot is not None:
                    elements.append(candle_plot)
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
            except Exception as stats_err:
                elements.append(pn.pane.Markdown(f'❌ Ошибка при построении графиков/сделок: {stats_err}'))
    except Exception as e:
        elements = [pn.pane.Markdown(f'**Ошибка:** {e}')]
    output.objects = elements
