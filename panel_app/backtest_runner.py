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
        # В некоторых случаях vectorbt.stats может бросать ошибку при пустых данных/частоте
        try:
            stats = pf.stats()
        except Exception:
            stats = {}
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

    # --- Расчёт МПП/МПУ (макс. нереализ. прибыль/убыток) на интервале сделки) в % от цены входа ---
    def _calc_mpp_mpu(row):
        try:
            eidx = int(row.get('entry_idx', -1))
            xidx = int(row.get('exit_idx', -1))
            if eidx < 0 or xidx < 0 or xidx < eidx:
                return pd.Series({'MPP%': None, 'MPU%': None})
            # Срез по барам сделки [eidx:xidx] включительно
            highs = df['high'].iloc[eidx:xidx+1]
            lows = df['low'].iloc[eidx:xidx+1]
            entry_price = float(row.get('entry_price', None))
            if entry_price is None:
                return pd.Series({'MPP%': None, 'MPU%': None})
            side = row.get('side', None)
            # Если side ещё не вычислен, определим как выше
            if side is None and all(k in row for k in ['entry_price','exit_price','pnl']):
                if row['entry_price'] == row['exit_price'] or row['pnl'] == 0:
                    side = 'flat'
                elif (row['exit_price'] > row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] < row['entry_price'] and row['pnl'] < 0):
                    side = 'long'
                elif (row['exit_price'] < row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] > row['entry_price'] and row['pnl'] < 0):
                    side = 'short'
                else:
                    side = 'unknown'
            # Прайс-дельты
            if str(side).lower() == 'short':
                max_unreal = (entry_price - lows.min()) / entry_price * 100.0
                min_unreal = (entry_price - highs.max()) / entry_price * 100.0  # отрицат. (макс. просадка)
            else:  # long/unknown/flat трактуем как long
                max_unreal = (highs.max() - entry_price) / entry_price * 100.0
                min_unreal = (lows.min() - entry_price) / entry_price * 100.0   # отрицат. (макс. просадка)
            return pd.Series({'MPP%': float(max_unreal), 'MPU%': float(min_unreal)})
        except Exception:
            return pd.Series({'MPP%': None, 'MPU%': None})

    if {'entry_idx','exit_idx'}.issubset(trades.columns) and {'high','low'}.issubset(df.columns):
        mpp_mpu = trades.apply(_calc_mpp_mpu, axis=1)
        trades = pd.concat([trades, mpp_mpu], axis=1)

    # --- Z-Score на момент открытия сделки (значение на баре сигнала: entry_idx-1) ---
    try:
        z_window = None
        # Пытаемся взять окно из виджета
        if isinstance(widgets, dict) and ('ZScore Window' in widgets):
            z_w = widgets.get('ZScore Window')
            z_w = get_first_field(z_w)
            z_window = int(getattr(z_w, 'value', None) or z_w)
        if not z_window:
            # Часто по умолчанию 30; если нет виджета, используем дефолт
            z_window = 30
        if 'close' in df.columns:
            close_ser = df['close'].astype('float64')
            mean = close_ser.rolling(z_window).mean()
            std = close_ser.rolling(z_window).std().replace(0, pd.NA)
            z_ser = (close_ser - mean) / std
            def _z_at_entry(row):
                try:
                    eidx = int(row.get('entry_idx', -1))
                    idx = eidx - 1 if eidx > 0 else eidx
                    if idx < 0 or idx >= len(z_ser):
                        return None
                    val = z_ser.iloc[idx]
                    return float(val) if pd.notna(val) else None
                except Exception:
                    return None
            if 'entry_idx' in trades.columns:
                trades['zscore_entry'] = trades.apply(_z_at_entry, axis=1)
    except Exception:
        pass

    # --- nATR на момент выхода из позиции (exit_idx) ---
    try:
        if {'high','low','close'}.issubset(df.columns) and 'exit_idx' in trades.columns:
            import pandas as pd
            close_f = df['close'].astype('float64')
            high_f = df['high'].astype('float64')
            low_f = df['low'].astype('float64')
            prev_close = close_f.shift(1)
            tr = pd.concat([(high_f - low_f).abs(), (high_f - prev_close).abs(), (low_f - prev_close).abs()], axis=1).max(axis=1)
            period = 30
            # ATR (Wilder) = RMA(TR, 30) = EMA с alpha=1/period
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            natr_ser = (atr / close_f) * 100.0
            def _natr_at_exit(row):
                try:
                    xidx = int(row.get('exit_idx', -1))
                    if xidx < 0 or xidx >= len(natr_ser):
                        return None
                    val = natr_ser.iloc[xidx]
                    return float(val) if pd.notna(val) else None
                except Exception:
                    return None
            trades['NATR% (exit)'] = trades.apply(_natr_at_exit, axis=1)
    except Exception:
        pass

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
    # Определяем символ для таблицы сделок: из df['symbol'] (TV) или из виджета (Binance)
    symbol_val_trades = None
    try:
        if 'symbol' in df.columns and len(df['symbol']) > 0:
            # В TradingView parquet символ обычно одинаков во всём файле
            symbol_val_trades = str(df['symbol'].iloc[0])
    except Exception:
        pass
    if symbol_val_trades is None and symbol is not None:
        try:
            symbol_val_trades = str(getattr(symbol, 'value', symbol))
        except Exception:
            pass
    if symbol_val_trades is not None:
        trades['symbol'] = symbol_val_trades
    trade_cols = ['symbol'] + ['entry_time', 'entry_price', 'exit_time', 'exit_price'] if 'symbol' in trades.columns else ['entry_time', 'entry_price', 'exit_time', 'exit_price']
    # Вставляем nATR% на выход рядом с ценой выхода
    if 'NATR% (exit)' in trades.columns:
        try:
            insert_pos = trade_cols.index('exit_price') + 1
            trade_cols.insert(insert_pos, 'NATR% (exit)')
        except Exception:
            trade_cols.append('NATR% (exit)')
    if all(col in trades.columns for col in ['entry_price', 'exit_price', 'pnl']):
        trades['side'] = trades.apply(detect_side, axis=1)
        trade_cols.append('side')
    if 'pnl' in trades.columns:
        trade_cols.append('pnl')
        # Добавим PNL% рядом с pnl
        def _calc_pnl_pct(row):
            try:
                ep = row.get('entry_price', None)
                xp = row.get('exit_price', None)
                if ep is None or xp is None:
                    return None
                ep = float(ep)
                xp = float(xp)
                if ep == 0:
                    return None
                side = row.get('side', None)
                if side is None and all(k in row for k in ['entry_price','exit_price','pnl']):
                    if row['entry_price'] == row['exit_price'] or row['pnl'] == 0:
                        side = 'flat'
                    elif (row['exit_price'] > row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] < row['entry_price'] and row['pnl'] < 0):
                        side = 'long'
                    elif (row['exit_price'] < row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] > row['entry_price'] and row['pnl'] < 0):
                        side = 'short'
                    else:
                        side = 'unknown'
                if str(side).lower() == 'short':
                    return (ep - xp) / ep * 100.0
                else:
                    return (xp - ep) / ep * 100.0
            except Exception:
                return None
        trades['PNL%'] = trades.apply(_calc_pnl_pct, axis=1)
        if 'PNL%' in trades.columns:
            trade_cols.append('PNL%')
    # Z-Score at entry рядом с ценой входа
    if 'zscore_entry' in trades.columns:
        try:
            insert_pos = trade_cols.index('entry_price') + 1
            trade_cols.insert(insert_pos, 'zscore_entry')
        except Exception:
            trade_cols.append('zscore_entry')
    # Добавляем МПП% и МПУ%, если посчитали
    if 'MPP%' in trades.columns:
        trade_cols.append('MPP%')
    if 'MPU%' in trades.columns:
        trade_cols.append('MPU%')
    if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns:
        trades['entry_time'] = trades['entry_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
        trades['exit_time'] = trades['exit_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
    else:
        trades['entry_time'] = pd.NaT
        trades['exit_time'] = pd.NaT
    # Таблица сделок: пагинация по 25 строк, широкая
    trades_table = pn.widgets.Tabulator(
        trades[trade_cols],
        show_index=False,
        header_filters=True,
        # width убран, чтобы не конфликтовать со stretch_width
        height=None,
        pagination='local',
        page_size=25,
        sizing_mode='stretch_width',
    )
    elements.append(trades_table)

# Универсальный запуск бэктеста для binance и tradingview parquet
def run_backtest_unified(
    source='binance', parquet_path=None, parquet_paths=None, multi_asset=False,
    symbol=None, timeframe=None, start=None, end=None,
    strategy_options=None, params_widgets=None, strategy_select=None,
    deposit=None, commission=None, leverage=None, enable_grid_search=None, disable_trades_chart=None, output=None,
    progress_bar=None, progress_text=None, enable_natr_sl_tp=None
):
    elements = []
    import panel as pn
    import pandas as pd
    import os
    import json
    import importlib
    # Режим мультиассета для TradingView: собираем список файлов и обработаем их агрегированно ниже
    files_to_process = None
    if source == 'tradingview' and (parquet_paths or multi_asset):
        # Если список не передан, соберём из кеша TradingView
        if not parquet_paths:
            try:
                from panel_app.tv_cache.tradingview_cache import get_tradingview_parquet_files
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cache_tradingview'))
                files = get_tradingview_parquet_files()
                parquet_paths = [os.path.join(base_dir, f) for f in files]
            except Exception:
                parquet_paths = []
        files_to_process = parquet_paths[:]
    # Настройка частоты для vectorbt по выбранному таймфрейму
    def _map_tf_to_freq(tf):
        mapping = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
        }
        return mapping.get(str(tf).lower(), None)
    from panel_app.data_utils.loader import load_ohlcv
    from panel_app.param_grid.grid_search import grid_search_params
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
        if files_to_process is None:
            if source == 'binance':
                df = get_ohlcv_df('binance', symbol=symbol, timeframe=timeframe, start=start, end=end)
            elif source == 'tradingview':
                df = get_ohlcv_df('tradingview', parquet_path=parquet_path)
            else:
                raise ValueError('Unknown data source')
        # Устанавливаем глобальную частоту для vectorbt
        try:
            import vectorbt as vbt
            tf_val = timeframe.value if hasattr(timeframe, 'value') else timeframe
            freq_str = _map_tf_to_freq(tf_val)
            if freq_str:
                vbt.settings['freq'] = freq_str
        except Exception:
            pass
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

        # --- Динамическое разрешение функции стратегии из реестра --- #
        def _load_registry():
            registry_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategies', 'registry.json'))
            if not os.path.exists(registry_path):
                return { 'strategies': [] }
            with open(registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        def _resolve_strategy_func(selected_name, options):
            # Преобразуем отображаемое имя -> ключ
            selected_key = None
            for k, v in options:
                if v == selected_name:
                    selected_key = k
                    break
            if not selected_key:
                raise ValueError(f'Unknown strategy: {selected_name}')

            reg = _load_registry()
            for s in reg.get('strategies', []):
                if s.get('key') == selected_key:
                    module = s.get('module')
                    func_name = s.get('function')
                    if not module or not func_name:
                        break
                    mod = importlib.import_module(module)
                    if not hasattr(mod, func_name):
                        break
                    return getattr(mod, func_name)
            raise ValueError(f"Strategy '{selected_name}' not found in registry or function missing")

        strategy_func = _resolve_strategy_func(strategy_key, strategy_options)

        def _resolve_arg_map(selected_name, options):
            selected_key = None
            for k, v in options:
                if v == selected_name:
                    selected_key = k
                    break
            reg = _load_registry()
            for s in reg.get('strategies', []):
                if s.get('key') == selected_key:
                    return s.get('arg_map', {})
            return {}

        arg_map = _resolve_arg_map(strategy_key, strategy_options)

        if files_to_process is not None:
            # ----- Агрегированный мультиассет (TV): один набор графиков/таблиц -----
            strategy_key = strategy_select.value
            widgets = params_widgets[list(params_widgets.keys())[0]]
            params = extract_strategy_params(strategy_key, strategy_options, widgets)
            arg_map = _resolve_arg_map(strategy_key, strategy_options)
            kwargs_common = { arg_map.get(name, name): val for name, val in params.items() }
            kwargs_common.update(dict(fee=commission.value / 100, init_cash=deposit.value, leverage=leverage.value))
            if (
                enable_natr_sl_tp is not None and getattr(enable_natr_sl_tp, 'value', False)
                and strategy_key == 'Z-Score + SMA with SL/TP'
            ):
                kwargs_common.update(dict(use_natr_sl_tp=True, natr_len=30))

            from panel_app.plotting.trade_plots import plot_cumulative_profit, plot_trade_profits
            from panel_app.metrics.user_metrics import collect_user_metrics
            import numpy as _np

            # --- Поддержка перебора параметров в мультиассете TV ---
            if enable_grid_search is not None and getattr(enable_grid_search, 'value', False):
                def run_strategy_func(params, deposit, commission, leverage):
                    # Агрегируем метрики по всем файлам для набора params
                    import numpy as _np
                    start_val = float(deposit.value)
                    end_val = start_val
                    total_trades = 0
                    gains = 0.0
                    losses_abs = 0.0
                    wins_cnt = 0
                    total_fees_local = 0.0
                    pnl_pct_list = []  # для Avg Winning/Losing Trade
                    trade_time_pnls = []  # (entry_time, pnl) для DD/Sortino
                    for p in files_to_process:
                        try:
                            _df = get_ohlcv_df('tradingview', parquet_path=p)
                            kwargs = { arg_map.get(name, name): val for name, val in params.items() }
                            kwargs.update(dict(fee=commission.value / 100, init_cash=deposit.value, leverage=leverage.value))
                            if (
                                enable_natr_sl_tp is not None and getattr(enable_natr_sl_tp, 'value', False)
                                and strategy_key == 'Z-Score + SMA with SL/TP'
                            ):
                                kwargs.update(dict(use_natr_sl_tp=True, natr_len=30))
                            pf, _ = strategy_func(_df, **kwargs)
                            tr = pf.trades.records
                            total_trades += int(len(tr))
                            if 'pnl' in tr.columns:
                                gains += float(tr.loc[tr['pnl'] > 0, 'pnl'].sum())
                                losses_abs += float(-tr.loc[tr['pnl'] < 0, 'pnl'].sum())
                                wins_cnt += int((tr['pnl'] > 0).sum())
                            # Сбор PNL% и времени входа для агрегированных метрик
                            try:
                                import pandas as pd
                                # Восстановим entry_time при наличии индексов
                                if 'entry_idx' in tr.columns:
                                    tr = tr.copy()
                                    tr['entry_time'] = tr['entry_idx'].apply(lambda idx: _df.index[idx] if 0 <= idx < len(_df.index) else pd.NaT)
                                # Рассчёт PNL% (как в других ветках)
                                if all(col in tr.columns for col in ['entry_price','exit_price','pnl']):
                                    def _detect_side(row):
                                        if row['entry_price'] == row['exit_price'] or row['pnl'] == 0:
                                            return 'flat'
                                        if (row['exit_price'] > row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] < row['entry_price'] and row['pnl'] < 0):
                                            return 'long'
                                        if (row['exit_price'] < row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] > row['entry_price'] and row['pnl'] < 0):
                                            return 'short'
                                        return 'unknown'
                                    if 'side' not in tr.columns:
                                        tr['side'] = tr.apply(_detect_side, axis=1)
                                    def _pnl_pct(row):
                                        try:
                                            ep = float(row.get('entry_price', None))
                                            xp = float(row.get('exit_price', None))
                                            if ep == 0:
                                                return None
                                            if str(row.get('side','')).lower() == 'short':
                                                return (ep - xp) / ep * 100.0
                                            else:
                                                return (xp - ep) / ep * 100.0
                                        except Exception:
                                            return None
                                    tr['PNL%'] = tr.apply(_pnl_pct, axis=1)
                                # Накапливаем проценты и времена
                                if 'PNL%' in tr.columns:
                                    pnl_pct_list.extend([float(v) for v in tr['PNL%'].dropna().tolist()])
                                if {'entry_time','pnl'}.issubset(tr.columns):
                                    for t, v in zip(tr['entry_time'], tr['pnl']):
                                        try:
                                            if pd.notna(t):
                                                trade_time_pnls.append((pd.Timestamp(t), float(v)))
                                        except Exception:
                                            continue
                            except Exception:
                                pass
                            try:
                                ords = pf.orders.records
                                fee_col = 'fees' if 'fees' in ords.columns else ('fee' if 'fee' in ords.columns else None)
                                if fee_col:
                                    total_fees_local += float(ords[fee_col].sum())
                            except Exception:
                                pass
                            try:
                                value = pf.value()
                                if len(value) > 0:
                                    end_val += float(value.iloc[-1] - value.iloc[0])
                            except Exception:
                                pass
                        except Exception:
                            continue
                    # Подсчёты
                    profit_factor = float(gains / losses_abs) if losses_abs > 0 else (float('inf') if total_trades > 0 else None)
                    winrate_pct = float(wins_cnt / total_trades * 100.0) if total_trades > 0 else None
                    # Avg Winning/Losing Trade из собранных процентов
                    try:
                        import numpy as _np
                        if len(pnl_pct_list) > 0:
                            arr = _np.array(pnl_pct_list, dtype=float)
                            pos = arr[arr > 0]
                            neg = arr[arr < 0]
                            avg_win_pct = float(pos.mean()) if pos.size > 0 else None
                            avg_loss_pct = float(neg.mean()) if neg.size > 0 else None
                        else:
                            avg_win_pct = None
                            avg_loss_pct = None
                    except Exception:
                        avg_win_pct = None
                        avg_loss_pct = None

                    # Оценка DD и Sortino по последовательности сделок, упорядоченной по времени входа
                    try:
                        import pandas as pd
                        equity = [start_val]
                        if len(trade_time_pnls) > 0:
                            trade_time_pnls.sort(key=lambda x: x[0])
                            for _, pnl_v in trade_time_pnls:
                                equity.append(equity[-1] + float(pnl_v))
                        else:
                            equity.append(end_val)
                        v = _np.array(equity, dtype=float)
                        peaks = _np.maximum.accumulate(v)
                        dd = _np.where(peaks > 0, v / peaks - 1.0, 0.0)
                        max_dd_pct = float(abs(dd.min()) * 100.0)
                        rets = _np.diff(v) / v[:-1] if v.size > 1 else _np.array([0.0])
                        neg_rets = rets[rets < 0]
                        sortino = float(rets.mean() / (neg_rets.std(ddof=0) if neg_rets.size > 0 else _np.nan)) if _np.isfinite(rets.mean()) else None
                    except Exception:
                        max_dd_pct = None
                        sortino = None
                    stats = {
                        'Start Value': start_val,
                        'End Value': end_val,
                        'Total Trades': total_trades,
                        'Profit Factor': profit_factor,
                        'Sortino Ratio': sortino,
                        'Total Fees Paid': total_fees_local if total_fees_local else None,
                        'Max Drawdown [%]': max_dd_pct,
                        'Win Rate [%]': winrate_pct,
                        'Avg Winning Trade [%]': avg_win_pct,
                        'Avg Losing Trade [%]': avg_loss_pct,
                    }
                    return None, stats

                from panel_app.param_grid.grid_search import grid_search_params
                stats_df = grid_search_params(widgets, strategy_key, run_strategy_func, deposit, commission, leverage, start, end, progress_bar=progress_bar, progress_text=progress_text)
                # Добавим столбец символа для наглядности
                if 'Symbol' not in stats_df.columns:
                    stats_df.insert(0, 'Symbol', 'MULTI_TV')
                else:
                    cols = ['Symbol'] + [c for c in stats_df.columns if c != 'Symbol']
                    stats_df = stats_df[cols]
                stats_table = pn.widgets.Tabulator(stats_df, show_index=False, header_filters=True, disabled=True, width=700, pagination='local', page_size=25)
                if output is not None:
                    output.objects = []
                    output.append(stats_table)
                # Выберем лучшие параметры по Net Profit и отрисуем агрегированные визуализации
                if not stats_df.empty and output is not None:
                    best_idx = stats_df['Net Profit'].idxmax() if 'Net Profit' in stats_df.columns else stats_df.index[0]
                    best_row = stats_df.loc[best_idx]
                    ui_param_names = set(arg_map.keys())
                    best_params = { name: best_row[name] for name in best_row.index if name in ui_param_names }
                    # Построим агрегированные трейды по лучшим параметрам
                    all_trades = []
                    total_fees = 0.0
                    for p in files_to_process:
                        try:
                            _df = get_ohlcv_df('tradingview', parquet_path=p)
                            _kwargs = { arg_map.get(name, name): val for name, val in best_params.items() }
                            _kwargs.update(dict(fee=commission.value / 100, init_cash=deposit.value, leverage=leverage.value))
                            if (
                                enable_natr_sl_tp is not None and getattr(enable_natr_sl_tp, 'value', False)
                                and strategy_key == 'Z-Score + SMA with SL/TP'
                            ):
                                _kwargs.update(dict(use_natr_sl_tp=True, natr_len=30))
                            pf, _ = strategy_func(_df, **_kwargs)
                            tr = pf.trades.records.reset_index(drop=True)
                            import pandas as pd
                            if 'entry_idx' in tr.columns and 'exit_idx' in tr.columns:
                                tr['entry_time'] = tr['entry_idx'].apply(lambda idx: _df.index[idx] if 0 <= idx < len(_df.index) else pd.NaT)
                                tr['exit_time'] = tr['exit_idx'].apply(lambda idx: _df.index[idx] if 0 <= idx < len(_df.index) else pd.NaT)
                            else:
                                tr['entry_time'] = pd.NaT
                                tr['exit_time'] = pd.NaT
                            # side/PNL%
                            def _detect_side(row):
                                if 'entry_price' not in row or 'exit_price' not in row or 'pnl' not in row:
                                    return 'unknown'
                                if row['entry_price'] == row['exit_price'] or row['pnl'] == 0:
                                    return 'flat'
                                if (row['exit_price'] > row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] < row['entry_price'] and row['pnl'] < 0):
                                    return 'long'
                                if (row['exit_price'] < row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] > row['entry_price'] and row['pnl'] < 0):
                                    return 'short'
                                return 'unknown'
                            if all(col in tr.columns for col in ['entry_price','exit_price','pnl']):
                                tr['side'] = tr.apply(_detect_side, axis=1)
                                def _pnl_pct(row):
                                    try:
                                        ep = float(row.get('entry_price', None))
                                        xp = float(row.get('exit_price', None))
                                        if ep == 0:
                                            return None
                                        if str(row.get('side','')).lower() == 'short':
                                            return (ep - xp) / ep * 100.0
                                        else:
                                            return (xp - ep) / ep * 100.0
                                    except Exception:
                                        return None
                                tr['PNL%'] = tr.apply(_pnl_pct, axis=1)
                            # символ
                            sym = None
                            try:
                                if 'symbol' in _df.columns and len(_df['symbol']) > 0:
                                    sym = str(_df['symbol'].iloc[0])
                            except Exception:
                                pass
                            if sym is None:
                                sym = os.path.splitext(os.path.basename(p))[0]
                            tr['symbol'] = sym
                            all_trades.append(tr)
                            try:
                                ords = pf.orders.records
                                fee_col = 'fees' if 'fees' in ords.columns else ('fee' if 'fee' in ords.columns else None)
                                if fee_col:
                                    total_fees += float(ords[fee_col].sum())
                            except Exception:
                                pass
                        except Exception:
                            continue
                    if all_trades:
                        import pandas as pd
                        trades = pd.concat(all_trades, ignore_index=True)
                        # Метрики и визуализации как в одиночном пути
                        stats_dict = {
                            'Start Value': float(deposit.value),
                            'End Value': float(deposit.value) + float(trades['pnl'].sum()) if 'pnl' in trades.columns else float(deposit.value),
                            'Total Trades': int(len(trades)),
                            'Profit Factor': None,
                            'Sortino Ratio': None,
                            'Total Fees Paid': total_fees if total_fees else None,
                            'Max Drawdown [%]': None,
                            'Win Rate [%]': None,
                            'Avg Winning Trade [%]': None,
                            'Avg Losing Trade [%]': None,
                        }
                        # Не выводим отдельную таблицу метрик, чтобы избежать лишней "шапки" между таблицей грида и графиками
                        # stats_df_best = collect_user_metrics(stats_dict, widgets, deposit, start, end, get_first_field)
                        # output.append(stats_df_best)
                        output.append(plot_cumulative_profit(trades))
                        output.append(plot_trade_profits(trades))
                        # Таблица сделок
                        trade_cols = ['symbol'] if 'symbol' in trades.columns else []
                        trade_cols += ['entry_time', 'entry_price', 'exit_time', 'exit_price']
                        if 'PNL%' in trades.columns:
                            trade_cols += ['side', 'pnl', 'PNL%'] if 'side' in trades.columns else ['pnl', 'PNL%']
                        else:
                            trade_cols += ['side', 'pnl'] if 'side' in trades.columns else ['pnl']
                        trade_cols = [c for c in trade_cols if c in trades.columns]
                        trades_table = pn.widgets.Tabulator(trades[trade_cols], show_index=False, header_filters=True, pagination='local', page_size=25, sizing_mode='stretch_width')
                        output.append(trades_table)
                return None

            # --- Обычный (без перебора) путь агрегирования мультиассета ---
            all_trades = []
            total_fees = 0.0
            for p in files_to_process:
                try:
                    _df = get_ohlcv_df('tradingview', parquet_path=p)
                    pf, _ = strategy_func(_df, **kwargs_common)
                    tr = pf.trades.records.reset_index(drop=True)
                    # --- Восстановим entry_time/exit_time ---
                    import pandas as pd
                    if 'entry_idx' in tr.columns and 'exit_idx' in tr.columns:
                        tr['entry_time'] = tr['entry_idx'].apply(lambda idx: _df.index[idx] if 0 <= idx < len(_df.index) else pd.NaT)
                        tr['exit_time'] = tr['exit_idx'].apply(lambda idx: _df.index[idx] if 0 <= idx < len(_df.index) else pd.NaT)
                    else:
                        tr['entry_time'] = pd.NaT
                        tr['exit_time'] = pd.NaT
                    # --- MPP% / MPU% ---
                    def _calc_mpp_mpu(row):
                        try:
                            eidx = int(row.get('entry_idx', -1))
                            xidx = int(row.get('exit_idx', -1))
                            if eidx < 0 or xidx < 0 or xidx < eidx:
                                return pd.Series({'MPP%': None, 'MPU%': None})
                            highs = _df['high'].iloc[eidx:xidx+1]
                            lows = _df['low'].iloc[eidx:xidx+1]
                            entry_price = float(row.get('entry_price', None))
                            if entry_price is None:
                                return pd.Series({'MPP%': None, 'MPU%': None})
                            side = row.get('side', None)
                            if side is None and all(k in row for k in ['entry_price','exit_price','pnl']):
                                if row['entry_price'] == row['exit_price'] or row['pnl'] == 0:
                                    side = 'flat'
                                elif (row['exit_price'] > row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] < row['entry_price'] and row['pnl'] < 0):
                                    side = 'long'
                                elif (row['exit_price'] < row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] > row['entry_price'] and row['pnl'] < 0):
                                    side = 'short'
                                else:
                                    side = 'unknown'
                            if str(side).lower() == 'short':
                                max_unreal = (entry_price - lows.min()) / entry_price * 100.0
                                min_unreal = (entry_price - highs.max()) / entry_price * 100.0
                            else:
                                max_unreal = (highs.max() - entry_price) / entry_price * 100.0
                                min_unreal = (lows.min() - entry_price) / entry_price * 100.0
                            return pd.Series({'MPP%': float(max_unreal), 'MPU%': float(min_unreal)})
                        except Exception:
                            return pd.Series({'MPP%': None, 'MPU%': None})
                    if {'entry_idx','exit_idx'}.issubset(tr.columns) and {'high','low'}.issubset(_df.columns):
                        mpp_mpu = tr.apply(_calc_mpp_mpu, axis=1)
                        tr = pd.concat([tr, mpp_mpu], axis=1)
                    # --- zscore_entry на момент входа (entry_idx-1) ---
                    try:
                        z_window = 30
                        close_ser = _df['close'].astype('float64') if 'close' in _df.columns else None
                        if close_ser is not None:
                            mean = close_ser.rolling(z_window).mean()
                            std = close_ser.rolling(z_window).std().replace(0, pd.NA)
                            z_ser = (close_ser - mean) / std
                            def _z_at_entry(row):
                                try:
                                    eidx = int(row.get('entry_idx', -1))
                                    idx = eidx - 1 if eidx > 0 else eidx
                                    if idx < 0 or idx >= len(z_ser):
                                        return None
                                    val = z_ser.iloc[idx]
                                    return float(val) if pd.notna(val) else None
                                except Exception:
                                    return None
                            if 'entry_idx' in tr.columns:
                                tr['zscore_entry'] = tr.apply(_z_at_entry, axis=1)
                    except Exception:
                        pass
                    # --- nATR% на выходе ---
                    try:
                        if {'high','low','close'}.issubset(_df.columns) and 'exit_idx' in tr.columns:
                            close_f = _df['close'].astype('float64')
                            high_f = _df['high'].astype('float64')
                            low_f = _df['low'].astype('float64')
                            prev_close = close_f.shift(1)
                            trng = pd.concat([(high_f - low_f).abs(), (high_f - prev_close).abs(), (low_f - prev_close).abs()], axis=1).max(axis=1)
                            period = 30
                            atr = trng.ewm(alpha=1/period, adjust=False).mean()
                            natr_ser = (atr / close_f) * 100.0
                            def _natr_at_exit(row):
                                try:
                                    xidx = int(row.get('exit_idx', -1))
                                    if xidx < 0 or xidx >= len(natr_ser):
                                        return None
                                    val = natr_ser.iloc[xidx]
                                    return float(val) if pd.notna(val) else None
                                except Exception:
                                    return None
                            tr['NATR% (exit)'] = tr.apply(_natr_at_exit, axis=1)
                    except Exception:
                        pass
                    # --- side и PNL% ---
                    def _detect_side(row):
                        if 'entry_price' not in row or 'exit_price' not in row or 'pnl' not in row:
                            return 'unknown'
                        if row['entry_price'] == row['exit_price'] or row['pnl'] == 0:
                            return 'flat'
                        if (row['exit_price'] > row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] < row['entry_price'] and row['pnl'] < 0):
                            return 'long'
                        if (row['exit_price'] < row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] > row['entry_price'] and row['pnl'] < 0):
                            return 'short'
                        return 'unknown'
                    if all(col in tr.columns for col in ['entry_price','exit_price','pnl']):
                        tr['side'] = tr.apply(_detect_side, axis=1)
                        def _pnl_pct(row):
                            try:
                                ep = float(row.get('entry_price', None))
                                xp = float(row.get('exit_price', None))
                                if ep == 0:
                                    return None
                                if str(row.get('side','')).lower() == 'short':
                                    return (ep - xp) / ep * 100.0
                                else:
                                    return (xp - ep) / ep * 100.0
                            except Exception:
                                return None
                        tr['PNL%'] = tr.apply(_pnl_pct, axis=1)
                    # символ из файла
                    sym = None
                    try:
                        if 'symbol' in _df.columns and len(_df['symbol']) > 0:
                            sym = str(_df['symbol'].iloc[0])
                    except Exception:
                        pass
                    if sym is None:
                        sym = os.path.splitext(os.path.basename(p))[0]
                    tr['symbol'] = sym
                    all_trades.append(tr)
                    # fees
                    try:
                        ords = pf.orders.records
                        fee_col = 'fees' if 'fees' in ords.columns else ('fee' if 'fee' in ords.columns else None)
                        if fee_col:
                            total_fees += float(ords[fee_col].sum())
                    except Exception:
                        pass
                except Exception as _e:
                    if output is not None:
                        output.append(pn.pane.Markdown(f"❌ Ошибка для {os.path.basename(p)}: {_e}"))
            if not all_trades:
                if output is not None:
                    output.append(pn.pane.Markdown('❌ Нет данных для агрегирования'))
                return None
            import pandas as pd
            trades = pd.concat(all_trades, ignore_index=True)

            # Доп. вычисления для таблицы (как в add_pf_visuals)
            if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns and 'close' in _df.columns:
                # Для MPP/MPU и др. нужны high/low; возьмём из последнего df, иначе пропустим
                pass  # для агрегата опустим расчёт на барной истории, оставим основные поля

            # Метрики на базе всех сделок
            start_val = float(deposit.value)
            end_val = start_val
            if 'pnl' in trades.columns:
                try:
                    end_val = start_val + float(trades['pnl'].sum())
                except Exception:
                    end_val = start_val
            total_trades = int(len(trades))
            profit_factor = None
            winrate_pct = None
            avg_win_pct = None
            avg_loss_pct = None
            try:
                if 'pnl' in trades.columns:
                    gains = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
                    losses_abs = -trades.loc[trades['pnl'] < 0, 'pnl'].sum()
                    profit_factor = float(gains / losses_abs) if losses_abs > 0 else (float('inf') if total_trades > 0 else None)
                wins = len(trades.loc[trades['pnl'] > 0]) if 'pnl' in trades.columns else 0
                winrate_pct = float(wins / total_trades * 100.0) if total_trades > 0 else None
                if {'pnl','entry_price'}.issubset(trades.columns):
                    pnl_pct = trades['pnl'] / trades['entry_price'].replace(0, _np.nan) * 100.0
                    if _np.isfinite(pnl_pct).any():
                        if (pnl_pct > 0).any():
                            avg_win_pct = float(pnl_pct[pnl_pct > 0].mean())
                        if (pnl_pct < 0).any():
                            avg_loss_pct = float(pnl_pct[pnl_pct < 0].mean())
            except Exception:
                pass

            # Max Drawdown и Sortino по кривой сделок
            try:
                v = [start_val]
                if 'pnl' in trades.columns:
                    for x in trades['pnl'].to_list():
                        v.append(v[-1] + float(x))
                import numpy as _np
                v = _np.array(v, dtype=float)
                peaks = _np.maximum.accumulate(v)
                dd = _np.where(peaks > 0, v / peaks - 1.0, 0.0)
                max_dd_pct = float(abs(dd.min()) * 100.0)
                rets = _np.diff(v) / v[:-1] if v.size > 1 else _np.array([0.0])
                neg_rets = rets[rets < 0]
                sortino = float(rets.mean() / (neg_rets.std(ddof=0) if neg_rets.size > 0 else _np.nan)) if _np.isfinite(rets.mean()) else None
            except Exception:
                max_dd_pct = None
                sortino = None

            # Таблица метрик (агрегат) — тот же шаблон, что и в add_pf_visuals
            stats_dict = {
                'Start Value': start_val,
                'End Value': end_val,
                'Total Trades': total_trades,
                'Profit Factor': profit_factor,
                'Sortino Ratio': sortino,
                'Total Fees Paid': total_fees if total_fees else None,
                'Max Drawdown [%]': max_dd_pct,
                'Win Rate [%]': winrate_pct,
                'Avg Winning Trade [%]': avg_win_pct,
                'Avg Losing Trade [%]': avg_loss_pct,
            }
            # collect_user_metrics ожидает stats как словарь метрик vectorbt
            stats_df = collect_user_metrics(stats_dict, widgets, deposit, start, end, get_first_field)
            # Добавим столбец Symbol первым
            if 'Symbol' not in stats_df.columns:
                stats_df.insert(0, 'Symbol', 'ALL_TV')
            else:
                cols = ['Symbol'] + [c for c in stats_df.columns if c != 'Symbol']
                stats_df = stats_df[cols]
            stats_table = pn.widgets.Tabulator(stats_df, show_index=False, header_filters=True, disabled=True, width=700)
            if output is not None:
                output.objects = []  # очищаем и показываем один комплект
                output.append(stats_table)
                output.append(plot_cumulative_profit(trades))
                output.append(plot_trade_profits(trades))
                # Таблица сделок — тот же порядок, что и add_pf_visuals
                trade_cols = ['symbol'] if 'symbol' in trades.columns else []
                trade_cols += ['entry_time', 'entry_price', 'exit_time', 'exit_price']
                if 'NATR% (exit)' in trades.columns:
                    try:
                        insert_pos = trade_cols.index('exit_price') + 1
                        trade_cols.insert(insert_pos, 'NATR% (exit)')
                    except Exception:
                        trade_cols.append('NATR% (exit)')
                if all(col in trades.columns for col in ['entry_price', 'exit_price', 'pnl']):
                    trade_cols.append('side') if 'side' in trades.columns else None
                if 'pnl' in trades.columns:
                    trade_cols.append('pnl')
                    if 'PNL%' in trades.columns:
                        trade_cols.append('PNL%')
                if 'zscore_entry' in trades.columns:
                    try:
                        insert_pos = trade_cols.index('entry_price') + 1
                        trade_cols.insert(insert_pos, 'zscore_entry')
                    except Exception:
                        trade_cols.append('zscore_entry')
                if 'MPP%' in trades.columns:
                    trade_cols.append('MPP%')
                if 'MPU%' in trades.columns:
                    trade_cols.append('MPU%')
                # Отфильтруем на случай отсутствующих колонок
                trade_cols = [c for c in trade_cols if c in trades.columns]
                trades_table = pn.widgets.Tabulator(trades[trade_cols], show_index=False, header_filters=True, pagination='local', page_size=25, sizing_mode='stretch_width')
                output.append(trades_table)
            return None

        if not enable_grid_search.value:
            # Используем автоматическое извлечение параметров
            params = extract_strategy_params(strategy_key, strategy_options, widgets)
            # Сбор kwargs по arg_map
            kwargs = { arg_map.get(name, name): val for name, val in params.items() }
            # Базовые аргументы
            kwargs.update(dict(fee=commission.value / 100, init_cash=deposit.value, leverage=leverage.value))
            # Режим SL/TP по nATR только для стратегии Z-Score + SMA with SL/TP
            if (
                enable_natr_sl_tp is not None and getattr(enable_natr_sl_tp, 'value', False)
                and strategy_key == 'Z-Score + SMA with SL/TP'
            ):
                kwargs.update(dict(use_natr_sl_tp=True, natr_len=30))
            pf, _ = strategy_func(df, **kwargs)
        else:
            def run_strategy_func(params, deposit, commission, leverage):
                # Быстрый путь для перебора: избегаем pf.stats(), считаем ключевые метрики вручную
                import numpy as _np
                kwargs = { arg_map.get(name, name): val for name, val in params.items() }
                kwargs.update(dict(fee=commission.value / 100, init_cash=deposit.value, leverage=leverage.value))
                if (
                    enable_natr_sl_tp is not None and getattr(enable_natr_sl_tp, 'value', False)
                    and strategy_key == 'Z-Score + SMA with SL/TP'
                ):
                    kwargs.update(dict(use_natr_sl_tp=True, natr_len=30))
                pf, _ = strategy_func(df, **kwargs)
                # Equity curve и базовые значения
                try:
                    value = pf.value()
                    start_val = float(value.iloc[0]) if len(value) > 0 else float(deposit.value)
                    end_val = float(value.iloc[-1]) if len(value) > 0 else start_val
                    # Max Drawdown [%]
                    v = value.to_numpy(dtype=float)
                    if v.size > 0:
                        peaks = _np.maximum.accumulate(v)
                        dd = _np.where(peaks > 0, v / peaks - 1.0, 0.0)
                        max_dd_pct = float(abs(dd.min()) * 100.0)
                    else:
                        max_dd_pct = None
                    # Sortino (не годовой): сред. доходность / std отриц. доходностей
                    rets = _np.diff(v) / v[:-1] if v.size > 1 else _np.array([0.0])
                    neg_rets = rets[rets < 0]
                    sortino = float(rets.mean() / (neg_rets.std(ddof=0) if neg_rets.size > 0 else _np.nan)) if _np.isfinite(rets.mean()) else None
                except Exception:
                    start_val = float(deposit.value)
                    end_val = start_val
                    max_dd_pct = None
                    sortino = None
                # Trades-based метрики
                profit_factor = None
                winrate_pct = None
                avg_win_pct = None
                avg_loss_pct = None
                total_trades = None
                try:
                    tr = pf.trades.records
                    total_trades = int(len(tr))
                    if 'pnl' in tr.columns:
                        gains = tr.loc[tr['pnl'] > 0, 'pnl'].sum()
                        losses_abs = -tr.loc[tr['pnl'] < 0, 'pnl'].sum()
                        if losses_abs and losses_abs > 0:
                            profit_factor = float(gains / losses_abs) if gains is not None else None
                        else:
                            profit_factor = float('inf') if total_trades and total_trades > 0 else None
                    wins = len(tr.loc[tr['pnl'] > 0]) if 'pnl' in tr.columns else None
                    if total_trades and total_trades > 0 and wins is not None:
                        winrate_pct = float(wins / total_trades * 100.0)
                    # Попытка оценить средние % по сделке
                    if {'pnl', 'entry_price'}.issubset(tr.columns):
                        pnl_pct = tr['pnl'] / tr['entry_price'].replace(0, _np.nan) * 100.0
                        if _np.isfinite(pnl_pct).any():
                            if (pnl_pct > 0).any():
                                avg_win_pct = float(pnl_pct[pnl_pct > 0].mean())
                            if (pnl_pct < 0).any():
                                avg_loss_pct = float(pnl_pct[pnl_pct < 0].mean())
                except Exception:
                    pass
                # Fees
                total_fees = None
                try:
                    # Предпочтительно из orders.records
                    ords = pf.orders.records
                    fee_col = 'fees' if 'fees' in ords.columns else ('fee' if 'fee' in ords.columns else None)
                    if fee_col:
                        total_fees = float(ords[fee_col].sum())
                except Exception:
                    try:
                        total_fees = float(getattr(pf, 'total_fees', None)) if getattr(pf, 'total_fees', None) is not None else None
                    except Exception:
                        pass
                fast_stats = {
                    'Start Value': start_val,
                    'End Value': end_val,
                    'Total Trades': total_trades,
                    'Profit Factor': profit_factor,
                    'Sortino Ratio': sortino,
                    'Total Fees Paid': total_fees,
                    'Max Drawdown [%]': max_dd_pct,
                    'Win Rate [%]': winrate_pct,
                    'Avg Winning Trade [%]': avg_win_pct,
                    'Avg Losing Trade [%]': avg_loss_pct,
                }
                return pf, fast_stats
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
                # Берём только те имена, которые есть в arg_map (т.е. UI-параметры)
                ui_param_names = set(arg_map.keys())
                best_params = { name: best_row[name] for name in best_row.index if name in ui_param_names }
                kwargs = { arg_map.get(name, name): val for name, val in best_params.items() }
                kwargs.update(dict(fee=commission.value / 100, init_cash=deposit.value, leverage=leverage.value))
                if (
                    enable_natr_sl_tp is not None and getattr(enable_natr_sl_tp, 'value', False)
                    and strategy_key == 'Z-Score + SMA with SL/TP'
                ):
                    kwargs.update(dict(use_natr_sl_tp=True, natr_len=30))
                pf, _ = strategy_func(df, **kwargs)
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
                # nATR на момент выхода из позиции (exit_idx) для грид-ветки
                try:
                    if {'high','low','close'}.issubset(df.columns) and 'exit_idx' in trades.columns:
                        import pandas as pd
                        close_f = df['close'].astype('float64')
                        high_f = df['high'].astype('float64')
                        low_f = df['low'].astype('float64')
                        prev_close = close_f.shift(1)
                        tr = pd.concat([(high_f - low_f).abs(), (high_f - prev_close).abs(), (low_f - prev_close).abs()], axis=1).max(axis=1)
                        period = 30
                        # ATR (Wilder) = RMA(TR, 30)
                        atr = tr.ewm(alpha=1/period, adjust=False).mean()
                        natr_ser = (atr / close_f) * 100.0
                        def _natr_at_exit(row):
                            try:
                                xidx = int(row.get('exit_idx', -1))
                                if xidx < 0 or xidx >= len(natr_ser):
                                    return None
                                val = natr_ser.iloc[xidx]
                                return float(val) if pd.notna(val) else None
                            except Exception:
                                return None
                        trades['NATR% (exit)'] = trades.apply(_natr_at_exit, axis=1)
                except Exception:
                    pass
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
                # Определяем символ для таблицы сделок (грид-ветка)
                symbol_val_trades = None
                try:
                    if 'symbol' in df.columns and len(df['symbol']) > 0:
                        symbol_val_trades = str(df['symbol'].iloc[0])
                except Exception:
                    pass
                if symbol_val_trades is None and symbol is not None:
                    try:
                        symbol_val_trades = str(getattr(symbol, 'value', symbol))
                    except Exception:
                        pass
                if symbol_val_trades is not None:
                    trades['symbol'] = symbol_val_trades
                trade_cols = ['symbol'] + ['entry_time', 'entry_price', 'exit_time', 'exit_price'] if 'symbol' in trades.columns else ['entry_time', 'entry_price', 'exit_time', 'exit_price']
                # Вставляем NATR% (exit) рядом с ценой выхода (грид-ветка)
                if 'NATR% (exit)' in trades.columns:
                    try:
                        insert_pos = trade_cols.index('exit_price') + 1
                        trade_cols.insert(insert_pos, 'NATR% (exit)')
                    except Exception:
                        trade_cols.append('NATR% (exit)')
                if all(col in trades.columns for col in ['entry_price', 'exit_price', 'pnl']):
                    trades['side'] = trades.apply(detect_side, axis=1)
                    trade_cols.append('side')
                if 'pnl' in trades.columns:
                    trade_cols.append('pnl')
                    # PNL% для грид-ветки
                    def _calc_pnl_pct(row):
                        try:
                            ep = row.get('entry_price', None)
                            xp = row.get('exit_price', None)
                            if ep is None or xp is None:
                                return None
                            ep = float(ep)
                            xp = float(xp)
                            if ep == 0:
                                return None
                            side = row.get('side', None)
                            if side is None and all(k in row for k in ['entry_price','exit_price','pnl']):
                                if row['entry_price'] == row['exit_price'] or row['pnl'] == 0:
                                    side = 'flat'
                                elif (row['exit_price'] > row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] < row['entry_price'] and row['pnl'] < 0):
                                    side = 'long'
                                elif (row['exit_price'] < row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] > row['entry_price'] and row['pnl'] < 0):
                                    side = 'short'
                                else:
                                    side = 'unknown'
                            if str(side).lower() == 'short':
                                return (ep - xp) / ep * 100.0
                            else:
                                return (xp - ep) / ep * 100.0
                        except Exception:
                            return None
                    trades['PNL%'] = trades.apply(_calc_pnl_pct, axis=1)
                    if 'PNL%' in trades.columns:
                        trade_cols.append('PNL%')
                # Z-Score at entry для грид-ветки
                try:
                    z_window = None
                    if isinstance(widgets, dict) and ('ZScore Window' in widgets):
                        z_w = widgets.get('ZScore Window')
                        z_w = get_first_field(z_w)
                        z_window = int(getattr(z_w, 'value', None) or z_w)
                    if not z_window:
                        z_window = 30
                    if 'close' in df.columns:
                        close_ser = df['close'].astype('float64')
                        mean = close_ser.rolling(z_window).mean()
                        std = close_ser.rolling(z_window).std().replace(0, pd.NA)
                        z_ser = (close_ser - mean) / std
                        def _z_at_entry(row):
                            try:
                                eidx = int(row.get('entry_idx', -1))
                                idx = eidx - 1 if eidx > 0 else eidx
                                if idx < 0 or idx >= len(z_ser):
                                    return None
                                val = z_ser.iloc[idx]
                                return float(val) if pd.notna(val) else None
                            except Exception:
                                return None
                        if 'entry_idx' in trades.columns:
                            trades['zscore_entry'] = trades.apply(_z_at_entry, axis=1)
                except Exception:
                    pass
                if 'zscore_entry' in trades.columns:
                    try:
                        insert_pos = trade_cols.index('entry_price') + 1
                        trade_cols.insert(insert_pos, 'zscore_entry')
                    except Exception:
                        trade_cols.append('zscore_entry')
                # МПП/МПУ в % для грид-ветки
                def _calc_mpp_mpu(row):
                    try:
                        eidx = int(row.get('entry_idx', -1))
                        xidx = int(row.get('exit_idx', -1))
                        if eidx < 0 or xidx < 0 or xidx < eidx:
                            return pd.Series({'MPP%': None, 'MPU%': None})
                        highs = df['high'].iloc[eidx:xidx+1]
                        lows = df['low'].iloc[eidx:xidx+1]
                        entry_price = float(row.get('entry_price', None))
                        if entry_price is None:
                            return pd.Series({'MPP%': None, 'MPU%': None})
                        side = row.get('side', None)
                        if side is None and all(k in row for k in ['entry_price','exit_price','pnl']):
                            if row['entry_price'] == row['exit_price'] or row['pnl'] == 0:
                                side = 'flat'
                            elif (row['exit_price'] > row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] < row['entry_price'] and row['pnl'] < 0):
                                side = 'long'
                            elif (row['exit_price'] < row['entry_price'] and row['pnl'] > 0) or (row['exit_price'] > row['entry_price'] and row['pnl'] < 0):
                                side = 'short'
                            else:
                                side = 'unknown'
                        if str(side).lower() == 'short':
                            max_unreal = (entry_price - lows.min()) / entry_price * 100.0
                            min_unreal = (entry_price - highs.max()) / entry_price * 100.0
                        else:
                            max_unreal = (highs.max() - entry_price) / entry_price * 100.0
                            min_unreal = (lows.min() - entry_price) / entry_price * 100.0
                        return pd.Series({'MPP%': float(max_unreal), 'MPU%': float(min_unreal)})
                    except Exception:
                        return pd.Series({'MPP%': None, 'MPU%': None})
                if {'entry_idx','exit_idx'}.issubset(trades.columns) and {'high','low'}.issubset(df.columns):
                    mpp_mpu = trades.apply(_calc_mpp_mpu, axis=1)
                    trades = pd.concat([trades, mpp_mpu], axis=1)
                if 'MPP%' in trades.columns:
                    trade_cols.append('MPP%')
                if 'MPU%' in trades.columns:
                    trade_cols.append('MPU%')
                if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns:
                    trades['entry_time'] = trades['entry_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
                    trades['exit_time'] = trades['exit_idx'].apply(lambda idx: df.index[idx] if 0 <= idx < len(df.index) else pd.NaT)
                else:
                    trades['entry_time'] = pd.NaT
                    trades['exit_time'] = pd.NaT
                trades_table = pn.widgets.Tabulator(
                    trades[trade_cols],
                    show_index=False,
                    header_filters=True,
                    # width убран, чтобы не конфликтовать со stretch_width
                    height=None,
                    pagination='local',
                    page_size=25,
                    sizing_mode='stretch_width',
                )
                elements.append(trades_table)
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
        import json
        import importlib
        from panel_app.data_utils.loader import load_ohlcv
        from panel_app.param_grid.grid_search import grid_search_params
        from panel_app.strategies.core import run_vbt_strategy
        from panel_app.plotting.trade_plots import plot_cumulative_profit, plot_trade_profits, plot_candlestick_with_trades

        df = load_ohlcv(symbol.value, timeframe.value, str(start.value), str(end.value))
        # Устанавливаем глобальную частоту для vectorbt
        try:
            import vectorbt as vbt
            def _map_tf_to_freq(tf):
                mapping = {
                    '1m': '1T',
                    '5m': '5T',
                    '15m': '15T',
                    '1h': '1H',
                    '4h': '4H',
                    '1d': '1D',
                }
                return mapping.get(str(tf).lower(), None)
            freq_str = _map_tf_to_freq(timeframe.value)
            if freq_str:
                vbt.settings['freq'] = freq_str
        except Exception:
            pass
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

        # --- Динамическое разрешение функции стратегии из реестра --- #
        def _load_registry():
            registry_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategies', 'registry.json'))
            if not os.path.exists(registry_path):
                return { 'strategies': [] }
            with open(registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        def _resolve_strategy_func(selected_name, options):
            # Преобразуем отображаемое имя -> ключ
            selected_key = None
            for k, v in options:
                if v == selected_name:
                    selected_key = k
                    break
            if not selected_key:
                raise ValueError(f'Unknown strategy: {selected_name}')

            reg = _load_registry()
            for s in reg.get('strategies', []):
                if s.get('key') == selected_key:
                    module = s.get('module')
                    func_name = s.get('function')
                    if not module or not func_name:
                        break
                    mod = importlib.import_module(module)
                    if not hasattr(mod, func_name):
                        break
                    return getattr(mod, func_name)
            raise ValueError(f"Strategy '{selected_name}' not found in registry or function missing")

        strategy_func = _resolve_strategy_func(strategy_key, strategy_options)

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
                # Быстрый путь для перебора: избегаем pf.stats(), считаем ключевые метрики вручную
                import numpy as _np
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
                # Equity curve и базовые значения
                try:
                    value = pf.value()
                    start_val = float(value.iloc[0]) if len(value) > 0 else float(deposit.value)
                    end_val = float(value.iloc[-1]) if len(value) > 0 else start_val
                    v = value.to_numpy(dtype=float)
                    if v.size > 0:
                        peaks = _np.maximum.accumulate(v)
                        dd = _np.where(peaks > 0, v / peaks - 1.0, 0.0)
                        max_dd_pct = float(abs(dd.min()) * 100.0)
                    else:
                        max_dd_pct = None
                    rets = _np.diff(v) / v[:-1] if v.size > 1 else _np.array([0.0])
                    neg_rets = rets[rets < 0]
                    sortino = float(rets.mean() / (neg_rets.std(ddof=0) if neg_rets.size > 0 else _np.nan)) if _np.isfinite(rets.mean()) else None
                except Exception:
                    start_val = float(deposit.value)
                    end_val = start_val
                    max_dd_pct = None
                    sortino = None
                # Trades-based метрики
                profit_factor = None
                winrate_pct = None
                avg_win_pct = None
                avg_loss_pct = None
                total_trades = None
                try:
                    tr = pf.trades.records
                    total_trades = int(len(tr))
                    if 'pnl' in tr.columns:
                        gains = tr.loc[tr['pnl'] > 0, 'pnl'].sum()
                        losses_abs = -tr.loc[tr['pnl'] < 0, 'pnl'].sum()
                        if losses_abs and losses_abs > 0:
                            profit_factor = float(gains / losses_abs) if gains is not None else None
                        else:
                            profit_factor = float('inf') if total_trades and total_trades > 0 else None
                    wins = len(tr.loc[tr['pnl'] > 0]) if 'pnl' in tr.columns else None
                    if total_trades and total_trades > 0 and wins is not None:
                        winrate_pct = float(wins / total_trades * 100.0)
                    if {'pnl', 'entry_price'}.issubset(tr.columns):
                        pnl_pct = tr['pnl'] / tr['entry_price'].replace(0, _np.nan) * 100.0
                        if _np.isfinite(pnl_pct).any():
                            if (pnl_pct > 0).any():
                                avg_win_pct = float(pnl_pct[pnl_pct > 0].mean())
                            if (pnl_pct < 0).any():
                                avg_loss_pct = float(pnl_pct[pnl_pct < 0].mean())
                except Exception:
                    pass
                # Fees
                total_fees = None
                try:
                    ords = pf.orders.records
                    fee_col = 'fees' if 'fees' in ords.columns else ('fee' if 'fee' in ords.columns else None)
                    if fee_col:
                        total_fees = float(ords[fee_col].sum())
                except Exception:
                    try:
                        total_fees = float(getattr(pf, 'total_fees', None)) if getattr(pf, 'total_fees', None) is not None else None
                    except Exception:
                        pass
                fast_stats = {
                    'Start Value': start_val,
                    'End Value': end_val,
                    'Total Trades': total_trades,
                    'Profit Factor': profit_factor,
                    'Sortino Ratio': sortino,
                    'Total Fees Paid': total_fees,
                    'Max Drawdown [%]': max_dd_pct,
                    'Win Rate [%]': winrate_pct,
                    'Avg Winning Trade [%]': avg_win_pct,
                    'Avg Losing Trade [%]': avg_loss_pct,
                }
                return pf, fast_stats
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
