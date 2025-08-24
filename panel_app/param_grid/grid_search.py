import numpy as np
import itertools
import panel as pn
import pandas as pd

def extract_param_ranges(widgets):
    param_names = []
    param_ranges = []
    for w in widgets:
        # label
        label = None
        if isinstance(w, pn.Column) and len(w) > 0:
            label_obj = w[0]
            if hasattr(label_obj, 'object'):
                label = label_obj.object
            elif hasattr(label_obj, 'text'):
                label = label_obj.text
            else:
                label = str(label_obj)
        else:
            label = str(w)
        if isinstance(label, str):
            label = label.replace('**', '').strip()
        # поля
        row = None
        if isinstance(w, pn.Column):
            for sub in w:
                if isinstance(sub, pn.Row):
                    row = sub
                    break
        # Поддержка булевых параметров: Column(label, Checkbox) без Row
        if row is None:
            if isinstance(w, pn.Column) and len(w) >= 2:
                checkbox = w[1]
                # Чекбокс имеет .value; добавляем как фиксированное значение (вне перебора)
                if hasattr(checkbox, 'value'):
                    param_names.append(label)
                    param_ranges.append([bool(checkbox.value)])
                    continue
            # Иначе пропускаем элемент
            continue
        w_from, w_to, w_step, w_checkbox = row[0], row[1], row[2], row[3]
        v_from, v_to, v_step = w_from.value, w_to.value, w_step.value
        # Если чекбокс включён — перебираем диапазон, иначе используем ЛЕВОЕ поле 'от' как фиксированное значение
        if w_checkbox.value:
            # float/int auto
            if isinstance(v_from, float) or isinstance(v_to, float) or isinstance(v_step, float):
                arr = np.arange(float(v_from), float(v_to)+v_step/2, float(v_step))
            else:
                arr = np.arange(int(v_from), int(v_to)+int(v_step), int(v_step))
        else:
            # Раньше здесь использовалось среднее поле ("до"), что
            # приводило к неожиданному поведению. Теперь берём левое поле "от".
            arr = [v_from]
        param_names.append(label)
        param_ranges.append(arr)
    return param_names, param_ranges

def grid_search_params(widgets, strategy_key, run_strategy_func, deposit, commission, leverage, start, end, progress_bar=None, progress_text=None):
    param_names, param_ranges = extract_param_ranges(widgets)
    results = []
    total = np.prod([len(r) for r in param_ranges])
    if progress_bar is not None:
        progress_bar.max = int(total)
        progress_bar.value = 0
        progress_bar.visible = True
    if progress_text is not None:
        progress_text.object = f'0 из {total} / 0 из 100%'
        progress_text.visible = True
    for i, values in enumerate(itertools.product(*param_ranges), 1):
        params = dict(zip(param_names, values))
        try:
            pf, stats = run_strategy_func(params, deposit, commission, leverage)
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
            user_metrics.update(params)
            results.append(user_metrics)
        except Exception as e:
            results.append({'Ошибка': str(e), **params})
        if progress_bar is not None:
            progress_bar.value = i
        if progress_text is not None:
            percent = int(i / total * 100)
            progress_text.object = f'{i} из {total} / {percent} из 100%'
    if progress_bar is not None:
        progress_bar.value = progress_bar.max
        progress_bar.visible = True
    if progress_text is not None:
        progress_text.visible = True
    stats_df = pd.DataFrame(results)
    return stats_df
