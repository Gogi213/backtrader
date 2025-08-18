import panel as pn

# --- UI parameter row generator and widgets for strategies ---
def make_param_row(widget):
    width = 70
    w_from = widget.__class__(name='', value=widget.value, step=getattr(widget, 'step', 1), width=width)
    w_to = widget.__class__(name='', value=widget.value, step=getattr(widget, 'step', 1), width=width)
    w_step = widget.__class__(name='', value=getattr(widget, 'step', 1), step=getattr(widget, 'step', 1), width=width)
    param_checkbox = pn.widgets.Checkbox(name='', value=False, width=18, margin=(13, 0, 0, 0))
    label = pn.pane.Markdown(f"**{widget.name}**", margin=(0,0,2,0))
    row = pn.Row(w_from, w_to, w_step, param_checkbox, width=350)
    return pn.Column(label, row, margin=(0,0,8,0))


# Описание параметров для автогенерации
strategy_param_specs = {
    'ZScoreATRVolume': [
        {'name': 'ZScore Window', 'type': 'int', 'default': 30, 'step': 1},
        {'name': 'ZScore Threshold', 'type': 'float', 'default': 2.0, 'step': 0.1},
        {'name': 'ATR Length', 'type': 'int', 'default': 30, 'step': 1},
        {'name': 'Volume Z Threshold', 'type': 'float', 'default': 0.5, 'step': 0.1},
        {'name': 'Min ATR', 'type': 'float', 'default': 0.01, 'step': 0.01},
    ],
    'ZScoreATRVolume1235': [
        {'name': 'ZScore Window', 'type': 'int', 'default': 30, 'step': 1},
        {'name': 'ZScore Threshold', 'type': 'float', 'default': 2.0, 'step': 0.1},
        {'name': 'ATR Length', 'type': 'int', 'default': 30, 'step': 1},
        {'name': 'Volume Z Threshold', 'type': 'float', 'default': 0.5, 'step': 0.1},
        {'name': 'Min ATR', 'type': 'float', 'default': 0.01, 'step': 0.01},
    ],
}

def make_param_row_auto(param):
    width = 70
    if param['type'] == 'int':
        widget_cls = pn.widgets.IntInput
    else:
        widget_cls = pn.widgets.FloatInput
    w_from = widget_cls(name='', value=param['default'], step=param['step'], width=width)
    w_to = widget_cls(name='', value=param['default'], step=param['step'], width=width)
    w_step = widget_cls(name='', value=param['step'], step=param['step'], width=width)
    param_checkbox = pn.widgets.Checkbox(name='', value=False, width=18, margin=(13, 0, 0, 0))
    label = pn.pane.Markdown(f"**{param['name']}**", margin=(0,0,2,0))
    row = pn.Row(w_from, w_to, w_step, param_checkbox, width=350)
    return pn.Column(label, row, margin=(0,0,8,0))

def get_params_widgets(strategy_key, strategy_options):
    for k, v in strategy_options:
        if v == strategy_key:
            return [make_param_row_auto(param) for param in strategy_param_specs.get(k, [])]
    return []

def extract_strategy_params(strategy_key, strategy_options, widgets):
    """
    Универсальная функция для извлечения параметров стратегии из виджетов
    """
    params = {}
    
    # Найдем ключ стратегии
    strategy_spec_key = None
    for k, v in strategy_options:
        if v == strategy_key:
            strategy_spec_key = k
            break
    
    if not strategy_spec_key or strategy_spec_key not in strategy_param_specs:
        return params
    
    param_specs = strategy_param_specs[strategy_spec_key]
    
    # Извлекаем значения из виджетов
    for i, param_spec in enumerate(param_specs):
        if i < len(widgets):
            widget = widgets[i]
            # Получаем первый виджет из Column (это основной input)
            if hasattr(widget, 'objects') and len(widget.objects) > 1:
                # widget.objects[1] это Row с виджетами [from, to, step, checkbox]
                row = widget.objects[1]
                if hasattr(row, 'objects') and len(row.objects) > 0:
                    value_widget = row.objects[0]  # первый виджет - это значение
                    if hasattr(value_widget, 'value'):
                        params[param_spec['name']] = value_widget.value
    
    return params

def extract_grid_search_params(strategy_key, strategy_options, widgets):
    """
    Извлечение параметров для grid search из виджетов
    """
    grid_params = {}
    
    # Найдем ключ стратегии
    strategy_spec_key = None
    for k, v in strategy_options:
        if v == strategy_key:
            strategy_spec_key = k
            break
    
    if not strategy_spec_key or strategy_spec_key not in strategy_param_specs:
        return grid_params
    
    param_specs = strategy_param_specs[strategy_spec_key]
    
    # Извлекаем диапазоны для grid search
    for i, param_spec in enumerate(param_specs):
        if i < len(widgets):
            widget = widgets[i]
            # Получаем Row с виджетами [from, to, step, checkbox]
            if hasattr(widget, 'objects') and len(widget.objects) > 1:
                row = widget.objects[1]
                if hasattr(row, 'objects') and len(row.objects) >= 4:
                    from_widget = row.objects[0]
                    to_widget = row.objects[1]
                    step_widget = row.objects[2]
                    checkbox_widget = row.objects[3]
                    
                    if (hasattr(checkbox_widget, 'value') and checkbox_widget.value and
                        hasattr(from_widget, 'value') and hasattr(to_widget, 'value') and 
                        hasattr(step_widget, 'value')):
                        
                        grid_params[param_spec['name']] = {
                            'from': from_widget.value,
                            'to': to_widget.value,
                            'step': step_widget.value
                        }
    
    return grid_params
