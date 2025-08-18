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
