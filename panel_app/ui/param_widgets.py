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

def get_params_widgets(strategy_key, strategy_options):
    for k, v in strategy_options:
        if v == strategy_key:
            return params_widgets[k]
    return []
