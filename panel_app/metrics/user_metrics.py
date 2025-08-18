import pandas as pd

def collect_user_metrics(stats, widgets, deposit, start, end, get_first_field):
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
    for w in widgets:
        label = None
        try:
            if hasattr(w, '__len__') and len(w) > 0 and hasattr(w[0], 'object'):
                label = w[0].object
            else:
                label = str(w)
        except Exception:
            label = str(w)
        try:
            first_field_widget = get_first_field(w)
            value = first_field_widget.value if hasattr(first_field_widget, 'value') else None
        except Exception:
            value = None
        if isinstance(label, str):
            label = label.replace('**', '').strip()
        user_metrics[label] = value
    stats_df = pd.DataFrame([user_metrics])
    return stats_df
