import pandas as pd
import warnings

pd.options.mode.chained_assignment = None
warnings.simplefilter("ignore")


def window_timeseries(epoch, timeseries):
    window_size, skip_initial_trials = 216, 3
    if epoch in ['rest', 'baseline', 'late']:
        return timeseries[-1 * window_size:]
    elif epoch == 'early':
        return timeseries[skip_initial_trials: skip_initial_trials + window_size]
    elif epoch == 'learning':
        return timeseries[window_size: 2 * window_size]


