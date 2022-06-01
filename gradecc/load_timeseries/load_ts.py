import warnings
import pandas as pd

from gradecc.load_timeseries.utils import _try_load_filenames, _window_timeseries

pd.options.mode.chained_assignment = None
warnings.simplefilter("ignore")


def _integrate_ts_cortex_subc(ts_cortex, ts_subc):
    if ts_cortex.shape[0] == ts_subc.shape[0]:
        return pd.concat([ts_cortex, ts_subc], axis=1)
    else:
        print('not the same size') # todo raise exception not proper size. maybe try except
        return


def load_ts_cortex(subject: int, epic: str) -> pd.DataFrame:
    """fMRI timeseries data
    Args:
        subject (int): subject number - or can be `dicom`
        epic (str): time period during the task, such as `rest`, `baseline`, `early` learning
    Returns:
        pd.DataFrame: columns=regions, rows=time trials
    """
    # todo make subject invariant to int or str
    timeseries = _try_load_filenames(epic, subject)
    return _window_timeseries(epic, timeseries)


# todo hmm. can separate followings. what correct structure? based on semantics! where I used it?
