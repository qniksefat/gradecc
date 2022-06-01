import pandas as pd

from gradecc.load_timeseries.load_ts import load_ts_cortex, _integrate_ts_cortex_subc
from gradecc.load_timeseries.subcortical import load_ts_subc


def load_ts(subject, epic: str, include_subcortex=False) -> pd.DataFrame:
    if include_subcortex:
        ts_subc = load_ts_subc(subject, epic)
        ts_cortex = load_ts_cortex(subject, epic)
        return _integrate_ts_cortex_subc(ts_cortex, ts_subc)
    else:
        return load_ts_cortex(subject, epic)


def all_region_names():
    df = load_ts_cortex(1, 'rest')
    return df.columns.tolist()
