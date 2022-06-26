import pandas as pd

from gradecc.load_timeseries.load_ts import load_ts_cortex, _integrate_ts_cortex_subc
from gradecc.load_timeseries.subcortical import load_ts_subc

# todo bad smell, needs class
INCLUDE_SUBC = True


def load_ts(subject, epoch: str, include_subcortex=INCLUDE_SUBC) -> pd.DataFrame:
    if include_subcortex:
        ts_subc = load_ts_subc(subject, epoch)
        ts_cortex = load_ts_cortex(subject, epoch)
        return _integrate_ts_cortex_subc(ts_cortex, ts_subc)
    else:
        return load_ts_cortex(subject, epoch)


def all_region_names(include_subcortex=INCLUDE_SUBC):
    if include_subcortex:
        df = load_ts(1, 'rest')
    else:
        df = load_ts_cortex(1, 'rest')
    return df.columns.tolist()
