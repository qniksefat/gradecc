import typing

import pandas as pd

from gradecc.load_data.subject import Subject
from gradecc.load_data.utils import window_timeseries
from gradecc.load_data.cortex import try_load_filenames
from gradecc.load_data.subcortex import (_handle_if_subject_id,
                                         _make_subc_filename,
                                         _rename_columns_to_region_names)


def integrate_cortex_subcortex(ts_cortex, ts_subcortex):
    if ts_cortex.shape[0] == ts_subcortex.shape[0]:
        return pd.concat([ts_cortex, ts_subcortex], axis=1)
    else:
        raise ValueError('cortex and subcortex timeseries have not the same shape.')


class Timeseries:
    def __init__(self, subject_id: typing.Union[str, int], epoch: str, include_subcortex=True):
        # todo q: where to define type? in params or when casting
        self.subject = Subject(subject_id)
        self.epoch: str = epoch
        self.include_subcortex: bool = include_subcortex
        self.data: pd.DataFrame = pd.DataFrame()
        self.region_names = None
    # todo feature: if epoch is `whole`, combine all

    def load(self):
        if self.include_subcortex:
            ts_subcortex = self._load_ts_subc()
            ts_cortex = self._load_ts_cortex()
            self.data = integrate_cortex_subcortex(ts_cortex, ts_subcortex)
        else:
            self.data = self._load_ts_cortex()

        self.region_names = self.data.columns.tolist()

    def _load_ts_cortex(self) -> pd.DataFrame:
        """ loads timeseries for cortical regions
        Args:
            self.subject.int (int): subject number - or can be `dicom`
            self.epoch (str): time period during the task, such as `rest`, `baseline`, `early` learning
        Returns:
            pd.DataFrame: columns=regions, rows=time trials
        """
        timeseries = try_load_filenames(self.subject.int, self.epoch)
        return window_timeseries(self.epoch, timeseries)

    def _load_ts_subc(self) -> pd.DataFrame:
        """ loads timeseries for subcortical regions
        """
        subject_filename = _make_subc_filename(self.subject.str, self.epoch)
        ts_subc = pd.read_csv(subject_filename)
        ts_subc = _rename_columns_to_region_names(ts_subc)
        return window_timeseries(self.epoch, ts_subc)
    # todo having TS cortex class that inherits from TS

    def load_region_names(self):
        pass


def all_region_names(include_subcortex=True) -> list:
    ts = Timeseries(1, 'rest', include_subcortex)
    ts.load()
    return ts.data.columns.tolist()
