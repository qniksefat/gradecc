import numpy as np

from .conn_mat import conn_mat_from_timeseries
from .center import riemann_centered_conn_mat
from ...load_data import Timeseries
from ...load_data.subject import SUBJECTS_INT


class ConnectivityMatrix:
    def __init__(self, timeseries: Timeseries, centered=False):
        self.timeseries = timeseries
        self.centered = centered
        self.data: np.array = None
        self.region_names = None

    def load(self):
        self.timeseries.load()
        self.region_names = self.timeseries.region_names
        if self.centered:
            self.data = riemann_centered_conn_mat(self.timeseries)
        else:
            self.data = conn_mat_from_timeseries(self.timeseries)


class ConnectivityMatrixMean:
    """ connectivity matrix for an epoch averaged over subjects
    """
    # todo: does NOT handle include_subc
    def __init__(self, epoch: str, subjects: list[int] = SUBJECTS_INT):
        self.epoch = epoch
        if not isinstance(subjects, list):
            subjects = [subjects]
        self.subjects = subjects
        self.data = None
        self.region_names = None

    def load(self):
        timeseries_sample = Timeseries(self.subjects[0], self.epoch)
        timeseries_sample.load()
        self.region_names = timeseries_sample.region_names
        conn_mat_avg = self._compute_mean_conn_mat(timeseries_sample)
        self.data = conn_mat_avg

    def _compute_mean_conn_mat(self, timeseries_sample):
        conn_mat_to_init = ConnectivityMatrix(timeseries=timeseries_sample)
        conn_mat_to_init.load()
        conn_mat_avg = conn_mat_to_init.data
        conn_mat_avg = np.zeros(conn_mat_avg.shape)
        for subject in self.subjects:
            timeseries = Timeseries(subject, self.epoch)
            conn_mat = ConnectivityMatrix(timeseries=timeseries)
            conn_mat.load()
            conn_mat = conn_mat.data
            conn_mat_avg += conn_mat
        conn_mat_avg /= len(self.subjects)
        return conn_mat_avg
