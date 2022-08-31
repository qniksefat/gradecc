import numpy as np

from gradecc.compute.conn_mat.conn_mat import conn_mat_from_timeseries
from gradecc.compute.conn_mat.center import riemann_centered_conn_mat
from gradecc.load_data import Timeseries
from gradecc.load_data.subject import SUBJECTS_INT

# todo feature: select part of regions


class ConnectivityMatrix(np.ndarray):
    def __new__(cls, timeseries: Timeseries, centered=False, **kwargs):
        data = ConnectivityMatrix.load(timeseries, centered, **kwargs)
        # Defined s.t. you can instantiate it like np ndarray.
        # e.g. ConnectivityMatrix([1]) + ConnectivityMatrix([3]) exists
        obj = np.asarray(data).view(cls)
        obj.timeseries = timeseries
        obj.centered = centered
        obj.region_names = obj.timeseries.region_names
        return obj

    @staticmethod
    def load(timeseries, centered, **kwargs):
        timeseries.load()
        if centered:    return riemann_centered_conn_mat(timeseries)
        else:   return conn_mat_from_timeseries(timeseries, **kwargs)

    # def __array_finalize__(self, obj):
    #     if obj is None: return
    #     self.centered = getattr(obj, 'centered', None)


class ConnectivityMatrixMean:
    """ connectivity matrix for an epoch averaged over subjects
    """
    # todo: does NOT handle include_subc
    def __init__(self, epoch: str, subjects: list[int] = SUBJECTS_INT):
        self.epoch = epoch
        if not isinstance(subjects, list):  subjects = [subjects]
        self.subjects = subjects
        self.data = None
        self.region_names = None

    def load(self):
        ts_sample = Timeseries(self.subjects[0], self.epoch)
        ts_sample.load()
        self.region_names = ts_sample.region_names
        self.data = self._compute_mean_conn_mat()

    def _compute_mean_conn_mat(self):    # tested
        return np.stack([ConnectivityMatrix(Timeseries(s, self.epoch)) for s in self.subjects]).mean(axis=0)


if __name__ == '__main__':
    from gradecc.load_data import Subject

    print(ConnectivityMatrix(Timeseries(Subject(1), 'baseline'), centered=True))
