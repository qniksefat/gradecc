import numpy as np
from nilearn.connectome import ConnectivityMeasure

from gradecc.compute.store_conn_mat import load_centered_mat
from gradecc.load_data import Timeseries
from gradecc.load_data.subject import SUBJECTS_INT


# todo q: classes act a bit different.
# todo: superclass
class ConnectivityMatrixMean:
    """ connectivity matrix for an epoch averaged over subjects
        """

    # todo: does NOT handle include_subc

    def __init__(self, epoch: str, subjects: list[int] = SUBJECTS_INT, kind='covariance'):
        self.epoch = epoch
        if not isinstance(subjects, list):
            subjects = [subjects]
        self.subjects = subjects
        self.kind = kind
        self.data = None
        self.region_names = None

    def load(self):
        timeseries_sample = Timeseries(epoch=self.epoch, subject_id=self.subjects[0])
        timeseries_sample.load()
        self.region_names = timeseries_sample.region_names
        conn_mat_avg = self._compute_conn_mat_mean(timeseries_sample)
        self.data = conn_mat_avg

    def _compute_conn_mat_mean(self, timeseries_sample):
        conn_mat_to_init = ConnectivityMatrix(timeseries=timeseries_sample, kind=self.kind)
        conn_mat_to_init.load()
        conn_mat_avg = conn_mat_to_init.data
        conn_mat_avg = np.zeros(conn_mat_avg.shape)
        for subject in self.subjects:
            timeseries = Timeseries(subject_id=subject, epoch=self.epoch)
            conn_mat = ConnectivityMatrix(timeseries=timeseries, kind=self.kind)
            conn_mat.load()
            conn_mat = conn_mat.data
            conn_mat_avg += conn_mat
        conn_mat_avg /= len(self.subjects)
        return conn_mat_avg


# todo q: inherit from ConnectivityMeasure
class ConnectivityMatrix:
    def __init__(self, timeseries: Timeseries, kind='correlation', centered=True):
        self.timeseries = timeseries
        self.data = None
        self.kind = kind
        self.region_names = None
        self.centered = centered

    def load(self):
        """connectivity matrix within an epoch for a subject
        """
        self.timeseries.load()
        self.region_names = self.timeseries.region_names
        if self.centered:
            self.data = load_centered_mat(self.timeseries.subject, self.timeseries.epoch)
        else:
            timeseries_ndarray = self.timeseries.data.to_numpy()
            correlation_measure = ConnectivityMeasure(kind=self.kind)
            connectivity_matrix = correlation_measure.fit_transform([timeseries_ndarray])[0]
            # np.fill_diagonal(connectivity_matrix, 0)
            self.data = connectivity_matrix

# todo feature: select part of regions


if __name__ == '__main__':
    from tqdm import tqdm

    for _ in tqdm(range(3)):
        ts = Timeseries(1, 'baseline')
        c = ConnectivityMatrix(ts, centered=True)
        c.load()
        print(c.data.shape)
