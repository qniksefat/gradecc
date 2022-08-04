import numpy as np
from nilearn.connectome import ConnectivityMeasure

from gradecc.load_data import Timeseries
from gradecc.load_data.subject import SUBJECTS_INT


# todo feature: centering https://pyriemann.readthedocs.io/en/latest/index.html

# todo q: classes act a bit different.
# todo: superclass
class ConnectivityMatrixMean:
    """ connectivity matrix for an epoch averaged over subjects
        """
    # todo: does NOT handle include_subc

    def __init__(self, epoch: str, subjects: list[int] = SUBJECTS_INT, kind='correlation'):
        self.epoch = epoch
        if not isinstance(subjects, list):
            subjects = [subjects]
        self.subjects = subjects
        self.kind = kind
        self.data = None
        self.region_names = None

    def load(self):
        timeseries_sample = Timeseries(epoch=self.epoch, subject_id=self.subjects[0])
        self.region_names = timeseries_sample.region_names
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
        self.data = conn_mat_avg


# todo q: inherit from ConnectivityMeasure
class ConnectivityMatrix:
    def __init__(self, timeseries: Timeseries, kind='correlation'):
        self.timeseries = timeseries
        self.data = None
        self.kind = kind
        self.region_names = None

    def load(self):
        """connectivity matrix within an epoch for a subject
        """
        self.timeseries.load()
        self.region_names = self.timeseries.region_names
        timeseries_ndarray = self.timeseries.data.to_numpy()
        correlation_measure = ConnectivityMeasure(kind=self.kind)
        connectivity_matrix = correlation_measure.fit_transform([timeseries_ndarray])[0]
        # np.fill_diagonal(connectivity_matrix, 0)
        self.data = connectivity_matrix


def get_conn_mat(epoch: str, subject=None, **kwargs):
    """get connectivity matrix. If subject is None, matrix averaged over all subjects.
    """
    # todo feature: select part of regions
    if subject is not None:
        include_subcortex = kwargs.get('include_subcortex', True)
        timeseries = Timeseries(epoch=epoch, subject_id=subject, include_subcortex=include_subcortex)
        connectivity_matrix, regions = compute_conn_mat(timeseries, **kwargs)
    else:
        connectivity_matrix, regions = _average_conn_mat(epoch, **kwargs)
    return connectivity_matrix, regions


def compute_conn_mat(timeseries: Timeseries, **kwargs):
    """connectivity matrix within an epoch for a subject
    """
    timeseries.load()
    timeseries_ndarray = timeseries.data.to_numpy()
    correlation_measure = ConnectivityMeasure(kind='correlation')
    connectivity_matrix = correlation_measure.fit_transform([timeseries_ndarray])[0]
    # np.fill_diagonal(connectivity_matrix, 0)
    return connectivity_matrix, timeseries.region_names


def _average_conn_mat(epoch: str, subjects=SUBJECTS_INT, **kwargs):
    """connectivity matrix for an epoch averaged over all subjects
    """
    include_subcortex = kwargs.get('include_subcortex', True)
    timeseries_sample = Timeseries(epoch=epoch, subject_id=subjects[0], include_subcortex=include_subcortex)
    matrix_avg, rois = compute_conn_mat(timeseries=timeseries_sample, **kwargs)
    matrix_avg = np.zeros(matrix_avg.shape)

    for subject in subjects:
        timeseries = Timeseries(epoch=epoch, subject_id=subject)
        connectivity_matrix, _ = compute_conn_mat(timeseries, **kwargs)
        matrix_avg += connectivity_matrix
    matrix_avg /= len(subjects)
    return matrix_avg, rois
