import numpy as np
from nilearn.connectome import ConnectivityMeasure

from gradecc.load_data import Timeseries
from gradecc.load_data.subject import SUBJECTS_INT


# todo q: inherit from ConnectivityMeasure
class ConnectivityMatrix:
    def __init__(self, kind='correlation'):
        self.data = None


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
    timeseries_ndarray = timeseries.data.to_numpy()
    correlation_measure = ConnectivityMeasure(kind='correlation')
    connectivity_matrix = correlation_measure.fit_transform([timeseries_ndarray])[0]
    np.fill_diagonal(connectivity_matrix, 0)
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
