from os import path
import numpy as np
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt

from gradecc.load_timeseries import load_ts, INCLUDE_SUBC
from gradecc.load_timeseries.utils import SUBJECTS
from gradecc.utils.filenames import dir_images


def plot_conn_mat(epoch: str, subject=None, significant_regions=True, output_file=None, **kwargs):
    connectivity_matrix, regions = get_conn_mat(epoch, subject, **kwargs)
    if significant_regions:
        connectivity_matrix, regions = _mask_conn_mat(connectivity_matrix, regions)
    fig = plt.figure(figsize=(15, 10))
    plotting.plot_matrix(connectivity_matrix, labels=regions,
                         reorder=kwargs.get('reorder', True),
                         colorbar=True, vmax=0.8, vmin=-0.8,
                         figure=fig)
    if output_file:
        fig.savefig(path.join(dir_images, output_file + '.png'))


def _mask_conn_mat(conn_mat, regions):
    var_threshold = _variance_threshold(conn_mat)
    conn_mat_mask = np.where(np.std(conn_mat, axis=1) > var_threshold)[0]
    conn_mat_masked = conn_mat[conn_mat_mask][:, conn_mat_mask]
    regions_masked = [regions[i] for i in conn_mat_mask]
    return conn_mat_masked, regions_masked


def _variance_threshold(conn_mat):
    sd_threshold = np.std(conn_mat, axis=1).mean()
    return sd_threshold


def get_conn_mat(epoch: str, subject: int = None, **kwargs):
    """get connectivity matrix. If subject is None, matrix averaged over all subjects.
    """
    # todo select part of regions
    if subject is not None:
        connectivity_matrix, regions = _get_conn_mat_subject(epoch, subject, **kwargs)
    else:
        connectivity_matrix, regions = _get_conn_mat_averaged(epoch, **kwargs)
    return connectivity_matrix, regions


def _get_conn_mat_subject(epoch: str, subject: int, **kwargs):
    """connectivity matrix within an epoch for a subject
    """
    include_subc = kwargs.get('include_subcortex', INCLUDE_SUBC)
    timeseries = load_ts(epoch=epoch, subject=subject, include_subcortex=include_subc)
    regions = timeseries.columns.tolist()
    timeseries = timeseries.to_numpy()
    correlation_measure = ConnectivityMeasure(kind='correlation')
    connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]
    np.fill_diagonal(connectivity_matrix, 0)

    return connectivity_matrix, regions


def _get_conn_mat_averaged(epoch: str, subjects=SUBJECTS, **kwargs):
    """connectivity matrix for an epoch averaged over all subjects
    """
    avg_matrix, rois = _get_conn_mat_subject(epoch=epoch, subject=subjects[0], **kwargs)
    avg_matrix = np.zeros(avg_matrix.shape)
    for subject in subjects:
        connectivity_matrix, _ = _get_conn_mat_subject(epoch=epoch, subject=subject, **kwargs)
        avg_matrix += connectivity_matrix
    avg_matrix /= len(subjects)
    return avg_matrix, rois
