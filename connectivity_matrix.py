import numpy as np
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure

from load_timeseries import SUBJECTS, load_timeseries


def plot_conn_mat(epic: str, subject=None, significant_regions=True):
    connectivity_matrix, regions = get_conn_mat(epic, subject)
    if significant_regions:
        connectivity_matrix, regions = _mask_conn_mat(connectivity_matrix, regions)
    plotting.plot_matrix(connectivity_matrix, labels=regions, reorder=True,
                         colorbar=True, figure=(10, 15), vmax=0.8, vmin=-0.8)


def _mask_conn_mat(conn_mat, regions):
    sd_threshold = _variance_threshold(conn_mat)
    conn_mat_mask = np.where(np.std(conn_mat, axis=1) > sd_threshold)[0]
    conn_mat_masked = conn_mat[conn_mat_mask][:, conn_mat_mask]
    regions_masked = [regions[i] for i in conn_mat_mask]
    return conn_mat_masked, regions_masked


def _variance_threshold(conn_mat):
    sd_threshold = np.std(conn_mat, axis=1).mean()
    return sd_threshold


def get_conn_mat(epic: str, subject: int = None):
    """get connectivity matrix. If subject is None, matrix averaged over all subjects.
    """
    if subject is not None:
        connectivity_matrix, regions = _get_conn_mat_subject(epic, subject)
    else:
        connectivity_matrix, regions = _get_conn_mat_averaged(epic)
    return connectivity_matrix, regions


def _get_conn_mat_subject(epic: str, subject: int):
    """connectivity matrix within an epic for a subject
    """
    timeseries = load_timeseries(epic=epic, subject=subject)
    regions = timeseries.columns.tolist()
    timeseries = timeseries.to_numpy()
    correlation_measure = ConnectivityMeasure(kind='correlation')
    connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]
    np.fill_diagonal(connectivity_matrix, 0)

    return connectivity_matrix, regions


def _get_conn_mat_averaged(epic: str):
    """connectivity matrix for an epic averaged over all subjects
    """
    avg_matrix, rois = _get_conn_mat_subject(epic=epic, subject=SUBJECTS[0])
    avg_matrix = np.zeros(avg_matrix.shape)
    for subject in SUBJECTS:
        connectivity_matrix, _ = _get_conn_mat_subject(epic=epic, subject=subject)
        avg_matrix += connectivity_matrix
    avg_matrix /= len(SUBJECTS)
    return avg_matrix, rois
