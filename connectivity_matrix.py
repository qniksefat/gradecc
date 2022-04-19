import numpy as np
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure

from load_timeseries import SUBJECTS, load_timeseries


def plot_conn_mat(epic: str, subject=None):
    connectivity_matrix, rois = get_conn_mat(epic, subject)
    
    ### todo Reduce conn matrix size, only for visualization purposes
    # mat_mask = np.where(np.std(correlation_matrix, axis=1) > 0.2)[0]
    # c = correlation_matrix[mat_mask][:, mat_mask]
    # # Create corresponding region names
    # regions_list = ['%s_%s' % (h, r.decode()) for h in ['L', 'R'] for r in regions]
    # masked_regions = [regions_list[i] for i in mat_mask]

    plotting.plot_matrix(connectivity_matrix, labels=rois,
                         colorbar=True, figure=(10, 15), vmax=0.8, vmin=-0.8, reorder=True)


def get_conn_mat(epic: str, subject: int=None):
    """get connectivity matrix. If subject is None, matrix averaged over all subjects.

    Args:
        epic (str): _description_
        subject (int, optional): when specified, get the resutl for one subject. Defaults to None.

    Returns:
        _type_: matrix in ndarray, list of ROIs
    """
    if subject is not None:
        connectivity_matrix, rois = _get_conn_mat_subject(epic, subject)
    else:
        connectivity_matrix, rois = _get_conn_mat_averaged(epic)
    return connectivity_matrix, rois


def _get_conn_mat_subject(epic: str, subject: int):
    """connectivity matrix within an epic for a subject 

    Args:
        epic (str): time period such as `rest`, `baseline`, etc
        subject (int): subject identification number

    Returns:
        (np.array, list): connectivity matrix and regions of interest (ROIs)
    """
    timeseries = load_timeseries(epic=epic, subject=subject)
    rois = timeseries.columns.tolist()
    timeseries = timeseries.to_numpy()
    correlation_measure = ConnectivityMeasure(kind='correlation')
    connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]
    np.fill_diagonal(connectivity_matrix, 0)

    return connectivity_matrix, rois


def _get_conn_mat_averaged(epic: str):
    """connectivity matrix for an epic averaged over all subjects

    Args:
        epic (str): time period such as `rest`, `baseline`, etc

    Returns:
        (np.array, list): connectivity matrix and regions of interest (ROIs)
    """
    avg_matrix, rois = _get_conn_mat_subject(epic=epic, subject=SUBJECTS[0])
    avg_matrix = np.zeros(avg_matrix.shape)
    for subject in SUBJECTS:
            connectivity_matrix, _ = _get_conn_mat_subject(epic=epic, subject=subject)
            avg_matrix += connectivity_matrix
    avg_matrix /= len(SUBJECTS)
    return avg_matrix, rois
