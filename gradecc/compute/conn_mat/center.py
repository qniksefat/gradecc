import numpy as np
from pyriemann.utils.base import sqrtm, invsqrtm, logm, expm
from pyriemann.utils.mean import mean_riemann

from gradecc.compute.conn_mat.conn_mat import conn_mat_from_timeseries, stack_conn_mats
from gradecc.load_data import Subject, Timeseries
from gradecc.load_data.subject import SUBJECTS
from gradecc.load_data import EPOCHS
from gradecc.utils.filenames import memory

MAX_ITER_RIEMANN = 5


def _to_tangent(s, mean):
    # Covariance centering
    p = sqrtm(mean)
    p_inv = invsqrtm(mean)
    return p @ logm(p_inv @ s @ p_inv) @ p


def _gl_transport(t, sub_mean, grand_mean):
    g = sqrtm(grand_mean) @ invsqrtm(sub_mean)
    return g @ t @ g.T


def _from_tangent(t, grand_mean):
    p = sqrtm(grand_mean)
    p_inv = invsqrtm(grand_mean)
    return p @ expm(p_inv @ t @ p_inv) @ p


def center_cmat(c, sub_mean, grand_mean):
    """Center covariance matrix using tangent transporting procedure
    https://github.com/danjgale/adaptation-manifolds/blob/main/adaptman/connectivity.py

    Parameters
    ----------
    c : numpy.ndarray
        Single MxM covariance matrix of a single subject
    sub_mean : numpy.ndarray
        Geometric mean covariance matrix of the subject
    grand_mean : numpy.ndarray
        Geometric mean across all subjects and matrices

    Returns
    -------
    numpy.ndarray
        Centered covariance matrix
    """
    t = _to_tangent(c, sub_mean)
    tc = _gl_transport(t, sub_mean, grand_mean)
    return _from_tangent(tc, grand_mean)


@memory.cache
def mean_riemann_conn_mats(conn_mats_stacked: np.ndarray) -> np.ndarray:
    """
    You should modify any change (even position) in memory cached codes to func_code.py in memory dir.
    @param conn_mats_stacked: shape num_matrices * num_regions*num_regions
    @return:
    """
    assert conn_mats_stacked.shape[1] == conn_mats_stacked.shape[2]
    # I made this method to cache output using joblib. mean_riemann takes long.
    return mean_riemann(conn_mats_stacked, maxiter=MAX_ITER_RIEMANN)


@memory.cache   # could make redundant files on disk
def riemann_centered_conn_mat(timeseries: Timeseries) -> np.ndarray:
    conn_mat = conn_mat_from_timeseries(timeseries)
    subject_mean = mean_riemann_conn_mats(stack_conn_mats(EPOCHS, timeseries.subject))
    grand_mean = mean_riemann_conn_mats(stack_conn_mats(EPOCHS, SUBJECTS))
    return center_cmat(conn_mat, subject_mean, grand_mean)


if __name__ == '__main__':
    c1 = riemann_centered_conn_mat(Timeseries(Subject(3), 'baseline'))
    print(c1)
