import numpy as np
from pyriemann.utils.base import sqrtm, invsqrtm, logm, expm
from pyriemann.utils.mean import mean_riemann
from joblib import Memory

from gradecc.compute.conn_mat.conn_mat import conn_mat_from_timeseries
from gradecc.load_data import Subject, Timeseries
from gradecc.load_data.subject import SUBJECTS
from gradecc.load_data import EPOCHS
from gradecc.utils.filenames import data_outside

memory = Memory(data_outside)
MAX_ITER = 10

# should be inside class. gets data gives centered data


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
    """Center covariance matrix using tangent transporting procedure\
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
def mean_riemann_conn_mat(epochs, subjects) -> np.ndarray:
    """
    you should modify any change (even position) in memory cached codes to func_code.py in memory dir.
    @param epochs:
    @param subjects:
    @return:
    """
    if not isinstance(epochs, list):    epochs = [epochs]
    if not isinstance(subjects, list):  subjects = [subjects]
    conn_mats_epochs = np.stack([conn_mat_from_timeseries(Timeseries(s, e))
                                 for e in epochs for s in subjects])
    # mean_riemann takes long, so I cache it using joblib
    return mean_riemann(conn_mats_epochs, maxiter=MAX_ITER)


def riemann_centered_conn_mat(timeseries: Timeseries) -> np.ndarray:
    conn_mat = conn_mat_from_timeseries(timeseries)
    subject_mean = mean_riemann_conn_mat(EPOCHS, timeseries.subject)
    grand_mean = mean_riemann_conn_mat(EPOCHS, SUBJECTS)
    return center_cmat(conn_mat, subject_mean, grand_mean)


if __name__ == '__main__':
    print(riemann_centered_conn_mat(Timeseries(Subject(6), 'baseline')))
