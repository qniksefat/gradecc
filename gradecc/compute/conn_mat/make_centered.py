from os import path
import pickle
import numpy as np
from tqdm import tqdm
from pyriemann.utils.mean import mean_riemann

from gradecc.utils.filenames import dir_conn_mat
from gradecc.compute.conn_mat.conn_mat import ConnectivityMatrixMean
from gradecc.load_data import EPOCHS
from gradecc.load_data.subject import SUBJECTS_INT

MAX_ITER = 10


def dump_pkl(obj, filename):
    with open(path.join(dir_conn_mat, filename + '.pickle'), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_conn_mats(epochs, subjects_int=SUBJECTS_INT):
    conn_mats = {s: {} for s in subjects_int}
    for s in tqdm(subjects_int):
        for e in epochs:
            c = ConnectivityMatrixMean(epoch=e, subjects=s)
            c.load()
            conn_mats[s][e] = c.data
    return conn_mats


def flat(conn_mats: dict):
    """
    flatten each element from num_regions*num_regions* matrix to num_regions**2 ndarray
    @param conn_mats:
    @return:
    """
    subjects_int = conn_mats.keys()
    cm_flat = {s: {} for s in subjects_int}
    for s in tqdm(subjects_int):
        for e in conn_mats[s].keys():
            cm_flat[s][e] = conn_mats[s][e].flatten()
    return cm_flat


def stack(conn_mats_flat) -> np.array:
    """
    vertically stack all epochs for all subjects
    @param conn_mats_flat: each element is flattened to num_regions*num_regions
    @return: np.array shape len(s)*len(e) by num_regions**2
    """
    return np.vstack([
        np.stack([
            conn_mats_flat[s][e]
            for e in conn_mats_flat[s].keys()
        ])
        for s in SUBJECTS_INT
    ])


def make_subj_riemann_mean(conn_mats):
    subjects_riemann_mean = {}
    subjects_int = conn_mats.keys()
    for s in tqdm(subjects_int):
        subjects_riemann_mean[s] = mean_riemann(np.stack(conn_mats[s].values()),
                                                maxiter=MAX_ITER)
    return subjects_riemann_mean


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
    from pyriemann.utils.base import sqrtm, logm, expm, invsqrtm

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

    t = _to_tangent(c, sub_mean)
    tc = _gl_transport(t, sub_mean, grand_mean)
    return _from_tangent(tc, grand_mean)


def compute_conn_mats_cnt(conn_mats, subject_mean, grand_mean):
    conn_mats_cnt = {}
    subjects = conn_mats.keys()
    for s in tqdm(subjects):
        conn_mats_cnt[s] = {
            e: center_cmat(conn_mats[s][e], subject_mean[s], grand_mean)
            for e in conn_mats[s].keys()
        }
    return conn_mats_cnt


def make_conn_mats_centered(epochs_list, subjects_int=SUBJECTS_INT):
    assert epochs_list == EPOCHS or epochs_list == EPOCHS + ['rest']

    dct = {}
    conn_mat = make_conn_mats(epochs_list, subjects_int)
    subject_riemann_mean = make_subj_riemann_mean(conn_mat)
    dct['subject_riemann_mean'] = subject_riemann_mean

    conn_mat_stacked = stack(conn_mat)
    grand_mean = mean_riemann(conn_mat_stacked, maxiter=MAX_ITER)
    dct['grand_mean'] = grand_mean

    conn_mat_cnt = compute_conn_mats_cnt(
        conn_mat, subject_riemann_mean, grand_mean)
    dct['conn_mat_centered'] = conn_mat_cnt

    return dct


if __name__ == '__main__':
    conn_mat_riemannian = make_conn_mats_centered(EPOCHS)
    conn_mat_riemannian['include_rest'] = make_conn_mats_centered(EPOCHS+['rest'])
    dump_pkl(conn_mat_riemannian, 'conn_mat_riemannian')
