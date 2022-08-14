from os import path
import pickle
import numpy as np

from gradecc.utils.filenames import dir_conn_mat
from gradecc.load_data.subject import Subject

conn_mat_riemannian = {}
conn_mat_is_loaded = False


def load_pkl(filename):
    with open(path.join(dir_conn_mat, filename + '.pickle'), 'rb') as handle:
        return pickle.load(handle)


def load_all_conn_mat():
    global conn_mat_riemannian
    conn_mat_riemannian = load_pkl('conn_mat_riemannian')
    global conn_mat_is_loaded
    conn_mat_is_loaded = True


def load_conn_mat_cnt_indv(subject: Subject, epoch: str, include_rest=False):
    if not conn_mat_is_loaded:
        load_all_conn_mat()

    subject = subject.int
    if include_rest:
        return conn_mat_riemannian['include_rest']['conn_mat_centered'][subject][epoch]
    else:
        return conn_mat_riemannian['conn_mat_centered'][subject][epoch]


def load_grand_mean(epoch, include_rest=False):
    if include_rest:
        return conn_mat_riemannian['include_rest']['grand_mean'][epoch]
    else:
        return conn_mat_riemannian['grand_mean'][epoch]


def get_conn_mat_centered(epoch, include_rest=False, subject: Subject = None) -> np.array:
    """
    @param subject:
    @param epoch:
    @param include_rest:
    @return:
    """
    if subject is None:
        load_grand_mean(epoch)
    else:
        assert isinstance(subject, Subject)
        return load_conn_mat_cnt_indv(subject, epoch, include_rest)
