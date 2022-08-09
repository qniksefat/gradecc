from os import path
import pickle

from gradecc.utils.filenames import dir_conn_mat
from gradecc.load_data.subject import Subject


global conn_mat_centered, conn_mat_centered_include_rest

LOADED = False


def load_pkl(filename):
    with open(path.join(dir_conn_mat, filename + '.pickle'), 'rb') as handle:
        return pickle.load(handle)


def load_centered_matrices_data():
    global conn_mat_centered, conn_mat_centered_include_rest

    conn_mat_centered_include_rest = load_pkl('conn_mat_centered')
    # subject_riemann_mean_include_rest = load_pkl('subject_riemann_mean')
    # grand_mean_include_rest = load_pkl('grand_mean')

    conn_mat_centered = load_pkl('conn_mat_centered2')
    # subject_riemann_mean = load_pkl('subject_riemann_mean2')
    # grand_mean = load_pkl('grand_mean2')

    global LOADED
    LOADED = True


def load_centered_mat(subject: Subject, epoch: str, include_rest=False):
    global LOADED
    if not LOADED:
        load_centered_matrices_data()
        LOADED = True

    subject = subject.int
    if include_rest:
        return conn_mat_centered_include_rest[subject][epoch]
    else:
        return conn_mat_centered[subject][epoch]
