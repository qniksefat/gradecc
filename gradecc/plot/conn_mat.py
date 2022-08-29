from os import path
import typing
import numpy as np
from matplotlib import pyplot as plt
from nilearn import plotting

from gradecc.compute.conn_mat import ConnectivityMatrixMean, ConnectivityMatrix
from gradecc.load_data import Timeseries
from gradecc.utils.filenames import dir_images


def plot_conn_mat(connectivity_matrix: typing.Union[ConnectivityMatrix, ConnectivityMatrixMean],
                  significant_regions=True, output_filename=None, **kwargs):
    conn_mat_ndarray, regions = _prep_conn_mat(connectivity_matrix, significant_regions)

    fig = plt.figure(figsize=(15, 10))
    plotting.plot_matrix(conn_mat_ndarray, labels=regions,
                         reorder=kwargs.get('reorder', True),
                         colorbar=True, vmax=0.8, vmin=-0.8,
                         figure=fig)
    if output_filename:
        fig.savefig(path.join(dir_images, output_filename + '.png'))


def _prep_conn_mat(connectivity_matrix, significant_regions):
    if isinstance(connectivity_matrix, ConnectivityMatrix):
        regions = connectivity_matrix.timeseries.region_names
        conn_mat_ndarray = connectivity_matrix
    else:
        assert isinstance(connectivity_matrix, ConnectivityMatrixMean)

        connectivity_matrix.load()
        conn_mat_ndarray = connectivity_matrix.data
        sample_subject = connectivity_matrix.subjects[0]
        # assert isinstance(sample_subject, int) or isinstance(sample_subject, str)
        ts = Timeseries(sample_subject, epoch=connectivity_matrix.epoch)
        ts.load()
        regions = ts.region_names

    if significant_regions:
        conn_mat_ndarray, regions = _mask_conn_mat(conn_mat_ndarray, regions)
    return conn_mat_ndarray, regions


def _mask_conn_mat(conn_mat, regions):
    var_threshold = _variance_threshold(conn_mat)
    conn_mat_mask = np.where(np.std(conn_mat, axis=1) > var_threshold)[0]
    conn_mat_masked = conn_mat[conn_mat_mask][:, conn_mat_mask]
    regions_masked = [regions[i] for i in conn_mat_mask]
    return conn_mat_masked, regions_masked


def _variance_threshold(conn_mat):
    sd_threshold = np.std(conn_mat, axis=1).mean()
    return sd_threshold
