from os import path
import numpy as np
from matplotlib import pyplot as plt
from nilearn import plotting

from gradecc.compute.conn_mat import get_conn_mat
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
