import numpy as np
from nilearn.connectome import ConnectivityMeasure

from gradecc.load_data import Timeseries


def conn_mat_from_timeseries(timeseries: Timeseries, fill_diag=False) -> np.array:
    if not timeseries.data: timeseries.load()
    timeseries_ndarray = timeseries.data.to_numpy()
    correlation_measure = ConnectivityMeasure()
    connectivity_matrix = correlation_measure.fit_transform([timeseries_ndarray])[0]
    if fill_diag:   np.fill_diagonal(connectivity_matrix, 0)
    return connectivity_matrix

# todo feature: select part of regions
