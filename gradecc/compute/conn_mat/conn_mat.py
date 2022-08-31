import numpy as np
from nilearn.connectome import ConnectivityMeasure

from gradecc.load_data import Timeseries


def conn_mat_from_timeseries(timeseries: Timeseries, fill_diag=False, **kwargs) -> np.ndarray:
    if timeseries.data is None: timeseries.load()
    timeseries_ndarray = timeseries.data.to_numpy()
    correlation_measure = ConnectivityMeasure(kind=kwargs.get('kind', 'covariance'))
    connectivity_matrix = correlation_measure.fit_transform([timeseries_ndarray])[0]
    if fill_diag:   np.fill_diagonal(connectivity_matrix, 0)
    return connectivity_matrix


def stack_conn_mats(epochs, subjects) -> np.ndarray:    # tested. same with ConnectivityMatrixMean
    if not isinstance(epochs, list):    epochs = [epochs]
    if not isinstance(subjects, list):  subjects = [subjects]
    conn_mats_stacked = np.stack([conn_mat_from_timeseries(Timeseries(s, e))
                                  for e in epochs for s in subjects])
    return conn_mats_stacked


if __name__ == '__main__':
    c1 = stack_conn_mats(epochs='baseline', subjects=SUBJECTS).mean(axis=0)
    print(c1)
