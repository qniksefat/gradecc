from gradecc.load_data import Timeseries, Subject
from gradecc.plot.conn_mat import plot_conn_mat
from gradecc.compute.conn_mat import ConnectivityMatrix, ConnectivityMatrixMean

if __name__ == '__main__':
    ts = Timeseries(Subject('AB1'), 'rest', include_subcortex=True)
    c = ConnectivityMatrix(ts)
    plot_conn_mat(c, reorder=True, significant_regions=True, output_filename='asdfasdf')

