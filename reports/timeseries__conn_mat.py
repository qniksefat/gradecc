from gradecc.load_data import Timeseries
from gradecc.plot.conn_mat import plot_conn_mat
from gradecc.compute.conn_mat import ConnectivityMatrix, ConnectivityMatrixMean

if __name__ == '__main__':
    ts = Timeseries(subject_id='AB1', epoch='rest', include_subcortex=True)
    # ts.load()
    # print('timeseries data', ts.data)
    #
    c = ConnectivityMatrix(ts)
    # c.load()
    # print('', c.data.shape)
    #
    c1 = ConnectivityMatrixMean(epoch='rest')
    # c1.load()
    # print(c1.data)

    plot_conn_mat(c, reorder=True, significant_regions=True, output_filename='asdfasdf')

