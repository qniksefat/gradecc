from gradecc.load_data import Timeseries
from gradecc.plot.conn_mat import plot_conn_mat
from gradecc.compute.conn_mat import get_conn_mat, compute_conn_mat


if __name__ == '__main__':
    ts = Timeseries(subject_id='AB1', epoch='rest', include_subcortex=False)
    print('timeseries data', ts.data)

    plot_conn_mat(epoch='late', include_subcortex=True, reorder=True, significant_regions=True, output_file='late epoch')

    print(compute_connx_mat(timeseries=ts)[0].shape)
