from gradecc.load_timeseries import load_ts
from gradecc.connectivity_matrix import plot_conn_mat

if __name__ == '__main__':
    print('timeseries data',
          load_ts(subject=1, epoch='rest', include_subcortex=True))

    plot_conn_mat(epoch='late', include_subcortex=True, reorder=True, significant_regions=True, output_file='late epoch')

