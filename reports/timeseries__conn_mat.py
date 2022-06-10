from gradecc.load_timeseries import load_ts
from gradecc.connectivity_matrix import plot_conn_mat

if __name__ == '__main__':
    print(
        load_ts(subject=1, epic='rest', include_subcortex=True)
    )

    plot_conn_mat(epic='late', include_subcortex=True, reorder=True, significant_regions=True, output_file='late epoch')

