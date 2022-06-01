import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from gradecc.load_timeseries.load_ts import load_ts_cortex
from gradecc.load_timeseries.utils import SUBJECTS
from gradecc.load_timeseries import all_region_names


def seed_connectivity(seed_regions, epic_list=None):
    epic_list, seed_regions = _init_inputs_seed_conn(epic_list, seed_regions)
    df_seed = pd.DataFrame()
    for epic in epic_list:
        for subject in SUBJECTS:
            for seed_region in seed_regions:
                _df = _compute_seed_conn(seed_region, subject, epic)
                df_seed = pd.concat([df_seed, _df], axis=0)
    df_seed = df_seed.set_index(['seed_region', 'epic', 'subject'])
    return df_seed


def _init_inputs_seed_conn(epic_list, seed_regions):
    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    if not isinstance(seed_regions, list):
        seed_regions = [seed_regions]
    return epic_list, seed_regions


def _compute_seed_conn(seed_region, subject, epic):
    timeseries_subject = load_ts_cortex(subject, epic)
    timeseries_subject = timeseries_subject.transpose()
    seed_timeseries = timeseries_subject.loc[seed_region]
    timeseries_subject = timeseries_subject.to_numpy()
    stat_map = np.zeros(timeseries_subject.shape[0])
    for i in range(timeseries_subject.shape[0]):
        stat_map[i] = pearsonr(seed_timeseries, timeseries_subject[i])[0]
    _remask_previously_masked(stat_map, timeseries_subject)
    _df = pd.DataFrame([stat_map], columns=all_region_names())
    _df['subject'] = subject
    _df['epic'] = epic
    _df['seed_region'] = seed_region
    return _df


def _remask_previously_masked(stat_map, timeseries_subject):
    stat_map[np.where(np.mean(timeseries_subject, axis=1) == 0)] = 0


def seed_average(df_seed):
    return df_seed.reset_index() \
        .groupby(['seed_region', 'epic']) \
        .mean().drop('subject', axis=1)


if __name__ == '__main__':
    from gradecc.plot.plot_cortex import plot_cortex
    # df_seed = seed_connectivity('7Networks_LH_Default_PFC_19')
    # df_seed_avg = seed_average(df_seed)

    import pickle

    # with open('seed_avg.pickle', 'wb') as handle:
    #     pickle.dump(df_seed_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('seed_avg.pickle', 'rb') as handle:
        df_seed_avg = pickle.load(handle)

    text = 'average seed connectivity \n in Late epoch for \n'
    plot_cortex(df_seed_avg.loc['7Networks_LH_Default_PFC_19', 'late'].T,
                color_range=(-1, 1), color_map='bwr', text=text)

