import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from gradecc.load_timeseries import load_ts
from gradecc.load_timeseries.utils import SUBJECTS
from gradecc.load_timeseries import all_region_names


def seed_connectivity(seed_regions, epoch_list=None):
    epoch_list, seed_regions = _init_inputs_seed_conn(epoch_list, seed_regions)
    df_seed = pd.DataFrame()
    for epoch in epoch_list:
        for subject in SUBJECTS:
            for seed_region in seed_regions:
                _df = _compute_seed_conn(seed_region, subject, epoch)
                df_seed = pd.concat([df_seed, _df], axis=0)
    df_seed = df_seed.set_index(['seed_region', 'epoch', 'subject'])
    return df_seed


def _init_inputs_seed_conn(epoch_list, seed_regions):
    if epoch_list is None:
        epoch_list = ['baseline', 'early', 'late']
    if not isinstance(seed_regions, list):
        seed_regions = [seed_regions]
    return epoch_list, seed_regions


def _compute_seed_conn(seed_region, subject, epoch):
    timeseries_subject = load_ts(subject, epoch)
    timeseries_subject = timeseries_subject.transpose()
    seed_timeseries = timeseries_subject.loc[seed_region]
    timeseries_subject = timeseries_subject.to_numpy()
    stat_map = np.zeros(timeseries_subject.shape[0])
    for i in range(timeseries_subject.shape[0]):
        stat_map[i] = pearsonr(seed_timeseries, timeseries_subject[i])[0]
    _remask_previously_masked(stat_map, timeseries_subject)
    _df = pd.DataFrame([stat_map], columns=all_region_names())
    _df['subject'] = subject
    _df['epoch'] = epoch
    _df['seed_region'] = seed_region
    return _df


def _remask_previously_masked(stat_map, timeseries_subject):
    stat_map[np.where(np.mean(timeseries_subject, axis=1) == 0)] = 0


def seed_average(df_seed):
    return df_seed.reset_index() \
        .groupby(['seed_region', 'epoch']) \
        .mean().drop('subject', axis=1)
