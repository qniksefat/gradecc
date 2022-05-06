import pandas as pd
import numpy as np
from load_timeseries import load_timeseries, all_region_names, SUBJECTS
from scipy.stats import pearsonr


def seed_connectivity(seed_regions, epic_list=None):
    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    if not isinstance(seed_regions, list):
        seed_regions = [seed_regions]
    df_seed = pd.DataFrame()
    for epic in epic_list:
        for subject in SUBJECTS:
            timeseries_subject = load_timeseries(subject, epic)
            timeseries_subject = timeseries_subject.to_numpy().transpose()
            for seed_region in seed_regions:
                seed_timeseries = timeseries_subject[seed_region]
                stat_map = np.zeros(timeseries_subject.shape[0])
                for i in range(timeseries_subject.shape[0]):
                    stat_map[i] = pearsonr(seed_timeseries, timeseries_subject[i])[0]
                # Re-mask previously masked nodes (medial wall)
                stat_map[np.where(np.mean(timeseries_subject, axis=1) == 0)] = 0
                _df = pd.DataFrame([stat_map], columns=all_region_names())
                _df['subject'] = subject
                _df['epic'] = epic
                _df['seed_region'] = seed_region
                df_seed = pd.concat([df_seed, _df], axis=0)
    df_seed = df_seed.set_index(['seed_region','epic', 'subject'])
    return df_seed


def seed_average(df_seed):
    return df_seed.reset_index() \
        .groupby(['seed_region', 'epic']) \
        .mean().drop('subject', axis=1)

