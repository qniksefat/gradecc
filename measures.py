import pandas as pd
import numpy as np
from gradient import make_gradients
from load_timeseries import SUBJECTS
import os.path

DATA_FNAME = 'data/'


def get_measures(measures=None, epic_list=['baseline', 'early', 'late'],
                 subjects=SUBJECTS, **kwargs) -> pd.DataFrame:
    """get measures for a brain region

    Args:
        measures (list): different measures such as gradients or eccentricity

    Returns:
        pd.DataFrame: _description_
    """
    fname = DATA_FNAME + 'measures.csv'
    file_exists = os.path.isfile(fname)
    
    if file_exists:
        df = pd.read_csv(fname)
        measures_list = _measures_list(measures)
        return df[(df.measure.isin(measures_list)) &
                  (df.subject.isin(subjects)) &
                  (df.epic.isin(epic_list))]
    
    else:
        # todo make_measures save in that, access it outside
        df_gradients = make_gradients(epic_list=epic_list,
            subjects=subjects)
        df_ecc = _make_constructs(df_gradients)
        df_measures = pd.concat([df_gradients, df_ecc], axis=0)
        if _save_file(kwargs.get('save', False), subjects):
            df_measures.to_csv(DATA_FNAME + 'measures.csv', index=False)
            # todo def make fname

        return get_measures(measures=measures, epic_list=epic_list,
                            subjects=subjects, **kwargs)


def _make_constructs(dff):
    ecc = dff\
        .assign(v=lambda r: r['value'] ** 2, aixs=1)\
            .groupby(['subject', 'epic', 'region']).agg({'v': sum})\
                .v.apply(np.sqrt).reset_index()\
                    .rename(columns={'v': 'value'})
    ecc['measure'] = 'eccentricity'
    return ecc


def _save_file(save: bool, subjects: list):
    includes_all_subjects = set(subjects) == set(SUBJECTS)
    return includes_all_subjects and save

def _measures_list(measures):
    """should've two sets for gradients and grad+ecc

    Args:
        measures (_type_): _description_

    Returns:
        _type_: _description_
    """
    if measures is None: return ['gradient' + str(i + 1) for i in range(4)] + ['eccentricity']
    else: return measures
    