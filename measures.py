import pandas as pd
import numpy as np
from gradient import make_gradients
from load_timeseries import SUBJECTS
from utils import file_exists, DATA_FILENAME
from tqdm import tqdm
tqdm.pandas()


# decorator cache
def get_measures(measures=None, epic_list=None,
                 subjects=SUBJECTS) -> pd.DataFrame:
    """get measures for a brain region

    Args:
        epic_list: list of epics such as `baseline`
        measures (list): different measures such as gradients or eccentricity
        subjects (list): if not specified, default to all subjects

    Returns:
        pd.DataFrame: measures corresponding to each region, subject, epic
    """
    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    if measures is None:
        measures = ['gradient' + str(i + 1) for i in range(4)] + ['eccentricity']
    if subjects is None:
        subjects = SUBJECTS

    filename = DATA_FILENAME + 'measures.csv'
    if file_exists(filename):
        print('Reading data from', filename)
        df = pd.read_csv(filename)
        return df[(df.measure.isin(measures)) &
                  (df.subject.isin(subjects)) &
                  (df.epic.isin(epic_list))]
    else:
        make_measures()
        return get_measures(measures=measures, epic_list=epic_list, subjects=subjects)


# todo cache make_measures
def make_measures(epic_list=None, subjects=SUBJECTS):
    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    df_gradients = make_gradients(epic_list=epic_list, subjects=subjects)
    df_ecc = _make_eccentricity(df_gradients)
    df_measures = pd.concat([df_gradients, df_ecc], axis=0)
    filename = DATA_FILENAME + 'measures.csv'
    df_measures.to_csv(filename, index=False)
    print('Saved the data to', filename)


def _make_eccentricity(df):
    """makes eccentricity with Euclidean distance of ALL gradients available.

    Args:
        df (pd.DataFrame): values for gradients, typically 3 to 4, for each region

    Returns:
        pd.DataFrame: the eccentricity values for each region
    """
    print('Making eccentricity...')
    eccentricity = df \
        .assign(v=lambda r: r['value'] ** 2, aixs=1) \
        .groupby(['subject', 'epic', 'region']).agg({'v': sum}) \
        .v.progress_apply(np.sqrt).reset_index() \
        .rename(columns={'v': 'value'})
    eccentricity['measure'] = 'eccentricity'
    return eccentricity


def get_measures_avg(measures=None, epic_list=None,
                     subjects=SUBJECTS) -> pd.DataFrame:
    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    values = get_measures(measures, epic_list, subjects)
    values = values.groupby(['region', 'epic', 'measure']) \
        .mean().drop('subject', axis=1) \
        .reset_index()
    return values
