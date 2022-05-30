import pandas as pd
import numpy as np
from tqdm import tqdm

from gradecc.gradient import make_gradients, NUM_COMPONENTS
from gradecc.load_timeseries import SUBJECTS
from gradecc.utils import file_exists
from gradecc.filenames import measures_filename

tqdm.pandas()


# todo decorator cache
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
    epic_list, measures, subjects = _init_inputs(epic_list, measures, subjects)

    if file_exists(measures_filename):
        print('Reading data from', measures_filename)
        df = pd.read_csv(measures_filename)
        return df[(df.measure.isin(measures)) &
                  (df.subject.isin(subjects)) &
                  (df.epic.isin(epic_list))]
    else:
        make_measures()
        return get_measures(measures=measures, epic_list=epic_list, subjects=subjects)


def _init_inputs(epic_list, measures, subjects):
    if measures is None:
        measures = _make_measures_list()
    elif not isinstance(measures, list):
        measures = [measures]

    if subjects is None:
        subjects = SUBJECTS
    elif not isinstance(subjects, list):
        subjects = [subjects]

    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    elif not isinstance(epic_list, list):
        epic_list = [epic_list]

    return epic_list, measures, subjects


def _make_measures_list():
    return ['gradient' + str(i + 1) for i in range(NUM_COMPONENTS)] + ['eccentricity']


# todo cache make_measures
def make_measures(epic_list=None, subjects=SUBJECTS):
    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    df_gradients = make_gradients(epic_list=epic_list, subjects=subjects)
    df_ecc = _make_eccentricity(df_gradients)
    df_measures = pd.concat([df_gradients, df_ecc], axis=0)
    df_measures.to_csv(measures_filename, index=False)
    print('Data saved to', measures_filename)
    return df_measures


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


def get_measures_avg(epic_list=None, measures=None,
                     subjects=SUBJECTS) -> pd.DataFrame:
    # todo handle non valid inputs like LATE
    values = get_measures(measures, epic_list, subjects)
    values = values.groupby(['region', 'epic', 'measure']) \
        .mean().drop('subject', axis=1) \
        .reset_index()
    return values


if __name__ == '__main__':
    pass
