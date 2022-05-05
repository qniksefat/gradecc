import pandas as pd
import numpy as np


EPICS_FILENAME = {'rest': 'rest', 'baseline': 'RLbaseline',
'learning': 'RLlearning', 'early': 'RLlearning', 'late': 'RLlearning'}

DATA_DIR = '/Users/qasem/Dropbox/JasonANDQasem_SHARED/codes/RL_dataset_Mar2022/'

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 35, 36, 38,
    39, 40, 42, 43, 44, 45, 46, ] # ? excluded: 19, 27, 32, 34, 37, 41


def load_timeseries(subject: int, epic: str) -> pd.DataFrame:
    """fMRI timeseries data 

    Args:
        subject (int): subject number
        epic (str): time period during the task, such as `rest`, `baseline`, `early` learning

    Returns:
        pd.DataFrame: columns=regions, rows=time trials
    """
    filename = make_filename(subject, epic)
    timeseries = pd.read_csv(filename, delimiter='\t')
    window_size, start_window = 216, 3
    if epic in ['rest', 'baseline', 'late']:
        return timeseries[-1 * window_size:]
    elif epic == 'early':
        return timeseries[start_window: start_window + window_size]
    elif epic == 'learning':
        return timeseries[window_size: 2 * window_size]


def make_filename(subject: int, epic: str) -> str:
    epic = EPICS_FILENAME[epic]
    #? ses-02 or 01
    if subject < 10:
        try:
            filename = '0' + str(subject) + '_ses-01_task-' + epic + '_run-1'
        except:
            filename = '0' + str(subject) + '_ses-01_task-' + epic + '_run-2'
    else:
        try:
            filename = str(subject) + '_ses-01_task-' + epic + '_run-1'
        except:
            filename = str(subject) + '_ses-01_task-' + epic + '_run-2'
    filename = DATA_DIR + 'sub-' + filename + '_space-fsLR_den-91k_bold_timeseries.tsv'
    return filename


def get_regions_names(idx):
    if not isinstance(idx, list):
        idx = [idx]
    df = load_timeseries(1, 'rest')
    return df.iloc[:0, idx].columns.tolist()


def spot_region(region):
    df = load_timeseries(1, 'rest')
    return np.array(range(df.shape[1])) == region


# todo name all regions
def region_names():
    df = load_timeseries(1, 'rest')
    return df.columns.tolist()