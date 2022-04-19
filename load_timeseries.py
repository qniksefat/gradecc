import pandas as pd

EPICS_FNAME = {'rest': 'rest', 'baseline': 'RLbaseline',
'learning': 'RLlearning', 'early': 'RLlearning', 'late': 'RLlearning'}

DATA_DIR = '/Users/qasem/Dropbox/JasonANDQasem_SHARED/codes/RL_dataset_Mar2022/'

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 35, 36, 38,
    39, 40, 42, 43, 44, 45, 46, ] # ? excluded: 19, 27, 32, 34, 37, 41


def load_timeseries(subject: int, epic: str) -> pd.DataFrame:
    fname = make_fname(subject, epic)
    timeseries = pd.read_csv(fname, delimiter='\t')
    window_size, start_window = 216, 3
    if epic in ['rest', 'baseline', 'late']:
        return timeseries[-1 * window_size:]
    elif epic == 'early':
        return timeseries[start_window: start_window + window_size]
    elif epic == 'learning':
        return timeseries[window_size: 2 * window_size]


def make_fname(subj: int, epic: str) -> str:
    epic = EPICS_FNAME[epic]
    #? ses-02 or 01
    if subj < 10:
        try:
            fname = '0' + str(subj) + '_ses-01_task-' + epic + '_run-1'
        except:
            fname = '0' + str(subj) + '_ses-01_task-' + epic + '_run-2'
    else:
        try:
            fname = str(subj) + '_ses-01_task-' + epic + '_run-1'
        except:
            fname = str(subj) + '_ses-01_task-' + epic + '_run-2'
    fname = DATA_DIR + 'sub-' + fname + '_space-fsLR_den-91k_bold_timeseries.tsv'
    return fname
