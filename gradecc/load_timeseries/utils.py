from os import path
import pandas as pd

from gradecc.utils.filenames import dir_dataset, subjects_filename

CORTICAL_FILENAME_SUFFIX = '_space-fsLR_den-91k_bold_timeseries.tsv'

EPICS_FILENAME_CORTEX = {'rest': 'rest', 'baseline': 'RLbaseline',
                         'learning': 'RLlearning', 'early': 'RLlearning', 'late': 'RLlearning'}

EPICS_FILENAME_SUBC = {'rest': 'Rest_Rest', 'baseline': 'RL_Baseline',
                       'learning': 'RL_Learning', 'early': 'RL_Learning', 'late': 'RL_Learning'}


def _make_subjects_list():
    subjects = pd.read_csv(subjects_filename, delimiter='\t')
    subjects = subjects[subjects.exclude == False].participant_id.to_list()
    return subjects


SUBJECTS = _make_subjects_list()


def _window_timeseries(epic, timeseries):
    window_size, start_window = 216, 3
    if epic in ['rest', 'baseline', 'late']:
        return timeseries[-1 * window_size:]
    elif epic == 'early':
        return timeseries[start_window: start_window + window_size]
    elif epic == 'learning':
        return timeseries[window_size: 2 * window_size]


def _try_load_filenames(epic, subject):
    try:
        filename = _make_filename(subject, epic, run=1)
        timeseries = pd.read_csv(filename, delimiter='\t')
    # todo except FileNotFoundError:
    except:
        filename = _make_filename(subject, epic, run=2)
        timeseries = pd.read_csv(filename, delimiter='\t')
    return timeseries


def _make_filename(subject: int, epic: str, run: int) -> str:
    epic = EPICS_FILENAME_CORTEX[epic]
    filename = _filename_two_digits(epic, subject, run)
    filename = 'sub-' + filename + CORTICAL_FILENAME_SUFFIX
    filename = path.join(dir_dataset, filename)
    return filename


def _filename_two_digits(epic, subject, run):
    filename = str(subject) + '_ses-01_task-' + epic + '_run-' + str(run)
    if subject < 10:
        return '0' + filename
    else:
        return filename