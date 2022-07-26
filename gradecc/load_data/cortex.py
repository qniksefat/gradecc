from os import path
import pandas as pd

from gradecc.utils.filenames import dir_dataset

CORTEX_FILENAME_SUFFIX = '_space-fsLR_den-91k_bold_timeseries.tsv'
EPOCH_FILENAME_CORTEX = {'rest': 'rest', 'baseline': 'RLbaseline',
                         'learning': 'RLlearning', 'early': 'RLlearning', 'late': 'RLlearning'}


def try_load_filenames(subject: int, epoch: str):
    try:
        filename = _make_filename_with_run(subject, epoch, run=1)
        return pd.read_csv(filename, delimiter='\t')
    except FileNotFoundError:
        filename = _make_filename_with_run(subject, epoch, run=2)
        return pd.read_csv(filename, delimiter='\t')


def _make_filename_with_run(subject: int, epoch: str, run: int) -> str:
    epoch = EPOCH_FILENAME_CORTEX[epoch]
    filename = _filename_two_digits(epoch, subject, run)
    filename = 'sub-' + filename + CORTEX_FILENAME_SUFFIX
    filename = path.join(dir_dataset, filename)
    return filename


def _filename_two_digits(epoch, subject, run):
    filename = str(subject) + '_ses-01_task-' + epoch + '_run-' + str(run)
    if subject < 10:
        return '0' + filename
    else:
        return filename
