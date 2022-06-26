from os import path
import pandas as pd
from gradecc.utils.filenames import subjects_filename, dir_subcortical, atlas_subc_filename
from gradecc.load_timeseries.utils import _window_timeseries, EPOCH_FILENAME_SUBC


def _handle_if_subject_id(subject) -> str:
    """
    Args:
        subject: is either subject_id in int or a two-letter words plus one number
    Returns:
        subject str as it's saved
    """
    if isinstance(subject, str):
        return subject
    elif isinstance(subject, int):
        subjects_match = pd.read_csv(subjects_filename, delimiter='\t')
        subject = subjects_match.loc[subjects_match.participant_id == subject, 'dicom_dir']
        subject = subject.values[0]
        return subject
    # todo raise type exception


def _make_subc_filename(subject: str, epoch):
    filename = subject + '_' + EPOCH_FILENAME_SUBC[epoch] + '.csv'
    return path.join(dir_subcortical, filename)


def _rename_columns_to_region_names(ts_subc):
    atlas_subc = pd.read_csv(atlas_subc_filename)
    subcortical_regions_ordered = atlas_subc['Label'].tolist()
    ts_subc.columns = subcortical_regions_ordered
    return ts_subc


def load_ts_subc(subject, epoch: str) -> pd.DataFrame:
    """ loads timeseries for subcortical regions
    """
    subject: str = _handle_if_subject_id(subject)
    subject_filename = _make_subc_filename(subject, epoch)
    ts_subc = pd.read_csv(subject_filename)
    ts_subc = _rename_columns_to_region_names(ts_subc)
    return _window_timeseries(epoch, ts_subc)


if __name__ == '__main__':
    load_ts_subc(1, 'rest')
