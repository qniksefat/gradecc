from os import path
import pandas as pd

from gradecc.utils.filenames import subjects_filename, dir_subcortical, atlas_subc_filename


# todo decide: perhaps deprecated
def _handle_if_subject_id(subject) -> str:
    """
    Args:
        subject: is either subject_id in int or a two-letter words plus one number
    Returns:
        subject str as it's saved
    """
    # todo move it such that it works for cortex too
    if isinstance(subject, str):
        return subject
    elif isinstance(subject, int):
        subjects_match = pd.read_csv(subjects_filename, delimiter='\t')
        subject = subjects_match.loc[subjects_match.participant_id == subject, 'dicom_dir']
        subject = subject.values[0]
        return subject


def _make_subc_filename(subject: str, epoch):
    filename = subject + '_' + EPOCH_FILENAME_SUBCORTEX[epoch] + '.csv'
    return path.join(dir_subcortical, filename)


def _rename_columns_to_region_names(ts_subc):
    atlas_subc = pd.read_csv(atlas_subc_filename)
    subcortical_regions_ordered = atlas_subc['Label'].tolist()
    ts_subc.columns = subcortical_regions_ordered
    return ts_subc


EPOCH_FILENAME_SUBCORTEX = {'rest': 'Rest_Rest', 'baseline': 'RL_Baseline',
                            'learning': 'RL_Learning', 'early': 'RL_Learning', 'late': 'RL_Learning'}
