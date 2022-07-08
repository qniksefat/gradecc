import pandas as pd

from gradecc.utils.filenames import subjects_filename


class Subject:
    @staticmethod
    def make_subject_id_mapping():
        subject_id_match = pd.read_csv(subjects_filename, delimiter='\t')
        subject_id_match = subject_id_match[subject_id_match.exclude == False]

        subject_id_dicts = []
        for key_value in [['participant_id', 'dicom_dir'],
                          ['dicom_dir', 'participant_id']]:
            subject_id_match_ = subject_id_match[key_value].to_dict('split')['data']
            subject_id_dicts.append(dict(subject_id_match_))
        return subject_id_dicts[0], subject_id_dicts[1]

    _subject_id_mappings = make_subject_id_mapping()
    SUBJECT_ID_INT2STR: dict[int, str] = _subject_id_mappings[0]
    SUBJECT_ID_STR2INT: dict[str, int] = _subject_id_mappings[1]

    def __init__(self, subject_id):
        self._subject_id = subject_id
        self.int = None
        self.str = None
        self.adapt_format_subject_id()

    def subject_id_is_int(self):
        return str(self._subject_id).isdigit()

    def adapt_format_subject_id(self):
        if self.subject_id_is_int():
            self.int = int(self._subject_id)
            self.str = self.SUBJECT_ID_INT2STR[self.int]
        else:
            self.str = self._subject_id
            self.int = self.SUBJECT_ID_STR2INT[self.str]


SUBJECTS_INT = list(Subject.SUBJECT_ID_INT2STR.keys())