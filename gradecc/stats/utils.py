from os import path
from tqdm import tqdm

from gradecc.utils.filenames import data_inside

tqdm.pandas()

ALPHA = 0.05
FDR_method = 'fdr_bh'


def _make_filename(file):
    filename = file + '.csv'
    filename = path.join(data_inside, filename)
    return filename
