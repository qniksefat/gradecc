import pandas as pd
import pingouin as pg

from gradecc.stats.false_discovery import _fdr_correction
from gradecc.stats.utils import _make_filename
from gradecc.utils.utils import file_exists


def rm_anova(df=None):
    """computes repeated measures ANOVA for each region in epics
    Returns:
        pd.DataFrame with F-statistic and p-values
    """
    filename = _make_filename('rm_anova')
    if file_exists(filename):
        return pd.read_csv(filename)
    else:
        if _has_one_epic(df):
            raise ValueError('file Needs more than one epic to compute ANOVA within them')

        return make_rm_anova(df)


def make_rm_anova(df):
    df_stats = _compute_rm_anova(df)
    df_stats = _fdr_correction(df_stats)
    filename = _make_filename('rm_anova')
    df_stats.to_csv(filename, index=False)
    return df_stats


def _compute_rm_anova(df):
    print('Computing repeated measures ANOVA...')
    df = df.groupby(['region', 'measure']).progress_apply(pg.rm_anova, dv='value', within='epic', subject='subject')
    df = df.rename(columns={'p-unc': 'pvalue'})
    df = df.reset_index()[['measure', 'region', 'F', 'pvalue']]
    return df


def _has_one_epic(df):
    return df.epic.nunique() < 2
