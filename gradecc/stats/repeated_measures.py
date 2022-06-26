import pandas as pd
import pingouin as pg

from gradecc.stats.false_discovery import _fdr_correction
from gradecc.utils.filenames import rm_anova_filename
from gradecc.utils.utils import file_exists


def rm_anova(df=None):
    """computes repeated measures ANOVA for each region in epochs
    Returns:
        pd.DataFrame with F-statistic and p-values
    """
    if file_exists(rm_anova_filename):
        return pd.read_csv(rm_anova_filename)
    else:
        if _has_one_epoch(df):
            raise ValueError('file Needs more than one epoch to compute ANOVA within them')

        return make_rm_anova(df)


def make_rm_anova(df):
    df_stats = _compute_rm_anova(df)
    df_stats = _fdr_correction(df_stats)
    df_stats.to_csv(rm_anova_filename, index=False)
    return df_stats


def _compute_rm_anova(df, within='epoch'):
    print('Computing repeated measures ANOVA...')
    df = df.groupby(['region', 'measure']).progress_apply(pg.rm_anova, dv='value',
                                                          within=within, subject='subject')
    df = df.rename(columns={'p-unc': 'pvalue'})
    df = df.reset_index()[['measure', 'region', 'F', 'pvalue']]
    return df


def _has_one_epoch(df):
    return df['epoch'].nunique() < 2
