import pandas as pd
import pingouin as pg
from tqdm import tqdm

tqdm.pandas()

ALPHA = 0.05
FDR_method = 'fdr_bh'


def rm_anova(df: pd.DataFrame):
    """computes repeated measures ANOVA for each region in epics
    Returns:
        pd.DataFrame with F-statistic and p-values
    """
    if _has_one_epic(df):
        raise ValueError('Needs more than one epic to compute ANOVA within them')
    print('Computing repeated measures ANOVA...')
    df = df.groupby(['region', 'measure']).progress_apply(pg.rm_anova, dv='value', within='epic', subject='subject')
    df = df.rename(columns={'p-unc': 'pvalue'})
    df = df.reset_index()[['measure', 'region', 'F', 'pvalue']]
    df = _fdr_correction(df)
    return df


def _has_one_epic(df):
    return df.epic.nunique() < 2


def _fdr_correction(df):
    df_list = []
    for measure in df.measure.unique():
        _df = df[df.measure == measure]
        _df['fdr_significant'], _df['pvalue_corrected'] = \
            pg.multicomp(_df.pvalue.tolist(), alpha=ALPHA, method=FDR_method)
        _df = _df.set_index('region')
        _df = _df.reset_index()
        df_list.append(_df)
    df = pd.concat(df_list, axis=0)
    return df


def pairwise_ttests(df):
    # todo in input, if it's pd.df you should just provide column names
    """ FDR-corrected
    """
    df_grouped, index = _seed_ttests(df)
    df_stats_pairwise = df_grouped.progress_apply(pg.pairwise_ttests, dv='value', between='epic',
                                                  subject='subject', padjust=FDR_method)
    df_stats_pairwise = df_stats_pairwise.reset_index().set_index(index)[['region', 'T', 'p-corr']]
    df_stats_pairwise = df_stats_pairwise.rename(columns={'p-corr': 'pvalue_corrected', 'T': 'tstat'})
    return df_stats_pairwise


def _seed_ttests(df):
    index = ['A', 'B']
    if 'measure' in df.columns:
        df_grouped = df.groupby(['measure', 'region'])
        index = ['measure'] + index
    else:
        # for seed connectivity t-tests
        df_grouped = df.groupby(['region'])
    return df_grouped, index
