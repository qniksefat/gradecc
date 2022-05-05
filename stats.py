import pandas as pd
import pingouin as pg
from tqdm import tqdm
tqdm.pandas()

ALPHA = 0.05
FDR_method = 'fdr_bh'


def get_rm_anova(df: pd.DataFrame):
    """computes repeated measures ANOVA for each region in epics

    Args:
        df:

    Returns:
        pd.DataFrame with F-statistic and p-values
    """
    if df.epic.nunique() < 2:
        raise ValueError('Needs more than one epic to compute ANOVA within them')
    print('Computing repeated measures ANOVA...')
    _df = df.groupby(['region', 'measure']).progress_apply(pg.rm_anova, dv='value', within='epic', subject='subject')
    _df = _df.rename(columns={'p-unc': 'pvalue'})
    _df = _df.reset_index()[['measure', 'region', 'F', 'pvalue']]
    return _df


def append_fdr(df):
    df_list = []
    for measure in df.measure.unique():
        _df = df[df.measure == measure]
        reject, pvalue_corrected = pg.multicomp(_df.pvalue.tolist(), alpha=ALPHA, method=FDR_method)
        _df = _df.set_index('region')
        _df['fdr_significant'] = reject
        _df['pvalue_corrected'] = pvalue_corrected
        _df = _df.reset_index()
        df_list.append(_df)
    df = pd.concat(df_list, axis=0)
    return df


def pairwise_ttests(df):
    """ FDR-corrected

    Args:
        df:

    Returns:

    """
    df_grouped = df.groupby(['measure', 'region'])
    df_stats_pairwise = df_grouped.progress_apply(pg.pairwise_ttests,
                                                                dv='value', between='epic',
                                                                subject='subject', padjust=FDR_method)
    df_stats_pairwise = df_stats_pairwise.reset_index().set_index(['measure', 'A', 'B'])[['region', 'T', 'p-corr']]
    df_stats_pairwise = df_stats_pairwise.rename(columns={'p-corr': 'pvalue_corrected', 'T': 'tstat'})
    return df_stats_pairwise
