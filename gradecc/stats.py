import pandas as pd
import pingouin as pg
from tqdm import tqdm

from gradecc.utils import file_exists, melt_df
from gradecc.filenames import DATA_FILENAME

tqdm.pandas()

ALPHA = 0.05
FDR_method = 'fdr_bh'


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


def _make_filename(file):
    return DATA_FILENAME + file + '.csv'


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


def _fdr_correction(df_stats):
    df_list = []
    for measure in df_stats.measure.unique():
        _df = df_stats[df_stats.measure == measure]
        reject, pvalue_corr = pg.multicomp(_df.pvalue.tolist(), alpha=ALPHA, method=FDR_method)
        _df = _df.set_index('region')
        _df['fdr_significant'] = reject
        _df['pvalue_corrected'] = pvalue_corr
        _df = _df.reset_index()
        df_list.append(_df)
    df_stats = pd.concat(df_list, axis=0)
    return df_stats


def seed_ttests(df_seed, **kwargs):
    df_seed = melt_df(df_seed)
    return make_ttests(df_seed, save=True, **kwargs)


# todo separate ttests


def pairwise_ttests(df):
    """ FDR-corrected
    In input, if it's pd.df you should just provide column names
    """
    filename = _make_filename('ttests')
    if file_exists(filename):
        _, index = _prepare_for_seed_conn(df)
        return pd.read_csv(filename).set_index(index)
    else:
        return make_ttests(df, save=True)


def make_ttests(df, save: bool, **kwargs):
    # todo bad smell. no need to involve seed conn when making ttests
    df_grouped, index = _prepare_for_seed_conn(df)
    print('Computing ttests...')
    df_stats_pairwise = df_grouped.progress_apply(pg.pairwise_ttests, dv='value', between='epic',
                                                  subject='subject', padjust=FDR_method)
    df_stats_pairwise = df_stats_pairwise.reset_index().set_index(index)[['region', 'T', 'p-corr']]
    df_stats_pairwise = df_stats_pairwise.rename(columns={'p-corr': 'pvalue_corrected', 'T': 'tstat'})
    df_stats_pairwise = df_stats_pairwise.sort_index()
    if save:
        filename = _make_filename(kwargs.get('filename', 'ttests'))
        df_stats_pairwise.to_csv(filename)
    return df_stats_pairwise


def _prepare_for_seed_conn(df):
    index = ['A', 'B']
    if 'measure' in df.columns:
        df_grouped = df.groupby(['measure', 'region'])
        index = ['measure'] + index
    else:
        # means it's seed conn
        df_grouped = df.groupby(['region'])
    return df_grouped, index


if __name__ == '__main__':
    from measures import get_measures
    df = get_measures()
    df_stats_pairwise = pairwise_ttests(df)