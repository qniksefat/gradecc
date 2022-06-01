import pandas as pd
import pingouin as pg

from gradecc.stats.utils import _make_filename, FDR_method
from gradecc.utils.utils import file_exists, melt_df


def ttests(df):
    """ FDR-corrected
    In input, if it's pd.df you should just provide column names
    """
    filename = _make_filename('ttests')
    if file_exists(filename):
        _, index = _prepare_for_seed_conn(df)
        print('index', index)
        print(filename)
        print(
            pd.read_csv(filename)
        )
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


def seed_ttests(df_seed, **kwargs):
    df_seed = melt_df(df_seed)
    return make_ttests(df_seed, save=False, **kwargs)
