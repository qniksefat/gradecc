import pandas as pd
import pingouin as pg

from gradecc.stats.utils import FDR_method, melt_df
from gradecc.utils.filenames import ttests_filename
from gradecc.utils import file_exists


def ttests(df):
    """ FDR-corrected
    In input, if it's pd.df you should just provide column names
    """
    if file_exists(ttests_filename):
        _, index = _prepare_for_seed_conn(df)
        return pd.read_csv(ttests_filename).set_index(index)
    else:
        return make_ttests(df, save=True)


def make_ttests(df, save: bool, between='epoch'):
    # todo bad smell. no need to involve seed conn when making ttests
    df_grouped, index = _prepare_for_seed_conn(df)
    print('Computing t-tests...')
    df_pairwise = df_grouped.progress_apply(pg.pairwise_ttests, dv='value', between=between,
                                            subject='subject', padjust=FDR_method)
    df_pairwise = df_pairwise.reset_index().set_index(index)[['region', 'T', 'p-corr']]
    df_pairwise = df_pairwise.rename(columns={'p-corr': 'pvalue_corrected', 'T': 'tstat'})
    df_pairwise = df_pairwise.sort_index()
    if save:    df_pairwise.to_csv(ttests_filename)
    return df_pairwise


def _prepare_for_seed_conn(df):
    index = ['A', 'B']
    if 'measure' in df.columns:
        df_grouped = df.groupby(['measure', 'region'])
        index = ['measure'] + index
    # else means it's seed conn pairs
    else:   df_grouped = df.groupby(['region'])
    return df_grouped, index


def seed_ttests(df_seed):
    df_seed = melt_df(df_seed)
    return make_ttests(df_seed, save=False)
