import pandas as pd
import os.path

from load_timeseries import all_region_names

DATA_FILENAME = 'data/'


def melt_df(df):
    df.columns = all_region_names()
    df = pd.melt(df.reset_index(),
                 id_vars=['seed_region', 'epic', 'subject'],
                 value_vars=list(df.columns),
                 var_name='region', value_name='value')
    return df


def file_exists(filename):
    return os.path.isfile(filename)
