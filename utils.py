import pandas as pd


def melt_df(df):
    df.columns = list(range(df.shape[1]))
    df = pd.melt(df.reset_index(),
                 id_vars=['seed_region', 'epic', 'subject'],
                 value_vars=list(df.columns),
                 var_name='region', value_name='value')
    return df