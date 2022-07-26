import pandas as pd
from tqdm import tqdm

from gradecc.load_data import all_region_names

tqdm.pandas()

ALPHA = 0.05
FDR_method = 'fdr_bh'


def melt_df(df):
    df.columns = all_region_names()
    df = pd.melt(df.reset_index(),
                 id_vars=['seed_region', 'epoch', 'subject'],
                 value_vars=list(df.columns),
                 var_name='region', value_name='value')
    return df
