import pandas as pd
import pingouin as pg

from gradecc.stats.utils import ALPHA, FDR_method


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
