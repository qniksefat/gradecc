from gradecc.compute.measures import get_measures
from gradecc.stats.repeated_measures import rm_anova
from gradecc.stats.pairwise_ttests import ttests
from gradecc.plot import plot_cortex, plot_subc


def prepare_data():
    df = get_measures()
    df_stats = rm_anova(df)
    df_stats_ecc = df_stats[df_stats.measure == 'eccentricity']
    return df_stats_ecc


# todo think of multilevel anova. each region has four values. instead of one Eccentricity.
def _plot_rm_anova_regions():
    df_stats_ecc = prepare_data()

    plot_cortex(df_stats_ecc, 'F', 'pvalue_corrected',
                color_range=(1, 15), layout='row',
                text='cortical ecc rm-anova FDR-cor', save_figure=True
                )

    plot_subc(df_stats_ecc, 'F', color_range=None,
              text='subcortical ecc rm-anova',
              )

    print('Number of significant regions:', df_stats_ecc.fdr_significant.sum())
    print('Regions of interest:',
          df_stats_ecc[df_stats_ecc.fdr_significant == True].region.tolist())


def _plot_ttests(mask_FDR=False):
    # these areas show expansion vs contraction
    df = get_measures()
    df_stats_pairwise = ttests(df)
    df_ecc_pairs = df_stats_pairwise.xs('eccentricity', level=0)
    for pair in df_ecc_pairs.index.unique():
        text = 'ecc t-tests ' + pair[0] + ' to ' + pair[1]

        plot_cortex(df_ecc_pairs.loc[pair], 'tstat',
                    text='cortical ' + text, color_range=(-4, 4), color_map='bwr', layout='grid', save_figure=True)

        plot_subc(df_ecc_pairs.loc[pair], 'tstat',
                  text='subcortical ' + text, color_range=(-2, 2), color_map='bwr')

    if mask_FDR:
        pair = ('baseline', 'early')
        plot_cortex(df_ecc_pairs.loc[pair], 'tstat', 'pvalue_corrected',
                    text='region-wise t-tests FDR-corrected significant',
                    color_range=(-4, 4), color_map='bwr',
                    save_figure=True)


if __name__ == '__main__':
    _plot_rm_anova_regions()
    _plot_ttests()
