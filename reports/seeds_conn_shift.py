import operator

from gradecc.compute.measures import get_measures
from gradecc.plot import plot_cortex, plot_subc
from gradecc.stats import rm_anova
from gradecc.stats.pairwise_ttests import seed_ttests
from gradecc.seed_conn import seed_connectivity
from gradecc.plot.utils import spot_region


def _find_seeds():
    df = get_measures()
    df_stats = rm_anova(df)
    df_stats_ecc = df_stats[df_stats.measure == 'eccentricity']

    plot_cortex(df_stats_ecc, 'F', 'pvalue_corrected',
                color_range=(1, 10), layout='grid')

    sig_regions = df_stats_ecc[df_stats_ecc.fdr_significant == True].region.tolist()
    # spotted by eyes
    # regions_of_interest = [1, 4, 7, 15, 17, 20, 27, 30, 45]
    regions_of_interest = [1, 4]
    regions_of_interest = operator.itemgetter(*regions_of_interest)(sig_regions)
    return list(regions_of_interest)


def _plot_shifts(seeds):
    for seed in seeds:
        # pass seeds to seed_connectivity
        df_seed = seed_connectivity(seed)
        df_seed_shift = seed_ttests(df_seed)

        for pair in df_seed_shift.index.unique():
            text = seed[10:] + ' seed ' + pair[0][0].upper() + ' to ' + pair[1][0].upper()
            # todo make plot_brain
            plot_cortex(df_seed_shift.loc[pair], 'tstat',
                        text=('cortex ' + text), color_range=(-4, 4), color_map='bwr', layout='grid',
                        save_figure=True)

            plot_subc(df_seed_shift.loc[pair], 'tstat',
                      text=('subc ' + text), color_range=(-4, 4), color_map='bwr')


if __name__ == '__main__':
    regions_of_interest = _find_seeds()

    for region in regions_of_interest:
        plot_cortex(spot_region(region), layout='grid',
                    text=region[10:], as_outline=True, save_figure=True)

    _plot_shifts(regions_of_interest)
