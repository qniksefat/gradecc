from gradecc.compute.measures import get_measures_avg
from gradecc.plot import plot_cortex, plot_subcortex


def plot_measure(measures):
    color_range = {('cortex', 'eccentricity'): (0, 4),
        ('subc', 'eccentricity'): (.6, 1)}
    color_map = {'eccentricity': 'viridis'}

    for m in measures:
        for epoch in ['baseline', 'early', 'late']:
            grad_data = get_measures_avg(epoch_list=epoch, measures=m)

            plot_cortex(grad_data, color_map=color_map.get(m, 'bwr'),
                        color_range=color_range.get(('cortex', m), (-3.7, 3.7)),
                        text=('cortical avg ' + m + ' ' + epoch.upper()),
                        save_figure=True)

            plot_subcortex(grad_data, color_map=color_map.get(m, 'bwr'),
                           color_range=color_range.get(('subc', m), (-.5, .5)),
                           text=('subcortical avg ' + m + ' ' + epoch.upper()))


if __name__ == '__main__':
    plot_measure(['eccentricity'])
    plot_measure(['gradient1', 'gradient2'])
