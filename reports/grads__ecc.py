from gradecc.compute.measures import get_measures_avg
from gradecc.plot import plot_cortex, plot_subc


def _plot_grads():
    for grad in ['gradient1', 'gradient2', 'gradient3', 'gradient4']:
        for epoch in ['baseline', 'early', 'late']:
            grad_data = get_measures_avg(epoch_list=epoch, measures=grad)

            plot_cortex(grad_data, color_map='bwr', color_range=(-3.7, 3.7),
                        text=('cortical avg ' + grad + ' ' + epoch.upper()),
                        save_figure=True)

            plot_subc(grad_data, color_map='bwr', color_range=(-.5, .5),
                      text=('subcortical avg ' + grad + ' ' + epoch.upper()))


def _plot_ecc():
    for grad in ['eccentricity']:
        for epoch in ['baseline', 'early', 'late']:
            grad_data = get_measures_avg(epoch_list=epoch, measures=grad)

            plot_cortex(grad_data, color_map='viridis', color_range=(0, 4),
                        text=('cortical avg ' + grad + ' ' + epoch.upper()),
                        save_figure=True)

            plot_subc(grad_data, color_map='viridis', color_range=(.6, 1),
                      text=('subcortical avg ' + grad + ' ' + epoch.upper()))


if __name__ == '__main__':
    _plot_ecc()
    _plot_grads()
