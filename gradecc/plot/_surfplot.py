from os import path
from surfplot import Plot

from gradecc.plot.utils import ATLAS
from gradecc.utils.filenames import dir_images


def _surf_plot(data, **kwargs):
    layout, size, text = _init_surfplot(**kwargs)
    p = Plot(surf_lh=ATLAS['left_surface'], surf_rh=ATLAS['right_surface'],
             label_text=[text], layout=layout, size=size)
    p.add_layer(data, cbar=True,
                cmap=kwargs.get('color_map', 'viridis'),
                as_outline=kwargs.get('as_outline', False),
                color_range=kwargs.get('color_range', None))
    figure = p.build()
    output_filename = kwargs.get('output_filename', False)
    if output_filename:  _save_figure(figure, output_filename)
    elif kwargs.get('save_figure', False):    _save_figure(figure, text)


def _save_figure(figure, text):
    figure_filename = path.join(dir_images, text)
    figure.savefig(figure_filename)


def _init_surfplot(**kwargs):
    text = kwargs.get('text', '')
    layout = kwargs.get('layout', 'row')
    size_layout = {'row': (1600, 300), 'column': (700, 1100), 'grid': (900, 700)}
    size = size_layout[layout]
    return layout, size, text
