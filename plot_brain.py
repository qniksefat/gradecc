import pandas as pd
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels
from surfplot import Plot
from stats import ALPHA
import numpy as np
import nibabel as nib

nib.imageglobals.logger.level = 40

IMAGE_DIR = '../grad_results/'
ATLAS_FILENAME = 'data/Schaefer2018_1000Parcels_7Networks_order.dlabel.nii'
ATLAS = {}


def load_atlas(filename=ATLAS_FILENAME):
    vertices = nib.load(filename).get_fdata()
    vertices = vertices[0]
    mask = vertices != 0
    return vertices, mask


def _init_atlas():
    if ATLAS == {}:
        ATLAS['left_surface'], ATLAS['right_surface'] = load_conte69()
        ATLAS['vertex_labels'], ATLAS['vertex_masked'] = load_atlas()


def plot_brain(data, value='value', save_figure=False, **kwargs):
    """ if pd.DataFrame, should have `value` and `region`. if pd.Series, should set index to `region`
    Args:
        value: if data is pd.df, what column to plot
        save_figure: saves the figure to DIR with text filename
        data: values with regions index
    """
    _init_atlas()
    data = _handle_df_series(data, value)
    data = _sort_to_plot(data, value)
    data = map_to_labels(data, ATLAS['vertex_labels'], mask=ATLAS['vertex_masked'])
    # todo consider other methods like brainspace hemi plot
    _surf_plot(data, save_figure, **kwargs)


def _handle_df_series(data, value):
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame):
        return data.set_index('region')[value]
    # todo except with type input error


def _sort_to_plot(data, value):
    data = data.reset_index()
    fname = 'data/Schaefer2018_1000Parcels_labels.csv'
    region_index_map = pd.read_csv(fname)
    # todo make class. should not load each time.

    data = data.merge(region_index_map, how='left', left_on='region', right_on='name_7networks')
    data = data.sort_values('index_7networks')
    data = np.array(data[value])
    return data


def _surf_plot(data, save_figure=False, **kwargs):
    text = kwargs.get('text', '')
    p = Plot(surf_lh=ATLAS['left_surface'], surf_rh=ATLAS['right_surface'],
             size=(1600, 300), layout='row', label_text=[text])
    p.add_layer(data, cbar=True, cmap=kwargs.get('color_map', 'viridis'),
                color_range=kwargs.get('color_range', None))
    figure = p.build()
    # figure.show()
    if save_figure:
        figure.savefig(IMAGE_DIR + text)


# todo name plot significant
def plot_brain_masked(data, value: str, mask: str, threshold=ALPHA, **kwargs):
    mask = data[mask]
    value = data[value]
    significance = np.array(mask) < threshold
    data['plot_value'] = np.where(significance, np.array(value), 0)
    plot_brain(data, 'plot_value', **kwargs)


if __name__ == '__main__':
    pass
