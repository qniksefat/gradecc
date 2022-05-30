from os import path
import pandas as pd
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels
from surfplot import Plot
import numpy as np
import nibabel as nib

from gradecc.stats import ALPHA
from gradecc.utils import melt_df
from gradecc.filenames import dir_images, atlas_filename


nib.imageglobals.logger.level = 40

ATLAS = {}


def load_atlas(filename=atlas_filename):
    vertices = nib.load(filename).get_fdata()
    vertices = vertices[0]
    mask = vertices != 0
    return vertices, mask


def _init_atlas():
    if ATLAS == {}:
        ATLAS['left_surface'], ATLAS['right_surface'] = load_conte69()
        ATLAS['vertex_labels'], ATLAS['vertex_masked'] = load_atlas()
        # todo inflated brain maps
        """
        from neuromaps.datasets import fetch_fsaverage
        surfaces = fetch_fsaverage(density='164k')
        ATLAS['left_surface'], ATLAS['right_surface'] = surfaces['inflated']
        from nilearn.datasets import fetch_surf_fsaverage
        ATLAS['left_surface'] = fetch_surf_fsaverage(mesh='fsaverage')['curv_left']
        ATLAS['right_surface'] = fetch_surf_fsaverage(mesh='fsaverage')['curv_right']
        """


def plot_brain(data, value='value', mask=None, **kwargs):
    if mask is None:
        _plot_brain(data, value, **kwargs)
    else:
        _plot_brain_masked(data, value, mask, **kwargs)


# todo involve masked to plot.
def _plot_brain(data, value, **kwargs):
    """
    Args:
        data: if pd.DataFrame, should have `value` and `region`. if pd.Series, should set index to `region`
        value: if data is pd.df, what column to plot
        data: values with regions index
    """
    _init_atlas()
    data = _handle_df_series(data, value)
    data = _sort_to_plot(data, value)
    data = map_to_labels(data, ATLAS['vertex_labels'], mask=ATLAS['vertex_masked'])
    # todo consider other methods like brainspace hemi plot
    _surf_plot(data, **kwargs)


def _handle_df_series(data, value):
    if isinstance(data, pd.Series):
        return data.rename(value).rename_axis('region')
    elif isinstance(data, pd.DataFrame):
        if 'region' in data.columns:
            return data.set_index('region')[value]
        else:
            # should not work
            return melt_df(data)
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


def _surf_plot(data, **kwargs):
    layout, size, text = _init_surfplot(**kwargs)
    p = Plot(surf_lh=ATLAS['left_surface'], surf_rh=ATLAS['right_surface'],
             label_text=[text], layout=layout, size=size)
    p.add_layer(data, cbar=True,
                cmap=kwargs.get('color_map', 'viridis'),
                as_outline=kwargs.get('as_outline', False),
                color_range=kwargs.get('color_range', None))
    figure = p.build()
    # figure.show()
    if kwargs.get('save_figure', False):
        _save_figure(figure, text)


def _save_figure(figure, text):
    figure_filename = path.join(dir_images, text)
    figure.savefig(figure_filename)


def _init_surfplot(**kwargs):
    text = kwargs.get('text', '')
    layout = kwargs.get('layout', 'row')
    size_layout = {'row': (1600, 300), 'column': (700, 1100), 'grid': (900, 700)}
    size = size_layout[layout]
    return layout, size, text


# todo name plot significant
def _plot_brain_masked(data, value: str, mask: str, **kwargs):
    mask = data[mask]
    value = data[value]
    significance = np.array(mask) < kwargs.get('threshold', ALPHA)
    data['plot_value'] = np.where(significance, np.array(value), 0)
    _plot_brain(data, 'plot_value', **kwargs)


if __name__ == '__main__':
    # %%
    from gradecc.measures import get_measures

    df = get_measures()
    # %%
    from gradecc.stats import rm_anova

    df_stats = rm_anova(df)
    df_stats_ecc = df_stats[df_stats.measure == 'eccentricity']

    plot_brain(df_stats_ecc, 'F', 'pvalue_corrected', color_range=(1, 10))
