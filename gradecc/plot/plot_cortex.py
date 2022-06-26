import pandas as pd
from brainspace.utils.parcellation import map_to_labels
import numpy as np

from gradecc.plot.utils import ATLAS, _init_atlas
from gradecc.plot._surfplot import _surf_plot
from gradecc.stats.utils import ALPHA
from gradecc.utils.utils import melt_df
from gradecc.utils.filenames import labels_filename


# todo cortex and subcortex
def plot_brain():
    pass


def plot_cortex(data, value='value', mask=None, **kwargs):
    if mask is None:
        _plot_cortex(data, value, **kwargs)
    else:
        _plot_brain_masked(data, value, mask, **kwargs)


def _plot_cortex(data, value, **kwargs):
    """
    Args:
        data: if pd.DataFrame, should have `value` and `region`. if pd.Series, should set index to `region`
        value: if data is pd.df, what column to plot
        data: values with regions index
    """
    _init_atlas()
    data = _handle_df_series(data, value)
    data = _sort_regarding_regions(data, value)
    data_mapped_parcels = map_to_labels(data, ATLAS['vertex_labels'], mask=ATLAS['vertex_masked'])
    # todo consider other methods like brainspace hemi plot
    _surf_plot(data_mapped_parcels, **kwargs)


def _handle_df_series(data, value):
    """ If it gets Series, outputs data with region on index
    if it gets DataFrame, outputs a series with region as index
    """
    if isinstance(data, pd.Series):
        return data.rename(value).rename_axis('region')
    elif isinstance(data, pd.DataFrame):
        if 'region' in data.columns:
            return data.set_index('region')[value]
        else:
            # should not work
            return melt_df(data)
    # todo except with type input error


def _sort_regarding_regions(data, value):
    data = data.reset_index()
    fname = labels_filename
    region_index_map = pd.read_csv(fname)
    # todo make class. should not load each time.

    data = data.merge(region_index_map, how='left', left_on='region', right_on='name_7networks')
    data = data.sort_values('index_7networks')
    data = np.array(data[value])
    return data


# todo name plot significant
def _plot_brain_masked(data, value: str, mask: str, **kwargs):
    mask = data[mask]
    value = data[value]
    significance = np.array(mask) < kwargs.get('threshold', ALPHA)
    data['plot_value'] = np.where(significance, np.array(value), 0)
    _plot_cortex(data, 'plot_value', **kwargs)
