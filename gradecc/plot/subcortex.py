from os import path

import pandas as pd
from enigmatoolbox.plotting import plot_subcortical

from gradecc.utils.filenames import atlas_subc_order_filename, dir_images


def _init_enigma_plot(**kwargs):
    text = kwargs.get('text', ' ')
    size = kwargs.get('size', (1600, 800))
    return size, text


def _enigma_plot(data, value, **kwargs):
    size, text = _init_enigma_plot(**kwargs)
    plot_subcortical(array_name=data[value], size=size,
                     cmap=kwargs.get('color_map', 'viridis'),
                     color_range=kwargs.get('color_range', None),
                     filename=path.join(dir_images, text + '.png'),
                     color_bar=True, ventricles=False,
                     nan_color=(.8, .8, .8, .4), interactive=False, screenshot=True,
                     )


def _sort_re_enigma_order(data):
    subc_ordered_to_plot = pd.read_csv(atlas_subc_order_filename)
    # todo should not load each time
    subc_ordered_to_plot['region'] = subc_ordered_to_plot['Hemisphere'] + ' ' + subc_ordered_to_plot['Structure']
    subc_ordered_to_plot = subc_ordered_to_plot[['region']]
    data = pd.merge(subc_ordered_to_plot, data, how='left', on='region')
    return data


def _handle_df_series(data, value):
    if isinstance(data, pd.Series):
        return data.rename(value).rename_axis('region')
    elif isinstance(data, pd.DataFrame):
        if 'region' in data.columns:
            return data.set_index('region')[value]


def plot_subcortex(data, value='value', **kwargs):
    data = _handle_df_series(data, value)
    data = _sort_re_enigma_order(data)
    _enigma_plot(data, value, **kwargs)


if __name__ == '__main__':
    value = 'value'
    from gradecc.load_data import load_ts_subc

    a = load_ts_subc(1, 'rest').var(axis=0)

    a = a.rename(value).rename_axis('region')
    plot_subcortex(a)
