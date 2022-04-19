from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels
from surfplot import Plot
import nibabel as nib

import pandas as pd
import numpy as np

IMAGE_DIR = '../grad_results/'
ATLAS_FNAME = 'data/Schaefer2018_1000Parcels_7Networks_order.dlabel.nii'
ATLAS = {}


def load_atlas(filename=ATLAS_FNAME):
    vertices = nib.load(filename).get_fdata()
    vertices = vertices[0]
    mask = vertices != 0
    return vertices, mask


def _init_atlas():
    if ATLAS == {}:
        ATLAS['left_surface'], ATLAS['right_surface'] = load_conte69()
        ATLAS['vertex_labels'], ATLAS['vertex_masked'] = load_atlas()


def plot_brain(data: pd.Series, save_figure=False, **kwargs):
    """plot values on brain regions

    Args:
        data (pd.Series): pandas series that have values with rois index
    """
    _init_atlas()
    data = np.array(data)
    data = map_to_labels(data, ATLAS['vertex_labels'], mask=ATLAS['vertex_masked'])
    # todo consider other methods like brainspace hemi plot
    _surf_plot(data, save_figure, **kwargs)


def _surf_plot(data, save_figure=False, **kwargs):
    text = kwargs.get('text', '')
    p = Plot(surf_lh=ATLAS['left_surface'], surf_rh=ATLAS['right_surface'], 
        size=(1600, 300), layout='row',
        label_text=[text])
    p.add_layer(data, cbar=True, cmap=kwargs.get('color_map', None),
                color_range=kwargs.get('color_range', None))
    
    figure = p.build()
    
    if save_figure: figure.savefig(IMAGE_DIR + text)
