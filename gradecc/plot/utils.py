import nibabel as nib
import pandas as pd
from brainspace.datasets import load_conte69

from gradecc.load_data import all_region_names
from gradecc.utils.filenames import atlas_filename, labels_filename

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

        ATLAS['CORTEX_LABELS'] = pd.read_csv(labels_filename)
        # todo inflated brain maps
        """
        from neuromaps.datasets import fetch_fsaverage
        surfaces = fetch_fsaverage(density='164k')
        ATLAS['left_surface'], ATLAS['right_surface'] = surfaces['inflated']
        from nilearn.datasets import fetch_surf_fsaverage
        ATLAS['left_surface'] = fetch_surf_fsaverage(mesh='fsaverage')['curv_left']
        ATLAS['right_surface'] = fetch_surf_fsaverage(mesh='fsaverage')['curv_right']
        """


def spot_region(region):
    df = pd.DataFrame(columns=all_region_names(), index=['value'])
    df.loc['value', :] = 0
    df.loc['value', region] = 1
    return pd.melt(df, value_vars=list(df.columns), var_name='region')
