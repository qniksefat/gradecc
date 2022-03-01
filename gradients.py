import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

#todo refactor
# make some classes

EPICS_FNAME = {'rest': 'rest', 'baseline': 'RLbaseline',
'learning': 'RLlearning', 'early_learning': 'RLlearning', 'late_learning': 'RLlearning', 
}

# func: load by fname
def load_data(subj: int, epic: str) -> pd.DataFrame:
    DATA_DIR = '~/Downloads/subjects/'
    #todo restrict epics to {'rest', 'RLbaseline', 'learning', 'early', 'late'}

    cond = EPICS_FNAME[epic]

    if subj < 10:
        # handle 01, 02, ..., 09
        try:
            # handle run-1 and run-2
            fname = 'sub-0' + str(subj) + '_ses-01_task-' + cond + '_run-1_space-fsLR_den-91k_bold_timeseries.tsv'
        except:
            fname = 'sub-0' + str(subj) + '_ses-01_task-' + cond + '_run-2_space-fsLR_den-91k_bold_timeseries.tsv'
    else:
        try:
            fname = 'sub-' + str(subj) + '_ses-01_task-' + cond + '_run-1_space-fsLR_den-91k_bold_timeseries.tsv'
        except:
            fname = 'sub-' + str(subj) + '_ses-01_task-' + cond + '_run-2_space-fsLR_den-91k_bold_timeseries.tsv'

    fname = DATA_DIR + fname
    #todo ? can be ses-02 or 01
    data = pd.read_csv(fname, delimiter='\t')

    # handle windowing
    # real sizes: 297 219 609
    window_size = 216
    start_period = 3

    if epic == 'rest' or epic == 'baseline' or epic == 'late_learning':
        return data[-1 * window_size:]
    elif epic == 'early_learning':
        return data[start_period: start_period + window_size]
    elif epic == 'learning':
        return data[200: 200 + window_size]
    
    return data[start_window:start_window + window_size]

    #todo for baseline -> just remove the first 3 trs
    #todo for resting -> last trs
    #todo for learnint: early: 4toEND; late: last trs


def make_mat(data: pd.DataFrame) -> np.array:
    correlation_measure = ConnectivityMeasure(kind='correlation')
    corr_mat = correlation_measure.fit_transform([data.values])[0]
    return corr_mat


# ? pass regions
# from nilearn import plotting
# from matplotlib import pyplot as plt


# def plot_conn_mat(conn_mat: np.array, reduce=False):
    # reduce mat size for visualization
#     if reduce:

#     else:
#         fig = plt.figure(figsize=(15,15))
#         ax = plotting.plot_matrix(conn_mat, labels=masked_regions,
#                                         vmax=0.8, vmin=-0.8, reorder=True,
#                                         figure=fig, # figure=(15, 15), 
#                                         )

#         fig.savefig('con-mat.png')
    # ? if reordering



import nibabel as nib

# add dependencies to git repo: a conda is great

# OK, we make a HUGE shitty function and
#todo break func `load_atlas` later
def load_atlas1(timeseriesT):
    # fork read from a file on github: Schaefer et al. 2018
    ATLAS_DIR = '../atlas/Schaefer2018_1000Parcels_7Networks/'

    atlas_lh = nib.freesurfer.read_annot(ATLAS_DIR + 'lh.Schaefer2018_1000Parcels_7Networks_order.annot')
    surf_labels_lh = atlas_lh[0]
    atlas_rh = nib.freesurfer.read_annot(ATLAS_DIR + 'rh.Schaefer2018_1000Parcels_7Networks_order.annot')
    surf_labels_rh = atlas_rh[0]
    
    surf_labels_rh[surf_labels_rh != 0] += 500  # different labels for lh and rh
    surf_labels = np.concatenate([surf_labels_lh, surf_labels_rh])

    # ? func: to get labels (with removed regions)

    labels_lh = [x.decode() for x in atlas_lh[2]]
    labels_rh = [x.decode() for x in atlas_rh[2]]
    labels_rh.remove('Background+FreeSurfer_Defined_Medial_Wall')
    regions = labels_lh + labels_rh

    # should remove the regions not in the timeseriesT
    REMOVED_REGIONS = set(regions) - set(timeseriesT.columns.tolist())
    # should first find their labels, then remove them
    removed_labels = [regions.index(r) for r in REMOVED_REGIONS]
    for r in REMOVED_REGIONS:
        regions.remove(r)
    
    mask_removed = ~np.isin(surf_labels, removed_labels)

    return surf_labels, mask_removed


def load_atlas(filename='Schaefer2018_1000Parcels_7Networks_order.dlabel.nii'):
    vertices = nib.load(filename).get_fdata()
    vertices = vertices[0]
    mask = vertices != 0
    return vertices, mask


from brainspace.gradient import GradientMaps

#todo should become a class
def make_gradients(subj: int, epics=list(EPICS_FNAME.keys()), DIM_RED_APPROACH='pca', gm_ref=None):
    # if ref is None, takes mean as ref

    #todo it's not epic specific
    #todo variance ratio

    data_epics = {epic: load_data(subj=subj, epic=epic) for epic in epics}
    corr_matrices = {epic: make_mat(data_epics[epic]) for epic in epics}

    if gm_ref is None:
        REF_EPIC = 'rest'
        gm_ref = GradientMaps(random_state=0, approach=DIM_RED_APPROACH)
        gm_ref.fit(corr_matrices[REF_EPIC], sparsity=0.9)

    gm_aligned = GradientMaps(random_state=0, alignment="procrustes", approach=DIM_RED_APPROACH)
    # matrices = [corr_mat_rest, corr_mat_baseline, corr_mat_lrn, corr_mat_lrn_early, corr_mat_lrn_late]
    matrices = [corr_matrices[epic] for epic in epics]

    # removing rest and learning
    del matrices[0], matrices[1]

    gm_aligned.fit(matrices, reference=gm_ref.gradients_, sparsity=0.9)

    return gm_aligned


from brainspace.datasets import load_fsa5, load_conte69
from brainspace.utils.parcellation import map_to_labels

IMAGE_DIR = '../grad_results/'

surf_lh, surf_rh = load_conte69()

# with brainspace plot_hemi

import sys
import warnings

# with surfplot 
from surfplot import Plot

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# ? func inside/outside another func
def stack_surfplot(data_to_show, text_bar, color_map, data_range, not_save_fig=False):
    p = Plot(surf_lh=surf_lh, surf_rh=surf_rh,
            size=(1600, 300),
            layout='row',
            label_text=[text_bar],
            )

    #todo fix zero?
    p.add_layer(
        data_to_show, cbar=True, cmap=color_map, color_range=data_range,
    )
    fig = p.build()
    if not_save_fig:
        fig.show()
    else:
        fig.savefig(IMAGE_DIR + text_bar)


if __name__ == "__main__":
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 42, 43, 44, 45, 46, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30,
    31, 33, 35, 36, 38, 39, 40,]
    # excluded subjects: 41, 19, 32, 27, 34, 37
    
    for s in subjects:
        data_rs = load_data(subj=s, epic='rest')
        surf_labels, mask_removed = load_atlas(data_rs)
        gm = make_gradients(subj=s)
        plot_gradients(s, gm, surf_labels, mask_removed)
