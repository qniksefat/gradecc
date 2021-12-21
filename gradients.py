# subject
# conditions:
    # fnames
# plot conn:
    # respct to rs
# plot brain:
    # respct to rs
    # with surfplot
# grads


class Subject:
    masked_regions = ['7Networks_RH_Cont_Cing_1', '7Networks_RH_Vis_33']
    
    def __init__(self, number) -> None:
        self.number = number

    def __str__(self) -> str:
        pass

    def load(self):
        self.fname = 'sub-' + str(self.number) + '_ses-01_task-RLbaseline_run-1_space-fsLR_den-91k_bold_timeseries.tsv'

 
###


import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

#todo refactor
# make some classes

# func: load by fname

def make_mat(subj: int, cond: str) -> np.array:
    DATA_DIR = '../sub43data/'
    #todo restrict cond {'rest', 'RLbaseline', 'RLlearning'}
    fname = 'sub-' + str(subj) + '_ses-01_task-' + cond + '_run-1_space-fsLR_den-91k_bold_timeseries.tsv'
    fname = DATA_DIR + fname
    #todo ? can be ses-02 or 01
    data = pd.read_csv(fname, delimiter='\t')
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
def load_atlas(timeseriesT):
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
    for r in REMOVED_REGIONS:
        regions.remove(r)
    
    removed_labels = [regions.index(r) for r in REMOVED_REGIONS]
    mask_removed = ~np.isin(surf_labels, removed_labels)

    return surf_labels, mask_removed


from brainspace.gradient import GradientMaps

#todo should become a class
def make_gradients(subj: int, ref=None):
    # if ref is None, takes mean as ref
    
    # where should i put these:
    DIM_RED_APPROACH = 'pca'

    corr_mat_rest = make_mat(subj=subj, cond='rest')
    corr_mat_lrn = make_mat(subj=subj, cond='RLlearning')
    corr_mat_baseline = make_mat(subj=subj, cond='RLbaseline')

    gm_ref = GradientMaps(random_state=0, approach=DIM_RED_APPROACH)
    gm_ref.fit(corr_mat_rest, sparsity=0.9)

    gm_aligned = GradientMaps(random_state=0, alignment="procrustes", approach=DIM_RED_APPROACH)
    gm_aligned.fit([corr_mat_rest, corr_mat_baseline, corr_mat_lrn],
    reference=gm_ref.gradients_, sparsity=0.9)

    return gm_aligned



from brainspace.utils.parcellation import map_to_labels
from brainspace.datasets import load_fsa5


#todo break func ? to what
def plot_gradients(gm, surf_labels, mask_removed):
    grad = map_to_labels(gm.gradients_[2][:, 0], surf_labels, mask=mask_removed, fill=np.nan)
    grad2 = map_to_labels(gm.gradients_[2][:, 1], surf_labels, mask=mask_removed, fill=np.nan)
    grad_aligned = map_to_labels(gm.aligned_[2][:, 0], surf_labels, mask=mask_removed, fill=np.nan)
    grad2_aligned = map_to_labels(gm.aligned_[2][:, 1], surf_labels, mask=mask_removed, fill=np.nan)

    surf_lh, surf_rh = load_fsa5()

    # with brainspace plot_hemi
    
    # ? func inside another func
    # with surfplot 
    from surfplot import Plot
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    def stack_surfplot(data_to_show, text_bar, color_map):
        p = Plot(surf_lh=surf_lh, surf_rh=surf_rh,
                size=(1600, 300),
                layout='row',
                label_text=[text_bar]
                )

        p.add_layer(
            data_to_show, cbar=True, cmap=color_map,  
        )

        fig = p.build()
        fig = p.render(offscreen=True)
        fig.show()


    texts = ['Schaefer\n1000', '1st grad', '1st grad aligned', '2nd grad', '2nd grad aligned']
    data = [surf_labels, grad, grad_aligned, grad2, grad2_aligned]
    color_maps = ['tab20', 'viridis_r', 'viridis_r', 'viridis_r', 'viridis_r']
    z = zip(data, texts, color_maps)

    for data_to_show, text_bar, color_map in z:
    stack_surfplot(data_to_show, text_bar, color_map)
