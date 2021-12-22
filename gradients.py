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
    
    def __init__(self, number):
        self.number = number

    def __str__(self):
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
def load_data(subj: int, cond: str) -> pd.DataFrame:
    DATA_DIR = '~/Downloads/subjects/'
    #todo restrict cond {'rest', 'RLbaseline', 'RLlearning'}

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
    return data

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
    # should first find their labels, then remove them
    removed_labels = [regions.index(r) for r in REMOVED_REGIONS]
    for r in REMOVED_REGIONS:
        regions.remove(r)
    
    mask_removed = ~np.isin(surf_labels, removed_labels)

    return surf_labels, mask_removed


from brainspace.gradient import GradientMaps

#todo should become a class
def make_gradients(subj: int, ref=None):
    # if ref is None, takes mean as ref
    
    # where should i put these:
    DIM_RED_APPROACH = 'pca'

    #todo BAD SMELL
    data_rs = load_data(subj=subj, cond='rest')
    corr_mat_rest = make_mat(data_rs)
    data_lrn = load_data(subj=subj, cond='RLlearning')
    corr_mat_lrn = make_mat(data_lrn)
    data_baseline = load_data(subj=subj, cond='RLbaseline')
    corr_mat_baseline = make_mat(data_baseline)

    gm_ref = GradientMaps(random_state=0, approach=DIM_RED_APPROACH)
    gm_ref.fit(corr_mat_rest, sparsity=0.9)

    gm_aligned = GradientMaps(random_state=0, alignment="procrustes", approach=DIM_RED_APPROACH)
    gm_aligned.fit([corr_mat_rest, corr_mat_baseline, corr_mat_lrn],
    reference=gm_ref.gradients_, sparsity=0.9)

    return gm_aligned



from brainspace.utils.parcellation import map_to_labels
from brainspace.datasets import load_fsa5
IMAGE_DIR = 'img/'

#todo break func ? to what
def plot_gradients(subj: int, gm, surf_labels, mask_removed):
    # grad = map_to_labels(gm.gradients_[2][:, 0], surf_labels, mask=mask_removed, fill=np.nan)
    # grad2 = map_to_labels(gm.gradients_[2][:, 1], surf_labels, mask=mask_removed, fill=np.nan)
    # grad_aligned = map_to_labels(gm.aligned_[2][:, 0], surf_labels, mask=mask_removed, fill=np.nan)
    # grad2_aligned = map_to_labels(gm.aligned_[2][:, 1], surf_labels, mask=mask_removed, fill=np.nan)

    grad_aligned_rs = map_to_labels(gm.aligned_[0][:, 0], surf_labels, mask=mask_removed, fill=np.nan)
    grad_aligned_baseline = map_to_labels(gm.aligned_[1][:, 0], surf_labels, mask=mask_removed, fill=np.nan)
    grad_aligned_lrn = map_to_labels(gm.aligned_[2][:, 0], surf_labels, mask=mask_removed, fill=np.nan)


    surf_lh, surf_rh = load_fsa5()

    # with brainspace plot_hemi
    
    # with surfplot 
    from surfplot import Plot
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # ? func inside another func
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
        fig.show()
        fig.savefig(IMAGE_DIR + text_bar)


    texts = ['rs grad', 'baseline grad', 'learning grad']
    data = [grad_aligned_rs, grad_aligned_baseline, grad_aligned_lrn]
    # fill in `vir` in size of len(data)
    color_maps = ['viridis_r', 'viridis_r', 'viridis_r']
    z = zip(data, texts, color_maps)

    for data_to_show, text_bar, color_map in z:
        stack_surfplot(data_to_show, 'sub' + str(subj) + ' - ' + text_bar, color_map)


if __name__ == "__main__":
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 42, 43, 44, 45, 46, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30,
    31, 33, 35, 36, 38, 39, 40,]
    # didn't work for these subjects: 41, 19
    for s in subjects:
        data_rs = load_data(subj=s, cond='rest')
        surf_labels, mask_removed = load_atlas(data_rs)
        gm = make_gradients(subj=s)
        plot_gradients(s, gm, surf_labels, mask_removed)
