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

# def load by fname

def make_mat(subj: int, cond: str) -> np.array:
    DATA_DIR = '../sub43data/'
    #todo cond {'rest', 'RLbaseline', 'RLlearning'}
    fname = 'sub-' + str(subj) + '_ses-01_task-' + cond + '_run-1_space-fsLR_den-91k_bold_timeseries.tsv'
    fname = DATA_DIR + fname
    #todo can be ses-02 or 01
    data = pd.read_csv(fname, delimiter='\t')
    correlation_measure = ConnectivityMeasure(kind='correlation')
    corr_mat = correlation_measure.fit_transform([data.values])[0]
    return corr_mat




#todo maybe break it into files

from nilearn.datasets import fetch_atlas_schaefer_2018

# it's just for con-mat plot
def load_atlas():
    #todo is it the best way to define it?
    MASKED_REGIONS = [b'7Networks_RH_Cont_Cing_1', b'7Networks_RH_Vis_33']

    atlas = fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=2)
    labels = [x.decode() for x in atlas['labels']]
    regions = atlas['labels'].copy().tolist()
    masked_labels = [regions.index(r) for r in MASKED_REGIONS]
    for r in MASKED_REGIONS:
        regions.remove(r)
    
    regions_list = ['%s_%s' % (h, r.decode()) for h in ['L', 'R'] for r in regions]


import nibabel as nib

# add dependencies to git repo: a conda is great

def load_atlas_labels():
    # fork read from a file on github
    ATLAS_DIR = '../atlas/Schaefer2018_1000Parcels_7Networks/'
    surf_labels_lh = nib.freesurfer.read_annot(ATLAS_DIR + 'lh.Schaefer2018_1000Parcels_7Networks_order.annot')[0]
    surf_labels_rh = nib.freesurfer.read_annot(ATLAS_DIR + 'rh.Schaefer2018_1000Parcels_7Networks_order.annot')[0]
    
    surf_labels_rh[surf_labels_rh != 0] += 500  # different labels for lh and rh
    surf_labels = np.concatenate([surf_labels_lh, surf_labels_rh])


