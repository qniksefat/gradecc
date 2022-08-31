from functools import cache

import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps

from gradecc.compute.conn_mat import ConnectivityMatrix, riemann_centered_conn_mat
from gradecc.compute.conn_mat.center import mean_riemann_conn_mats
from gradecc.compute.conn_mat.conn_mat import stack_conn_mats
from gradecc.load_data import Subject, Timeseries, EPOCHS
from gradecc.load_data import EPOCH_REF
from gradecc.load_data.subject import SUBJECTS


class Gradients(GradientMaps):
    def __init__(self, subject: Subject, epoch_list: list[str] = EPOCHS, epoch_ref=EPOCH_REF,
                 n_components=4, approach='pca', alignment='procrustes', random_state=0, sparsity=0.9, **kwargs):

        super(Gradients, self).__init__(n_components, approach, None, alignment, random_state)

        self.subject = subject
        self.epoch_list = epoch_list
        self.sparsity = sparsity
        cnt = kwargs.get('centered', False)
        self.conn_mats = [ConnectivityMatrix(Timeseries(self.subject, e), centered=cnt) for e in self.epoch_list]
        self.region_names = self.conn_mats[0].region_names
        self.grads_ref = None if epoch_ref is None else ref_grad_model(epoch_ref, centered=cnt).gradients_

        self.fit_()
        self.df = self.make_df()

    def fit_(self, gamma=None, n_iter=10):
        x = self.conn_mats
        return super(Gradients, self).fit(x, gamma, self.sparsity, n_iter, reference=self.grads_ref)

    def make_df(self):
        df_ = pd.DataFrame()
        for epoch in self.epoch_list:
            for component in range(self.n_components):
                values = self.aligned_[self.epoch_list.index(epoch)][:, component]
                df = pd.DataFrame({'subject': self.subject.int, 'epoch': epoch, 'region': self.region_names,
                                   'measure': 'gradient' + str(component + 1),  'value': values})
                df_ = pd.concat([df_, df], axis=0)
        return df_


# @cache
def ref_grad_model(epoch_ref=EPOCH_REF, subjects=SUBJECTS, n_components=4, approach='pca',
                   random_state=0, sparsity=0.9, **kwargs):
    # we assume to compute gradient on euclidean mean not riemann mean
    # todo coupled to conn mat module
    if kwargs.get('centered', False):   cmat_mean_ref = mean_riemann_conn_mats(np.stack(
        [riemann_centered_conn_mat(Timeseries(s, epoch_ref)) for s in subjects]))
    else:   cmat_mean_ref = stack_conn_mats(epochs=epoch_ref, subjects=subjects).mean(axis=0)

    grad_model_ref = GradientMaps(n_components=n_components, random_state=random_state, approach=approach)
    grad_model_ref.fit(cmat_mean_ref, sparsity)
    return grad_model_ref
