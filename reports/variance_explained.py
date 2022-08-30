from os import path

import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from gradecc.compute.gradient import _make_reference_gradient, _make_subject_gradient_model
from gradecc.load_data.subject import SUBJECTS_INT

from gradecc.utils.filenames import dir_images


def variance_explained(epoch_list=None, subjects=SUBJECTS_INT,
                       reference_epoch='baseline', cum_sum=True
                       ) -> pd.DataFrame:
    """ the ratio of variance, explained by each principal component. for all epochs.
    """
    if epoch_list is None:
        epoch_list = ['baseline', 'early', 'late']
    num_comp_default = 10
    gradient_reference = _make_reference_gradient(reference_epoch)
    print('Computing variance explained...')
    subjects_lambdas = []
    for subject in tqdm(subjects):
        subject_gradient_model = _make_subject_gradient_model(subject=subject, epoch_list=epoch_list,
                                                              gradient_reference=gradient_reference,
                                                              dim_reduction_approach='pca')
        subjects_lambdas.append(np.stack(subject_gradient_model.lambdas_))
    subjects_lambdas_avg = np.array(subjects_lambdas).mean(axis=0) / num_comp_default
    variance = pd.DataFrame(subjects_lambdas_avg, index=epoch_list,
                            columns=list(np.arange(10)))
    if cum_sum: return _cum_sum(variance)
    else:   return variance


if __name__ == '__main__':
    variance = variance_explained(subjects=[1,2,3], epoch_list=None, cum_sum=False, reference_epoch='baseline')

    fig = sns.lineplot(y='value', x='variable', hue='index',
                       style='index', markers=True,
                       data=pd.melt(variance.reset_index(), ['index']))

    fig = fig.get_figure()

    fig.savefig(path.join(dir_images, 'var 123 cnt.png'))


def _cum_sum(variance):
    variance_cum_sum = pd.DataFrame()
    for epoch_idx in variance.index:
        variance_cum_sum[epoch_idx] = variance.loc[epoch_idx].to_numpy().cumsum()
    return variance_cum_sum.T
