import pandas as pd
import numpy as np
from brainspace.gradient import GradientMaps
from tqdm import tqdm

from gradecc.load_timeseries.utils import SUBJECTS
from gradecc.connectivity_matrix import get_conn_mat

NUM_COMPONENTS = 4
SPARSITY = 0.9
# can be a class variable, also not a semantically global var. what is it?


def make_gradients(epic_list=None, subjects=SUBJECTS,
                   num_components=NUM_COMPONENTS, reference_epic='baseline',
                   ) -> pd.DataFrame:
    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    gradient_reference = _make_reference_gradient(reference_epic)
    df = pd.DataFrame()
    print('Making gradients for subjects...')
    for subject in tqdm(subjects):
        subject_gradient_model = _make_subject_gradients(subject=subject, epic_list=epic_list,
                                                         gradient_reference=gradient_reference,
                                                         dim_reduction_approach='pca')
        for epic in epic_list:
            for component in range(num_components):
                df_part = _make_subject_values(subject, subject_gradient_model,
                                               component, epic, epic_list)
                df = pd.concat([df, df_part], axis=0)
    return df


def _make_subject_values(subject, subject_gradient_model, component, epic, epic_list):
    subject_gradients = _get_epic_component(subject_gradient_model,
                                            component, epic, epic_list)
    regions_epic_subject = get_conn_mat(epic, subject)[1]
    df_part = _fill_df(subject_gradients, regions_epic_subject,
                       subject, epic, component)
    return df_part


def _make_subject_gradients(subject, epic_list, gradient_reference: GradientMaps,
                            dim_reduction_approach='pca'):
    """ if ref is None, takes the first as ref
    """
    gradient_model = GradientMaps(random_state=0, alignment="procrustes",
                                  approach=dim_reduction_approach)
    conn_mat_epics = [get_conn_mat(epic, subject)[0] for epic in epic_list]
    gradient_model.fit(conn_mat_epics, sparsity=SPARSITY,
                       reference=gradient_reference.gradients_)
    return gradient_model


def _get_epic_component(gradient_model: GradientMaps, component, epic, epic_list):
    return gradient_model.aligned_[epic_list.index(epic)][:, component]


# todo is it enough to reference grad model? or we need normalize conn mats?
def _make_reference_gradient(reference_epic, dim_reduction_approach='pca'):
    global_conn_mat, _ = get_conn_mat(epic=reference_epic)
    global_gradient_reference = GradientMaps(random_state=0, approach=dim_reduction_approach)
    global_gradient_reference.fit(global_conn_mat, sparsity=SPARSITY)
    return global_gradient_reference


def _fill_df(values, regions, subject, epic, component):
    df = pd.DataFrame(values, columns=['value'])
    df['region'] = regions
    df['subject'] = subject
    df['epic'] = epic
    df['measure'] = 'gradient' + str(component + 1)
    return df


# todo separate


def variance_explained(epic_list=None, subjects=SUBJECTS,
                       reference_epic='baseline', cum_sum=True
                       ) -> pd.DataFrame:
    """ the ratio of variance, explained by each principal component. for all epics.
    """
    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    num_comp_default = 10
    gradient_reference = _make_reference_gradient(reference_epic)
    print('Computing variance explained...')
    subjects_lambdas = []
    for subject in tqdm(subjects):
        subject_gradient_model = _make_subject_gradients(subject=subject, epic_list=epic_list,
                                                         gradient_reference=gradient_reference,
                                                         dim_reduction_approach='pca')
        subjects_lambdas.append(np.stack(subject_gradient_model.lambdas_))
    subjects_lambdas_avg = np.array(subjects_lambdas).mean(axis=0) / num_comp_default
    # todo pass without averaging, plot with SD
    variance = pd.DataFrame(subjects_lambdas_avg, index=epic_list,
                            columns=list(np.arange(10)))
    if cum_sum:
        return _cum_sum(variance)
    else:
        return variance


def _cum_sum(variance):
    variance_cum_sum = pd.DataFrame()
    for epic_idx in variance.index:
        variance_cum_sum[epic_idx] = variance.loc[epic_idx].to_numpy().cumsum()
    return variance_cum_sum.T


if __name__ == '__main__':
    pass
