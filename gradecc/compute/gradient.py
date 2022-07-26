import pandas as pd
from brainspace.gradient import GradientMaps
from tqdm import tqdm

from gradecc.load_data.subject import SUBJECTS_INT
from gradecc.compute.conn_mat import get_conn_mat

NUM_COMPONENTS = 4
SPARSITY = 0.9
# can be a class variable, also not a semantically global var. what is it?


def make_gradients(epoch_list=None, subjects=SUBJECTS_INT,
                   num_components=NUM_COMPONENTS,
                   reference_epoch='baseline',  # todo change to `rest`
                   ) -> pd.DataFrame:
    if epoch_list is None:
        epoch_list = ['baseline', 'early', 'late']
    gradient_reference = _make_reference_gradient(reference_epoch)
    df = pd.DataFrame()
    print('Making gradients for subjects...')
    for subject in tqdm(subjects):
        subject_gradient_model = _make_subject_gradients(subject=subject, epoch_list=epoch_list,
                                                         gradient_reference=gradient_reference,
                                                         dim_reduction_approach='pca')
        for epoch in epoch_list:
            for component in range(num_components):
                df_part = _make_subject_values(subject, subject_gradient_model,
                                               component, epoch, epoch_list)
                df = pd.concat([df, df_part], axis=0)
    return df


def _make_subject_values(subject, subject_gradient_model, component, epoch, epoch_list):
    subject_gradients = _get_epoch_component(subject_gradient_model,
                                             component, epoch, epoch_list)
    regions_epoch_subject = get_conn_mat(epoch, subject)[1]
    df_part = _fill_df(subject_gradients, regions_epoch_subject,
                       subject, epoch, component)
    return df_part


def _make_subject_gradients(subject, epoch_list, gradient_reference: GradientMaps,
                            dim_reduction_approach='pca'):
    """ if ref is None, takes the first as ref
    """
    gradient_model = GradientMaps(random_state=0, alignment="procrustes",
                                  approach=dim_reduction_approach)
    conn_mat_epochs = [get_conn_mat(epoch, subject)[0] for epoch in epoch_list]
    gradient_model.fit(conn_mat_epochs, sparsity=SPARSITY,
                       reference=gradient_reference.gradients_)
    return gradient_model


def _get_epoch_component(gradient_model: GradientMaps, component, epoch, epoch_list):
    return gradient_model.aligned_[epoch_list.index(epoch)][:, component]


# todo is it enough to reference grad model? or we need normalize conn mats?
def _make_reference_gradient(reference_epoch, dim_reduction_approach='pca'):
    global_conn_mat, _ = get_conn_mat(epoch=reference_epoch)
    global_gradient_reference = GradientMaps(random_state=0, approach=dim_reduction_approach)
    global_gradient_reference.fit(global_conn_mat, sparsity=SPARSITY)
    return global_gradient_reference


def _fill_df(values, regions, subject, epoch, component):
    df = pd.DataFrame(values, columns=['value'])
    df['region'] = regions
    df['subject'] = subject
    df['epoch'] = epoch
    df['measure'] = 'gradient' + str(component + 1)
    return df
