import pandas as pd
from brainspace.gradient import GradientMaps
from tqdm import tqdm

from gradecc.load_data.subject import SUBJECTS_INT
from gradecc.compute.conn_mat import ConnectivityMatrix, ConnectivityMatrixMean
from gradecc.load_data import Timeseries

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
    print('Computing gradients for subjects...')
    for subject in tqdm(subjects):
        subject_gradient_model = _make_subject_gradient_model(subject=subject, epoch_list=epoch_list,
                                                              gradient_reference=gradient_reference,
                                                              dim_reduction_approach='pca')
        for epoch in epoch_list:
            for component in range(num_components):
                df_subject = _make_df_for_subject(subject, subject_gradient_model,
                                                  component, epoch, epoch_list)
                df = pd.concat([df, df_subject], axis=0)
    return df


def _make_df_for_subject(subject, subject_gradient_model, component, epoch, epoch_list):
    subject_gradients = subject_gradient_model.aligned_[epoch_list.index(epoch)][:, component]

    ts = Timeseries(epoch=epoch, subject_id=subject)
    conn_mat = ConnectivityMatrix(timeseries=ts)
    conn_mat.load()

    df_part = pd.DataFrame(subject_gradients, columns=['value'])
    df_part['region'] = conn_mat.region_names
    df_part['subject'] = subject
    df_part['epoch'] = epoch
    df_part['measure'] = 'gradient' + str(component + 1)
    return df_part


def _make_subject_gradient_model(subject, epoch_list, gradient_reference: GradientMaps,
                                 dim_reduction_approach='pca'):
    """ if ref is None, takes the first as ref
    """
    gradient_model = GradientMaps(random_state=0, alignment="procrustes",
                                  approach=dim_reduction_approach)

    conn_mat_epochs = []
    for epoch in epoch_list:
        ts = Timeseries(epoch=epoch, subject_id=subject)
        conn_mat = ConnectivityMatrix(timeseries=ts)
        conn_mat.load()
        conn_mat_epochs.append(conn_mat.data)

    gradient_model.fit(conn_mat_epochs, sparsity=SPARSITY,
                       reference=gradient_reference.gradients_)
    return gradient_model


# todo is it enough to reference grad model? or we need normalize conn mats?
def _make_reference_gradient(reference_epoch, dim_reduction_approach='pca'):
    # todo assume: compute gradient on conn mat mean not riemann mean
    global_conn_mat = ConnectivityMatrixMean(epoch=reference_epoch)
    global_conn_mat.load()
    global_conn_mat = global_conn_mat.data

    global_gradient_reference = GradientMaps(random_state=0, approach=dim_reduction_approach)
    global_gradient_reference.fit(global_conn_mat, sparsity=SPARSITY)
    return global_gradient_reference
