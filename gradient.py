import pandas as pd
from brainspace.gradient import GradientMaps
from tqdm import tqdm

from load_timeseries import EPICS_FILENAME, SUBJECTS
from connectivity_matrix import get_conn_mat


def make_gradients(epic_list=None, subjects=SUBJECTS,
                   num_components=4, reference_epic='baseline',
                   ) -> pd.DataFrame:
    """_summary_
        let's make n gradients for all subjects; over all epics
        needs 2 lists: baseline/early/late and all 

    Args:
        epic_list (list, optional): _description_. Defaults to ['baseline', 'early', 'late'].
        num_components (int, optional): _description_. Defaults to 4.
        reference_epic (str, optional): _description_. Defaults to 'baseline'.
        subjects (_type_, optional): _description_. Defaults to all SUBJECTS listed.

    Returns:
        pd.DataFrame: _description_
    """
    if epic_list is None:
        epic_list = ['baseline', 'early', 'late']
    gradient_reference = _make_reference_gradient(reference_epic)
    # what if it's a gradient object
    df = pd.DataFrame()
    print('Making gradients for subjects...')
    for subject in tqdm(subjects):
        subject_gradient_model = _make_subject_gradients(subject=subject, epic_list=epic_list,
                                                         gradient_reference=gradient_reference,
                                                         dim_reduction_approach='pca')
        for epic in epic_list:
            for component in range(num_components):
                subject_gradients = _get_epic_component(subject_gradient_model,
                                                        component, epic, epic_list)
                df_part = _make_df_part(subject_gradients, subject, epic, component)
                df = pd.concat([df, df_part], axis=0)
    return df


def _make_subject_gradients(subject, epic_list, gradient_reference: GradientMaps,
                            dim_reduction_approach='pca'):
    """ if ref is None, takes the first as ref

    Args:
        subject:
        epic_list:
        gradient_reference:
        dim_reduction_approach:

    Returns:

    """
    gradient_model = GradientMaps(random_state=0, alignment="procrustes",
                                  approach=dim_reduction_approach)
    conn_mat_epics = [get_conn_mat(epic, subject)[0] for epic in epic_list]
    gradient_model.fit(conn_mat_epics, sparsity=0.9,
                       reference=gradient_reference.gradients_)
    return gradient_model


def _get_epic_component(gradient_model: GradientMaps, component, epic, epic_list):
    return gradient_model.aligned_[epic_list.index(epic)][:, component]


# todo is it enough to reference grad model? or we need normalize conn mats?
def _make_reference_gradient(reference_epic, dim_reduction_approach='pca'):
    global_conn_mat, _ = get_conn_mat(epic=reference_epic)
    global_gradient_reference = GradientMaps(random_state=0, approach=dim_reduction_approach)
    global_gradient_reference.fit(global_conn_mat, sparsity=0.9)
    # todo sparsity?
    return global_gradient_reference


def _make_df_part(values, subject, epic, component):
    df = pd.DataFrame(values, columns=['value'])
    df.index = df.index.set_names(['region'])
    df = df.reset_index()
    df['subject'] = subject
    df['epic'] = epic
    df['measure'] = 'gradient' + str(component + 1)
    return df
