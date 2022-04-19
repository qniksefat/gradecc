import pandas as pd
from brainspace.gradient import GradientMaps

from load_timeseries import EPICS_FNAME, SUBJECTS
from connectivity_matrix import get_conn_mat


def make_gradients(epic_list=['baseline', 'early', 'late'],
                   num_components=4, reference_epic='baseline',
                   subjects=SUBJECTS) -> pd.DataFrame:
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
    gradient_reference = _make_reference_gradient(reference_epic)
    # what if it's a gradient object
    
    df = pd.DataFrame()
    for subject in subjects:
        subject_gradient_model = _make_gradients(subject=subject, epic_list=epic_list, 
                                          gradient_reference=gradient_reference,
                                          dim_reduction_approach='pca')
        for epic in epic_list:
            for component in range(num_components):
                subject_gradients = _get_subject_gradients(subject_gradient_model,
                                                                component, epic, epic_list)
                df = pd.concat([df,
                                _make_df_part(subject_gradients,
                                              subject, epic, component)
                                ], axis=0)
    
    return df


def _make_gradients(subject, epic_list, gradient_reference: GradientMaps,
                    dim_reduction_approach='pca'):
    """_summary_

    Args:
        subject (_type_): _description_
        epic_list (_type_): _description_
        gradient_reference (GradientMaps): _description_
        dim_reduction_approach (str, optional): _description_. Defaults to 'pca'.

    Returns:
        _type_: _description_
    """
    # if ref is None, takes the first as ref
    
    gradient_model = GradientMaps(random_state=0, alignment="procrustes",
                                 approach=dim_reduction_approach)
    conn_mat_epics = [get_conn_mat(epic)[0] for epic in epic_list]
    gradient_model.fit(conn_mat_epics, sparsity=0.9,
                       reference=gradient_reference.gradients_)
    return gradient_model


def _get_subject_gradients(gradient_model: GradientMaps, component, epic, epic_list):
    return gradient_model.aligned_[epic_list.index(epic)][:, component]


def _make_reference_gradient(reference_epic, dim_reduction_approach='pca'):
    """_summary_

    Args:
        reference_epic (_type_): _description_
        dim_reduction_approach (str, optional): _description_. Defaults to 'pca'.

    Returns:
        _type_: _description_
    """
    global_conn_mat, _ = get_conn_mat(epic=reference_epic)
    global_gradient_reference = GradientMaps(random_state=0, approach=dim_reduction_approach)
    global_gradient_reference.fit(global_conn_mat, sparsity=0.9)
    # todo sparsity?
    return global_gradient_reference


def _make_df_part(values, subject, epic, component):
    dff = pd.DataFrame(values, columns=['value'])
    dff.index = dff.index.set_names(['region'])
    dff = dff.reset_index()
    dff['subject'] = subject
    dff['epic'] = epic
    dff['measure'] = 'gradient'+ str(component + 1)
    return dff
