import pandas as pd

from ..utils import generate_params_for_grid_search, setup_random_seed, split_data
from ...utils import log_dataframe_artifact
from ...reporting.nodes import calculate_nll

from ....tfboost.flow import ContinuousNormalizingFlow
from ....tfboost.tree import MODELS
from ....tfboost.tfboost import TreeFlowBoost


def train_treeflow(x_train, y_train, x_val, y_val, flow_p, flow_params, tree_p, tree_params, tree_model_type,
                   n_epochs: int = 100, batch_size: int = 1000, random_seed: int = 42):
    """
    Train a TreeFlow model.
    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_val: Validation data.
    :param y_val: Validation labels.
    :param flow_p: Flow parameters from grid search.
    :param flow_params: Flow parameters.
    :param tree_p: Tree parameters from grid search.
    :param tree_params: Tree parameters.
    :param tree_model_type: Type of the Tree model (see tfboost.tree package).
    :param n_epochs: Number of epochs.
    :param batch_size: Batch size for Flow model.
    :param random_seed: Random seed.
    :return:
    """
    flow = ContinuousNormalizingFlow(conditional=True, **flow_p, **flow_params)
    tree = MODELS[tree_model_type](**tree_params, **tree_p, random_seed=random_seed)

    if x_val is not None and y_val is not None:
        x_val, y_val = x_val.values, y_val.values

    m = TreeFlowBoost(flow_model=flow, tree_model=tree, embedding_size=flow_p['context_dim'])
    m = m.fit(x_train.values, y_train.values, x_val, y_val, n_epochs=n_epochs, batch_size=batch_size)
    return m, m.flow_model.epoch_best


def modeling_treeflow(x_train: pd.DataFrame, y_train: pd.DataFrame, tree_model_type, flow_params, tree_params,
                      flow_hyperparams, tree_hyperparams, split_size=0.8, n_epochs: int = 100, batch_size: int = 1000,
                      random_seed: int = 42):
    setup_random_seed(random_seed)

    x_tr, x_val, y_tr, y_val = split_data(x_train=x_train, y_train=y_train, split_size=split_size)

    flow_hyperparams['hidden_dims'] = map(tuple, flow_hyperparams['hidden_dims'])
    results = []

    for flow_p in generate_params_for_grid_search(flow_hyperparams):
        for tree_p in generate_params_for_grid_search(tree_hyperparams):
            m, best_epoch = train_treeflow(x_tr, y_tr, x_val, y_val, flow_p, flow_params, tree_p, tree_params,
                                           tree_model_type, n_epochs, batch_size, random_seed)

            result_train = calculate_nll(m, x_tr, y_tr, batch_size=batch_size)
            result_val = calculate_nll(m, x_val, y_val, batch_size=batch_size)

            print(flow_p, tree_p, result_train, result_val, best_epoch)
            results.append([flow_p, tree_p, result_train, result_val, best_epoch])

    results = pd.DataFrame(results, columns=['flow_p', 'tree_p', 'log_prob_train', 'log_prob_val', 'best_epoch'])
    results = results.sort_values('log_prob_val', ascending=True)
    log_dataframe_artifact(results, 'grid_search_results')

    best_params = results.iloc[0].to_dict()
    best_flow_p = best_params['flow_p']
    best_tree_p = best_params['tree_p']

    m, _ = train_treeflow(x_tr, y_tr, x_val, y_val, best_flow_p, flow_params, best_tree_p, tree_params,
                          tree_model_type, n_epochs, batch_size, random_seed)
    return m
