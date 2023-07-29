import numpy as np
import pandas as pd

from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal

from sklearn.tree import DecisionTreeRegressor

from ..utils import generate_params_for_grid_search, setup_random_seed, split_data
from ...utils import log_dataframe_artifact
from ...reporting.nodes import calculate_nll_ngboost
from ....independent_multivariate_boosting import IndependentCatBoost, IndependentNGBoost


def train_multivariate(x_train, y_train, x_val, y_val, ngboost_p, ngboost_params, tree_p, tree_params, independent=False,
                       independent_model_type: str = "NGBoost", random_seed: int = 42):
    """
    Train a NGBoost model.
    :param x_train: Training data.
    :param y_train: Training labels.
    :param ngboost_p: NGBoost/CatBoost parameters from grid search.
    :param ngboost_params: NGBoost/CatBoost parameters.
    :param tree_p: Tree parameters for NGBoost from grid search.
    :param tree_params: Tree parameters for NGBoost.
    :param independent: Flag whether fit independent model (NGBoost/CatBoost) or Multivariate Gaussian with NGBoost.
    :param independent_model_type: Selected independent model.
    :param random_seed: Random seed.
    :return:
    """
    if independent:
        if independent_model_type == "CatBoost":
            model = IndependentCatBoost(
                params_tree={**tree_p, **tree_params, "random_state": random_seed},
                params_ngboost={**ngboost_p, **ngboost_params}
            )
        elif independent_model_type == "NGBoost":
            model = IndependentNGBoost(
                params_tree={**tree_p, **tree_params, "random_state": random_seed},
                params_ngboost={**ngboost_p, **ngboost_params}
            )
        else:
            raise ValueError(
                f"Independent model type: {independent_model_type} is not supported. "
                f"Select between CatBoost and NGBoost."
            )
    else:
        base_model = DecisionTreeRegressor(
            **tree_p,
            **tree_params,
            random_state=random_seed
        )
        model = NGBRegressor(
            Dist=MultivariateNormal(y_train.shape[1]),
            Base=base_model,
            **ngboost_p,
            **ngboost_params
        )
    model.fit(x_train.values, y_train.values, x_val.values, y_val.values, early_stopping_rounds=50)
    return model


def modeling_multivariate(x_train: pd.DataFrame, y_train: pd.DataFrame, ngboost_params, tree_params,
                          ngboost_hyperparams, tree_hyperparams, independent: bool = False,
                          independent_model_type: str = "NGBoost", split_size=0.8, random_seed: int = 42):
    setup_random_seed(random_seed)

    x_tr, x_val, y_tr, y_val = split_data(x_train=x_train, y_train=y_train, split_size=split_size)

    results = []

    for ngboost_p in generate_params_for_grid_search(ngboost_hyperparams):
        for tree_p in generate_params_for_grid_search(tree_hyperparams):
            try:
                m = train_multivariate(x_tr, y_tr, x_val, y_val, ngboost_p, ngboost_params, tree_p, tree_params,
                                       independent, independent_model_type, random_seed)

                result_train = calculate_nll_ngboost(m, x_tr, y_tr, independent=independent)
                result_val = calculate_nll_ngboost(m, x_val, y_val, independent=independent)

                results.append([ngboost_p, tree_p, result_train, result_val])
            except:
                results.append([ngboost_p, tree_p, np.nan, np.nan])

    results = pd.DataFrame(results, columns=['ngboost_p', 'tree_p', 'log_prob_train', 'log_prob_val'])
    results = results.sort_values('log_prob_val', ascending=True)
    log_dataframe_artifact(results, 'grid_search_results')

    best_params = results.iloc[0].to_dict()
    best_ngboost_p = best_params['ngboost_p']
    best_tree_p = best_params['tree_p']

    m = train_multivariate(x_train, y_train, x_val, y_val, best_ngboost_p, ngboost_params, best_tree_p, tree_params,
                           independent, independent_model_type, random_seed)
    return m
