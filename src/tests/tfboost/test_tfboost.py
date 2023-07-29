import itertools

import numpy as np
import pytest

from probabilistic_flow_boosting.tfboost.flow import ContinuousNormalizingFlow
from probabilistic_flow_boosting.tfboost.tree.engboost import EmbeddableNGBoost, EmbeddableNGBoost2, \
    EmbeddableNGBoostDecisionPath
from probabilistic_flow_boosting.tfboost.tree.ecatboost import EmbeddableCatBoost
from probabilistic_flow_boosting.tfboost.tfboost import TreeFlowBoost

models = [
    EmbeddableNGBoost,
    EmbeddableNGBoost2,
    EmbeddableNGBoostDecisionPath,
    EmbeddableCatBoost
]
outputs = [
    np.random.randn(100, 1),
]

test_cases = itertools.product(models, outputs)


@pytest.mark.parametrize("model,y", test_cases)
def test_fit_sample_1d(model, y):
    x = np.random.randn(y.shape[0], 10)

    flow = ContinuousNormalizingFlow(
        input_dim=y.shape[1],
        hidden_dims=(20,),
        context_dim=10,
        conditional=True
    )

    tree = model()
    tfb = TreeFlowBoost(flow_model=flow, tree_model=tree, embedding_size=10)
    tfb.fit(x, y, n_epochs=10)

    samples = tfb.sample(x, num_samples=10)
    assert samples.shape == (100, 10, y.shape[1])


def test_fit_sample_2d():
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 2)

    flow = ContinuousNormalizingFlow(
        input_dim=y.shape[1],
        hidden_dims=(20,),
        context_dim=10,
        conditional=True
    )

    tree = EmbeddableCatBoost(loss_function="MultiRMSE")
    tfb = TreeFlowBoost(flow_model=flow, tree_model=tree, embedding_size=10)
    tfb.fit(x, y, n_epochs=10)

    samples = tfb.sample(x, num_samples=10)
    assert samples.shape == (100, 10, y.shape[1])
