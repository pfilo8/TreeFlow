from typing import Union

import numpy as np
import torch.nn as nn

from src.probabilistic_flow_boosting.tfboost.tfboost import TreeFlowBoost


class TreeFlowWithoutShallow(TreeFlowBoost):

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Union[np.ndarray, None] = None,
            y_val: Union[np.ndarray, None] = None, n_epochs: int = 100, batch_size: int = 1000, verbose: bool = False):
        self.tree_model.fit(X, y)

        context: np.ndarray = self.tree_model.embed(X)
        params: np.ndarray = self.tree_model.pred_dist_param(X)
        y: np.ndarray = y if len(y.shape) == 2 else y.reshape(-1, 1)

        if X_val is not None and y_val is not None:
            context_val: np.ndarray = self.tree_model.embed(X_val)
            params_val: np.ndarray = self.tree_model.pred_dist_param(X_val)
            y_val: np.ndarray = y_val if len(y_val.shape) == 2 else y_val.reshape(-1, 1)
        else:
            context_val = None
            params_val = None
            y_val = None

        self.flow_model.setup_context_encoder(nn.Identity())

        self.flow_model.fit(y, context, params, y_val, context_val, params_val, n_epochs=n_epochs,
                            batch_size=batch_size, verbose=verbose)
        return self
