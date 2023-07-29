from typing import Union

import joblib
import numpy as np
import torch.nn as nn

from sklearn.base import BaseEstimator


class TreeFlowBoost(BaseEstimator):

    def __init__(self, tree_model, flow_model, embedding_size: int = 20):
        self.tree_model = tree_model
        self.flow_model = flow_model

        self.embedding_size = embedding_size

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

        self.flow_model.setup_context_encoder(nn.Sequential(
            nn.Linear(context.shape[1], self.embedding_size),
            nn.Tanh(),
        ))

        self.flow_model.fit(y, context, params, y_val, context_val, params_val, n_epochs=n_epochs,
                            batch_size=batch_size, verbose=verbose)
        return self

    def sample(self, X: np.ndarray, num_samples: int = 10, batch_size: int = 1000) -> np.ndarray:
        context: np.ndarray = self.tree_model.embed(X)
        params: np.ndarray = self.tree_model.pred_dist_param(X)
        samples: np.ndarray = self.flow_model.sample(num_samples=num_samples, context=context, params=params,
                                                     batch_size=batch_size)
        return samples

    def predict(self, X: np.ndarray, num_samples: int = 10, batch_size: int = 1000) -> np.ndarray:
        samples: np.ndarray = self.sample(X=X, num_samples=num_samples, batch_size=batch_size)
        y_hat: np.ndarray = samples.mean(axis=1)
        return y_hat

    def predict_tree(self, X: np.ndarray) -> np.ndarray:
        return self.tree_model.predict(X)

    def embed(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        context: np.ndarray = self.tree_model.embed(X)
        context_e: np.ndarray = self.flow_model.embed(context, batch_size=batch_size)
        return context_e

    def log_prob(self, X: np.ndarray, y: np.ndarray, batch_size: int = 1000):
        context: np.ndarray = self.tree_model.embed(X)
        params: np.ndarray = self.tree_model.pred_dist_param(X)
        logpx: np.ndarray = self.flow_model.log_prob(X=y, context=context, params=params, batch_size=batch_size)
        return logpx

    def save(self, filename):
        joblib.dump(self, f"{filename}-tfboost")
        joblib.dump(self.tree_model._leafs_encoder, f"{filename}-leafs_encoder")
        joblib.dump(self.tree_model._y_dims, f"{filename}-y-dims")

    @classmethod
    def load(cls, filename):
        model = joblib.load(f"{filename}-tfboost")
        model.tree_model._leafs_encoder = joblib.load(f"{filename}-leafs_encoder")
        model.tree_model._y_dims = joblib.load(f"{filename}-y-dims")
        return model
