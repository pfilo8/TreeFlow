import ngboost
import numpy as np

from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeRegressor


class IndependentNGBoost:
    def __init__(self, params_tree=None, params_ngboost=None):
        self.params_tree = params_tree
        self.params_ngboost = params_ngboost
        self.models = []

    def fit(self, x, y, x_val, y_val, early_stopping_rounds=50):
        for i in range(y.shape[1]):
            base = DecisionTreeRegressor(**self.params_tree)
            self.models.append(ngboost.NGBoost(Base=base, **self.params_ngboost))
            self.models[i].fit(
                X=x,
                Y=y[:, i],
                X_val=x_val,
                Y_val=y_val,
                early_stopping_rounds=early_stopping_rounds
            )
        return self

    def scipy_distribution(self, X_test, cmat_output=False):
        pred_dists = [
            model.pred_dist(X_test, max_iter=model.best_val_loss_itr)
            for model in self.models
        ]
        means = np.concatenate([dist.loc.reshape(-1, 1) for dist in pred_dists], axis=1)
        vars = np.concatenate([dist.var.reshape(-1, 1) for dist in pred_dists], axis=1)
        cmat = [np.diag(vars[i, :]) for i in range(vars.shape[0])]
        if cmat_output:
            out = [means, np.stack(cmat)]
        else:
            out = [
                multivariate_normal(means[i, :], cov=cmat[i])
                for i in range(vars.shape[0])
            ]
        return out
