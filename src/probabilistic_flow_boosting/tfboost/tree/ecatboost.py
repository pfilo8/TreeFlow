import sys

import catboost
import numpy as np

from sklearn.preprocessing import OneHotEncoder


class EmbeddableCatBoost(catboost.CatBoostRegressor):

    def fit(self, X, y=None, cat_features=None, sample_weight=None, baseline=None, use_best_model=None,
            eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None,
            verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
            save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None,
            log_cout=sys.stdout, log_cerr=sys.stderr):
        super().fit(
            X,
            y=y,
            cat_features=cat_features,
            sample_weight=sample_weight,
            baseline=baseline,
            use_best_model=use_best_model,
            eval_set=eval_set,
            verbose=verbose,
            logging_level=logging_level,
            plot=plot,
            column_description=column_description,
            verbose_eval=verbose_eval,
            metric_period=metric_period,
            silent=silent,
            early_stopping_rounds=early_stopping_rounds,
            save_snapshot=save_snapshot,
            snapshot_file=snapshot_file,
            snapshot_interval=snapshot_interval,
            init_model=init_model,
            callbacks=callbacks,
            log_cout=log_cout,
            log_cerr=log_cerr
        )
        leafs = self.calc_leaf_indexes(X)
        self._fit_encoder(leafs)
        self._y_dims = y.shape[1] if len(y.shape) == 2 else 1  # Improve that!

    def _fit_encoder(self, leafs):
        self._leafs_encoder = OneHotEncoder(
            categories=[range(leafs.max() + 1) for _ in range(leafs.shape[1])],
            drop='if_binary',
            sparse=False
        ).fit(leafs)

    def _transform_encoder(self, leafs):
        return self._leafs_encoder.transform(leafs)

    def embed(self, X):
        leafs = self.calc_leaf_indexes(X)
        embeddings = self._transform_encoder(leafs)
        return embeddings


class EmbeddableCatBoostPriorNormal(EmbeddableCatBoost):
    """
    Embeddable CatBoost with N(0, 1) prior.
    """

    def pred_dist_param(self, X):
        """ Method for predicting distribution parameters. """
        return np.zeros((X.shape[0], 2 * self._y_dims))


class EmbeddableCatBoostPriorPredicted(EmbeddableCatBoost):
    """
    Embeddable CatBoost with N(mu(x), sigma(x)) prior.
    """

    def pred_dist_param(self, X):
        params = self.predict(X)
        params[:, 1] = np.log(np.sqrt(params[:, 1]))
        return params


class EmbeddableCatBoostPriorAveraged(EmbeddableCatBoost):
    """
    Embeddable CatBoost with N(mean(mu(x)), mean(sigma(x)) prior.
    """

    def pred_dist_param(self, X):
        params = self.predict(X)
        params[:, 1] = np.log(np.sqrt(params[:, 1]))

        return np.repeat(
            params.mean(axis=0, keepdims=True),
            params.shape[0],
            axis=0
        )
