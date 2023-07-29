import numpy as np

from sklearn.preprocessing import OneHotEncoder


class EmbeddableOneHotEncoder(OneHotEncoder):

    def fit(self, X, y=None):
        """Method for fitting One Hot Encoder."""
        super().fit(X=X, y=y)
        self._y_dims = y.shape[1] if len(y.shape) == 2 else 1  # Improve that!

    def embed(self, X):
        """Method for embedding data using Tree model."""
        return self.transform(X).todense()

    def pred_dist_param(self, X):
        """ Method for predicting distribution parameters. """
        return np.zeros((X.shape[0], 2 * self._y_dims))
