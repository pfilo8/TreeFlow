class EmbeddableTree:

    def fit(self, X, y):
        """Method for fitting Tree model."""
        pass

    def embed(self, X):
        """Method for embedding data using Tree model."""
        pass

    def pred_dist_param(self, X):
        """
        Method for predicting distribution parameters.

        Distribution parameters will be later used as a prior for CNF model.
          - 2D array - First column should be mean and the second logstd.
        """
        pass
