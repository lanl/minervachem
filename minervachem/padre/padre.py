"""PADRE"""

from functools import partial 

import numpy as np

from .mappers import concatenate_and_subtract_features, subtract_targets

class PADRERegressor():
    """Pairwise difference regression with an arbitrary base learner

    :param base_regressor: a regressor that implements .fit and .predict
    :param feature_promote_fn: a function to promote features to pairwise features
        if None then canonical fn is used
    :param target_promote_fn: a function to promote targets to pairwise targets
        if None then canonical fn is used

    """

    def __init__(self, 
                 base_regressor, 
                 feature_promote_fn=None, 
                 target_promote_fn=None, 
                 random_state=42, 
                 x_transform_chunks=(100,-1), 
                 y_transform_chunks=(100,), 
                 x_output_chunks=(5000,-1), 
                 y_output_chunks=(5000,)):

        self.base_regressor = base_regressor
        if feature_promote_fn is None:
            feature_promote_fn = concatenate_and_subtract_features
        if target_promote_fn is None:
            target_promote_fn = subtract_targets
        self.feature_promote_fn = feature_promote_fn
        self.target_promote_fn = target_promote_fn
        self.random_state = np.random.seed(random_state) # todo: need to think about this

        self.x_transform_chunks=x_transform_chunks
        self.x_output_chunks=x_output_chunks
        self.y_transform_chunks=y_transform_chunks
        self.y_output_chunks=y_output_chunks
        if 'dask' in self.feature_promote_fn.__name__: 
            self.feature_promote_fn = partial(self.feature_promote_fn,
                                              transform_chunks=x_transform_chunks, 
                                              output_chunks=x_output_chunks)
        if 'dask' in self.target_promote_fn.__name__: 
            self.target_promote_fn = partial(self.target_promote_fn,
                                              transform_chunks=y_transform_chunks, 
                                              output_chunks=y_output_chunks)

    def fit(self, X1, y1, X2=None, y2=None, n_sample=None, **kwargs):
        """Fit `base_regressor` to the pairwise representation of X and y

        Approximates the function y1 - y2 = f(X1, X2) for all elements of X1, X2.
        If y2 and X2 are None, then fits y1 - y1 = f(X1, X1)

        :param X1: Training matrix of LHS pair elements
        :param X2: Training matrix of RHS pair elements
        :param y1: LHS training targets
        :param y2: RHS training targets
        :param n_sample: pair each element of X1 with n_sample elements of X2
        """
        self.X_train = X1.copy()
        self.y_train = y1.copy()

        if (X2 is None and y2 is not None) or \
           (X2 is not None and y2 is None):
            raise ValueError('Must provide both or neither X2 and y2')

        if X2 is None:
            X2, y2 = X1.copy(), y1.copy()
        else:
            X2, y2 = X2.copy(), y2.copy()

        if n_sample is not None:
            # todo: this implies that each X1 gets paired with the same X2. That's suboptimal.
            sample_idx = np.random.choice(np.arange(y2.shape[0]), size=n_sample, replace=False)
            X2 = X2[sample_idx, :]
            y2 = y2[sample_idx]

        X1X2_train = self.feature_promote_fn(X1, X2)
        y1y2_train = self.target_promote_fn(y1, y2)
        self.base_regressor.fit(X1X2_train, y1y2_train, **kwargs)
        return

    def predict(self, X, n_sample=None, return_std=False, return_cov=False):
        """Predict on X based on pairwise regression against the training set"""
        X1 = X
        X2 = self.X_train.copy()
        y2 = self.y_train.copy()
        if n_sample is not None:
            sample_idx = np.random.choice(np.arange(X2.shape[0]), size=n_sample, replace=False)
            X2 = X2[sample_idx, :]
            y2 = y2[sample_idx]

        n1 = X1.shape[0]
        n2 = X2.shape[0]
        X1X2_infer = self.feature_promote_fn(X1, X2)
        y1_minus_y2_hat = self.base_regressor.predict(X1X2_infer)
        y1_hat_distribution = y1_minus_y2_hat.reshape(n1, n2) + y2[np.newaxis, :]
        mu = y1_hat_distribution.mean(axis=1)
        sigma = y1_hat_distribution.std(axis=1)
        cov = np.cov(y1_hat_distribution, rowvar=False)

        if return_std and return_cov:
            return mu, sigma, cov
        elif return_std:
            return mu, sigma
        elif return_cov:
            return mu, cov
        else:
            return mu
            