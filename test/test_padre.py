import pytest
import numpy as np
import scipy as sp
import scipy.sparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor, DaskLGBMRegressor

from distributed import Client, LocalCluster

from minervachem.padre import PADRERegressor
from minervachem.padre.mappers import (concatenate_and_subtract_features,
                           concatenate_and_subtract_features_sparse, 
                           concatenate_and_subtract_features_sparse_dask, 
                           subtract_targets, 
                           subtract_targets_dask)


def test_padre_rf_sklearn_regression():
    """Test PADRE RF against an sklearn regression problem

    Checks output shapes and does a regression tests for some metrics

    TODO, if repeating similar tests, first factor:
        * problem generation
        * model fitting
        * expected values
        * tests
    """

    # setup problem
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    padre_rf = PADRERegressor(
        base_regressor=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    )
    rf.fit(X_train, y_train)
    padre_rf.fit(X_train, y_train)

    # test output shapes/sizes
    y_hat_train = padre_rf.predict(X_train)
    y_hat_test = padre_rf.predict(X_test)
    assert len(y_hat_train.shape) == 1, "PADRE should predict a vector in this case"
    assert len(y_hat_test.shape) == 1,  "PADRE should predict a vector in this case"
    assert y_hat_train.shape[0] == y_train.shape[0], "PADRE train prediction shape should match train targets"
    assert y_hat_test.shape[0] == y_test.shape[0],   "PADRE train prediction shape should match train targets"

    rf_err = mean_squared_error(rf.predict(X_test), y_test)
    padre_rf_err = mean_squared_error(padre_rf.predict(X_test), y_test)

    assert rf_err > padre_rf_err, "PADRE should do better on this task"

def test_sparse_feature_mapper(): 
    """Tests concatenate_and_subtract_features_sparse against non-sparse version"""

    X1 = sp.sparse.random(50, 100)
    X2 = sp.sparse.random(60, 100)

    X1_np, X2_np = [np.asarray(a.todense()) for a in (X1, X2)]

    expected = concatenate_and_subtract_features(X1_np, X2_np)
    result = concatenate_and_subtract_features_sparse(X1, X2)
    result_np = np.asarray(result.todense()) 

    assert (expected == result_np).all()

def test_padre_with_sparse(): 
    """Ensure sparse mapper is compatible with PADRERegressor"""

    X = sp.sparse.random(100, 100)
    y = np.random.randn(100)
    rfr = RandomForestRegressor(n_jobs=-1, n_estimators=50)
    pr = PADRERegressor(base_regressor=rfr, feature_promote_fn=concatenate_and_subtract_features_sparse)
    pr.fit(X, y)
    pr.predict(X)

class TestPadreDask(): 
    
    @classmethod
    def setup_class(cls): 
        cls.cluster = LocalCluster(n_workers=1)
        cls.client = Client(cls.cluster)
        return cls
    @classmethod
    def teardown_class(cls): 
        cls.cluster.close()
        cls.client.close()

    def test_padre_dask_feature_mapper(self): 
        X = sp.sparse.random(100, 100)
        expected = concatenate_and_subtract_features_sparse(X, X)
        actual = concatenate_and_subtract_features_sparse_dask(X, X, transform_chunks=(100, -1), output_chunks=(100,-1))
        actual = actual.compute()
        expected, actual = [np.asarray(a.todense()) for a in [expected, actual]]
        assert np.all(expected == actual)

    def test_padre_dask_target_mapper(self):
        y = np.random.randn(100)
        expected = subtract_targets(y, y)
        actual = subtract_targets_dask(y, y, transform_chunks=(100,), output_chunks=(100,))
        actual = actual.compute()
        assert np.all(actual == expected)

    def test_padre_dask_regressor(self): 
        X = sp.sparse.random(100, 100)
        y = np.random.randn(100)
        lgbm_regressor = PADRERegressor(LGBMRegressor(n_estimators=10, random_state=42),
                                        feature_promote_fn=concatenate_and_subtract_features_sparse)
        dask_regressor = PADRERegressor(DaskLGBMRegressor(n_estimators=10, random_state=42), 
                                        feature_promote_fn=concatenate_and_subtract_features_sparse_dask, 
                                        target_promote_fn=subtract_targets_dask,
                                        x_output_chunks=(50,-1),
                                        x_transform_chunks=(50, -1), 
                                        y_output_chunks=(50,), 
                                        y_transform_chunks=(50,))
        lgbm_regressor.fit(X, y)
        dask_regressor.fit(X, y)
        expected = lgbm_regressor.predict(X)
        actual = dask_regressor.predict(X).compute()
        assert np.allclose(expected, actual)