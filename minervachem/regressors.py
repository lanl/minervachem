"""Provides sklearn-style regressors
"""


from copy import deepcopy
from functools import cached_property
from time import time

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.metrics import mean_absolute_error
from flaml import tune
from flaml.automl.model import LGBMEstimator
import lightgbm

    
class HierarchicalResidualModel():
    def __init__(self,
                 regressor,
                 levels=None,
                 verbose=0):
        """Hierarchical residual model: fits regressor at level zero, residual regressors at levels 1..L.

        At level zero minimizes Loss(y, f_0(x)). At level 1 <= l <= L, minimizes Loss(y, f_{l-1}(x)).
        Prediction yhat_l = f_l(x) + yhat_{l-1}; yhat_0 = f_0(x).

        Args:
            regressor: sklearn regressor or cross validator, or a mapping or length n_levels iterable thereof
                       Regressor(s) to fit hierarchically. If iterable, assumes levels are 0..L.
                       If single regressor, it is copied.

            verbose: (int) verbosity
            levels: (np.ndarray[int]) of size n_features, indicates the level of the features. Can be passed to fit.
        
        """
        
        self._regressor = regressor # what is passed by user
        self.regressors = None      # what is constructed based on levels

        self.levels = levels
        self.verbose = verbose
        self.is_fitted_ = False
        self.cv_estimators_ = None


        if levels is not None:
            self._validate_levels()
            self._set_regressors()
        return

    def fit(self, X, y, levels=None, *args, **kwargs):
        """
        X is m x n
        y is m x 1
        levels is n x 1
        """
        # for recording fit times at each level
        self.fit_times = []

        if levels is not None:
            self.levels = levels
            self._validate_levels()
            self._set_regressors()

        unique_levels = self.unique_levels if not self.verbose else tqdm(self.unique_levels, 'Regressor levels:')
        for l in unique_levels:
            start = time()
            X_l = X[:, self.levels == l]
            if l == self.min_level:
                self.regressors[l].fit(X_l, y, *args, **kwargs)
            else:
                pred = self._predict_up_to_l(X, l - 1)
                resid = y - pred
                self.regressors[l].fit(X_l, resid, *args, **kwargs)
            elapsed = time() - start
            self.fit_times.append(elapsed)
        self._handle_cv_estimators()
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self._predict_up_to_l(X, self.max_level)
        
    @property
    def min_level(self): 
        return min(self.levels)

    @property
    def max_level(self):
        return max(self.levels)

    @property
    def unique_levels(self):
        if self.levels is not None:
            return sorted(list(set(list(self.levels))))
        else: 
            return []

    @property
    def n_levels(self):
        return len(self.unique_levels)

    def _predict_up_to_l(self, X, L):
        """Inclusive of L"""
        pred = np.zeros(X.shape[0])
        for l in self.unique_levels:
            if l <= L:
                X_l = X[:, self.levels == l]
                pred_l = self.regressors[l].predict(X_l)
                pred += pred_l
        return pred

    def _validate_levels(self):
        levels_match = True
        if isinstance(self._regressor, dict):
            levels_match = set(self.unique_levels) == set(list(self._regressor.keys()))
            message = 'If given dict of regressors, keys must match unique levels'
        elif isinstance(self._regressor, list):
            levels_match = self.n_levels == len(self._regressor)
            message = 'If given list of regressors, length of list must match n_levels'
        if not levels_match:
            raise RuntimeError(message)
        return

    def _set_regressors(self):
        """self.regressors will always be a dict to handle when unique levels arent just a range from 0..n"""
        if isinstance(self._regressor, dict):
            self.regressors = self._regressor
        elif isinstance(self._regressor, list):
            self.regressors = {l: self._regressor[i] for i, l in zip(range(self.n_levels), self.unique_levels)}
        else:
            self.regressors = {l: deepcopy(self._regressor) for l in self.unique_levels}
        return

    def _handle_cv_estimators(self):
        self.cv_estimators_ = {}
        for l, r in self.regressors.items():
            if hasattr(r, 'best_estimator_'):
                self.regressors[l] = r.best_estimator_
                self.cv_estimators_[l] = r
        return

    @cached_property
    def mu(self):
        """A vector of the mean coefficients at each level. Only works for linear models."""
        if not self.is_fitted_:
            raise NotFittedError()
        else:
            return np.hstack([self.regressors[l].coef_.mean() for l in self.unique_levels])

    @cached_property
    def var(self):
        """A vector of the variance of the coefficients at each level. Only works for linear models"""
        if not self.is_fitted_:
            raise NotFittedError()
        else:
            return np.hstack([self.regressors[l].coef_.var(ddof=1) for l in self.unique_levels])

    @property
    def weight_df(self):
        """A dataframe of coefficients and the log of their absolute values by level

        Useful for visualizing coeffs. Only works for linear models
        """
        df = []
        for level in self.unique_levels:
            df_l = pd.DataFrame()
            df_l['coef'] = self.regressors[level].coef_
            df_l['log_abs_coef'] = np.log10(np.abs(self.regressors[level].coef_))
            df_l['level'] = level
            df.append(df_l)
        return pd.concat(df)
    
    @property
    def coef_(self):
        """"""
        out = []
        for level in sorted(self.unique_levels): 
            model = self.regressors[level]
            if hasattr(model, 'coef_'):
                out.append(model.coef_)
            else: 
                out.append(model.best_estimator_.coef_)
        return np.hstack(out)

    def plot_coefs(self, log=True, *args, **kwargs):
        ax = sns.boxplot(data=self.weight_df,
                    x='level',
                    y='log_abs_coef' if log else 'coef')
        return ax

    def pred_var_unseen(self, X, levels_unseen):
        n = X.shape[0]
        pred = np.zeros(n)
        var = np.zeros(n)
        for l in levels_unseen:
            X_l = X[:, levels_unseen == l]
            pred += np.asarray(X_l.sum(1)).squeeze() * self.mu[l]
            var += np.asarray((X_l.power(2)).sum(1)).squeeze() * self.var[l]
        return pred, var

    def pred_var_seen_unseen(self, X_seen, X_unseen, levels_unseen):
        pred = self.predict(X_seen)
        correction, var = self.pred_var_unseen(X_unseen, levels_unseen)
        return pred + correction, var

class FlamlLGBM(BaseEstimator, RegressorMixin): 
    
    def __init__(self, 
                 train_size=.8, 
                 time_budget_s=10, 
                 random_state=None, 
                 metric='mae',
                 verbose=0):
        """Adapts FLAML HPO for LGBM to an sklearn interface

        :param train_size: the rest is used for val
        :param time_budget_s: number of seconds to do HPO
        :random_state: passed to train_test_split
        :metric: this is optimized during HPO
        """
        self.train_size = train_size
        self.time_budget_s = time_budget_s
        self.random_state = random_state
        self.metric = metric
        self.verbose=verbose
        
    def fit(self, X, y, X_val=None, y_val=None, val_ix=None, config=None, validate=True, refit=True): 
        #warnings.simplefilter("ignore")
        if validate:
            if X_val is None or y_val is None:
                (self.X_train, 
                 self.X_val,
                 self.y_train,  
                 self.y_val) = train_test_split(X, 
                                                y, 
                                                train_size=self.train_size, 
                                                random_state=self.random_state)
            elif val_ix is not None: 
                splitter = PredefinedSplit(val_ix)
                train_i, val_i = splitter.split(X, y)
                self.X_train = X[train_i]
                self.X_val   = X[val_i]
                self.y_train = y[train_i]
                self.y_val   = y[val_i]

            else: 
                self.X_train = X
                self.X_val = X_val
                self.y_train = y
                self.y_val = y_val
            self._test_tune_lgbm()
        else: 
            self.X_train = X
            self.y_train = y

        train_set = lightgbm.Dataset(
            data=self.X_train, 
            label=self.y_train
        )
        if validate:
            if refit:
                train_val = lightgbm.Dataset(
                    data=sp.sparse.vstack([self.X_train, self.X_val]),
                    label=np.hstack([self.y_train, self.y_val])
                )
                self.model = lightgbm.train(self.analysis_.best_config, train_val)
            else:
                self.model = lightgbm.train(self.analysis_.best_config, train_set)
        else: 
            self.model = lightgbm.train(config, train_set)
            
        #warnings.simplefilter("default")
        return self
                                    
    def predict(self, X): 
        return self.model.predict(X)

    def _train_lgbm(self, config: dict) -> dict:
        # convert config dict to lgbm params
        params = LGBMEstimator(**config).params
        # train the model
        train_set = lightgbm.Dataset(
            data=self.X_train, 
            label=self.y_train
        )
        model = lightgbm.train(params, train_set)
        # evaluate the model
        pred = model.predict(self.X_val)
        mae = mean_absolute_error(self.y_val, pred)
        # return eval results as a dictionary
        return {"mae": mae}

    def _test_tune_lgbm(self):
        # load a built-in search space from flaml
        flaml_lgbm_search_space = LGBMEstimator.search_space(self.X_train.shape)
        # specify the search space as a dict from hp name to domain; you can define your own search space same way
        config_search_space = {
            hp: space["domain"] for hp, space in flaml_lgbm_search_space.items()
        }
        self._vprint(config_search_space,verbose_thresh=2)
        # give guidance about hp values corresponding to low training cost, i.e., {"n_estimators": 4, "num_leaves": 4}
        low_cost_partial_config = {
            hp: space["low_cost_init_value"]
            for hp, space in flaml_lgbm_search_space.items()
            if "low_cost_init_value" in space
        }
        # self._vprint(low_cost_partial_config)
        # initial points to evaluate
        points_to_evaluate = [
            {
                hp: space["init_value"]
                for hp, space in flaml_lgbm_search_space.items()
                if "init_value" in space
            }
        ]
        self._vprint("AutoML points to evaluate:",points_to_evaluate)
        # run the tuning, minimizing mse, with total time budget 3 seconds
        self.analysis_ = tune.run(
            self._train_lgbm,
            metric="mae",
            mode="min",
            config=config_search_space,
            low_cost_partial_config=low_cost_partial_config,
            points_to_evaluate=points_to_evaluate,
            time_budget_s=self.time_budget_s,
            num_samples=-1,
            verbose=0
        )
        return 
    
    @property
    def best_params_(self): 
        return self.analysis_.best_result['config']
    
    
    @property
    def best_estimator_(self): 
        return self.model

    def _vprint(self,*args,verbose_thresh=1,**kwargs):
        if self.verbose>=verbose_thresh:
            print(*args,**kwargs)
