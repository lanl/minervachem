"""Unit tests for transformer *methods*

We do not currently test fit/predict or do any integration tests
but we can implement this next. 
"""

import pytest

import numpy as np
import scipy as sp
import scipy.sparse
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from rdkit.Chem import MolFromSmiles
from rdkit import Chem
from rdkit.Chem import AllChem

from minervachem import transformers as T
from minervachem.sparse_eliminator import OverflowWarning, map_useless_features_sparse


class BaseTestMapUselessFeatures: 
   
    @staticmethod
    def generate_random_problem(size=(1000,1000), 
                            n_cor=10, 
                            n_clusters=1): 
        """Generate a random matrix of size size with n_cor perfectly correlated columns
        :n_cor int: number of correlated variables
        :n_clusters: number of unique 'clusters' of variables among the n_correlated variables
        """
        
        solution = {}
        if not n_clusters <= n_cor: 
            raise ValueError("n_clusters must be less than n_cor")
        X = np.random.randn(size[0], size[1])
        X[X<0] = 0
        X = (X*10).astype(np.int64)

        if n_cor == 0: 
            return sp.sparse.csr_matrix(X).astype(np.uint64), solution

        cor_indices = sorted(np.random.choice(np.arange(size[1]), 
                            size=n_cor, 
                            replace=False))
        for cluster in np.array_split(cor_indices, n_clusters): 
            unique_feat_ix, redundant_indices = cluster[0], cluster[1:]
            if len(redundant_indices) == 0: 
                continue
            unique_feat = X[:, unique_feat_ix]
            solution[unique_feat_ix] = list(redundant_indices)
            for redundant_feat_ix in redundant_indices:
                sign_slope, sign_bias = np.random.choice([-1, 1], 2)
                slope, bias = np.random.uniform((1, 5), 2).astype(int)
                redundant_feat = sign_slope * slope * unique_feat + sign_bias * bias 
                if (redundant_feat<0).any(): 
                    redundant_feat = redundant_feat - redundant_feat.min()
                X[:, redundant_feat_ix] = redundant_feat
        
        X = sp.sparse.csr_matrix(X).astype(np.uint64)
        return X, solution


class TestMapUselessFeaturesSparse(BaseTestMapUselessFeatures): 

    @pytest.mark.parametrize(
        "size,n_cor,n_clusters",
        [
            ((100,1000), 10, 1),
            ((100,1001), 10, 1),
            ((100,999), 10, 1),
            ((100,1000), 10, 2),
            ((100,1000), 10, 1), 
            ((100,1000), 0, 0)
        ]
    )
    def test_map_useless_features_sparse(self, size, n_cor, n_clusters): 
        problem, expected = self.generate_random_problem(size, n_cor, n_clusters)
        _, solution = map_useless_features_sparse(problem)
        assert solution == expected

class TestSparseEliminatorErrors(BaseTestMapUselessFeatures): 
    """Assert that this eliminator raises the errors we defined it to"""
    
    def test_input_is_int(self):
        problem, _ = self.generate_random_problem()
        problem = problem.astype(np.float64)
        with pytest.raises(TypeError) as excinfo: 
            map_useless_features_sparse(problem)
        assert 'integer' in str(excinfo.value)

    def test_input_is_nonnegative(self): 
        problem, _ = self.generate_random_problem()
        problem[0,0] = -1
        problem = problem.astype(np.int64)
        with pytest.raises(ValueError) as excinfo: 
            map_useless_features_sparse(problem)
        assert 'nonnegative' in str(excinfo.value)

    def test_input_has_no_constants(self): 
        problem, _ = self.generate_random_problem()
        problem[:, 0] = 1
        with pytest.raises(ValueError) as excinfo: 
            map_useless_features_sparse(problem)
        assert 'constant' in str(excinfo.value)

    def test_overflow_error(self):
        """This is an unrealistic example designed to overflow in 
        computing R/L, but not in the sum. A direct test of the overflow 
        warning function may be desirable. 
        """
        val = np.iinfo(np.int64).max / (3 * 100)
        problem = (np.zeros((100, 1000)) + val)
        problem [0, :] = 0  # avoid constant columns
        problem = sp.sparse.csc_matrix(problem, dtype=np.int64)
        with pytest.raises(OverflowError) as excinfo:
            map_useless_features_sparse(problem, overflow_action='raise')
        with pytest.warns(OverflowWarning):
            map_useless_features_sparse(problem, overflow_action='warn')
