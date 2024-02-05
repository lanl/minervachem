"""Utils for working with sparse arrays"""

import numpy as np
import scipy as sp
import scipy.sparse

def corrcoef(A, B=None):
    """Like np.corrcoef but for scipy sparse matrices
    
    modified from https://stackoverflow.com/questions/19231268"""

    if B is not None:
        A = sp.sparse.vstack((A, B), format='csr')

    A = A.astype(np.float64)
    n = A.shape[1]

    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))

    return np.array(coeffs)

def variance(X, axis=0): 
    """Variance of a scipy sparse matrix"""
    E_X2 = np.array(X.power(2).mean(axis)).squeeze()
    EX_2 = np.array(X.mean(axis)).squeeze()**2
    var = E_X2 - EX_2
    return var 

def dropcols_coo(C, idx_to_drop, copy=True):
    """https://stackoverflow.com/questions/23966923/delete-columns-of-matrix-of-csr-format-in-python?lq=1"""
    if copy:
        C = C.copy()
    idx_to_drop = np.unique(idx_to_drop)
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C 

def drop_zero_columns_coo(S, copy=True):
    """https://github.com/scipy/scipy/issues/6754#issuecomment-258303764"""
    if copy:
        S = S.copy()
    M, N = S.shape
    order = np.argsort(S.col)
    tmp = np.cumsum((np.diff(S.col[order]) > 0) * 1)
    if sum(tmp.shape) == 0: 
        return S
    S.col *= 0
    S.col[1:] += tmp
    S.row = S.row[order]
    S.data = S.data[order]
    N = int(tmp.max()) + 1
    S._shape = (M, N)
    return S

def to_array(S):
    return np.asarray(S.todense())
