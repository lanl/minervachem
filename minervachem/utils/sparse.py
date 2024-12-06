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

def select_rows_from_matrix(csr_matrix, row_indices):
    """
    The function is used to select a subset of rows of `csr_matrix` and return
    the selected subset as a 'csr_matrix' of the same width, but different length.
    :param csr_matrix: the big matrix, from which we will select a subset of rows.
    :param row_indices: indices identifying rows to be selected.
    """
    if type(row_indices) != list:
        row_indices = list(row_indices)
    return csr_matrix[row_indices, :]

def combine_sparse_matrices(csr_matrix1, csr_matrix2):
    """
    The function is used to combine two sparse matrices into one. Two sparse matrices are combined if they have the same size.
    """
    if csr_matrix1.shape[1] != csr_matrix2.shape[1]:
        print('The two matrices must have the same number of columns.')
        return
    else:
        return sp.sparse.vstack([csr_matrix1, csr_matrix2])

def select_columns_from_matrix(csr_matrix, col_indices):
    """
    The function is used to select a subset of columns of `csr_matrix` and return
    the selected subset as a 'csr_matrix' of the same length, but different width.
    :param csr_matrix: the big matrix, from which we will select a subset of columns.
    :param col_indices: indices identifying columns to be selected.
    """
    csc_matrix = csr_matrix.tocsc()
    if type(col_indices) != list:
        col_indices = list(col_indices)
    subset_matrix = csc_matrix[:, col_indices]
    return subset_matrix.tocsr()

def find_nonzero_rows(csr_matrix, column_x):
    col_data = csr_matrix.getcol(column_x).toarray().flatten()
    return np.nonzero(col_data)[0]