"""Functions for promoting features and targets from single to double indexed quantities"""
import numpy as np
import scipy as sp
import sparse
import dask.array as da


def concatenate_and_subtract_features(X1, X2):
    """Returns all pairwise concatenations and differences
    
    Feature promoter described in original manuscript
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    X1 = X1[:, np.newaxis, :].repeat(n2, axis=1)
    X2 = X2[np.newaxis, :, :].repeat(n1, axis=0)
    X1X2_combined = np.concatenate([X1, X2, X1 - X2], axis=2)
    return X1X2_combined.reshape(n1 * n2, -1)


def concatenate_and_subtract_features_sparse(X1, X2):
    """Returns all pairwise concatenations and differences of sparse features
    
    Sparse implementation of feature promoter described in original manuscript
    """
    S1 = sparse.COO(X1)
    S2 = sparse.COO(X2)
    n1 = S1.shape[0]
    n2 = S2.shape[0]
    m = S1.shape[1]
    S1 = S1[:, np.newaxis, :].broadcast_to((n1, n2, m))
    S2 = S2[np.newaxis, :, :].broadcast_to((n1, n2, m))
    S1S2_combined = sparse.concatenate([S1, S2, S1-S2], axis=2)
    return S1S2_combined.reshape((n1*n2, -1)).to_scipy_sparse()


def concatenate_and_subtract_features_sparse_dask(X1, X2,
                                                  transform_chunks,
                                                  output_chunks): 
    """Returns all pairwise concatenations and differences of sparse features
    
    Sparse-dask implementation of feature promoter described in original manuscript
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    m = X1.shape[1]
    
    # convert to dask arrays 
    S1, S2 = [da.asarray(sparse.COO(a), chunks=transform_chunks)
              for a in [X1, X2]]
    
    # standard transformation 
    S1 = da.broadcast_to(S1[:, np.newaxis, :], (n1, n2, m))
    S2 = da.broadcast_to(S2[np.newaxis, :, :], (n1, n2, m))
    S1S2_combined = da.concatenate([S1, S2, S1-S2], axis=2)
    out = S1S2_combined.reshape((n1*n2, -1))
    
    # rechunk and change types to satisfy lgbm interface
    out = out.rechunk(output_chunks)
    out = out.map_blocks(lambda a: a.to_scipy_sparse(), dtype=np.float64)
    # hacky: the above line will give chunks of type np.ndarray for unknown reason
    # so we do a second map_blocks. My guess is this is highly wasteful
    out = out.map_blocks(lambda a: sp.sparse.csr_matrix(a, dtype=np.float64))
    return out

def subtract_targets(y1, y2):
    """All pairwise differences of a target vector
    
    Target promoter described in original manuscript
    """
    return (y1[:, np.newaxis] - y2[np.newaxis, :]).flatten()

def subtract_targets_dask(y1, y2, 
                          transform_chunks,
                          output_chunks):
    """All pairwise differences of a target vector
    
    Target promoter described in original manuscript
    """
    y1, y2 = [da.asarray(a, chunks=transform_chunks) 
              for a in [y1, y2]]
    out = (y1[:, np.newaxis] - y2[np.newaxis, :]).flatten()
    out = out.rechunk(output_chunks)
    return out
