import numpy as np
import warnings

class OverflowWarning(Warning): 
    pass

def vec_of_mat(a):
    if a.shape[0]!=1:
        raise ValueError(f"v shape is not vector: {a.shape}")
    return  np.asarray(a)[0,:]

def make_vecs(X):
    n = np.asarray(X.shape[0],dtype=X.dtype)
    sx = X.sum(axis=0)
    vx = X.power(2).sum(axis=0)
    qx = n*vx - np.power(sx, 2)
    if (qx == 0).any(): 
        raise ValueError("Features must not contain constant columns")
    return sx,vx

def sparse_correlated(X,Y=None, overflow_action='warn'):
    """
    Find correlated features in sparse, non-negative integer matrix X.
    We assume that constant value features have been removed ahead of time.
    X is shape (n_examples,n_features)
    Note: Integer overflow could cause problems for large N.
    If X and Y are the same matrix, we omit returning cases where i>=j.

    Rough math.
    r = 1 = <(x-m_x)(y-m_y)>/(sig_x * sig_y) # Pearson correlation
    sig_x * sig_y = <(x-m_x)(y-m_y)> = <xy> - m_x m_y
    var_x * var_y = (<xy> - m_x m_y)^2
    multiply by N^4.
    define for x and y:
    s_x := sum_i x_i
    v_x := x.x = sum_i x_i^2

    Write out variance term:
    var_x * N^2 = (<X^2> - <X>^2)* N^2 = (N x.x - s_x s_x ) = (N v_x - s_x^2)
    Write out right hand side
    (<xy> - m_x_my) * N^2 = (N x.y - s_x s_y)
    now have:
    (N v_x - s_x^2)(N v_y - s_y^2) = (N x.y - s_x*s_y)**2
    Expand out:
    N^2 v_x v_y - N*(v_x s_y^2 + v_y s_x^2) + s_x^2 s_y^2
        == N^2 (x.y)^2 + 2*N x.y*s_x*s_y + s_x^2 s_y^2
    Can cancel s_x^2 s_y^2 term, and then divide by N
    N v_x v_y - (v_x s_y^2 + v_y s_x^2) = N (x.y)^2 + 2*x.y *s_x*s_y
    L_xy := N v_x v_y - (v_x s_y^2 + v_y s_x^2)
    R_xy := N (x.y)^2 + 2*x.y*s_x*s_y

    
    This equation shows how it works for a single correlation.
    For multiple correlations we must vectorize.
    
    Find nonzero values of P_ij, and only compute those values of L_ij.
    
    We can rely on looking at nonzero values of P because:
        P_xy = N * x.y = 0:
        This can only occur if x and y are orthgonal (x.y) == 0.
        Since x and y are non-negative, this implies they have no overlapping nonzero elements.
        In order for this to be the case and still have them exactly correlated,
        both x and y must consist of entirely zeros.
        Thus they must be constant 0s, and so we have a contradiction.
    """
        
    if Y is None:
        Y = X
    same = (X is Y)

    for Z in X,Y:
        if not np.issubdtype(Z.dtype, np.integer): 
            raise TypeError("Features must be of an integer type")
    
    if (X<0).astype(int).sum() > 0: 
        raise ValueError("Features X must be nonnegative")
    
    # Convert ot int64 and csc representation
    # Smaller integer types may overflow more easily.
    n = X.shape[0]
    if same:
        X = Y = X.astype(np.int64).tocsc()
    else:
        if n != Y.shape[0]:
            raise ValueError("Features X and Y don't have same number of samples")
        if (Y<0).astype(int).sum() > 0: 
            raise ValueError("Features Y must be nonnegative")

        X = X.astype(np.int64).tocsc() 
        Y = Y.astype(np.int64).tocsc()

    
    def warn_overflow_possible(vars,
                               dtype=np.int64,
                               action='warn'):
        """Simple check for possibility of overflow after key quantities are computed"""
        XTY_vec, sx, sy, vx, vy = map(lambda a: int(a.max()), vars)
        R = n*XTY_vec**2 + (vx * sy**2 + vy * sx**2)
        L = n * vx * vy + XTY_vec*2*sx*sy
        ii = np.iinfo(dtype)
        msg = 'Encountered possible overflow'
        if L > ii.max or R > ii.max:
            if action == 'raise': 
                raise OverflowError(msg)
            elif action == 'warn': 
                warnings.warn(msg, OverflowWarning)

    n = np.asarray(n,dtype=X.dtype)

    sx, vx = make_vecs(X)
    
    sy, vy = (sx, vx) if same else make_vecs(Y)

    XTY = X.T*Y
    XTY.sort_indices() # This speeds up the operation of extracting indices later
    
    ind_x, ind_y = np.nonzero(XTY)
    if same:
        reduced = (ind_x < ind_y)
        ind_x, ind_y = ind_x[reduced],ind_y[reduced]
    
    # If nothing remains we are done.
    if len(ind_x) == 0:
        return [],[]

    # get vector versions
    vx, sx, vy, sy = map(vec_of_mat,(vx,sx, vy, sy))
    # get relevant indices
    vx, sx = vx[ind_x], sx[ind_x]
    vy, sy = vy[ind_y], sy[ind_y]

    
    XTY_vec = XTY[ind_x,ind_y]
    XTY_vec = vec_of_mat(XTY_vec)

    warn_overflow_possible((XTY_vec, sx, sy, vx, vy), action=overflow_action)
    
    R_flat = n * XTY_vec**2 + (vx * np.power(sy,2) + vy * np.power(sx,2))    
    L_flat = n * vx * vy + XTY_vec*2*sx*sy 
    
    eq = (L_flat == R_flat)
    
    return ind_x[eq], ind_y[eq]

def map_useless_features_sparse(X, overflow_action='warn'):
    """
    For API compatiblity
    """
    # Note: relies on ind_1 coming out sorted?
    ind_1,ind_2 = sparse_correlated(X, overflow_action=overflow_action)

    # ignore self-maps and i > j values.
    where_interested = ind_1 < ind_2
    ind_1, ind_2 = ind_1[where_interested], ind_2[where_interested]
    
    # Step one, get the map from the redundant indices to the lowest 
    # index available, which is non-redundant by the `where_interested` above.
    # Note: I think this converges exponentially so even a really bad set
    # would still be ok.
    
    # Map indices to -some- lower index that is redundant.
    lower_map = dict(zip(ind_2[::-1],ind_1[::-1]))
    updated = True
    while updated:
        # Map to an even lower index if possible!
        new_lower_map = {j:lower_map.get(k,k) for j,k in lower_map.items()}
        updated = (new_lower_map != lower_map)
        #total = len([k for (k,v) in lower_map.items() if new_lower_map[k]!=v])
        #print(total)
        lower_map = new_lower_map
        
    # Step two, collect the results of the map to the lowest feature, above
    collection = {i:[] for i in set(ind_1)}
    for j,i in lower_map.items():
        collection[i].append(j)
    collection = {i:list(sorted(set(arr))) for i,arr in collection.items() if len(arr)!=0}
        
    useful = np.ones(X.shape[1], dtype=bool)
    ind_redund = set(ind_2)
    if ind_redund:
        useless = np.asarray(list(ind_redund))
        useful[useless] = False

    return useful,collection
    
