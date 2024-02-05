import numpy as np

class FragmentLinearModelCorrector: 
    
    def __init__(self, params, param_sizes):
        """
        :params:      a (n, ) numpy array of linear model coefficients
        :param_sizes: a (n, ) numpy array of sizes of the substructures associated with the coefficients
        """
        
        self.params = params
        self.param_sizes = param_sizes
        self.unique_fragment_sizes = sorted(list(set(param_sizes)))
        
        # take average of coefficients at different sizes
        self.mu_by_size = {}
        for size in self.unique_fragment_sizes: 
            self.mu_by_size[size] = self.params[param_sizes == size].mean()
            
    def predict(self, X_seen, X_unseen, unseen_sizes):
        """
        :X_seen: (m,n) array of substructure counts, to be multiplied with self.params
        :X_unseen: (m,u) array of of unseen substructure counts
        :unseen_sizes: (u,) array of sizes of unseen features 
        """
        seen_prediction = self._predict_seen(X_seen)
        unseen_correction = self._predict_unseen(X_unseen, unseen_sizes)
        return seen_prediction + unseen_correction
        
    def _predict_seen(self, X):
        
        return X @ self.params
    
    def _predict_unseen(self, X, sizes): 
        correction = np.zeros(X.shape[0])
        for size in self.unique_fragment_sizes: 
            counts_of_size = X[:, sizes == size].sum(1) # gives an (m, 1) matrix
            correction_at_size = self.mu_by_size[size] * counts_of_size
            correction += np.asarray(correction_at_size).squeeze() 
        return correction
    