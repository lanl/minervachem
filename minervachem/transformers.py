"""Provides sklearn transformer interface to minerva's fingerprinting capabiliities
"""

from time import time

import numpy as np
import scipy as sp
import pandas as pd
import sigfig

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from rdkit import Chem

from tqdm.auto import tqdm

from .utils import misc
from minervachem.utils.misc import uniquify_lol
from minervachem.sparse_eliminator import map_useless_features_sparse


class MoleculeFeaturizer(BaseEstimator, TransformerMixin):
    """Creates RDKitMols from a pd dataframe column
    """

    def __init__(self,
                input_column='ligand',
                output_column='mol',
                n_jobs=None,
                verbose=0,
                transformer=Chem.MolFromSmiles):
        self.input_column = input_column
        self.output_column = output_column
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.transformer = transformer


    def fit(self, X, y=None):
        """Doesn't do anything, exists for compatibility"""
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Looks for smiles_column, finds unique, makes rdkit mols, joins in as output column"""

        # only convert each molecule from smiles once
        unique_mol_rows = X.drop_duplicates(subset=self.input_column)[[self.input_column]]

        n_mols = unique_mol_rows.shape[0]
        batch_size = misc.evenly_distribute_jobs(n_mols, self.n_jobs)
        print('Converting to RDKit Mol...')
        p = Parallel(n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    prefer='threads',
                    batch_size=batch_size,
                    )

        # RDKit functions cannot be pickled directly, so we wrap them
        def t(s):
            return self.transformer(s)
        f = delayed(t)

        mols = p(f(mol) for mol in unique_mol_rows[self.input_column])
        unique_mol_rows[self.output_column] = mols
        X = pd.merge(X, unique_mol_rows, how='left')
        return X

    def get_feature_names(self):
        return ['mol']

class FingerprintFeaturizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 fingerprinter, 
                 return_dense=False,
                 return_float=False,
                 n_jobs=None,
                 chunk_size='even',
                 verbose=0,
                ): 
        
        """sklearn transformer that applies a minervachem.fingerprinter to rdkit.Molecule objects

        Args:
            fingerprinter: a minerva.fingerprinters.Fingerprinter object
            return_dense (bool): whether transform returns a dense matrix
            return_float (bool):  whether transform returns a matrix of floats
            n_jobs (int): number of joblib jobs
            chunk_size (int or string): joblib chunk size. If "even", evaluates to n_mols // n_cores
            verbose (int): verbosity level (currently 0 or 1)

        Attributes:
            self.n_bits_ (int): number of fingerprint fragments found during fit. Sorted like self.bit_ids_
            self.bit_ids_ (list): list of fragment identifiers for all fragments found in fit. Sorted in increasing order
            self.bit_sizes_ (list): list of sizes of each fragment found during fit. Sorted like self.bit_ids_
            self.bit_indices_ (dict): maps bit IDs to their column index
            self.n_unseen_ (int): number of new fingerprint fragments found during transform. Sorted like self.bit_ids_
            self.X_unseen_ (np.ndarray or sp.sparse.matrix): matrix of fingerprint fragments found during transform
            self.bit_indices_unseen_ (dict): maps unseen bit IDs to their column index
            self.bit_sizes_unseen_ (list): list of sizes of fragments found during transform.
        """
        self.fingerprinter = fingerprinter
        self.return_dense = return_dense
        self.return_float = return_float
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        self.verbose = verbose
        self._cache = {}
        self._in_fit_transform = False

        # set during .fit
        self.n_bits_ = None
        self.bit_ids_ = None
        self.bit_sizes_ = None
        self.bit_indices_ = None

        # set during.transform
        self.n_unseen_ = None
        self.X_unseen_ = None
        self.bit_ids_unseen_ = None
        self.bit_indices_unseen_ = None
        self.bit_sizes_unseen_ = None

    def fit(self, X, y=None):
        """Find fingerprints in iterable of molecules

        Note: It is highly recommended to call from fit_transform with large datasets, as fingerprints
              must be recomputed during transform otherwise.

        Args:
            X: an iterable of RDKit molecules
            y: exists for compatibility with sklearn pipelines, does nothing

        Returns:
            self (FingerprintFeaturizer): this instance
        """

        if self.verbose: 
            print(f'Finding fingerprints in fit')
        mols = X
        fps, self.bi_fit_, bit_ids = self._get_fps(mols)
            
        # unnest, get unique BitIDs, and sort bit IDs
        
        self.bit_ids_ = sorted(uniquify_lol(bit_ids))
        (self.bit_indices_,
         self.bit_sizes_) = self._get_bit_indices_and_sizes(self.bit_ids_)
        self.n_bits_ = len(self.bit_ids_)
        
        if self.verbose:
            print(f'N bits: {self.n_bits_}')
            
        # the cache is used in between fit and transform 
        # during self.fit_transform to avoid computing the FPs twice
        if self._in_fit_transform:
            self._cache['fps'] = fps
        self.is_fitted_ = True
        return self

    def transform(self, X, return_unseen=False):
        """Transform an iterable of molecules to a matrix of fingerprints.

        Must be called after .fit

        Args:
            X: an iterable of molecules
            return_unseen: (bool) if True, return X_unseen containing fragments not seen during .fit
        Returns:
            X (np.ndarray or sp.sparse.matrix of int or float): a matrix, format depends on self.return_sparse and
                                                                return_float
            X_unseen (optional): like X, but for fragments not identified during fit
        """
        start = time()
        check_is_fitted(self)
        if self._in_fit_transform:
            fps = self._cache['fps']
            bits_seen = self.bit_ids_
            self.bit_ids_unseen_ = []
        else:
            mols = X
            self._print('finding fingerprints in transform')
            fps, self.bi_transform_, bits = self._get_fps(mols)
            bits = sorted(uniquify_lol(bits))
            self.bit_ids_unseen_ = sorted(list(set(bits) - set(self.bit_ids_)))
            self.n_unseen_ = len(self.bit_ids_unseen_)
            if self.n_unseen_ > 0:
                (self.bit_indices_unseen_,
                 self.bit_sizes_unseen_) = self._get_bit_indices_and_sizes(self.bit_ids_unseen_)
            else:
                self.bit_indices_unseen_, self.bit_sizes_unseen_ = {}, []
            self.n_unseen_ = len(self.bit_ids_unseen_)

        self._print('Converting bits to array form')
        M_seen = self._to_sparse(fps, self.bit_indices_)
        if self.bit_ids_unseen_:
            self._print('Converting unseen bits to array form')
            self.X_unseen_ = self._to_sparse(fps, self.bit_indices_unseen_)
        else:
            self.X_unseen_ = None
        if self.return_dense:
            self._print('Converting from sparse to dense')
            M_seen = M_seen.toarray()
            if self.X_unseen_ is not None:
                self.X_unseen_ = self.X_unseen_.toarray()
        else:
            M_seen = M_seen.tocsr()
            if self.X_unseen_ is not None:
                self.X_unseen_ = self.X_unseen_.tocsr()

        if self.return_float:
            M_seen = M_seen.astype(float)
            if self.X_unseen_ is not None:
                self.X_unseen_ = self.X_unseen_.astype(float)
        if return_unseen:
            return M_seen, self.X_unseen_
        self.transform_time_ = (time() - start) / len(X)  # .shape[0]
        self._print(f"Sparse Transform time: {sigfig.round(self.transform_time_, sigfigs=3)} s/mol")
        return M_seen

    def fit_transform(self, X, y=None, fit_params=None):
        """Fit to iterable of molecules, then transform it return a matrix of fingerprints

        Args:
            X: an iterable of RDKit molecules
            y: exists for compatibility with sklearn pipelines, does nothing
            fit_params: exists for compatibility with sklearn pipelines, does nothing

        Returns:
            X (np.ndarray or sp.sparse.matrix of int or float): a matrix, format depends on self.return_sparse and
                                                                return_float
            X_unseen (optional): like X, but for fragments not identified during fit
        """

        self._in_fit_transform = True
        self.fit(X, y)
        out = self.transform(X)
        self._in_fit_transform = False
        self._cache = {}
        return out
   
    def _print(self, msg): 
        if self.verbose: 
            print(msg)
            
    @staticmethod
    def _get_bit_indices_and_sizes(bit_ids):
        """Given a list of bit ids
        
        return 
            * dict mapping list elements to indexes (for constant time arg lookup)
            * np array  of size elements of the IDs
        """
        

        # this gives us constant time lookup of the bit IDS
        bit_indices = {v: i for i, v in enumerate(bit_ids)}

        # a list of bit sizes
        bit_sizes, _ = map(np.asarray, zip(*bit_ids))
        
        return bit_indices, bit_sizes
    
    def _get_fps(self, mols):
        """Applies self._get_fp in parallel over molecules
        
        :return: Tuple[List[Dict], List[Dict], List[List[Tuple]]]
        
                 The elemenets of these tuple are: 
                     * List of fingerprints 
                     * List of bit maps (map bit IDs to to lists of atoms) 
                     * List of bit IDs
        """
        if self.chunk_size == 'even':
            batch_size = misc.evenly_distribute_jobs(len(mols), self.n_jobs)
        else: 
            batch_size = self.chunk_size
        p = Parallel(n_jobs=self.n_jobs, 
                     verbose=self.verbose,
                     prefer='processes',
                     batch_size=batch_size,
                     return_as='generator',
                    )
        f = delayed(self._get_fp) 
        
        fps = tqdm(p(f(mol,
                  self.fingerprinter,
                 )
                for mol in mols), desc="Constructing Fingerprints", total=len(mols))
        (fps, 
         bis, 
         bits,
        ) = list(zip(*fps))
        return fps, bis, bits #, metal_bits, metal_sizes        
        
    @staticmethod
    def _get_fp(mol, 
                fingerprinter, 
               ):
        """Calls the fingerprinter on an individual molecule, returning 
        
        For an rdkit.Molecule, return a fingerprint and a list of tuples of form (bit, r)
        """        
        fp, bi = fingerprinter(mol)
        bits = list(fp.keys())
        return fp, bi, bits
    
    def _to_sparse(self, fps, used_bits):
        """Convert a list of morgan fingerprints (represented as dicts) to a sparse matrix 
        of shape (len(fps), self.n_bits)
        :param fps: List[Dict] 
        :param used_bits: the unique bits present in fps
        """
        
        M = sp.sparse.dok_matrix((len(fps), len(used_bits)), dtype=int)
        for i, fp in tqdm(enumerate(fps), 
                          'Converting FPs to sparse', 
                          total=len(fps)): 
            for bit, count in fp.items(): 
                if bit in used_bits.keys():
                    j = used_bits[bit]
                    M[(i, j)] = count
                    
        return M

class SparseFeatureEliminator(BaseEstimator, TransformerMixin): 

    def __init__(self, overflow_action='warn', verbose=False): 
        self.overflow_action = overflow_action
        self.verbose = verbose

    def fit(self, X, y=None): 
        if self.verbose: 
            print('sparse feature elim')
        self.useful_, self.mapr_ = map_useless_features_sparse(
                                        X, 
                                        overflow_action=self.overflow_action
                                    )
        self.n_useful_ = self.useful_.sum()
        self.frac_useful_ = self.useful_.mean()
        return self
    
    def transform(self, X): 
        return X[:, self.useful_]
