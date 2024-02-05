from rdkit.Chem.rdmolops import UnfoldedRDKFingerprintCountBased

class RDKitFingerprinter():
    """A minervachem fingerprinter for rdkit fingerprints

    Currently only supports unfolded count-based fingerprints. 

    The arguments are passed directly to UnfoldedRDKFingerprintCountBased

    Note that the size component of the bit IDs correspond to the number of bonds present in the fragment
    """
    
    def __init__(self, 
                 minPath=1, 
                 maxPath=7, 
                 useHs=True, 
                 branchedPaths=True, 
                 useBondOrder=True, 
                 atomInvariants=0, 
                 fromAtoms=0, 
                 atomBits=None): 
        
        self.minPath = minPath
        self.size = self.maxPath = maxPath
        self.useHs = useHs
        self.branchedPaths = branchedPaths
        self.useBondOrder = useBondOrder
        self.atomInvariants = atomInvariants
        self.fromAtoms = fromAtoms
        self.atomBits = atomBits
        
    def __call__(self, mol):
        """Fingerprint a molecule with rdkit fingerprints 
        
        :param mol: an RDKit Molecule
        :return fp: a dict mapping substructure IDs to counts
        :return bi: a dict mapping substructure IDs to a list of lists, where the inner lists are the set of bonds for a given occurence of the substructure
        """
        
        bi = {}
        fp = UnfoldedRDKFingerprintCountBased(mol,
                                              bitInfo=bi, 
                                              minPath=self.minPath,
                                              maxPath=self.maxPath,
                                              useHs=self.useHs,
                                              branchedPaths=self.branchedPaths,
                                              useBondOrder=self.useBondOrder,
                                              atomInvariants=self.atomInvariants,
                                              fromAtoms=self.fromAtoms,
                                              atomBits=self.atomBits,
                                             ).GetNonzeroElements()
        
        fp, bi = [self._append_lengths(d, bi) for d in [fp, bi]]
        return fp, bi
    
    @staticmethod
    def _append_lengths(d, bi): 
        return {(len(bi[k][0]), k): v for k, v in d.items()}
