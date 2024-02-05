from rdkit.Chem import AllChem

class MorganFingerprinter:
    """A minervachem wrapper to Morgan fingerprints from RDKit

    Note that here, the size component of the bit ID corresponds to radius. 

    :param radius: the maximum radius used for FP construction
    :use counts: indicator for count-based or boolean fingerprints
    :n_bits: if None, do not hash, else hash to the given size
    """
    
    def __init__(self, radius, use_counts=True, n_bits=None): 
        self.radius = radius
        self.use_counts = use_counts
        self.n_bits = n_bits
        self.size = self.radius
        
    def __call__(self, mol): 
        bi = {}
        if self.n_bits: 
            fp = self._hashed_fp(mol, bi)
        else: 
            fp = self._unhashed_fp(mol, bi)
            
        
        fp, bi = [self._append_radii(d, bi) for d in [fp, bi]]                           
        return fp, bi
    
    @staticmethod
    def _append_radii(d, bi): 
        return {(bi[k][0][1], k): v for k, v in d.items()}
        
    def _hashed_fp(self, mol, bi): 
        if self.use_counts:
            fp = (AllChem.GetHashedMorganFingerprint(mol,
                                                    self.radius,
                                                    nBits=self.n_bits,
                                                    bitInfo=bi)
                  .GetNonzeroElements())
        else: 
            fp = (AllChem.GetMorganFingerprintAsBitVect(mol,
                                                        self.radius,
                                                        nBits=self.n_bits,
                                                        bitInfo=bi)
                  .GetOnBits())
            fp = {b: 1 for b in list(fp)}
        
        return fp
    
    def _unhashed_fp(self, mol, bi):
        fp = AllChem.GetMorganFingerprint(mol, 
                                          self.radius, 
                                          bitInfo=bi, 
                                          useCounts=self.use_counts)
        return fp.GetNonzeroElements()
    