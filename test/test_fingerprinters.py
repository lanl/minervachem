import pytest

from rdkit.Chem import MolFromSmiles, AddHs

from minervachem.fingerprinters import MorganFingerprinter, RDKitFingerprinter, GraphletFingerprinter
from minervachem.utils.misc import counts_by_bit_by_size

@pytest.mark.parametrize('fingerprint_cls,smiles,use_Hs,fingerprinter_kws,expected',
                            [(MorganFingerprinter, 'C', False, {'radius': 1}, {0: (1,)}),
                            (MorganFingerprinter, 'C', True, {'radius': 1}, {0: (1, 4), 1: (1, 4)}),
                            (MorganFingerprinter, 'CC=C', False, {'radius': 2}, {1: (1, 1, 1), 0: (1, 1, 1)}),
                            (MorganFingerprinter, 'CC=C', True, {'radius': 2}, {2: (1, 1, 1), 1: (1, 1, 1, 1, 2, 3), 0: (1, 1, 1, 6)}),
                            (RDKitFingerprinter, 'C', False, {'maxPath': 4}, {}),
                            (RDKitFingerprinter, 'C', True, {'maxPath': 4}, {4: (1,), 2: (6,), 3: (4,), 1: (4,)}),
                            (RDKitFingerprinter, 'CC=C', False, {'maxPath': 2}, {1: (1, 1), 2: (1,)}),
                            (RDKitFingerprinter, 'CC=C', True, {'maxPath': 2}, {2: (1, 3, 4, 4), 1: (1, 1, 6)}),
                            (GraphletFingerprinter, 'C', False, {'max_len': 5, 'useHs': False}, {1: (1,)}),
                            (GraphletFingerprinter, 'C', True, {'max_len': 5, 'useHs': True}, {2: (4,), 3: (6,), 4: (4,), 1: (1, 4), 5: (1,)}),
                            (GraphletFingerprinter, 'CC=C', False, {'max_len': 3, 'useHs': False}, {3: (1,), 2: (1, 1), 1: (1, 1, 1)}),
                            (GraphletFingerprinter, 'CC=C', True, {'max_len': 3, 'useHs': True}, {3: (1, 3, 4, 4), 2: (1, 1, 6), 1: (3, 6)}),
                            ])
def test_fingerprinter(fingerprint_cls, smiles, use_Hs, fingerprinter_kws,expected):
    """Test that a fingerprinter producs the right counts of each fragment by size. doesnt assume hash is constant across runs"""
    fingerprinter = fingerprint_cls(**fingerprinter_kws)
    mol = MolFromSmiles(smiles)
    if use_Hs: 
        mol = AddHs(mol)
    _, bi = fingerprinter(mol)
    counts = counts_by_bit_by_size(bi)
    assert counts == expected
