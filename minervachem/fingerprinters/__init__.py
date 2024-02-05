"""Fingerprinters: a set of callables for creating molecular fingerprints

Fingerprinters are instantiated with parameters and then called on molecules. 
Once called they return two dictionaries, one mapping from substructures IDs to substructure counts, 
and the other mapping from substructure IDs to a list of lists containing the atomIDs 
of the satisfying substructures. We commonly call the first dict `fp` and the second `bi` (for bitInfo)
to keep with RDKit convention. 

An important note is that the fingerprint bit IDs in minervachem are a tuple of the substructure's hash and its size. 
This is convenient for DAG construction and hierarchical model fitting. The meaning of size differs across the
provided fingerprinters. 

Available fingerprinters: 
* GraphletFingerprinter 
* RDKitFingerprinter (calls RDKit's rdkit fingerprint routines)
* MorganFingerprinter (calls RDKit's Morgan fingerprint routines)
"""

from .morganfingerprinter import MorganFingerprinter
from .rdkitfingerprinter import RDKitFingerprinter
from .graphletfingerprinter import GraphletFingerprinter
