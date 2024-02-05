from functools import lru_cache
import re

import networkx as nx
from collections import Counter, defaultdict
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import Mol
import rdkit


class GraphletFingerprinter():     
    def __init__(self, max_len=3, useHs=False):
        """A class to produce graphlet fingerprints

        The size component of the bit ID corresponds to the number of atoms in the graphlet. 
        
        :param max_len: int, the largest number of atoms to consider for induced subgraphs
        :param useHs: bool, whether to use explicit hydrogens 
        """
        
        self.size = self.max_len = max_len
        self.useHs = useHs
        
    def __call__(self, mol): 
        fp, bi = ComputeGraphletFingerprint(mol, 
                                           self.max_len, 
                                           self.useHs)
        return fp, bi
    


def akey_noH(atom):
    return (atom.GetAtomicNum(),atom.GetFormalCharge(),atom.GetIsAromatic(),atom.GetNumImplicitHs())

def akey_withH(atom):
    return (atom.GetAtomicNum(),atom.GetFormalCharge())
    
def mol_to_nx(mol,explicit_h=False):
    """
    Modified from:
     https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py#L17
    """
    G = nx.Graph()
    if explicit_h:
        mol=rdmolops.AddHs(mol)
        akey = akey_withH
    else:
        akey = akey_noH 
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atom_key=akey(atom))
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_key = bond.GetBondType())
    return G

def mol2_to_nx(mol2str,explicit_h=False):
    """mol2_to_nx 
    Take in a mol2string and return a networkx version of the molecular graph.
    By default uses heavy elements only.
    """
    G = nx.Graph()
    # ptable=PTable()
    s = mol2str.splitlines()
    read_atoms = False
    read_bonds = False
    atoms = []
    bonds = []
    hydrogen_inds = []
    non_hydrogen_inds = []
    for line in s:
        # Get Atoms First
        if '<TRIPOS>BOND' in line:
            read_atoms = False
        if '<TRIPOS>SUBSTRUCTURE' in line:
            read_bonds = False
            read_atoms = False
        if read_atoms:
            s_line = line.split()
            acount += 1
            atom_symbol1 = re.sub('[0-9]+[A-Z]+', '', line.split()[1])
            atom_symbol1 = re.sub('[0-9]+', '', atom_symbol1)
            atom_type = line.split()[5]
            atoms.append((atom_symbol1,atom_type))
            if atom_symbol1 == 'H':
                hydrogen_inds.append(acount)
            else:
                non_hydrogen_inds.append(acount)
        if read_bonds: 
            s_line = line.split()
            bonds.append((int(s_line[1]),
                   int(s_line[2]),
                   s_line[3],
                   (int(s_line[1]) in hydrogen_inds) or (int(s_line[2]) in hydrogen_inds))) # If bond has H at either end.
        if '<TRIPOS>ATOM' in line:
            read_atoms = True
            acount = 0
        if '<TRIPOS>BOND' in line:
            read_bonds = True
    acount = 0
    # Translate to nx
    for i,atom in enumerate(atoms):
        if explicit_h:
            acount +=1 
            G.add_node(acount,atom_key=atom)
        else:
            if atom[0] != 'H':
                acount += 1
                h_count = len([x for x in bonds if ((i+1 in x) and (x[3]))])
                new_atom = (atom[0],atom[1],h_count)
                G.add_node(acount,atom_key=new_atom)
    for bond in bonds:
        if explicit_h:
            G.add_edge(bond[0],bond[1],bond_key = bond[2])
        elif (not bond[3]):
            start_ind = non_hydrogen_inds.index(bond[0]) + 1
            end_ind = non_hydrogen_inds.index(bond[1]) + 1
            G.add_edge(start_ind,end_ind,bond_key = bond[2])
    return G

# version with nice notation.
# def active_neighbors(G,node,cur_neighbors,whitelist,yields_with):
#     # The next things to check will be the neighbors of this neighbor one,
#     # so long as they aren't already going to be checked!
#     possible = cur_neighbors | set(G.neighbors(node))
#     allowed = (possible - yields_with) & whitelist
#     return allowed
def active_neighbors(G,node,cur_neighbors,whitelist,yields_with):
    """
    Things in current neighbors or neighbors of a given node,
    as long as in whitelist and not already in current cluster.
    """
    """note! this definition is inlined in search_subgraphs from cluster"""
    return {n for n in (*cur_neighbors,*G.neighbors(node))
                  if n in whitelist
                  and n not in yields_with}

def generate_subgraphs_from_node(G,node,depth,whitelist):
    """
    Generate subgraphs starting from a specific node.
    Essentially just sets up calculation for searching from cluster.
    """
    blacklist = ()
    yields_with = {node}
    neighbors = active_neighbors(G,node,set(),whitelist,yields_with)

    yield from search_subgraphs_from_cluster(G,depth,neighbors,yields_with,whitelist,set())

def search_subgraphs_from_cluster(G,depth,neighbors,yields_with,whitelist,found):
    "Generate subgraphs from a given cluster."
    # Depth-first type search
    f_yw = frozenset(yields_with)
    if f_yw in found:
        # Already seen this thing, ignore it
        return
    found.add(f_yw)

    yield yields_with
    if depth<=1: # stop recursion at some arbitrary integer.
        # Note: if you change this integer, you must change the hasher recursive call to generate_subgraphs!
        return

    # go deeper!

    for n in neighbors:
        # next_neighbors = active_neighbors(G, n, neighbors, whitelist, yields_with)
        # the above gets called enough that inlining actually helps.
        yields_with.add(n) # add to current set for yielding
        next_neighbors={n for n in (*neighbors, *G.neighbors(n))
         if n in whitelist
         and n not in yields_with}
        # inception!
        yield from search_subgraphs_from_cluster(G,depth-1,next_neighbors,yields_with,
                                                 whitelist,found)
        yields_with.remove(n) # remove from current set for yielding

    return
    

def generate_subgraphs(G,maxlen,hash_helper=None,whitelist=None):
    """
    G: networkx graph for molecule
    maxlen: size of
    hash_helper: used to compute hashes when this function is called recursively
    whitelist: subset of atoms in the molecule to consider.
    """
    # If hash_helper is None, we are generating fresh.
    # if hash_helper is not None, we are helping ourselves.

    retain_counts=(hash_helper is None)
    if retain_counts:
        hash_helper=HashHelper(G,maxlen)

    all_subsets = set()
    if whitelist is None:
        whitelist = set(G.nodes)
    else:
        whitelist = set(whitelist) # copies if set, removes 'frozen' if frozenset

    for n in list(whitelist): # changes during iteration
        for subset in generate_subgraphs_from_node(G,n,maxlen,whitelist):
            subset = frozenset(subset)
            h = hash_helper(subset)
            all_subsets.add((subset,h))         
            
        whitelist.remove(n) # now never look at that node again
    if retain_counts:
        res = Counter((len(ss),h) for ss,h in all_subsets)
    else:
        res = None
    return all_subsets,res

class HashHelper():
    def __init__(self,graph,maxlen):
        self.graph=graph
        self.maxlen=maxlen
        
    @lru_cache(maxsize=None) # very important, this is memoized.
    def __call__(self,indices):
        if len(indices)==1:
            # Just hash atom key
            a=list(indices)[0]
            akey = self.graph.nodes[a]['atom_key']
            h=hash(akey)
            return h
        if len(indices)==2:
            # Include bond key in hash
            a1,a2=indices
            h1, h2 = map(self,((a1,),(a2,)))
            btype = self.graph.edges[a1,a2]['bond_key']
            h3 = hash(btype)
            h = hash(tuple(sorted((h1,h2,h3))))
            return h
        
        # hash abd count of all substructures of this structure that are at most one smaller than this one.
        # Note that the his call passes itself back to generate_subgraphs to increase the efficiency
        # of memoization.
        sub_graph_set,_ = generate_subgraphs(self.graph,len(indices)-1,hash_helper=self,whitelist=indices)
        sub_graph_set = [sg for sg, h in sub_graph_set]
        sub_hashes = Counter(self(idxs) for idxs in sub_graph_set)
        # combine hashes from substructures to form new hash for this.
        this_hashkey = tuple(sorted(sub_hashes.items()))
        h = hash(this_hashkey)

        return h
                                 
def ComputeGraphletFingerprint(rdkit_mol, maxlen, explicit_h):
    """RDKit style wrapper to generate_subgraphs but keyed by (size, bit)"""
    if isinstance(rdkit_mol, Mol):
        G = mol_to_nx(rdkit_mol, explicit_h=explicit_h)
    elif isinstance(rdkit_mol,str):
        if 'TRIPOS' in rdkit_mol:
            G = mol2_to_nx(rdkit_mol,explicit_h=explicit_h)
        else:
            raise ValueError('Unknown molecule type for featurizing')
    else:
        raise ValueError('Unknown molecule type for featurizing')
    subsets, counter = generate_subgraphs(G, maxlen)
    bit_info = defaultdict(lambda: [])
    for atoms, bit in subsets:
        subset = list(atoms)
        size = len(subset)
        bit_info[(size, bit+2**64)].append(subset)
    fp = {(k[0], k[1]+2**64): v for k, v in counter.items()}
    return fp, dict(bit_info)
