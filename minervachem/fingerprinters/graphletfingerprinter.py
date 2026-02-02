from functools import lru_cache
import re
import hashlib

import networkx as nx
from collections import Counter, defaultdict
from rdkit.Chem import rdmolops, RWMol
from rdkit.Chem.rdchem import Mol
import rdkit

class GraphletFingerprinter():     
    def __init__(self, max_len=3, useHs=False, elements=(), filter_in=True, terminal_pos=False):
        """A class to produce graphlet fingerprints

        The size component of the bit ID corresponds to the number of atoms in the graphlet. 
        
        :param max_len: int, the largest number of atoms to consider for induced subgraphs
        :param useHs: bool, whether to use explicit hydrogens
        :param elements: tuple of str, the elements to filter for
        :param filter_in: bool, True if substructures with filtered elements should be kept,
        False if substructures with filtered elements should be removed
        :param terminal_pos: bool, True if substructures with terminal atoms should be kept,
        """
        
        self.size = self.max_len = max_len
        self.useHs = useHs
        self.elements = elements
        self.filter_in = filter_in
        self.terminal_pos = terminal_pos
        self.verbose = False

        if len(self.elements)==0:
            print('No elements to filter. Complete fingerprints are returned.') #should only appear once at the initialization
        
    def __call__(self, mol): 
        fp, bi = ComputeGraphletFingerprint(mol, 
                                            self.max_len,
                                            self.useHs,
                                            self.elements,
                                            self.filter_in,
                                            self.terminal_pos,
                                            self.verbose)
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

def _guess_element(atom_name: str, atom_type: str | None) -> str | None:
    elements = ('X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 
                'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 
                'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 
                'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 
                'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 
                'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 
                'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og')
    if atom_type:
        t0 = atom_type.split('.')[0]
        cand = t0[0].upper() + (t0[1:].lower() if len(t0) > 1 else "")
        if cand in elements:
            return cand
    m = re.match(r"([A-Za-z]{1,2})", atom_name or "")
    if m:
        cand = m.group(1)[0].upper() + (m.group(1)[1:].lower() if len(m.group(1)) == 2 else "")
        if cand in elements:
            return cand
    return None

def _bond_order_and_flags(bond_type: str | None):
    if not bond_type:
        return None, False
    bt = bond_type.lower()
    aromatic = (bt == "ar")
    if bt in ("1", "single"):   order = 1
    elif bt in ("2", "double"): order = 2
    elif bt in ("3", "triple"): order = 3
    elif bt == "ar":            order = 1.5
    elif bt == "am":            order = 1 # amide single
    elif bt == "un":            order = 0.5 # un should mostly be dative
    else:                       order = None
    return order, aromatic

def mol2_to_nx(mol2str, explicit_h=False):
    """mol2_to_nx 
    Take in a mol2string and return a networkx version of the molecular graph.
    By default uses heavy elements only.

    Node attrs (explicit_h=True):
        - name, element, sybyl_type, x,y,z, charge, subst_id, subst_name
        - atom_key = (symbol, atom_type)
        - original_index (1-based from MOL2)

    Node attrs (explicit_h=False; hydrogens collapsed):
        - all of the above EXCEPT element/name for removed H nodes
        - h_count (# directly bonded H)
        - atom_key = (symbol, atom_type, h_count)
        - original_index maps to original heavy-atom id list (1-based indices of heavy atoms)

    Edge attrs (both modes):
        - order (float|None), aromatic (bool), raw_type (str), bond_key (==raw_type)
        - has_h (bool)  # True if either endpoint in the original MOL2 bond was a hydrogen
        - original_indices = (i1, j1)  # original 1-based atom ids
    """
    atoms_tmp = {}  # id1 -> dict
    bonds_tmp = []  # (i1, j1, raw_type)

    s = mol2str.splitlines() # mol2 are expected to be small, can be replaced with io.StringIO(mol2str) for more efficiency
    in_atoms = in_bonds = False
    for line in s:
        if not line:
            continue
        if line.startswith("@<TRIPOS>") or line.startswith("<TRIPOS>"):
            tag = line.replace("@", "").upper()
            in_atoms = (tag == "<TRIPOS>ATOM")
            in_bonds = (tag == "<TRIPOS>BOND")
            continue
        if in_atoms:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                atom_id = int(parts[0])
                name = parts[1]
                x, y, z = map(float, parts[2:5])
                sybyl_type = parts[5]
                subst_id = int(parts[6])
                subst_name = parts[7]
                charge = float(parts[8]) if len(parts) >= 9 else 0.0
            except Exception:
                continue
            elem = _guess_element(name, sybyl_type)
            atoms_tmp[atom_id] = dict(
                name=name, element=elem, sybyl_type=sybyl_type,
                x=x, y=y, z=z, charge=charge,
                subst_id=subst_id, subst_name=subst_name,
                original_index=atom_id
            )
        elif in_bonds:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                i1 = int(parts[1]); j1 = int(parts[2]); btype = parts[3]
            except Exception:
                continue
            bonds_tmp.append((i1, j1, btype))

    is_h = {aid: (atoms_tmp[aid].get("element") == "H") for aid in atoms_tmp}
    heavy_ids = [aid for aid in sorted(atoms_tmp) if not is_h[aid]]
    heavy_reindex = {aid: idx for idx, aid in enumerate(heavy_ids, start=1)}  # 1..N

    h_count_map = {aid: 0 for aid in atoms_tmp}
    for i1, j1, _ in bonds_tmp:
        if is_h.get(i1, False) and not is_h.get(j1, False):
            h_count_map[j1] += 1
        elif is_h.get(j1, False) and not is_h.get(i1, False):
            h_count_map[i1] += 1

    G = nx.Graph()

    if explicit_h:
        # atoms with original ordering for node ids (1..N)
        for aid in sorted(atoms_tmp):
            a = atoms_tmp[aid]
            atom_key = (a["element"] or "?", a["sybyl_type"])
            G.add_node(
                aid,
                name=a["name"], element=a["element"], sybyl_type=a["sybyl_type"],
                x=a["x"], y=a["y"], z=a["z"], charge=a["charge"],
                subst_id=a["subst_id"], subst_name=a["subst_name"],
                original_index=a["original_index"],
                # "compat" attr
                atom_key=atom_key
            )
        for i1, j1, raw in bonds_tmp:
            order, aromatic = _bond_order_and_flags(raw)
            G.add_edge(
                i1, j1,
                order=order, aromatic=aromatic,
                raw_type=raw, bond_key=raw,
                has_h=is_h.get(i1, False) or is_h.get(j1, False),
                original_indices=(i1, j1)
            )
    else:
        # only heavy atoms become nodes, reindexed 1..N
        for aid in heavy_ids:
            a = atoms_tmp[aid]
            hc = h_count_map.get(aid, 0)
            atom_key = (a["element"] or "?", a["sybyl_type"], hc)
            new_id = heavy_reindex[aid]
            G.add_node(
                new_id,
                name=a["name"], element=a["element"], sybyl_type=a["sybyl_type"],
                x=a["x"], y=a["y"], z=a["z"], charge=a["charge"],
                subst_id=a["subst_id"], subst_name=a["subst_name"],
                h_count=hc, original_index=aid,
                atom_key=atom_key
            )
        for i1, j1, raw in bonds_tmp:
            if is_h.get(i1, False) or is_h.get(j1, False):
                continue
            i_new = heavy_reindex[i1]; j_new = heavy_reindex[j1]
            order, aromatic = _bond_order_and_flags(raw)
            G.add_edge(
                i_new, j_new,
                order=order, aromatic=aromatic,
                raw_type=raw, bond_key=raw,
                has_h=False,
                original_indices=(i1, j1)
            )
    G.graph["explicit_h"] = explicit_h
    G.graph["heavy_reindex"] = heavy_reindex if not explicit_h else None
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

def flatten_list(xss):
    return [x for xs in xss for x in xs]

class HashHelper():
    def __init__(self, graph, maxlen):
        self.graph=graph
        self.maxlen=maxlen

    @staticmethod
    def custom_hash(obj):
        s = str(obj).encode()
        hash_int = int(hashlib.sha1(s).hexdigest(), 16)
        return hash_int

    @lru_cache(maxsize=None) # very important, this is memoized.
    def __call__(self,indices):

        if len(indices)==1:
            # Just hash atom key
            a=list(indices)[0]
            akey = self.graph.nodes[a]['atom_key']
            h = self.custom_hash(akey)
            return h
        if len(indices)==2:
            # Include bond key in hash
            a1,a2=indices
            h1, h2 = map(self,((a1,),(a2,)))
            btype = self.graph.edges[a1,a2]['bond_key']
            h3 = self.custom_hash(btype)
            h = self.custom_hash(tuple(sorted((h1, h2, h3))))
            return h
        
        # hash abd count of all substructures of this structure that are at most one smaller than this one.
        # Note that the his call passes itself back to generate_subgraphs to increase the efficiency
        # of memoization.
        sub_graph_set,_ = generate_subgraphs(self.graph,len(indices)-1,hash_helper=self,whitelist=indices)
        sub_graph_set = [sg for sg, h in sub_graph_set]
        sub_hashes = Counter(self(idxs) for idxs in sub_graph_set)
        # combine hashes from substructures to form new hash for this.
        this_hashkey = tuple(sorted(sub_hashes.items()))
        h = self.custom_hash(this_hashkey)

        return h
                                 
# def ComputeGraphletFingerprint(rdkit_mol, maxlen, explicit_h):
#     """RDKit style wrapper to generate_subgraphs but keyed by (size, bit)"""
#     if isinstance(rdkit_mol, Mol):
#         G = mol_to_nx(rdkit_mol, explicit_h=explicit_h)
#     elif isinstance(rdkit_mol,str):
#         if 'TRIPOS' in rdkit_mol:
#             G = mol2_to_nx(rdkit_mol,explicit_h=explicit_h)
#         else:
#             raise ValueError('Unknown molecule type for featurizing')
#     else:
#         raise ValueError('Unknown molecule type for featurizing')
#     subsets, counter = generate_subgraphs(G, maxlen)
#     bit_info = defaultdict(lambda: [])
#     for atoms, bit in subsets:
#         subset = list(atoms)
#         size = len(subset)
#         bit_info[(size, bit+2**64)].append(subset)
#     fp = {(k[0], k[1]+2**64): v for k, v in counter.items()}
#     return fp, dict(bit_info)

def ComputeGraphletFingerprint(rdkit_mol, maxlen, explicit_h, elements=(), filter_in=True, terminal_pos=False, verbose=False):
    """RDKit style wrapper to generate_subgraphs but keyed by (size, bit)"""
    if isinstance(rdkit_mol, Mol):
        G = mol_to_nx(rdkit_mol, explicit_h=explicit_h)
        el_inds = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() in elements] if elements else None
    elif isinstance(rdkit_mol, str):
        if 'TRIPOS' in rdkit_mol:
            G = mol2_to_nx(rdkit_mol, explicit_h=explicit_h)
            el_inds = [n for n, d in G.nodes(data=True) if d.get('element') in elements] if elements else None
        else:
            raise ValueError('Unknown molecule type for featurizing')
    elif isinstance(rdkit_mol, nx.classes.graph.Graph):
        G = rdkit_mol
        el_inds = [n for n, d in G.nodes(data=True) if d.get('element') in elements] if elements else None
    else:
        raise ValueError('Unknown molecule type for featurizing')
    subsets, counter = generate_subgraphs(G, maxlen)
    bit_info = defaultdict(lambda: [])
    for atoms, bit in subsets:
        subset = list(atoms)
        size = len(subset)
        bit_info[(size, bit + 2 ** 64)].append(subset)
    fp = {(k[0], k[1] + 2 ** 64): v for k, v in counter.items()}
    bit_info = dict(bit_info)

    if len(elements)==0:
        return fp, bit_info
    else:
        # el_inds = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() in elements]
        if len(el_inds)==0:
            if len(elements)!=0:
                if verbose:
                    print("Elements to filter for not found OR incorrect element symbols are entered. Empty FP is returned")
                for k in bit_info.keys():
                    bit_info[k] = []
                    fp[k] = 0
        else:
            for k, v in bit_info.items():
                remove_flag = True
                if not filter_in:
                    remove_flag = False
                for bit_atom in flatten_list(v):
                    if bit_atom in el_inds:
                        remove_flag = not remove_flag
                        break
                if remove_flag:
                    bit_info[k] = []
                    fp[k] = 0
        if terminal_pos: # Not working for network x yet
            assert filter_in, "The functionality currently works for filtering in only." # TODO make it work for filter out too
            for k, v in bit_info.items():
                if len(v)==0 or k[0]<=2:
                    continue

                rwmol_temp = RWMol(rdkit_mol)
                for atom in reversed(list(rwmol_temp.GetAtoms())):
                    if atom.GetIdx() not in v[0]:
                        rwmol_temp.RemoveAtom(atom.GetIdx())

                non_terminal = []
                for element in elements:
                    local_elements = [atom for atom in rwmol_temp.GetAtoms() if atom.GetSymbol()==element]
                    local_nbrs = [len(atom.GetNeighbors()) for atom in local_elements] #this should support multi metal element core
                    non_terminal.append(any(nbr_number > 1 for nbr_number in local_nbrs) if local_nbrs else True) # makes sure that if other element filtered for is not is the substructure it does not get flagged as terminal

                if all(non_terminal):
                    bit_info[k] = []
                    fp[k] = 0

        return fp, bit_info
