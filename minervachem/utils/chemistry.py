import re
import ast

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdchem


def add_metal_center(mol, smicat, center): 
    """Add a metal center to a molecule at smicat"""
    emol = rdchem.EditableMol(mol)
    center = Chem.Atom(center)
    center_ix = emol.AddAtom(center)
    for i in smicat: 
        emol.AddBond(i-1, center_ix, rdchem.BondType.DATIVE)
    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def read_complex_mol2(s):
    """Reads a mol2 file of a metal ligand complex and returns a mol object
    
    Ignores explicit hydrogens, represents coordinating bonds as "ZERO" order bonds
    
    if .mol2 is not in s, assumes it is passed a string with lines delimited by \n
    """
    ## READ MOl2 FILE
    file = s.endswith('.mol2')
    template_str = '@<TRIPOS>{}'
    if file:   
        with open(f, 'r') as f: 
            l = f.readlines()
        template_str+='\n'
    else:
        if '\n' in s: 
            l = s.split('\n')
        else: 
            l = ast.literal_eval(s)
            

    a = np.asarray(l)
    
    ## pull out relevant parts of file
    atom_ix, bond_ix, struct_ix = [int(np.where(a == template_str.format(s))[0])
                                  for s in ['ATOM', 'BOND', 'SUBSTRUCTURE']]
    atoms = l[atom_ix+1:bond_ix]
    bonds = l[bond_ix+1:struct_ix]
    
    
    symbols = [] # tracks the heavy atom symbols
    imap = {}    # maps from the mol2 ix to the rdkit heavy atom ix
    mol = Chem.EditableMol(Chem.Mol())
    
    H_ix = []    # tracks hydrogen index
    atom_rows = [re.sub('\W+', 
                     ',', 
                     s_atom.strip()
                     ).split(',') for s_atom in atoms]
    for row in atom_rows: 
        atom_ix = int(row[0])
        symbol = row[1]
        symbol = re.sub('\d', '', symbol)
        if symbol != 'H': 
            imap[atom_ix] = len(symbols)
            symbols.append(symbol)
            at = Chem.Atom(symbol)
            #at.SetNoImplicit(True)
            mol.AddAtom(at)
        else: 
            H_ix.append(atom_ix)

    ## ADD BONDS
    bond_orders = {
        1: Chem.rdchem.BondType.SINGLE, 
        2: Chem.rdchem.BondType.DOUBLE, 
        3: Chem.rdchem.BondType.TRIPLE, 
    }
    for s_bond in bonds: 
        bond_id, start, stop, order = map(int, 
            re.sub('\W+', 
                   ',',
                   s_bond.strip()
                  ).split(','))
        order = bond_orders[order]
        has_hydrogen = ((start in H_ix) or (stop in H_ix))
        if not has_hydrogen:
            metal_bond = (start == 1) or (stop == 1) # one indexed!
            if metal_bond: 
                order = Chem.rdchem.BondType.ZERO
            start = imap[start]
            stop = imap[stop]
            mol.AddBond(start, stop, order)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol

