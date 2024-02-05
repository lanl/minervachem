#/bin/python
"""
This script is meant for preprocessing the QM9 dataset:

Ramakrishnan, R., Dral, P., Rupp, M. et al. Quantum chemistry structures and properties of 134 kilo molecules. Sci Data 1, 140022 (2014). https://doi.org/10.1038/sdata.2014.22


run this script using:
    python process_qm9.py
To run this script you will need to download a few things from the QM9 dataset, and place them in a new folder `/demos/qm9_data_files/`. The items needed are:

1) `qm9_data_files/dsgdb9nsd.xyz.tar.bz2` - The main file, a tarball which contains inside it the properties for each molecule.
2) `qm9_data_files/uncharacterized.txt` - A list of molecules which failed a consistency check in the data generation process. These are usually excluded from machine learning analyses.
3) `qm9_data_files/atomref.txt` - Reference atom values for different types of energy calculations.

These files can be found at: https://doi.org/10.6084/m9.figshare.978904

Run this script using the command:

    python preprocess_qm9.py

The script takes approximately 1 minute to run. When complete, you should now have a file called `/demos/qm9_processed.csv`.

"""
import tarfile
import pandas as pd
import numpy as np
from rdkit.Chem import MolFromSmiles, AddHs
from tqdm.auto import tqdm
import re
tqdm.pandas()

HARTREE_TO_KCAL = 627.50947406

def read_qm9_file(tardata, file):
    """Extracts computed quantities and smiles string from single tarfile element in qm9 archive"""
    with tardata.extractfile(file) as data: 
        lines = data.readlines()
    vals = (lines[1].decode()
        .replace('gdb ', '')
        .replace('\t\n', '')
        .split('\t'))
    smiles = lines[-2].decode().split('\t')[0]
    vals.insert(0, smiles)
    return vals

def count_elements(mol):
    """Return a dict of CHNOF counts in an rdkit molecule"""
    counts = {s: 0 for s in 'CHNOF'}
    for atom in mol.GetAtoms(): 
        symbol = atom.GetSymbol()
        counts[symbol] += 1
    return dict(counts)

def read_qm9_txt_row(row): 
    return re.split('[\ ]{2,}', row.strip(' #\n'))

def read_qm9_txt_file(file, header_row=2, data_start_row=5): 
    with open(file, 'r') as f: 
        lines = f.readlines()
    headers = read_qm9_txt_row(lines[header_row])
    data = [read_qm9_txt_row(r) for r in lines[data_start_row:-1]]
    return pd.DataFrame.from_records(data, columns=headers)



qm9_colnames = ['smiles', 
            'i', 
            'A', 
            'B', 
            'C', 
            'mu', 
            'alpha', 
            'e_homo', 
            'e_lumo', 
            'e_gap', 
            'E_R^2', 
            'zpve', 
            'U_0', 
            'U',
            'H', 
            'G',
            'C_v']

qm9_dtypes = {'smiles': str,
          'i': int, 
          'A': float, 
          'B': float, 
          'C': float, 
          'mu': float, 
          'alpha': float, 
          'e_homo': float, 
          'e_lumo': float, 
          'e_gap': float, 
          'E_R^2': float, 
          'zpve': float, 
          'U_0': float, 
          'U': float, 
          'H': float, 
          'G': float, 
          'C_v': float, 
}

if __name__ == "__main__":
    data_dir = './qm9_data_files'
    # Read every file in the tar archive
    with tarfile.open(f'{data_dir}/dsgdb9nsd.xyz.tar.bz2', 'r') as tardata: 
        files = tardata.getmembers()
        qm9 = [read_qm9_file(tardata, f) for f in tqdm(files, 'reading qm9 files')]  

    qm9_df = (pd.DataFrame.from_records(qm9, 
                                        columns=qm9_colnames, 
                                        coerce_float=True)
            .astype(qm9_dtypes))

    # convert total energies to kcal/mol
    qm9_df['U_0'] *= HARTREE_TO_KCAL
    qm9_df['U'] *= HARTREE_TO_KCAL

    # Compute atomization energies
    ## get rdkit mols to count elements
    qm9_df['mol'] = qm9_df.smiles.map(MolFromSmiles)
    qm9_df['mol_H'] = qm9_df.mol.map(AddHs) # need explicit Hs
    ## count the elements
    print('Calculating E_at')
    elem_counts = qm9_df['mol_H'].progress_map(count_elements)
    elem_counts = pd.json_normalize(elem_counts)
    ## Get self energies for each element
    atomref = read_qm9_txt_file(f'{data_dir}/atomref.txt').rename({'Ele-': 'Element'}, axis=1)
    ## convert to kcal/mol
    atomref['U (0 K)'] = atomref['U (0 K)'].astype(float) * HARTREE_TO_KCAL
    ## cast to np arrays
    elem_counts_a = elem_counts[atomref.Element].to_numpy() # ensure element orders match
    E_self_U0_a   = atomref['U (0 K)'].to_numpy()
    ## use self energies to compute atomization energy
    qm9_df['E_self_sum'] = elem_counts_a @ E_self_U0_a
    qm9_df['E_at'] = qm9_df['U_0'] - qm9_df['E_self_sum']
    qm9_df.set_index('i', inplace=True)
    # Drop molecules with bad geometries accorrding to QM9 authors
    drop_mols = read_qm9_txt_file(f'{data_dir}/uncharacterized.txt', header_row=7, data_start_row=9)
    drop_mols['Index'] = drop_mols['Index'].astype(int)
    drop_mask = qm9_df.index.isin(drop_mols['Index'])
    print("Dropping uncharacterized molecules:")
    print(qm9_df[drop_mask]['smiles'])
    qm9_df = qm9_df.iloc[~drop_mask, :]

    # save
    qm9_df.drop(['mol', 'mol_H'], axis=1).to_csv('qm9_processed.csv')
    print("All done!")
