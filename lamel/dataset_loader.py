import numpy as np
import pandas as pd

# Cheminformatics
from rdkit import Chem

# Custom
from minervachem import fingerprinters as fp
from minervachem import transformers as tf


class LoadedDataset:
    def __init__(self):
        self.data = pd.DataFrame()
        self.tasks = []

    def load(self, file_path):
        # detect format
        data_format = file_path.split('.')[-1]
        if data_format == 'csv':
            df = pd.read_csv(file_path)
        elif data_format == 'pkl':
            df = pd.read_pickle(file_path)
        elif data_format == 'xls' or data_format == 'xlsx':
            df = pd.read_excel(file_path)
        else:
            raise ValueError('Unrecognized dataset format')

        self._process_data(df)
        print(f'Available tasks: {self.tasks}')

    def _process_data(self, df):
        assert 'smiles' in [el.lower() for el in
                            df.columns.to_list()], "Molecules are required to have SMILES representation"
        if 'Smiles' in df.columns:
            df = df.rename(columns={'Smiles': 'SMILES'})
        elif 'smiles' in df.columns:
            df = df.rename(columns={'smiles': 'SMILES'})

        # removing unnamed index column (typical for the csv)
        for col_name in df.columns:
            if 'Unnamed' in col_name:
                df = df.drop(columns=[col_name])

        if self.data.empty:
            self.data = df
        else:  # we are adding new dataset from a new file
            new_smiles = set(df['SMILES']) - set(self.data['SMILES'])
            new_mols_df = df[df['SMILES'].isin(new_smiles)]
            old_mols_df = df[~df['SMILES'].isin(new_smiles)]

            self.data = pd.concat([self.data, new_mols_df], axis=0).reset_index(drop=True)

            temp_tasks = [pr for pr in df.columns.to_list() if pr != 'SMILES']
            for tsk in temp_tasks:
                for i in range(len(old_mols_df)):
                    ind = np.where(self.data['SMILES'] == old_mols_df['SMILES'].to_list()[i])[0][0]
                    self.data.at[ind, tsk] = old_mols_df[tsk].to_list()[i]

        self.tasks = [pr for pr in self.data.columns.to_list() if pr != 'SMILES']

    def get_fingerprint_descriptors(self, max_subgraph_size, n_jobs=-3, verbose=0):
        """
        The function turns SMILES strings into Graphlet fingerprints.
        :param data: pd.Dataframe. A Pandas dataframe for the dataset you are working on.
        :param max_subgraph_size: int. Size of the maximum subgraphs to use.
        :param n_jobs: int. Size of the parallel jobs. -3 means all n_cores-2 (joblib).
        :param verbose: int. Verbosity parameter.
        :return: scipy.sparse._csr.csr_matrix. A sparse matrix of fingerprints of size (n, m), where n is the number of
                molecules in the set, and m is the length of fingerprints.
        """
        assert 'SMILES' in self.data.columns, "There is no SMILES column in the dataframe"
        # adding hydrogen atoms
        self.data['mol'] = self.data['SMILES'].map(lambda s: Chem.AddHs(Chem.MolFromSmiles(s)))
        yfp = tf.FingerprintFeaturizer(
            fingerprinter=fp.GraphletFingerprinter(useHs=True, max_len=max_subgraph_size),
            n_jobs=n_jobs,
            verbose=verbose
        )
        return yfp.fit_transform(self.data['mol'])
