import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from tdc.single_pred import ADME
from rdkit.Chem import MolFromSmiles, AddHs
from minervachem.transformers import FingerprintFeaturizer
import minervachem.fingerprinters as fps
from copy import deepcopy

from minervachem.regressors import HierarchicalResidualModel


MAX_LEN=4



class TestHierarchicalRegressor:  
    @classmethod
    def setup_class(cls):
        data = ADME(name = 'Lipophilicity_AstraZeneca')
        split = data.get_split(method = 'scaffold',seed=42)
        train, valid, test = split['train'], split['valid'], split['test']
        train = pd.concat([train, valid])
        train['mol'] = train['Drug'].map(lambda s: AddHs(MolFromSmiles(s)))
        test['mol'] = test['Drug'].map(lambda s: AddHs(MolFromSmiles(s)))
        cls.fingerprinter = FingerprintFeaturizer(
            fps.GraphletFingerprinter(
                max_len=MAX_LEN, 
                useHs=True 
            ),
            n_jobs=1,
            verbose=1,
            chunk_size='auto',
        )
        cls.X_train = cls.fingerprinter.fit_transform(train['mol'])
        cls.X_test = cls.fingerprinter.transform(test['mol'])
        cls.y_train, cls.y_test = map(lambda df: df['Y'].to_numpy(), [train, test])


    @pytest.mark.parametrize('cv,repl_method',
                            [(False, 'implicit'),
                            (False, 'list'),
                            (False, 'dict'), 
                            (True, 'implicit'),
                            (True, 'list'), 
                            (True, 'dict')])
    def test_hierarchical_residual_model(self, cv, repl_method): 

        model = build_model(cv, repl_method)
        model = HierarchicalResidualModel(model)
        model.fit(self.X_train, self.y_train, levels=self.fingerprinter.bit_sizes_)
        y_test_hat = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_test_hat)
        assert r2 > 0.5 # we should do at least this well always

def build_model(cv, repl_method):
    """
    cv = t/f
    repl_method = either dict, list, or implicit

    we want to exercise all of these to make sure the model behaves as expected
    """
    model = Ridge(alpha=1e2,solver='sparse_cg',tol=1e-5)

    if cv:
        param_grid = {'alpha':10.**np.arange(-1,9,1/3)} # a range of alpha values to search
        model = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            refit=True,
                            verbose=1,
                            n_jobs=1)
    if repl_method == 'list': 
        model = [deepcopy(model) for i in range(MAX_LEN)] # todo: make these not trivial copies
    elif repl_method == 'dict': 
        model = {i+1: deepcopy(model) for i in range(MAX_LEN)}
    # else if its implicit, leave it up to the hrm class

    return model
