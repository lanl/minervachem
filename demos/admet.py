"""

Example for fitting to the admet tasks from the Therapeutics Data Commons (TDC)

Note: Requires installation of the tdc package, which can be found at:
https://tdc.readthedocs.io/en/main/index.html

A preprint containing these minervachem results and explaining the method is available at:
https://doi.org/10.26434/chemrxiv-2024-r81c8
"Linear Graphlet Models for Accurate and Interpretable Cheminformatics"
M. Tynes et al, 2024
"""

from tdc.benchmark_group import admet_group

from sklearn.linear_model import Ridge


def fit_model(X_train, y_train, bit_sizes, seed):

    from minervachem.regressors import HierarchicalResidualModel
    import sklearn.model_selection
    import numpy as np

    base_model = Ridge(alpha=1e2, solver="sparse_cg", tol=1e-5)
    param_grid = {"alpha": 10.0 ** np.arange(-1, 9, 1 / 3)}  # a range of alpha values to search

    cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_search = sklearn.model_selection.GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        refit=True,
        verbose=1,
        n_jobs=-3,
        cv=cv,
    )

    hmodel = HierarchicalResidualModel(cv_search, verbose=True)

    hmodel.fit(X_train, y_train, levels=bit_sizes)
    return hmodel


def featurize_data(train, test):
    from rdkit.Chem import MolFromSmiles, AddHs

    lamb = lambda x: AddHs(MolFromSmiles(x))
    train["mol"] = train["Drug"].map(lamb)
    test["mol"] = test["Drug"].map(lamb)

    from minervachem.transformers import FingerprintFeaturizer
    import minervachem.fingerprinters as fps

    fingerprinter = FingerprintFeaturizer(
        fps.GraphletFingerprinter(max_len=7, useHs=True),
        n_jobs=-3,
        verbose=0,
        chunk_size="auto",
    )
    X_train = fingerprinter.fit_transform(train["mol"])
    X_test, X_unseen = fingerprinter.transform(test["mol"], return_unseen=True)
    return X_train, X_test, fingerprinter.bit_sizes_


if __name__ == "__main__":
    from tdc import metadata

    group = admet_group(path="data/")
    benchmark_names = [name for name in metadata.admet_metrics.keys() if metadata.admet_metrics.get(name, None) in ("mae", "spearman")]
    print(benchmark_names)
    all_results = {}
    for bname in benchmark_names:
        predictions_list = []
        print("running", bname)

        for seed in [1, 2, 3, 4, 5]:
            print("Running seed", seed)
            benchmark = group.get(bname)
            # all benchmark names in a benchmark group are stored in group.dataset_names
            predictions = {}
            name = benchmark["name"]
            train_val, test = benchmark["train_val"], benchmark["test"]
            train, valid = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)

            X_train, X_test, bit_sizes = featurize_data(train_val, test)
            y_train = train_val["Y"]

            model = fit_model(X_train, y_train, bit_sizes, seed)

            y_pred_test = model.predict(X_test)

            predictions[name] = y_pred_test
            predictions_list.append(predictions)

        results = group.evaluate_many(predictions_list)
        print(bname)
        print(results)
        all_results[bname] = results
    print("All results")
    print(all_results)
