from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from minervachem.fingerprinters import GraphletFingerprinter
from minervachem.transformers import FingerprintFeaturizer
from sklearn.model_selection import train_test_split

def create_pipeline(df):
    """creates pipeline object of the MinervaChem ridge model.

    Args:
        df (pandas dataframe): a dataframe of qm9 molecules that have been processed with the /demos/preprocess_qm9.py

    Returns:
        pipeline: model pipeline object
    """

    train, test = train_test_split(
        df, 
        train_size=0.8, 
        random_state=42,
    )
    
    y_train, _ = [sub_df['E_at'] for sub_df in [train, test]]


    pipeline = Pipeline(
        [
            ("featurizer", FingerprintFeaturizer(fingerprinter=GraphletFingerprinter(max_len=3, useHs=True),
                                                verbose=0,
                                                n_jobs=-3,
                                                chunk_size='auto',)
                                                ),
            ("ridge", Ridge(fit_intercept=False, 
                            alpha=1e-5, 
                            solver='sparse_cg', 
                            tol=1e-5)
                            )
        ]
    )

    pipeline.fit(train['mol'], y_train)

    return pipeline
