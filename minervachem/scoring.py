import scipy as sp
import sklearn
from pprint import pprint

def prediction_metrics(true, pred, verbose=0):
    """scores in a dict"""
    r = sp.stats.pearsonr(true, pred)
    rho = sp.stats.spearmanr(true, pred)
    scores = {"MAE": sklearn.metrics.mean_absolute_error(true, pred),
              "RMSE": sklearn.metrics.mean_squared_error(true, pred, squared=False),
              "R^2": sklearn.metrics.r2_score(true, pred),
              "pearson-r": r[0],
              "pearson-p": r[1],
              "spearman-r": rho[0],
              "spearman-p": r[1]
             }
    if verbose:
        pprint({k: f"{v:0.4f}" for k, v in scores.items()}, width=1)
        print()
    return scores

def corrcoefs(x, y):
    r = sp.stats.pearsonr(x, y)
    rho = sp.stats.spearmanr(x, y)
    return {
        "pearson_r": r[0],
        "pearson_p": r[1],
        "spearman_r": rho[0],
        "spearman_p": rho[1],
     }