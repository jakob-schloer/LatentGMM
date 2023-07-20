"""Collection of metrics used in this project."""
import numpy as np
from sklearn import metrics

def cluster_performance(y_pred, y_target, metric='rand_index'):
    """Metrices for clustering performance given target."""
    assert y_pred.shape == y_target.shape
    if metric == 'rand_index':
        score = metrics.adjusted_rand_score(y_target, y_pred)
    elif metric == 'acc':
        score = np.mean(y_pred == y_target) * 100
    else:
        raise ValueError(f"Metric {metric} not defined!")
    
    return score


def wassertein_gaussians(mu_1, cov_1, mu_2, cov_2):
    """Wasserstein W2 loss between two Gaussians.
    
    d2 = |mu_1 - mu_2|^2 + |cov_1^1/2 - cov_2^1/2|^2
    
    Only valid under the assumption:
        cov_1 * cov_2 == cov_2 * cov_1

    Args:
        mu_1  (np.ndarray): Mean of Gaussian 1. Shape (n).
        cov_1 (np.ndarray): Covariance matrix of Gaussian 1. Shape (n,n).
        mu_2  (np.ndarray): Mean of Gaussian 2. Shape (n).
        cov_2 (np.ndarray): Covariance matrix of Gaussian 1. Shape (n,n).
    
    Reference:
        [1]: https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
    """
    d = np.linalg.norm(mu_1 - mu_2) \
        + np.linalg.norm( np.sqrt(cov_1) - np.sqrt(cov_2), ord='fro')
    
    return d


def fraction_unexplained_variance_partition(X, Y):
    """Fraction of unexplained variance. 

    Following definition by Monahan 2000:
        FUV = ( sum_M(var(X)) - sum_M (var(Y)) ) / sum_M(var(X))
    with M: number of features

    Args:
        X (np.ndarray): Original input data of shape (num_samples, num_features)
        Y (np.ndarray): Output of model, i.e. reconstruction of data
            of shape (num_samples, num_features)
    Returns:
        fuv (float): Fraction of unexplained variance
    """
    var_data = np.nansum( np.nanstd(X, axis=0) **2)
    fuv = ( var_data
            - np.nansum(np.nanstd(Y, axis=0) **2) ) / var_data

    return fuv


def fraction_unexplained_variance(X, Y):
    """Fraction of unexplained variance. 

    Following definition by Monahan 2000:
        FUV = ( sum_M(var(X)) - sum_M (var(Y)) ) / sum_M(var(X))
    with M: number of features

    Args:
        X (np.ndarray): Original input data of shape (num_samples, num_features)
        Y (np.ndarray): Output of model, i.e. reconstruction of data
            of shape (num_samples, num_features)
    Returns:
        fuv (float): Fraction of unexplained variance
    """
    var_data = np.nanstd(X) **2
    var_res =  np.nanstd(X-Y) **2
    fuv = var_res / var_data

    return fuv

    
    
    