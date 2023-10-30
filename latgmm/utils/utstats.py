''' Util functions for statistical tests.

@Author  :   Jakob Schlör 
@Time    :   2022/07/10 18:30:17
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import numpy as np
import xarray as xr
from tqdm import tqdm
import scipy.stats as stats
from statsmodels.tsa.ar_model import AutoReg

from tqdm import tqdm
from joblib import Parallel, delayed



def linear_regression_params(X, y):
    """
    Compute regression parameter between xarray DataArray X and
    1D time series y using least square method.

        y = alpha + beta * X
        min ( || X * beta - y ||^2 ) -> X y = beta * X^2
    
    Returns:
        beta (xr.DataArray): Regression coefficient.
        alpha (xr.DataArray): Intercept.
    """
    # Center variables
    X_anom = X - X.mean(dim='time')
    y_anom = y - y.mean(dim='time')

    # Compute sums needed for regression coefficiens
    cov_Xy = (X_anom * y_anom).sum(dim='time')
    var_X = (X_anom ** 2).sum(dim='time')
    var_y = (y_anom ** 2).sum(dim='time')

    # Calculate regression coefficient (beta)
    beta = cov_Xy / var_X
    alpha = y.mean(dim='time') - beta * X.mean(dim='time')

    return beta, alpha

# Statistical tests for composite analysis
# ======================================================================================

def holm(pvals, alpha=0.05, corr_type="dunn"):
    """
    Returns indices of p-values using Holm's method for multiple testing.
    """
    n = len(pvals)
    sortidx = np.argsort(pvals)
    p_ = pvals[sortidx]
    j = np.arange(1, n + 1)
    if corr_type == "bonf":
        corr_factor = alpha / (n - j + 1)
    elif corr_type == "dunn":
        corr_factor = 1. - (1. - alpha) ** (1. / (n - j + 1))
    try:
        idx = np.where(p_ <= corr_factor)[0][-1]
        idx = sortidx[:idx]
    except IndexError:
        idx = []
    return idx


def effective_sample_size(X):
    """Compute effective sample size by fitting AR-process to each location in time-series.

        n_eff = n * (1-coeff) / (1+coeff)

    Args:
        X (xr.Dataarray): Spatio-temporal data.
        order (int, optional): Order of process. Defaults to 1.

    Returns:
        nobs_eff (xr.Dataarray): Effective sample size. 
    """
    X = X.stack(z=('lat', 'lon'))

    print("Fit AR1-process to each location to obtain the effective sample size.")
    arcoeff = []
    for loc in tqdm(X['z']):
        x_loc = X.sel(z=loc)

        if np.isnan(x_loc[0]):
            arcoeff.append(np.nan) 
        else:
            mod = AutoReg(x_loc.data, 1)
            res = mod.fit()
            arcoeff.append(res.params[0])

    arcoeff = xr.DataArray(data=np.array(arcoeff),
                           dims=['z'], coords=dict(z=X['z']))

    nobs_eff =  (len(X['time'])* (1-arcoeff) / (1 + arcoeff))
    return nobs_eff.unstack()


def fit_ar1(X, i):
    x_loc = X.sel(z=X['z'][i])
    if np.isnan(x_loc[0]):
        arcoeff = np.nan 
    else:
        mod = AutoReg(x_loc.data, 1)
        res = mod.fit()
        arcoeff = res.params[0]
    return arcoeff, i


def effective_sample_size_parallel(X):
    """Compute effective sample size by fitting AR-process to each location in time-series.

        n_eff = n * (1-coeff) / (1+coeff)

    Args:
        X (xr.Dataarray): Spatio-temporal data.
        order (int, optional): Order of process. Defaults to 1.

    Returns:
        nobs_eff (xr.Dataarray): Effective sample size. 
    """
    X = X.stack(z=('lat', 'lon'))
    print("Fit AR1-process to each location to obtain the effective sample size.")
    # Run in parallel 
    n_processes = len(X['z'])
    results = Parallel(n_jobs=8)(
        delayed(fit_ar1)(X, i)
        for i in tqdm(range(n_processes))
    )
    # Read results
    arcoeff = []
    ids = []
    for r in results:
        coef, i = r
        arcoeff.append(coef)
        ids.append(i)
    # Sort z dimension to avoid errors in reshape
    sort_idx = np.sort(ids)
    arcoeff = xr.DataArray(data=np.array(arcoeff)[sort_idx],
                           dims=['z'], coords=dict(z=X['z']))

    nobs_eff =  (len(X['time'])* (1-arcoeff) / (1 + arcoeff))
    return nobs_eff.unstack()


def ttest_field(X, Y, weights=None, serial_data=False):
    """Point-wise t-test between means of samples from two distributions to test against
    the null-hypothesis that their means are equal.

    Args:
        X (xr.Dataarray): Samples of first distribution.
        Y (xr.Dataarray): Samples of second distribution to test against.
        serial_data (bool): If data is serial use effective sample size. Defaults to False.
        weights (xr.Dataarray): Weights of each time point of X. Defaults to None.
        weight_threshold (float): Threshold on probability weights,
            only used if weights are set. Defaults to 0.3.
        
    Returns:
        statistics (xr.Dataarray): T-statistics
        pvalues (xr.Dataarray): P-values
    """

    if weights is not None:
        # Weighted mean
        X_weighted = X.weighted(weights)
        mean_x = X_weighted.mean(dim='time').stack(z=('lat', 'lon'))
        std_x = X_weighted.std(dim='time').stack(z=('lat', 'lon'))

        # Use threshold on weights to crop X for a realistic sample size
        ids = np.where(weights.data > 0.0)[0]
        X = X.isel(time=ids) 
    else:
        mean_x = X.mean(dim='time', skipna=True).stack(z=('lat', 'lon'))
        std_x = X.std(dim='time', skipna=True).stack(z=('lat', 'lon'))

    mean_y = Y.mean(dim='time', skipna=True).stack(z=('lat', 'lon'))
    std_y = Y.std(dim='time', skipna=True).stack(z=('lat', 'lon'))

    # Effective sample size
    if serial_data:
        nobs_x = effective_sample_size_parallel(X).stack(z=('lat', 'lon')).data 
        nobs_y = effective_sample_size_parallel(Y).stack(z=('lat', 'lon')).data
    else:
        nobs_x = len(X['time'])
        nobs_y = len(Y['time'])

    statistic, pvalues = stats.ttest_ind_from_stats(mean_x.data, std_x.data, nobs_x,
                                                    mean_y.data, std_y.data, nobs_y,
                                                    equal_var=False,
                                                    alternative='two-sided')
    # Convert to xarray
    statistic = xr.DataArray(data=statistic, coords=mean_x.coords)
    pvalues = xr.DataArray(data=pvalues, coords=mean_x.coords)

    return statistic.unstack(), pvalues.unstack()


def kstest_field(X, Y):
    """Point-wise 2 sample Kolmogorov Smirnov test with 
    the null-hypothesis that the distributions are equal.

    Args:
        X (xr.Dataarray): Samples of first distribution.
        Y (xr.Dataarray): Samples of second distribution to test against.
        weight_threshold (float): Threshold on probability weights,
            only used if weights are set. Defaults to 0.3.
        
    Returns:
        statistics (xr.Dataarray): T-statistics
        pvalues (xr.Dataarray): P-values
    """
    X = X.stack(z=('lat', 'lon'))
    Y = Y.stack(z=('lat', 'lon'))

    statistics = xr.zeros_like(X.isel(samples=0))
    pvalues = xr.zeros_like(X.isel(samples=0))
    for i in tqdm(range(len(X['z']))): 
        statistics[i], pvalues[i] = stats.ks_2samp(
            X.isel(z=i).data, Y.isel(z=i).data, alternative='two-sided'
        )
    
    return statistics.unstack(), pvalues.unstack()


def mc_test_field(X, Y, n_bootstrap=2000):
    """Monte Carlo based statistical significant test of composites analysis.

    Args:
        X (xr.Dataarray): Samples of first distribution.
        Y (xr.Dataarray): Samples of second distribution to test against.
        n_bootstrap (int, optional): Number of sampling using bootstrapping.
            Defaults to 2000.

    Returns:
        mean_x (xr.Dataarray): Mean field of X. 
        pvalues (xr.Dataarray): Pvalues at each location. 
    """
    mean_x = X.mean(dim='time', skipna=True).stack(z=('lat', 'lon'))
    nobs_x = len(X['time'])

    Y_flat = Y.stack(z=('lat', 'lon'))
    # Test distribution
    y_mean_sample = []
    for i in range(n_bootstrap):
        ids = np.random.randint(0, len(Y['time']), size=nobs_x)
        y_mean_sample.append(Y_flat[ids].mean('time'))

    y_mean_sample = xr.concat(y_mean_sample, dim='n_sample')

    # Get p-values by approximating the dist using kde
    pvalues = []
    for loc in y_mean_sample['z']:
        if np.isnan(y_mean_sample.sel(z=loc)[0]):
            pvalues.append(np.nan) 
        else:
            y_kde = stats.gaussian_kde(y_mean_sample.sel(z=loc))
            pvalues.append(y_kde.pdf(mean_x.sel(z=loc))[0])

    pvalues = xr.DataArray(data=np.array(pvalues), coords=mean_x.coords)

    return mean_x.unstack(), pvalues.unstack()


def percentile_of_scores(X: xr.DataArray, y: xr.DataArray, 
                         stackdim: list=['lat', 'lon'], id: int=None):
    """Compute percentile of scores.

    Args:
        X (xr.DataArray): Samples of distribution.
        y (xr.DataArray): Value to compare against.
        stackdim (list, optional): Dimensions of xarray which should be stacked.
            Defaults to ['time', 'lon'].
        idx (int, optional): Index that will be returned for parallelization. Defaults to None.

    Returns:
        pvalues (xr.DataArray): Pvalues of dimensions of input y. 
        id (int): Input id for parallelization. 
    """
    X = X.stack(z=stackdim)
    y = y.stack(z=stackdim)
    pvalues = xr.ones_like(y) * np.nan
    for i in range(len(y['z'])):
        p = stats.percentileofscore(
            X.isel(z=i), y.isel(z=i), kind='weak')
        if p > 50:
            p = 100 - p
        pvalues[i] = p/100
    pvals = pvalues.unstack('z')

    return pvals, id


def field_significance_mask(pvalues, stackdim=('lat', 'lon'), alpha=0.05, corr_type="dunn"):
    """Create mask field with 1, np.NaNs for significant values
    using a multiple test correction.

    Args:
        pvalues (xr.Dataarray): Pvalues
        alpha (float, optional): Alpha value. Defaults to 0.05.
        corr_type (str, optional): Multiple test correction type. Defaults to "dunn".

    Returns:
        mask (xr.Dataarray): Mask
    """
    if corr_type is not None:
        pvals_flat = pvalues.stack(z=stackdim)
        mask_flat = xr.DataArray(data=np.zeros(len(pvals_flat), dtype=bool),
                                 coords=pvals_flat.coords)
        ids = holm(pvals_flat.data, alpha=alpha, corr_type=corr_type)
        mask_flat[ids] = True
        mask = mask_flat.unstack()
    else:
        mask = xr.where(pvalues <= alpha, True, False)

    return mask