"""PCA for spatio-temporal data."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def map2flatten(x_map: xr.Dataset) -> list:
    """Flatten dataset/dataarray and remove NaNs.

    Args:
        x_map (xr.Dataset/ xr.DataArray): Dataset or DataArray to flatten. 

    Returns:
        x_flat (xr.DataArray): Flattened dataarray without NaNs 
        ids_notNaN (xr.DataArray): Boolean array where values are on the grid. 
    """
    if type(x_map) == xr.core.dataset.Dataset:
        x_stack_vars = [x_map[var] for var in list(x_map.data_vars)]
        x_stack_vars = xr.concat(x_stack_vars, dim='var')
        x_stack_vars = x_stack_vars.assign_coords({'var': list(x_map.data_vars)})
        x_flatten = x_stack_vars.stack(z=('var', 'lat', 'lon')) 
    else:
        x_flatten = x_map.stack(z=('lat', 'lon'))

    # Flatten and remove NaNs
    if 'time' in x_flatten.dims:
        idx_notNaN = ~np.isnan(x_flatten.isel(time=0))
    else:
        idx_notNaN = ~np.isnan(x_flatten)
    x_proc = x_flatten.isel(z=idx_notNaN.data)

    return x_proc, idx_notNaN


def flattened2map(x_flat: np.ndarray, ids_notNaN: xr.DataArray, times: np.ndarray = None) -> xr.Dataset:
    """Transform flattened array without NaNs to gridded data with NaNs. 

    Args:
        x_flat (np.ndarray): Flattened array of size (n_times, n_points) or (n_points).
        ids_notNaN (xr.DataArray): Boolean dataarray of size (n_points).
        times (np.ndarray): Time coordinate of xarray if x_flat has time dimension.

    Returns:
        xr.Dataset: Gridded data.
    """
    if len(x_flat.shape) == 1:
        x_map = xr.full_like(ids_notNaN, np.nan, dtype=float)
        x_map[ids_notNaN.data] = x_flat
    else:
        temp = np.ones((x_flat.shape[0], ids_notNaN.shape[0])) * np.nan
        temp[:, ids_notNaN.data] = x_flat
        if times is None:
            times = np.arange(x_flat.shape[0]) 
        x_map = xr.DataArray(data=temp, coords={'time': times, 'z': ids_notNaN['z']})

    if 'var' in list(x_map.get_index('z').names):
        x_map = x_map.unstack()

        if 'var' in list(x_map.dims): # For xr.Datasset only
            da_list = [xr.DataArray(x_map.isel(var=i), name=var) 
                       for i, var in enumerate(x_map['var'].data)]
            x_map = xr.merge(da_list, compat='override')
            x_map = x_map.drop(('var'))
    else:
        x_map = x_map.unstack()
    
    return x_map



class SpatioTemporalPCA:
    """PCA of spatio-temporal data.

    Wrapper for sklearn.decomposition.PCA with xarray.DataArray input.

    Args:
        ds (xr.DataArray or xr.Dataset): Input dataarray to perform PCA on.
            Array dimensions (time, 'lat', 'lon')
        n_components (int): Number of components for PCA
    """
    def __init__(self, ds, n_components, **kwargs):
        self.ds = ds

        self.X, self.ids_notNaN = map2flatten(self.ds)

        # PCA
        self.pca = PCA(n_components=n_components, whiten=True)
        self.pca.fit(self.X.data)

        self.n_components = self.pca.n_components


    def get_eofs(self):
        """Return components of PCA.

        Parameters:
        -----------
        normalize: str 
            Normalization type of components.

        Return:
        -------
        components: xr.dataarray (n_components, N_x, N_y)
        """
        # EOF maps
        components = self.pca.components_
        eof_map = []
        for i, comp in enumerate(components):
            eof = flattened2map(comp, self.ids_notNaN)
            eof_map.append(eof)

        return xr.concat(eof_map, dim='eof')
    

    def get_principal_components(self):
        """Returns time evolution of components.

        Args:
        -----
        normalize: str or None
            Method to normalize the time-series

        Return:
        ------
        time_evolution: np.ndarray (n_components, time)
        """
        pc = self.pca.transform(self.X.data)
        da_pc = xr.DataArray(
                data=pc,
                coords=dict(time=self.X['time'], eof=np.arange(1, self.n_components+1)),
        )
        return da_pc
    

    def explained_variance(self):
        return self.pca.explained_variance_ratio_
    
    
    def transform(self, x: xr.Dataset):
        x_flat, ids_notNaN = map2flatten(x)
        assert len(x_flat['z']) == len(self.X['z'])
        z = self.pca.transform(x_flat.data)

        return z 
    
    def inverse_transform(self, z: np.ndarray, newdim='time') -> xr.Dataset:
        """Transform from eof space to data space.

        Args:
            z (np.ndarray): Principal components of shape (n_samples, n_components) 
            newdim (str, optional): Name of dimension of n_samples.
                Can be also a pd.Index object. Defaults to 'time'.

        Returns:
            xr.Dataset: Transformed PCs to grid space.
        """
        x_hat_flat = self.pca.inverse_transform(z) 

        x_hat_map = []
        for x_flat in x_hat_flat:
            x_hat = flattened2map(x_flat, self.ids_notNaN)
            x_hat = x_hat.drop([dim for dim in list(x_hat.dims) if dim not in ['lat', 'lon']])
            x_hat_map.append(x_hat)

        return xr.concat(x_hat_map, dim=newdim)
    

    def reconstruction(self, x: xr.Dataset) -> xr.Dataset:
        """Reconstruct the dataset from components and time-evolution."""
        stack_dim = [dim for dim in list(x.dims) if dim not in ['lat', 'lon']][0]
        z = self.transform(x)
        x_hat = self.inverse_transform(z, newdim=x[stack_dim])

        return x_hat
    
    

