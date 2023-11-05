"""Collection of functions to preprocess climate data."""
import os
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import xarray as xr
from tqdm import tqdm
from joblib import Parallel, delayed


def lon_to_180(longitudes):
    """Convert longitudes from [0, 360] to [-180, 180]."""
    return np.where(longitudes > 180, longitudes - 360, longitudes)

def lon_to_360(longitudes):
    """Convert longitudes from [-180, 180] to [0, 360]."""
    return np.where(longitudes < 0, longitudes + 360, longitudes)

def save_to_file(xArray, filepath, var_name=None):
    """Save dataset or dataarray to file."""
    if os.path.exists(filepath):
        print("File" + filepath + " already exists!")
    else:
        # convert xr.dataArray to xr.dataSet if needed
        if var_name is not None:
            ds = xArray.to_dataset(name=var_name)
        else:
            ds = xArray
        # Store to .nc file
        try:
            ds.to_netcdf(filepath)
        except OSError:
            print("Could not write to file!")
    return


def compute_anomalies(dataarray, group='dayofyear',
                      base_period=None):
    """Calculate anomalies.

    Parameters:
    -----
    dataarray: xr.DataArray
        Dataarray to compute anomalies from.
    group: str
        time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'
    base_period (list, None): period to calculate climatology over. Default None.

    Return:
    -------
    anomalies: xr.dataarray
    """
    if base_period is None:
        base_period = np.array(
            [dataarray.time.data.min(), dataarray.time.data.max()])

    climatology = dataarray.sel(time=slice(base_period[0], base_period[1])
                                ).groupby(f'time.{group}').mean(dim='time', skipna=True)
    anomalies = (dataarray.groupby(f"time.{group}")
                 - climatology)

    return anomalies


def detrend_dim(da, dim='time', deg=1, startyear=None):
    """Detrend data by subtracting a linear fit which is obtained
        for the lat-lon mean at each time.

    Args:
        da ([xr.DataArray]): Data to detrend.

    Returns:
        da_detrend (xr.DataArray): Detrended data.
        coef (list): Linear fit coefficients,
                    i.e. coef[0]*x + coef[1]
    """
    if startyear is not None:
        tmin, tmax = da.time.data.min(), da.time.data.max()
        ttrend = np.datetime64(f'{startyear}-01-01', 'D') -1
        da_notrend = da.sel(time=slice(tmin, ttrend))
        da_trend = da.sel(time=slice(ttrend, tmax))
    else:
        da_trend = da
    
    # Linear fit to data 
    p = da_trend.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da_trend[dim], p.polyfit_coefficients)
    da_detrend =  da_trend - fit + fit[0]

    if startyear is not None:
        da_detrend = xr.concat([da_notrend, da_detrend], dim='time')

    return da_detrend


def time_average(ds, group='1D'):
    """Downsampling of time dimension by averaging.

    Args:
    -----
    ds: xr.dataFrame 
        dataset
    group: str
        time group e.g. '1D' for daily average from hourly data
    """
    ds_average = ds.resample(time=group, label='left').mean(skipna=True)

    # Shift time to first of month
    if group == '1M':
        new_time = ds_average.time.data + np.timedelta64(1, 'D')
        new_coords = {}
        for dim in ds_average.dims:
            new_coords[dim] = ds_average[dim].data
        new_coords['time'] = new_time
        ds_average = ds_average.assign_coords(coords=new_coords)

    return ds_average


def get_antimeridian_coord(lons):
    """Change of coordinates from normal to antimeridian."""
    lons = np.array(lons)
#    lons_new = np.where(lons < 0, (lons % 180),(lons % 180 - 180))
    lons_new = np.where(lons < 0, (lons + 180), (lons - 180))
    return lons_new


def set_antimeridian2zero(ds, roll=True):
    """Set the antimeridian to zero.

    Easier to work with the pacific then.
    """
    if ds['lon'].data[0] <= -100 and roll is True:
        # Roll data such that the dateline is not at the corner of the dataset
        print("Roll longitudes.")
        ds = ds.roll(lon=(len(ds['lon']) // 2), roll_coords=True)

    # Change lon coordinates
    lons_new = get_antimeridian_coord(ds.lon)
    ds = ds.assign_coords(
        lon=lons_new
    )
    print('Set the dateline to the new longitude zero.')
    return ds


def reduce_map_resolution(ds, lon_factor, lat_factor):
    """Reduce resolution of map of dataarray."""
    ds_coursen = ds.coarsen(lon=lon_factor).mean().coarsen(
        lat=lat_factor).mean()
    print('Reduced the resolution of the map!')
    return ds_coursen


def set_grid(ds, step_lat=1, step_lon=1,
             lat_range=None, lon_range=None):
    """Interpolate grid.

    Args:
        ds (xr.Dataset): Dataset or dataarray to interpolate.
            Dataset is only supported for grid_type='mercato'.
        step_lat (float, optional): Latitude grid step. Defaults to 1.
        step_lon (float, optional): Longitude grid step. Defaults to 1.
        lat_range (list, optional): Latitude range. Defaults to None.
        lon_range (list, optional): Longitude range. Defaults to None.

    Returns:
        da (xr.Dataset): Interpolated dataset or dataarray.
        grid (dict): Grid used for interpolation, dict(lat=[...], lon=[...]).
    """
    lat_min = ds['lat'].min().data if lat_range is None else lat_range[0] 
    lat_max = ds['lat'].max().data if lat_range is None else lat_range[1] 
    lon_min = ds['lon'].min().data if lon_range is None else lon_range[0] 
    lon_max = ds['lon'].max().data if lon_range is None else lon_range[1] 
    init_lat = np.arange(
        lat_min, (lat_max + step_lat), step_lat
    )
    init_lon = np.arange(
        lon_min, lon_max, step_lon
    )
    grid = {'lat': init_lat, 'lon': init_lon}
    # Interpolate
    da = ds.interp(grid, method='nearest')

    return da, grid


def interp_points(i, da, points_origin, points_grid):
    """Interpolation of dataarray to a new set of points.

    Args:
        i (int): Index of time
        da (xr.Dataarray): Dataarray
        points_origin (np.ndarray): Array of origin locations.
        points_grid (np.ndarray): Array of locations to interpolate on.

    Returns:
        i (int): Index of time
        values_grid_flat (np.ndarray): Values on new points.
    """
    values_origin = da[i].data.flatten()
    values_grid_flat = interpolate.griddata(
        points_origin, values_origin, xi=points_grid, method='nearest'
    )
    return i, values_grid_flat


def interp_points2mercato(da, grid):
    """Interpolate Dataarray with non-rectangular grid to mercato grid.

    Args:
        da (xr.Dataarray): Dataarray with non-rectangular grid. 
        grid (dict): Grid to interpolate on dict(lat=[...], lon=[...]).

    Returns:
        da_grid (xr.Dataarray): Dataarray interpolated on mercato grid. 
    """
    # Create array of points from mercato grid
    xx, yy = np.meshgrid(grid['lon'], grid['lat'])
    points_grid = np.array([xx.flatten(), yy.flatten()]).T
    points_origin = np.array(
        [da['nav_lon'].data.flatten(), da['nav_lat'].data.flatten()]).T

    # Interpolation at each time step in parallel
    n_processes = len(da['time_counter'])
    results = Parallel(n_jobs=8)(
        delayed(interp_points)(i, da, points_origin, points_grid)
        for i in tqdm(range(n_processes))
    )
    # Read results
    ids = []
    values_grid_flat = []
    for r in results:
        i, data = r
        ids.append(i)
        values_grid_flat.append(data)
    ids = np.array(ids)
    values_grid_flat = np.array(values_grid_flat)

    # Store to new dataarray
    values_grid = np.reshape(
        values_grid_flat,
        newshape=(len(values_grid_flat), len(grid['lat']), len(grid['lon']))
    )
    times = da['time_counter'].data[ids]
    da_grid = xr.DataArray(
        data=values_grid,
        dims=['time', 'lat', 'lon'],
        coords=dict(time=times, lat=grid['lat'], lon=grid['lon']),
        name=da.name
    )
    return  da_grid


def cut_map(ds, lon_range=None, lat_range=None, shortest=True):
    """Cut an area in the map. Use always smallest range as default.
    It lon ranges accounts for regions (eg. Pacific) that are around the -180/180 region.

    Args:
    ----------
    lon_range: list [min, max]
        range of longitudes
    lat_range: list [min, max]
        range of latitudes
    shortest: boolean
        use shortest range in longitude (eg. -170, 170 range contains all points from
        170-180, -180- -170, not all between -170 and 170). Default is True.
    Return:
    -------
    ds_area: xr.dataset
        Dataset cut to range
    """
    if lon_range is not None:
        if (max(lon_range) - min(lon_range) <= 180) or shortest is False:
            ds = ds.sel(
                lon=slice(np.min(lon_range), np.max(lon_range)),
                lat=slice(np.min(lat_range), np.max(lat_range))
            )
        else:
            # To account for areas that lay at the border of -180 to 180
            ds = ds.sel(
                lon=ds.lon[(ds.lon < min(lon_range)) |
                           (ds.lon > max(lon_range))],
                lat=slice(np.min(lat_range), np.max(lat_range))
            )
    if lat_range is not None:
        ds = ds.sel(
            lat=slice(np.min(lat_range), np.max(lat_range))
        )

    return ds


def select_months(ds, months=[12, 1, 2]):
    """Select only some months in the data.

    Args:
        ds ([xr.DataSet, xr.DataArray]): Dataset of dataarray
        months (list, optional): Index of months to select. 
                                Defaults to [12,1,2]=DJF.
    """
    ds_months = ds.sel(time=np.in1d(ds['time.month'], months))

    return ds_months


def select_time_snippets(ds: xr.Dataset, time_periods: np.ndarray) -> xr.Dataset:
    """Select time periods from dataset.

    Args:
        ds (xr.Dataset or xr.Dataarray): Dataarray to select time-periods from.
        time_periods (np.ndarray): Time periods of shape (n,2) with n time periods.

    Returns:
        xr.Dataset: Dataset with selected time periods.
    """
    ds_lst = []
    for start_time, end_time in time_periods:
        # Check if the start and end times are within the DataArray's time range
        if (start_time > np.min(ds.time.values)) and (end_time < np.max(ds.time.values)):
            # Check that the start time is before the end time
            if start_time <= end_time:
                ds_lst.append(ds.sel(time=slice(start_time, end_time)))
            else:
                ValueError("Start time is after end time")
        else:
            continue
    ds_snip = xr.concat(ds_lst, dim='time')

    return ds_snip


def average_time_periods(ds, time_snippets):
    """Select time snippets from dataset and average them.

    Parameters:
    -----------
    time_snippets: np.datetime64  (n,2)
        Array of n time snippets with dimension (n,2).

    Returns:
    --------
    xr.Dataset with averaged times
    """
    ds_lst = []
    for time_range in time_snippets:
        temp_mean = ds.sel(time=slice(time_range[0], time_range[1])).mean('time')
        temp_mean['time'] = time_range[0] +  0.5 * (time_range[1] - time_range[0])
        ds_lst.append(temp_mean)

    ds_snip = xr.concat(ds_lst, dim='time')

    return ds_snip


def get_mean_time_series(da, lon_range, lat_range, time_roll=0):
    """Get mean time series of selected area.

    Parameters:
    -----------
    da: xr.DataArray
        Data
    lon_range: list
        [min, max] of longitudinal range
    lat_range: list
        [min, max] of latiduninal range
    """
    da_area = cut_map(da, lon_range, lat_range)
    ts_mean = da_area.mean(dim=('lon', 'lat'), skipna=True)
    ts_std = da_area.std(dim=('lon', 'lat'), skipna=True)
    if time_roll > 0:
        ts_mean = ts_mean.rolling(time=time_roll, center=True).mean()
        ts_std = ts_std.rolling(time=time_roll, center=True).mean()

    return ts_mean, ts_std


def check_dimensions(ds, sort=True):
    """
    Checks whether the dimensions are the correct ones for xarray!
    """
    dims = list(ds.dims)

    rename_dic = {
        'longitude': 'lon',
        'nav_lon': 'lon',
        'xt_ocean': 'lon',
        'latitude': 'lat',
        'nav_lat': 'lat',
        'yt_ocean': 'lat',
    }
    for c_old, c_new in rename_dic.items():
        if c_old in dims:
            print(f'Rename:{c_old} : {c_new} ')
            ds = ds.rename({c_old: c_new})
            dims = list(ds.dims)

    # Check for dimensions
    clim_dims = ['lat', 'lon']
    for dim in clim_dims:
        if dim not in dims:
            raise ValueError(
                f"The dimension {dim} not consistent with required dims {clim_dims}!")

    # If lon from 0 to 360 shift to -180 to 180
    if max(ds.lon) > 180:
        print("Shift longitude!")
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))

    if sort:
        print('Sort longitudes and latitudes in ascending order, respectively')
        ds = ds.sortby('lon')
        ds = ds.sortby('lat')

    # if 'time' in ds.dims:
        # ds = ds.transpose('time', 'lat', 'lon')

    return ds


def rotate_matrix(M, Theta):
    """Rotate 2d matrix by angle theta.

    Args:
        M (np.ndarray): (2,2) 2d matrix
        Theta (float): Angle in rad.

    Returns:
        (np.ndarray) (2,2) Rotated matrix.
    """
    R = np.array(
        [[np.cos(Theta), -np.sin(Theta)], [np.sin(Theta), np.cos(Theta)]]
    )
    return R @ M @ R.T


def process_data(f_data, vars, 
                 time_average=None, antimeridian=False,
                 lon_range=None, lat_range=None, 
                 climatology=None, detrend_from=None,
                 normalization=None, splity=None,
                 grid_step=None, **kwargs):
    """Load an   d preprocess data using xarray.

    Args:
        f_sst (str): Filename.
        vars (list): List of variable names.
        time_average (str, optional): Resample time and average, e.g. 'M'. 
            Defaults to None.
        antimeridian (bool, optional): Set the antimeridian to zero.
            Defaults to False.
        lon_range (list, optional): Longitude range to cut, e.g. [90, 120].
            Defaults to None.
        lat_range (list, optional): Latitude range to cut, e,g, [-10, 10]. 
            Defaults to None.
        grid_step (float, optional): Grid step for interpolation.
            Defaults to None.
        climatology (str, optional): If set anomalies are computed with respect to the 
            choosen climatology, e.g. 'day', 'month'. Defaults to None.
        detrend_from (int, optional): Year to detrend from. Defaults to None.
        normalization (str, optional): Normalize data by either 'minmax' or 'zscore'.
            Defaults to None.
        splity (int, optional): Year to split dataarray into training and test set. 
            Defaults to None.

    Returns:
        da (xr.DataArray): Processed dataarray.
        da_train (xr.DataArray): Processed dataarray until splity.
            None if splity=None.
        da_test (xr.DataArray): Processed dataarray from splity to end.
            None if splity=None.
    """
    ds = xr.open_dataset(f_data)
    ds = check_dimensions(ds)

    da_preprocessed = []
    attributes = {}
    for i, varname in enumerate(vars):
        print(f"Process {varname}:", flush=True)
        da = ds[varname]

        if time_average is not None:
            print(f"Resample time by {time_average} and compute mean.")
            da = da.resample(time=time_average, label='left').mean()
            da = da.assign_coords(
                dict(time=da['time'].data + np.timedelta64(1, 'D'))
            )

        # change coordinates to dateline == 0
        if antimeridian:
            da = set_antimeridian2zero(da)
            if (lon_range is not None) and (i == 0):
                lon_range = get_antimeridian_coord(lon_range)


        # Cut area of interest
        if lon_range is not None or lat_range is not None:
            print(f'Get selected area: lon={lon_range}, lat={lat_range}!')
            da = cut_map(
                da, lon_range=lon_range, lat_range=lat_range, shortest=False
            )

        # coarse grid if needed
        if grid_step is not None:
            print(f'Interpolate grid on res {grid_step}')
            da, grid = set_grid(da, step_lat=grid_step, step_lon=grid_step,
                                lat_range=lat_range, lon_range=lon_range)

        # Compute anomalies
        if climatology is not None:
            print("Detrend and compute anomalies:")
            da = detrend_dim(da, dim='time', startyear=detrend_from)
            da = compute_anomalies(da, group=climatology)
            da.name = f"{varname}a"

        # Normalize data
        if normalization is not None:
            scaler = Normalizer(method=normalization)
            da = scaler.fit_transform(da)
            attributes[da.name] = {'normalizer': scaler}
        
        da_preprocessed.append(da)
    
    ds_preproc = xr.merge(da_preprocessed)
    ds_preproc.attrs = attributes
    
    # Split into training and test
    if splity is None:
        return ds_preproc

    else:
        start_time = ds_preproc.time.min().data
        end_time = ds_preproc.time.max().data
        split_time = np.datetime64(f"{splity}-01-01", 'D')
        ds_train = ds_preproc.sel(time=slice(start_time, split_time))    
        ds_test = ds_preproc.sel(time=slice(split_time, end_time))    
        return ds_preproc, ds_train, ds_test



def time2timestamp(time):
    """Convert np.datetime64 to int."""
    return (time - np.datetime64('0001-01-01', 'ns')) / np.timedelta64(1, 'D') 


def timestamp2time(timestamp):
    """Convert timestamp to np.datetime64 object."""
    return (np.datetime64('0001-01-01', 'ns') 
            + timestamp * np.timedelta64(1, 'D') )

def time2idx(timearray, times):
    """Get index of t in times.
    
    Parameters:
        timearray (np.ndarray): Time array to search in.
        times (list): List of times.
    """
    idx = []
    for t in times:
        idx.append(np.argmin(np.abs(timearray - t)))
    return idx


class Normalizer():
    """Normalizing xr.DataArray.

    Args:
        method (str, optional): Normalization method, i.e.
            'minmax': Normalize data between [0,1], 
            'zscore': Standardizes the data,  
            'center': Centers the data around 0,  
            Defaults to 'zscore'.

    Raises:
        ValueError: If method is none of the listed above. 
    """
    
    def __init__(self, method: str = 'zscore') -> None:
        self.method = method

        if self.method not in ['zscore', 'minmax', 'center']:
            raise ValueError(f"Your selected normalization method "+
                             self.method + " does not exist.", flush=True)


    def fit(self, da: xr.DataArray, axis: int = None, dim: str = None, **kwargs) -> None:
        """Compute the parameters of the normalization for the given data.

        Args:
            da (xr.DataArray): Data to compute the normalization for.
            dim (str, optional): Dimension along the normalization should be performed.
                Defaults to None.
            axis (int, optional): Axis along the normalization should be performed. If
                dim is specified, axis will not be used. Defaults to None.
        """
        if dim is None and axis is not None:
            dim = da.dims[axis]
        elif dim is None and axis is None:
            dim = da.dims

        if self.method == 'minmax':
            self.min = da.min(dim=dim, skipna=True)
            self.max = da.max(dim=dim, skipna=True)

        elif self.method == 'zscore':
            self.mean = da.mean(dim=dim, skipna=True)
            self.std = da.std(dim=dim, skipna=True)

        elif self.method == 'center':
            self.mean = da.mean(dim=dim, skipna=True)

        return None
    

    def transform(self, da: xr.DataArray, **kwargs) -> xr.DataArray:
        """Normalize data using the normalization parameters.

        Args:
            da (xr.DataArray): Data to compute the normalization for.
        
        Returns:
            xr.DataArray: Normalized data. 
        """
        if self.method == 'minmax':
            return (da - self.min) / (self.max - self.min)

        elif self.method == 'zscore':
            return (da - self.mean) / self.std

        elif self.method == 'center':
            return da - self.mean 
    

    def fit_transform(self, da: xr.DataArray, axis: int = None,
                      dim: str = None, **kwargs) -> xr.DataArray:
        """Compute the normalization parameters and transform the data.

        Args:
            da (xr.DataArray): Data to compute the normalization for.
            dim (str, optional): Dimension along the normalization should be performed.
                Defaults to None.
            axis (int, optional): Axis along the normalization should be performed. If
                dim is specified, axis will not be used. Defaults to None.

        Returns:
            xr.DataArray: Normalized data. 
        """
        self.fit(da, axis=axis, dim=dim, **kwargs)
        da_norm = self.transform(da, **kwargs)
        return da_norm
    

    def inverse_transform(self, da_norm: xr.DataArray) -> xr.DataArray:
        """Inverse the normalization.

        Args:
            da_norm (xr.DataArray): Normalized data. 

        Returns:
            xr.DataArray: Unnormalized data.
        """
        if self.method == 'minmax':
            return da_norm * (self.max - self.min) + self.min
        elif self.method == 'zscore':
            return da_norm * self.std + self.mean
        elif self.method == 'center':
            return da_norm + self.mean
    

    def to_dict(self):
        """Save variables to dict."""
        config = dict(method=self.method)
        if self.method == 'minmax':
            config['min']= np.array(self.min)
            config['max']= np.array(self.max)
        elif self.method == 'zscore':
            config['mean']= np.array(self.mean)
            config['std']= np.array(self.std)
        elif self.method == 'center':
            config['mean']= np.array(self.mean)

        return config


def normalizer_from_dict(config: dict) -> Normalizer:
    """Create Normalizer object from dictionary.

    Args:
        config (dict): Dictionary with class parameters.

    Returns:
        Normalizer: Normalizer class object
    """
    normalizer = Normalizer(method=config['method'])
    if config['method'] == 'minmax':
        normalizer.min = config['min']  
        normalizer.max = config['max'] 
    elif config['method'] == 'zscore':
        normalizer.mean = config['mean']
        normalizer.std = config['std'] 
    elif config['method'] == 'center':
        normalizer.mean = config['mean']

    return normalizer
