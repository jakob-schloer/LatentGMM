''' Load ensemble of reanalysis datasets, compute anomalies and merge to one dataset.

@Author  :   Jakob Schlör 
@Time    :   2023/08/07 13:55:00
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
# Import packages
# ======================================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Preprocessing functions 
# ======================================================================================
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

def get_antimeridian_coord(lons):
    """Change of coordinates from normal to antimeridian."""
    lons = np.array(lons)
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
    ds = ds.interp(grid, method='nearest')

    return ds, grid

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

# %%
# Main part of the script
# ======================================================================================
if __name__ == "__main__":

    # Paths and parameters
    varname='sst'
    lon_range=[130, -70]
    lat_range=[-31, 32]
    grid_step = 1
    climatology='month'
    detrend_from = 1950
    dirpath = "../../data/reanalysis/monthly/"
    filelist = {
        'COBE2':   dirpath+"/COBE/sst_cobe2_month_1850-2019.nc",
        'ErSSTv5': dirpath+"/ERSSTv5/sst_ersstv5_month_1854-present.nc",
        'HadISST': dirpath+"/HadISST/sst_hadisst_month_1870-present.nc",
        'ORAS5':   dirpath+"/ORAS5/sea_surface_temperature_oras5_single_level_1958_2023_1x1.nc",
        'GODAS':   dirpath+"/GODAS/sst_godas_month_1980-present.nc",
        'SODA':    dirpath+"/SODA/sst_SODA_month_1980-2017.nc",
        'ERA5':    dirpath+"/ERA5/sea_surface_temperature_era5_monthly_sp_1940-2022_1.0x1.0.nc",
        'CERA-20c':dirpath+"/CERA-20C/sst_cera20c_1901-2009_r1x1.nc",
    }
    outfile = "../../data/ssta_merged_dataset.nc"

    # Load datasets and apply preprocessing
    data_list = []
    for i, (name, path) in enumerate(filelist.items()):
        print(f"Process {name}:", flush=True)
        ds = xr.open_dataset(path)
        ds = check_dimensions(ds)
        da = ds[varname]

        # change coordinates to dateline == 0
        da = set_antimeridian2zero(da)
        if (lon_range is not None) and (i == 0):
            lon_range = get_antimeridian_coord(lon_range)

        # Cut area of interest
        print(f'Get selected area: lon={lon_range}, lat={lat_range}!')
        da = cut_map(
            da, lon_range=lon_range, lat_range=lat_range, shortest=False
        )

        # coarse grid if needed
        print(f'Interpolate grid on res {grid_step}')
        da, grid = set_grid(da, step_lat=grid_step, step_lon=grid_step,
                                lat_range=lat_range, lon_range=lon_range)

        # Compute anomalies
        print("Detrend and compute anomalies:")
        da = detrend_dim(da, dim='time', startyear=detrend_from)
        da = compute_anomalies(da, group=climatology)
        da.name = f"{varname}a"

        # Drop coordinates which are not needed
        for coord_name in list(da.coords.keys()):
            if coord_name not in ['time', 'lat', 'lon', 'member']:
                da = da.reset_coords(coord_name, drop=True)

        # Add label for data source
        da = da.assign_coords(
            member=('time', len(da.time) * [name])
        )

        # Convert timescales to month
        da['time'] = np.array(da['time'].data, dtype='datetime64[M]')

        # Mask NaN: Make sure NaNs are consistnetn across files
        print("Update NaN mask!")
        if i == 0:
            mask_nan = xr.zeros_like(da.isel(time=0))
        for t in range(len(da['time'])):
            mask_nan = np.logical_or(mask_nan, np.isnan(da.isel(time=t)))

        data_list.append(da)

    # Concatenate data
    print("Concatenate datasets!")
    data = xr.concat(data_list, dim='time')

    # Drop coordinates which are not needed
    for coord_name in list(data.coords.keys()):
        if coord_name not in ['time', 'lat', 'lon', 'member']:
            data = data.reset_coords(coord_name, drop=True)

    # Mask all lat lon locations where there are NaNs
    data = xr.where(mask_nan == False, data, np.nan)

    # Store to file
    data.to_netcdf(outfile)

# %%
