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
from latgmm.utils import preproc

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
outfile = "../../data/reanalysis/monthly/ssta_merged_dataset_3.nc"


# %%
# Load datasets and apply preprocessing
data_list = []
for i, (name, path) in enumerate(filelist.items()):
    print(f"Process {name}:", flush=True)
    ds = xr.open_dataset(path)
    ds = preproc.check_dimensions(ds)
    da = ds[varname]

    # change coordinates to dateline == 0
    da = preproc.set_antimeridian2zero(da)
    if (lon_range is not None) and (i == 0):
        lon_range = preproc.get_antimeridian_coord(lon_range)

    # Cut area of interest
    print(f'Get selected area: lon={lon_range}, lat={lat_range}!')
    da = preproc.cut_map(
        da, lon_range=lon_range, lat_range=lat_range, shortest=False
    )

    # coarse grid if needed
    print(f'Interpolate grid on res {grid_step}')
    da, grid = preproc.set_grid(da, step_lat=grid_step, step_lon=grid_step,
                            lat_range=lat_range, lon_range=lon_range)

    # Compute anomalies
    print("Detrend and compute anomalies:")
    da = preproc.detrend_dim(da, dim='time', startyear=detrend_from)
    da = preproc.compute_anomalies(da, group=climatology)
    da.name = f"{varname}a"

    # Drop coordinates which are not needed
    for coord_name in list(da.coords.keys()):
        if coord_name not in ['time', 'lat', 'lon', 'member']:
            da = da.reset_coords(coord_name, drop=True)

    # Convert timescales to month
    da['time'] = np.array(da['time'].data, dtype='datetime64[M]')

    # Add label for data source
    da = da.assign_coords(
        member=('time', len(da.time) * [name])
    )

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

# Mask all lat lon locations where there are NaNs
data = xr.where(mask_nan.drop('member') == False, data, np.nan)


# %% 
# Get index of time for each member and year (July-June)
ids_time = []
init_years = []
members = []
for member in np.unique(data['member']):
    idx_member = np.where(data['member'] == member)[0]
    x_member = data.isel(time=idx_member)
    unique_years = np.unique(x_member['time'].dt.year)
    if np.max(unique_years) > 2022:
        unique_years = unique_years[:-1]
    for y in unique_years[:-1]:
        ids_time.append(np.where(
            (data['member'].data == member)
            & (data['time'] >= np.datetime64(f'{y}-07-01'))
            & (data['time'] < np.datetime64(f'{y+1}-07-01'))
        )[0])
        init_years.append(y)
        members.append(member)

ids_time = xr.DataArray(np.array(ids_time),
                        coords={'year': init_years,
                                'month': np.arange(1,13)})
ids_time = ids_time.assign_coords(member=('year', members))
    
# %%
# Sample index to get 4 samples of each year
sampled_indices = []
unique_years = np.arange(1901, 2022)
for year in unique_years:
    indices = np.where(ids_time['year'] == year)[0]
    sampled_ids_year = np.random.choice(indices, size=4, replace=False)
    sampled_ids_time = ids_time.isel(year=sampled_ids_year).data.flatten()
    sampled_indices.append(sampled_ids_time)

data_sampled = data.isel(time=np.array(sampled_indices).flatten())

# %%
# Store to file
data_sampled.to_netcdf(outfile)

# %%
