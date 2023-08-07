# %%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from latgmm.utils import preproc

PATH = os.path.dirname(os.path.abspath(__file__))

source = 'ORAS5'
if source == 'SODA':
    infiles = [
        dict(var='sst',  fname=PATH +f"/../../data/multivar/{source}/sst_SODA_month_1980-2017.nc"),
        dict(var='ssh',  fname=PATH +f"/../../data/multivar/{source}/ssh_SODA_month_1980-2017.nc"),
        dict(var='ucur', fname=PATH +f"/../../data/multivar/{source}/ucur_SODA_month_1980-2017.nc"),
        dict(var='vcur', fname=PATH +f"/../../data/multivar/{source}/vcur_SODA_month_1980-2017.nc"),
        dict(var='taux', fname=PATH +f"/../../data/multivar/{source}/taux_SODA_month_1980-2017.nc"),
        dict(var='tauy', fname=PATH +f"/../../data/multivar/{source}/tauy_SODA_month_1980-2017.nc"),
    ]
    lon_range = [-180, 180]
    lat_range = [-75, 75]
elif source == 'GODAS':
    infiles = [
        dict(var='sst',  fname=PATH +f"/../../data/multivar/{source}/sst_godas_month_1980-present_raw.nc"),
        dict(var='ssh',  fname=PATH +f"/../../data/multivar/{source}/ssh_godas_month_1980-present_raw.nc"),
        dict(var='ucur', fname=PATH +f"/../../data/multivar/{source}/ucur_godas_month_1980-present_raw.nc"),
        dict(var='vcur', fname=PATH +f"/../../data/multivar/{source}/vcur_godas_month_1980-present_raw.nc"),
        dict(var='taux', fname=PATH +f"/../../data/multivar/{source}/taux_godas_month_1980-present_raw.nc"),
        dict(var='tauy', fname=PATH +f"/../../data/multivar/{source}/tauy_godas_month_1980-present_raw.nc"),
    ]
    lon_range = [-180, 180]
    lat_range = [-65, 65]
elif source == 'ORAS5':
    infiles = [
        dict(var='sst',  fname=PATH +f"/../../data/multivar/{source}/sea_surface_temperature_oras5_single_level_1958_2023_1x1.nc"),
        dict(var='ssh',  fname=PATH +f"/../../data/multivar/{source}/sea_surface_height_oras5_single_level_1958_2023_1x1.nc"),
#        dict(var='ucur', fname=PATH +f"/../../data/multivar/{source}/zonal_velocity_oras5_level_0_1958_2022_1x1.nc"),
#        dict(var='vcur', fname=PATH +f"/../../data/multivar/{source}/meridional_velocity_oras5_level_0_1958_2022_1x1.nc"),
#        dict(var='taux', fname=PATH +f"/../../data/multivar/{source}/zonal_wind_stress_oras5_single_level_1958_2022_1x1.nc"),
#        dict(var='tauy', fname=PATH +f"/../../data/multivar/{source}/meridional_wind_stress_oras5_single_level_1958_2022_1x1.nc"),
    ]
    lon_range = [-180, 180]
    lat_range = None 
elif source == 'CERA20C':
    infiles = [
        dict(var='sst',  fname=PATH +f"/../../data/sst/monthly/sst_cera20c_1901-2009_r1x1.nc"),
        dict(var='ssh',  fname=PATH +f"/../../data/multivar/CERA20c/ssh_cera20c_1901-2009_r1x1.nc"),
    ]
    lon_range = [-180, 180]
    lat_range = None 
    


# Preprocess
vars = [f['var'] for f in infiles]
antimeridian = False
climatology = None  # 'month'
normalization = None  # 'zscore'
grid_step = 1

outfile = fname = (PATH + f"/../../data/multivar/"
                   + f"oceanvars_{source}_{grid_step}x{grid_step}.nc")


da_lst = []
for i, f in enumerate(infiles):
    ds_pp = preproc.process_data(
        f['fname'], vars=[f['var']], antimeridian=antimeridian,
        lon_range=lon_range, lat_range=lat_range, grid_step=grid_step,
        climatology=climatology, detrend_from=None, normalization=normalization)

    da = ds_pp[f"{f['var']}"]
    # Times
    print(f"Dims: {da.shape}")
    print(f"Time: {da['time'].min().data}, {da['time'].max().data}")

    # Interpolate NaNs
    print("Replace NaN gaps in data.")
    da = da.interpolate_na(dim='lon', method='nearest', max_gap=3)

    # Mask NaNs consistently
    if i == 0:
        mask_nan = xr.zeros_like(da.isel(time=0))
    for t in range(len(da['time'])):
        mask_nan = np.logical_or(mask_nan, np.isnan(da.isel(time=t).data))
    
    da_lst.append(da)

print('Remove Nans')
for i, da in enumerate(da_lst):
    da_lst[i] = xr.where(mask_nan.data == False, da, np.nan)

print('Merge to dataset')
ds = xr.merge(da_lst)
ds = ds.sel(time=slice(None, '2022-01-01'))

# %%
print(f'Save to {outfile}')
ds.to_netcdf(outfile)

# %%
# Plot for checking
for var in list(ds.data_vars):
    plt.figure()
    ds[var].isel(time=0).plot()

    plt.figure()
    ds[var].sel(lat=0, lon=179, method='nearest').plot()

# %%
var = 'ucur'
idx_var = 3
count = 0
for i in range(len(ds['time'])):
    n_nan = np.count_nonzero(np.isnan(da_lst[idx_var].isel(time=i).data))
    if n_nan > count or n_nan < count:
        print(f"idx: {i}, n_nan: {n_nan}")
        count = n_nan
# %%
