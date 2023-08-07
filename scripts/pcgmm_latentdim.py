''' PCA on ensemble of reanalysis datasets

@Author  :   Jakob Schlör 
@Time    :   2023/08/07 10:34:34
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
import os
from importlib import reload
import xarray as xr
import numpy as np
import pandas as pd
import bottleneck
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import mixture, decomposition

from latgmm.utils import utenso, preproc, eof, utdata, utstats, metric
import latgmm.geoplot as gpl

PATH = os.path.dirname(os.path.abspath(__file__))
plt.style.use(PATH + "/../paper.mplstyle")

# %%
# Load data
# ======================================================================================
data_config = dict(
    multivar=False,
    variables=['sst'],
    source='reanalysis',
    timescale='monthly',
    lon_range=[130, -70],
    lat_range=[-31, 32],
)
dirpath = "../data/reanalysis/monthly/"
if (data_config['multivar'] is False) & (len(data_config['variables']) == 1):
    data_config['filenames'] = [
        dict(name='COBE2',   path=dirpath+"/COBE/sst_cobe2_month_1850-2019.nc"),
        dict(name='ErSSTv5', path=dirpath+"/ERSSTv5/sst_ersstv5_month_1854-present.nc"),
        dict(name='HadISST', path=dirpath+"/HadISST/sst_hadisst_month_1870-present.nc"),
        dict(name='ORAS5',   path=dirpath+"/ORAS5/oceanvars_ORAS5_1x1.nc"),
        dict(name='GODAS',   path=dirpath+"/GODAS/sst_godas_month_1980-present.nc"),
        dict(name='SODA',    path=dirpath+"/SODA/sst_SODA_month_1980-2017.nc"),
        dict(name='ERA5',    path=dirpath+"/ERA5/sea_surface_temperature_era5_monthly_sp_1940-2022_1.0x1.0.nc"),
        dict(name='CERA-20c',path=dirpath+"/CERA-20C/sst_cera20c_1901-2009_r1x1.nc"),
    ]
    data_config['splity'] = ['2005-01-01', '2022-01-01']
elif data_config['multivar'] & (data_config['timescale'] == 'monthly'):
    data_config['filenames']=[
        dict(name='SODA',     path=dirpath+f"/SODA/oceanvars_SODA_1x1.nc"),
        dict(name='GODAS',    path=dirpath+f"/CODAS/oceanvars_GODAS_1x1.nc"),
        dict(name='ORAS5',    path=dirpath+f"/ORAS5/oceanvars_ORAS5_1x1.nc"),
        dict(name='CERA-20c', path=dirpath+f"/CERA-20C/oceanvars_CERA20C_1x1.nc"),
    ]
else:
    raise ValueError(f"No data are loaded due to specified timescale and variables!")

data_config['detrend_from'] = '1950'
data_config['normalization'] ='zscore'

data = utdata.load_data(**data_config)
ds = data['full']

# %%
# Scan number of eofs
# ======================================================================================
djf_event = True
gmm_bic = []
n_eofs = [2,3,4,5,6]
for n_components in n_eofs:
    sppca = eof.SpatioTemporalPCA(ds, n_components=n_components)
    eofs = sppca.get_eofs()
    pcs = sppca.get_principal_components()
    print(f"EOFs {n_components}, exp. variance: {np.sum(sppca.explained_variance())}")

    # Preselect data for GMM
    x_enso = []
    z_enso = []
    for member in np.unique(ds['member']):
        idx_member = np.where(ds['member'] == member)[0]
        x_member = ds.isel(time=idx_member)
        z_member = pcs.isel(time=idx_member)

        nino_ids = utenso.get_nino_indices(x_member['ssta'], antimeridian=True)
        enso_classes = utenso.get_enso_flavors_N3N4(nino_ids)
        enso_classes = enso_classes.loc[enso_classes['type'] != 'Normal']
        x_member_enso = []
        z_member_enso = []
        times = []
        for i, time_period in enso_classes.iterrows():
            if djf_event:
                x_member_enso.append(
                    x_member.sel(time=slice(time_period['start'], time_period['end'])).mean(dim='time')
                )
                z_member_enso.append(
                    z_member.sel(time=slice(time_period['start'], time_period['end'])).mean(dim='time')
                )
            else:
                x_member_enso.append(
                    x_member.sel(time=slice(time_period['start'], time_period['end']))
                )
                z_member_enso.append(
                    z_member.sel(time=slice(time_period['start'], time_period['end']))
                )
            times.append(time_period['start'])

        if djf_event:
            x_member_enso = xr.concat(x_member_enso, dim=pd.Index(times, name='time'))
            z_member_enso = xr.concat(z_member_enso, dim=pd.Index(times, name='time'))
            assert len(np.unique(x_member_enso.time.dt.year)) == len(x_member_enso.time)
        else:
            x_member_enso = xr.concat(x_member_enso, dim='time')
            z_member_enso = xr.concat(z_member_enso, dim='time')

        x_member_enso = x_member_enso.assign_coords(member=('time', len(x_member_enso['time']) * [member]))
        x_member_enso = z_member_enso.assign_coords(member=('time', len(z_member_enso['time']) * [member]))

        x_enso.append(x_member_enso)
        z_enso.append(z_member_enso)

    x_enso = xr.concat(x_enso, dim='time')
    z_enso = xr.concat(z_enso, dim='time').transpose("eof", 'time')

    # ## Gaussian mixture 
    # ### Scan number of cluster
    n_classes = np.arange(1, 10, 1)
    n_runs = 100
    for k in n_classes:
        for r in range(n_runs):
            gmm = mixture.GaussianMixture(n_components=k, 
                                          covariance_type='full', max_iter=100)
            gmm.fit(z_enso.data.T)
            gmm_bic.append(
                {'k': k, 'bic': gmm.bic(z_enso.data.T), 'n_eof': n_components}
            )
gmm_bic = pd.DataFrame(gmm_bic)
# %%
# Boxplots for each number of eofs
# ======================================================================================
fig, axs = plt.subplots(len(n_eofs), 1, figsize=(5,len(n_eofs)*2), sharex=True)
for i, n in enumerate(n_eofs):
    temp = gmm_bic.loc[gmm_bic['n_eof'] == n]
    sns.boxplot(data=temp, x='k', y='bic', ax=axs[i])
    axs[i].set_title(f"EOFs: {n}")

# %%
# Rank the mean BIC
# ======================================================================================
rank_bic = []
for i, n in enumerate(n_eofs):
    temp = gmm_bic.loc[gmm_bic['n_eof'] == n]
    rank_bic.append(
        xr.DataArray(
            data=pd.DataFrame([temp.loc[temp['k']==k].median() for k in temp['k'].unique()])['bic'],
            coords=dict(k=temp['k'].unique())
        ).rank(dim='k')
    )
rank_bic = xr.concat(rank_bic, dim=pd.Index(n_eofs, name='eof'))
# %%
# Plot boxplots for EOF=2 and ranked BIC 
# ======================================================================================
fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))

# Boxplots
ax = axs[0]
temp = gmm_bic.loc[gmm_bic['n_eof'] == 2]
sns.boxplot(data=temp, x='k', y='bic', ax=ax, color='cyan', fliersize=0.0)
ax.set_ylabel('BIC')
ax.text(0.7, 0.8, rf"# of EOFs = 2", va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes, 
        bbox=dict(facecolor='none', edgecolor='grey', linewidth=0.1))

# Plot ranked BIC
ax = axs[1]
bounds = np.arange(1,np.max(rank_bic.data)+2)
cmap = plt.get_cmap("cividis_r")
norm = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N)
im = ax.pcolormesh(
    rank_bic['k'], rank_bic['eof'], rank_bic.data, cmap=cmap, norm=norm
)
# Colorbar
cbar = plt.colorbar(im, ax=ax, label='Ranked BIC', ticks=bounds[:-1]+0.5)
# cbar.set_ticks(bounds[:-1]+0.5)
cbar.set_ticklabels(np.unique(rank_bic.data).astype(int))

ax.set_xticks(rank_bic['k'])
ax.set_xlabel('k')
ax.set_yticks(rank_bic['eof'])
ax.set_ylabel('# of EOFs')

gpl.enumerate_subplots(axs, pos_x=-.2, pos_y=1.05, fontsize=10)

# %%
