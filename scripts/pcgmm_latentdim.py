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
datafile = "../data/reanalysis/monthly/ssta_merged_dataset_1.nc"
normalization = 'zscore'

ds = xr.open_dataset(datafile)
ds = ds.sel(lat=slice(-15, 15))

# Normalization
if normalization is not None:
    attributes = {}
    ds_norm = []
    for var in list(ds.data_vars):
        scaler = preproc.Normalizer(method=normalization)
        buff = scaler.fit_transform(ds[var])
        buff.attrs = {'normalizer': scaler}
        ds_norm.append(buff)

    ds = xr.merge(ds_norm) 

# %%
# Scan number of eofs
# ======================================================================================
reload(metric)
month_range=[12, 2]
gmm_bic = []
n_eofs = [2,3,4,5,6]
for n_components in n_eofs:
    sppca = eof.SpatioTemporalPCA(ds, n_components=n_components)
    eofs = sppca.get_eofs()
    pcs = sppca.get_principal_components()
    print(f"EOFs {n_components}, exp. variance: {np.sum(sppca.explained_variance())}")

    # Preselect data for GMM
    x_enso, x_events = utenso.select_enso_events(ds, month_range=month_range, threshold=0.5)
    z_enso = xr.DataArray(
        data=sppca.transform(x_enso),
        coords={'time': x_enso['time'].data, 'eof': np.arange(1, sppca.n_components+1)}
    ).assign_coords(member=('time', x_enso['member'].data))
    z_events = xr.DataArray(
        data=sppca.transform(x_events),
        coords={'time': x_events['time'].data, 'eof': np.arange(1, sppca.n_components+1)}
    ).assign_coords(member=('time', x_events['member'].data))

    # ## Gaussian mixture 
    print("Scan number of cluster")
    n_classes = np.arange(1, 10, 1)
    n_runs = 50
    for k in n_classes:
        for r in range(n_runs):
            gmm = mixture.GaussianMixture(n_components=k, 
                                          covariance_type='full', max_iter=100)
            gmm.fit(z_enso.data)
            gmm_bic.append(
                {'k': k, 'bic': gmm.bic(z_enso.data), 
                 'n_eof': n_components, 'parameters': gmm._n_parameters()}
            )
gmm_bic = pd.DataFrame(gmm_bic)
# %%
# Boxplots for each number of eofs
# ======================================================================================
clrs = sns.color_palette('magma', len(n_eofs))
fig, axs = plt.subplots(len(n_eofs), 1, figsize=(5,len(n_eofs)*2), sharex=True)
for i, n in enumerate(n_eofs):
    temp = gmm_bic.loc[gmm_bic['n_eof'] == n]
    temp['logbic'] = np.log(temp['bic'])
    sns.boxplot(data=temp, x='k', y='bic', ax=axs[i], color=clrs[i], fliersize=0.0)
    axs[i].set_title(f"EOFs: {n}")

# %%
# Boxplots in one plot
# ======================================================================================
import matplotlib.patches as mpatches
clrs = sns.color_palette('cividis', len(n_eofs))
fig, ax = plt.subplots(1, 1, figsize=(5, 8))
labels = []
for i, n in enumerate(n_eofs):
    temp = gmm_bic.loc[gmm_bic['n_eof'] == n]
    temp['logbic'] = np.log(temp['bic'])
    sns.boxplot(data=temp, x='k', y='logbic', ax=ax, color=clrs[i],
                fliersize=0.0)
    labels.append(mpatches.Patch(color=clrs[i], label=f"EOFs={n}"))
ax.legend(handles=labels, loc=4)


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
# BIC over number of eofs 
# ======================================================================================
bic = []
for i, n in enumerate(n_eofs):
    temp = gmm_bic.loc[gmm_bic['n_eof'] == n]
    bic.append(xr.DataArray(
            data=pd.DataFrame([temp.loc[temp['k']==k].median() for k in temp['k'].unique()])['bic'],
            coords=dict(k=temp['k'].unique())
        )
    )
bic = xr.concat(bic, dim=pd.Index(n_eofs, name='eof'))

# Number of parameters model 
# ======================================================================================
params = []
for i, n in enumerate(n_eofs):
    temp = gmm_bic.loc[gmm_bic['n_eof'] == n]
    params.append(xr.DataArray(
            data=pd.DataFrame([temp.loc[temp['k']==k].mean() for k in temp['k'].unique()])['parameters'],
            coords=dict(k=temp['k'].unique())
        )
    )
params = xr.concat(params, dim=pd.Index(n_eofs, name='eof'))

# %%
reload(gpl)
# Number of parameters and BIC
# ======================================================================================
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm, BoundaryNorm
fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True, sharey=True)

ax = axs[0]
im = gpl.plot_matrix(params, xcoord='k', ycoord='eof', ax=ax, cmap='viridis',
                vmin=np.min(params), vmax=np.max(params), eps=25, add_bar=False)
cbar = plt.colorbar(im['im'], ax=ax, orientation='horizontal',
                    label='# of parameters', pad=0.17)

ax.set_xticks(params['k'])
ax.set_xlabel('k')
ax.set_yticks(params['eof'])
ax.set_ylabel('# of EOFs')

# Plot log(BIC)
ax = axs[1]
logbic = np.log(bic)
levels = np.linspace(np.min(logbic)-0.061, np.max(logbic), 20)
cmap = plt.get_cmap('cividis_r', len(levels) +1)
norm = BoundaryNorm(levels, ncolors=cmap.N, extend='both')
c = ax.pcolormesh(logbic.k, logbic.eof, logbic, norm=norm, cmap=cmap)
cbar = plt.colorbar(c, orientation='horizontal', pad=0.17, label='log(BIC)')
ticks = cbar.ax.get_xticks()
cbar.set_ticks(ticks=ticks[1::2], labels=[f"{t:.1f}" for t in ticks[1::2]])
ax.set_xticks(logbic['k'])
ax.set_xlabel('k')

gpl.enumerate_subplots(axs, pos_x=.0, pos_y=1.05, fontsize=10)


plt.savefig("../output/plots/pcgmm_nparams_bic.png", dpi=300, bbox_inches='tight')

# %%
