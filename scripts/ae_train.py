'''  Autoencoder 
Nonlinear decoding using an AE before applying a GMM on the latent pace

@Author  :   Jakob Schlör 
@Time    :   2023/03/17 15:10:27
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
import os, time
import xarray as xr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cartopy as ctp
import seaborn as sns
from sklearn import mixture, decomposition
from importlib import reload

from latgmm.utils import utdata 
from latgmm.model import ae 
import latgmm.geoplot as gpl

PATH = os.path.dirname(os.path.abspath(__file__))
plt.style.use(PATH + "/../../paper.mplstyle")

# Load data
# ======================================================================================
reload(utdata)
param = dict(
    multivar=False,
    variables=['sst'],
    source='reanalysis',
    timescale='monthly',
    lon_range=[130, -70],
    lat_range=[-31, 32],
#    lat_range=[-15, 16],
)
if (param['multivar'] is False) & (len(param['variables']) == 1):
    if param['timescale'] == 'monthly':
        dirpath = "../../data/sst/monthly"
        param['filenames'] = [
            dict(name='cobe2',   path=dirpath+"/sst_cobe2_month_1850-2019.nc"),
            dict(name='ersstv5', path=dirpath+"/sst_ersstv5_month_1854-present.nc"),
            dict(name='hadisst', path=dirpath+"/sst_hadisst_month_1870-present.nc"),
            dict(name='oras5',   path=dirpath+"/sst_t300_oras5_1958-2018.nc"),
            dict(name='godas',   path=dirpath+"/sst_godas_month_1980-present.nc"),
            dict(name='soda',    path=dirpath+"/sst_SODA_month_1980-2017.nc"),
            dict(name='era5',    path=dirpath+"/sst_era5_monthly_sp_1959-2021_1x1.nc"),
            # dict(name='tropflux',path=dirpath+"/sst_tropflux_month_1979-2018.nc"),
        ]
        param['splity'] = ['2005-01-01', '2022-01-01']
    elif param['timescale'] == 'daily':
        param['filenames'] = [dict(
            name='era5',
            path=("../data/sst/daily/sea_surface_temperature_daily_coarse_1950_2021.nc")
        )]
    
elif param['multivar'] & (param['timescale'] == 'monthly'):
    param['filenames']=[
        dict(name='soda',  path=f"../../data/multivar/oceanvars_SODA_1x1.nc"),
        dict(name='godas', path=f"../../data/multivar/oceanvars_GODAS_1x1.nc"),
        dict(name='oras5', path=f"../../data/multivar/oceanvars_ORAS5_1x1.nc")
    ]
    param['splity']=['2013-01-01', '2022-01-01']
else:
    raise ValueError(f"No data are loaded due to specified timescale and variables!")

param['detrend_from'] = '1950'
param['normalization'] = 'zscore'

data = utdata.load_data(**param)
train_loader = data['train_loader']
val_loader = data['val_loader']

# %%
# Training
# ======================================================================================
def train(n_epochs, model, optimiser, loss_fn,
          train_loader, val_loader, scheduler=None, save_folder=None):
    """Training loop of AE."""
    train_loss = []
    val_loss = []
    val_loss_min = 5e5

    # Move to GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    print(f"Training on {device}!", flush=True)
    for epoch in range(n_epochs):
        tstart = time.time()

        # 1. Validation
        model.eval()
        val_loss_epoch = []
        for data in val_loader:
            x, l = data
            x = x.to(device)
            with torch.no_grad():
                x_hat, z = model(x)
                val_loss_epoch.append(
                    loss_fn(x_hat, x)
                )
        val_loss_epoch = torch.Tensor(val_loss_epoch).mean()

        # 2. Training
        model.train()
        train_loss_epoch = []
        for data in train_loader:
            x, l = data
            x = x.to(device)

            optimiser.zero_grad()
            x_hat, z = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimiser.step()
            train_loss_epoch.append(loss)
        train_loss_epoch = torch.Tensor(train_loss_epoch).mean()

        # Scheduler
        if scheduler is not None:
            scheduler.step()

        # 3. Print loss
        tend = time.time()
        print(f"Epoch {epoch}, train: {train_loss_epoch:.2e}, "
              + f"val. horiz.: {val_loss_epoch:.2e}, "
              + f"time: {tend - tstart}", flush=True)

        # 4. Store losses and create checkpoint
        train_loss.append(train_loss_epoch)
        val_loss.append(val_loss_epoch)
        checkpoint = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
        }

        # 5. Save checkpoint
        if save_folder is not None:
            if val_loss_epoch < val_loss_min:
                print("Save checkpoint!", flush=True)
                torch.save(checkpoint, save_folder + f"/min_checkpoint.pt")
                val_loss_min = val_loss_epoch

    # Save model at the end
    if save_folder is not None:
        torch.save(checkpoint, save_folder + f"/final_checkpoint.pt")
        print("Finished training and save model!", flush=True)
    else:
        print("Finished training!", flush=True)
        
    return checkpoint

# %%
# Sample datapoint
# ======================================================================================
x,l = data['train'][0]
x_dim = x.shape

# Define model
param['z_dim'] = 2
param['hid_channel'] = 16
model = ae.CNNAE(x_dim, z_dim=param['z_dim'], hid_channel=param['hid_channel'])

print("Number of trainable parameters of our model:",
      sum(p.numel() for p in model.parameters() if p.requires_grad)) 

# Forward pass
for datapoint in data['train_loader']:
    x, l = datapoint
    x_hat, z = model(x)
    print(f"Input size: {x.shape}")
    print(f"Output size: {x_hat.shape}")
    break


# %%
# Train model
# ======================================================================================
param['lr'] = 0.0001
param['postfix'] = ''
param['path'] = "../../output/reanalysis/ae/ssta/"
param['epochs'] = 50
criterion = lambda input, target: torch.mean((input - target)**2)
optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])

save_folder = (param['path']
               + f"/{'_'.join(data['full'].data_vars)}_eof_{param['z_dim']}"
               + f"_ae_hidden_{param['hid_channel']}_layers_3"
               + f"_ep_{param['epochs']}" 
               + f"{param['postfix']}")

# Save config
if not os.path.exists(save_folder):
    print(f"Create directoty {save_folder}", flush=True)
    os.makedirs(save_folder)
torch.save(param, save_folder + "/config.pt")

checkpoint = train(n_epochs=param['epochs'], model=model,
                   optimiser=optimizer, loss_fn=criterion, train_loader=train_loader,
                   val_loader=val_loader, save_folder=save_folder)

# %%
# Plot loss
# ======================================================================================
fig, ax = plt.subplots()
ax.plot(checkpoint['train_loss'],
        label=f"training {checkpoint['train_loss'][-1]:.2e}")
ax.plot(checkpoint['val_loss'],
        label=f"validation {checkpoint['val_loss'][-1]:.2e}")
ax.set_yscale('log')
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.legend()

plt.savefig(save_folder + f"/loss.png", dpi=300, bbox_inches='tight')

# %%



