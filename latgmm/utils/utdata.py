"""Utilities for data."""
import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from latgmm.utils.dataset import SpatialData, MultiVarSpatialData
from latgmm.utils import utdata, utenso, preproc 

PATH = os.path.dirname(os.path.abspath(__file__))

# Pytorch specific
# ======================================================================================
class ForLinear(object):
    """Flattens data map and remove NaNs.
    Transformation for linear input layers."""

    def __call__(self, sample):
        buff = sample['data'].flatten()
        idx_nan = np.isnan(buff)
        sample['data'] = buff[~idx_nan]

        return sample


class ForConvolution(object):
    """Transformation for convolutional input layers.
    Replace NaNs by zeros."""

    def __call__(self, sample):
        buff = np.copy(sample['data'])
        # set NaNs to value
        SETNANS = 0.0
        idx_nan = np.isnan(buff)
        buff[idx_nan] = float(SETNANS)

        # change dim from (n_lon, n_lat) to (1, n_lon, n_lat)
        if len(sample['data'].shape) == 2:
            sample['data'] = np.array([buff])
        else:
            sample['data'] = buff

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        torch_sample = sample
        for key in sample.keys():
            try:
                torch_sample[key] = torch.from_numpy(sample[key]).float()
            except:
                None
        
        return torch_sample


def data2dataset(dataarray, dataclass, batch_size=64,
                 data_2d=True, shuffle=True, split_size_rand=None,
                 **class_kwargs):
    """Load data and split into train, validation and test set.

    Parameters:
    -----
    dataarray: xr.Dataarray
        Dataarray to create dataset class from
    dataclass: class of type torch.utils.data.Dataset 
        Class to the corresponding dataset
    batch_size: int
        Batch size for torch.dataloader
    data_2d: bool
        Whether data should be 2d or flattened.
    shuffle: bool
        Whether data should be shuffled for training.
    split_size: list
        List of fraction of data to split. Default: [0.8, 0.2]
        (train_size + test_size) < 1.0
    **class_kwargs:
        Arguments passed to the dataclass.

    Returns:
    -------
    dataset: utils.SpatioTemporalDataset
    dataloaders: list
        List containing the splitted dataloaders
    """
    transform_func = ForConvolution() if data_2d == True else ForLinear()
    transformations = transforms.Compose([transform_func, ToTensor()])
    dataset = dataclass(dataarray,
                        transform=transformations,
                        **class_kwargs
                        )
    if split_size_rand is None:
        dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        # split training and validation set
        len_split = np.array(np.multiply(len(dataset), split_size_rand), dtype=int) 
        # in case of wrong rounding
        if sum(len_split) != len(dataset):
            len_split[-1] = len(dataset) - sum(len_split[:-1])

        splitted_data = torch.utils.data.random_split(
                dataset, len_split
        )

        dataloaders = []
        for data in splitted_data:
            # Create dataloader
            dataloaders.append(
                DataLoader(data, batch_size=batch_size, shuffle=shuffle)
            )

    return dataset, dataloaders 


# %%
# Load data
# ======================================================================================
def load_data(filenames=[], variables=['sst'], source='reanalysis', **kwargs):
    """Load data.

    Args:
        variable (str, optional): Variable name. Defaults to 'ssta'.
        source (str, optional): Source type, e.g. 'reanalysis', 'cmip6'. Defaults to 'reanalysis'.
        filenames (list, optional): List of filenames. Defaults to [].

    Returns:
        _type_: _description_
    """

    if source == 'reanalysis':
        data = load_reanalysis(filenames=filenames, vars=variables, **kwargs)
    elif source == 'cmip':
        data = load_cmip(filenames=filenames, vars=variables, **kwargs)
    else:
        ValueError(f"Given source={source} are not known.")

    return data


def load_reanalysis(
    filenames, vars,
    timescale='monthly',
    lon_range=[130, -70], lat_range=[-31, 32],
    normalization=None, detrend_from=1950,
    splity=['2005-01-01', '2016-01-01'], batchsize=32,
    enso_types=None, enso_months=[12,2], threshold=0.5, 
    **kwargs):
    """Load reanalysis datasets from different sources and merge them into one dataset.

    Args:
        filenames (list, optional): Filenames of datasets with dataset names.
            Defaults to relative path.
        lon_range (list, optional): Longitude range. Defaults to [130, -70].
        lat_range (list, optional): Latitude range. Defaults to [-31, 32].
        enso_types (list, optional): ENSO types. Defaults to ['Nino_EP', 'Nino_CP', 'Nina_EP', 'Nina_CP'].
        splity (int, optional): Year to split training and test data.
            Defaults to ['2005-01-01', '2016-01-01'].

    Returns:
        data (dict): Dictionary with xr.DataArray, i.e. 
            ['full', 'train', 'val', 'test']
    """
    if timescale == 'monthly':
        climatology = 'month'
    elif timescale == 'daily':
        climatology = 'dayofyear'
    else:
        climatology = timescale

    data = {'full': [], 'train': [], 'val': [], 'test': []}
    for i, f in enumerate(filenames):
        print(f"Open file {f['name']}", flush=True)
        ds = preproc.process_data(
            f['path'], vars=vars, antimeridian=True,
            lon_range=lon_range, lat_range=lat_range, grid_step=1,
            climatology=climatology, detrend_from=detrend_from)
        # Assign new coordinates
        ds = ds.assign_coords(
            member=('time', len(ds.time) * [f['name']])
        )

        if timescale=='daily':
            ds['time'] = np.array(ds['time'], dtype='datetime64[D]')
        else:
            ds['time'] = np.array(ds['time'], dtype='datetime64[M]')
        
        # Mask NaN
        if i == 0:
            mask_nan = xr.zeros_like(ds[list(ds.data_vars)[0]].isel(time=0))

        for t in range(len(ds['time'])):
            for var in list(ds.data_vars):
                mask_nan = np.logical_or(mask_nan, np.isnan(ds[var].isel(time=t)))


        # Select ENSO types
        if enso_types is not None:
            print(f"Load SST-data to select ENSO years!", flush=True)
            enso_classes = utenso.get_enso_flavors_obs(
                definition='N3N4', fname=f['path'], vname='sst', climatology=climatology,
                month_range=enso_months, threshold=threshold,
            )
            buff_enso = []
            for flavor in enso_types:
                time_snippets = np.array(
                    [enso_classes.loc[enso_classes['type'] == flavor]['start'],
                     enso_classes.loc[enso_classes['type'] == flavor]['end']]
                ).T
                ds_enso = preproc.select_time_snippets(ds, time_snippets)
                ds_enso = ds_enso.assign_coords(
                    enso=('time', len(ds_enso.time) * [flavor])
                )
                buff_enso.append(ds_enso)

            ds = xr.concat(buff_enso, dim='time')
            ds = ds.sortby('time')

        # Drop coordinates which are not needed
        for coord_name in list(ds.coords.keys()):
            if coord_name not in ['time', 'lat', 'lon', 'member', 'enso']:
                ds = ds.reset_coords(coord_name, drop=True)

        # Add to list
        data['full'].append(ds)

        # Split into training, validation and test set
        if splity is not None:
            start_time = ds.time.min()
            end_time = ds.time.max()
            split_time = np.array(splity, dtype='datetime64[D]')
            data['train'].append(ds.sel(time=slice(start_time, split_time[0])))
            data['val'].append(ds.sel(time=slice(split_time[0], split_time[1])))
            data['test'].append(ds.sel(time=slice(split_time[1], end_time)))

    # Merge data sources and apply mask
    data['full'] = xr.concat(data['full'], dim='time')
    for var in list(ds.data_vars):
        data['full'][var] = xr.where(mask_nan == False, data['full'][var], np.nan)

    # Normalization
    if normalization is not None:
        attributes = {}
        for var in list(data['full'].data_vars):
            buff = data['full'][var]
            scaler = preproc.Normalizer(method=normalization)
            data['full'][var] = scaler.fit_transform(buff)
            attributes[var] = {'normalizer': scaler}
        data['full'].attrs = attributes
        
    if splity is not None:
        for key in ['train', 'val', 'test']:

            print(f"Convert {key} data to torch.Dataset!", flush=True)
            ds_label = xr.concat(data[key], dim='time')
            for var in list(ds_label.data_vars):
                ds_label[var] = xr.where(mask_nan == False, ds_label[var], np.nan)

                if normalization is not None:
                    ds_label[var] = data['full'].attrs[var]['normalizer'].transform(ds_label[var])

            ds_label.attrs = data['full'].attrs

            if len(ds_label) == 0:
                data[key] = None
                data[f"{key}_loader"] = None 
            else:
                dataset_label, label_loader = utdata.data2dataset(
                    ds_label, MultiVarSpatialData, batch_size=batchsize,
                    shuffle=True, vars=list(ds_label.data_vars)
                )
                # Overwrite
                data[key] = dataset_label
                data[f"{key}_loader"] = label_loader

    return data


def load_cmip(
    filenames, vars,
    timescale='monthly',
    lon_range=[130, -70], lat_range=[-31, 32],
    enso_types=['Nino_EP', 'Nino_CP', 'Nina_EP', 'Nina_CP'],
    splity=None, batchsize=32,
    normalization='zscore', detrend_from=None, **kwargs):
    """Load cmip datasets from different sources and merge them into one dataset.

    Args:
        filenames (list, optional): Filenames of datasets with dataset names.
            Defaults to relative path.
        lon_range (list, optional): Longitude range. Defaults to [130, -70].
        lat_range (list, optional): Latitude range. Defaults to [-31, 32].
        enso_types (list, optional): ENSO types. Defaults to ['Nino_EP', 'Nino_CP', 'Nina_EP', 'Nina_CP'].
        splity (int, optional): Year to split training and test data.
            Defaults to ['2005-01-01', '2016-01-01'].

    Returns:
        data (dict): Dictionary with xr.DataArray, i.e. 
            ['full', 'train', 'val', 'test']
    """
    if timescale == 'monthly':
        climatology = 'month'
    elif timescale == 'daily':
        climatology = 'dayofyear'
    else:
        climatology = timescale

    data = {'full': [], 'train': [], 'val': [], 'test': []}
    for i, f in enumerate(filenames):
        print(f"Open file {f['name']}", flush=True)
        ds = preproc.process_data(
            f['path'], vars=vars, antimeridian=True,
            lon_range=lon_range, lat_range=lat_range, grid_step=1,
            climatology=climatology, normalization=normalization, detrend_from=detrend_from)
        # Assign new coordinates
        ds = ds.assign_coords(
            member=('time', len(ds.time) * [f['name']])
        )

        # Mask NaN
        if i == 0:
            mask_nan = xr.zeros_like(ds[list(ds.data_vars)[0]].isel(time=0))

        # Make sure NaNs are consistnetn across files
        for t in range(len(ds['time'])):
            for var in list(ds.data_vars):
                mask_nan = np.logical_or(mask_nan, np.isnan(ds[var].isel(time=t)))


        # Select ENSO types
        if enso_types is not None:
            print(f"Load SST-data to select ENSO years!", flush=True)
            vname = 'tsa' if climatology is None else 'ts'
            enso_classes = utenso.get_enso_flavors_cmip(
                f['path'], definition='N3N4', vname=vname, climatology=climatology,
                month_range=[12, 2], detrend_from=None
            )
            buff_enso = []
            for flavor in enso_types:
                time_snippets = np.array(
                    [enso_classes.loc[enso_classes['type'] == flavor]['start'],
                     enso_classes.loc[enso_classes['type'] == flavor]['end']]
                ).T
                ds_enso = preproc.select_time_snippets(ds, time_snippets)
                ds_enso = ds_enso.assign_coords(
                    enso=('time', len(ds_enso.time) * [flavor])
                )
                buff_enso.append(ds_enso)

            ds = xr.concat(buff_enso, dim='time')
            ds = ds.sortby('time')

        # Add to list
        data['full'].append(ds)

        # Split into training, validation and test set
        if splity is not None:
            start_time = ds.time.min()
            end_time = ds.time.max()
            split_time = np.array(splity, dtype='datetime64[D]')
            data['train'].append(ds.sel(time=slice(start_time, split_time[0])))
            data['val'].append(ds.sel(time=slice(split_time[0], split_time[1])))
            data['test'].append(ds.sel(time=slice(split_time[1], end_time)))

    # Merge data sources and apply mask
    data['full'] = xr.concat(data['full'], dim='time')
    for var in list(ds.data_vars):
        data['full'][var] = xr.where(mask_nan == False, data['full'][var], np.nan)

    if splity is not None:
        for key in ['train', 'val', 'test']:
            print(f"Convert {key} data to torch.Dataset!", flush=True)
            ds_label = xr.concat(data[key], dim='time')
            for var in list(ds_label.data_vars):
                ds_label[var] = xr.where(mask_nan == False, ds_label[var], np.nan)

            if len(ds_label) == 0:
                data[key] = None
                data[f"{key}_loader"] = None 
            else:
                dataset_label, label_loader = utdata.data2dataset(
                    ds_label, MultiVarSpatialData, batch_size=batchsize,
                    shuffle=True, vars=list(ds_label.data_vars)
                )
                # Overwrite
                data[key] = dataset_label
                data[f"{key}_loader"] = label_loader

    return data