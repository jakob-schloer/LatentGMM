"""
Loading and preprocessing of SST data.

@author: Jakob Schlör
"""
import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

import latgmm.utils.preproc as utpp
PATH = os.path.dirname(os.path.abspath(__file__))


class SpatialData(Dataset):
    """Spatial data class. 

    Args:
        dataarray ([type]): [description]
        transform ([type], optional): [description]. Defaults to None.
    """

    def __init__(self, dataarray, transform=None):
        # Normalize data
        if 'norm' in dataarray.attrs.keys():
            self.normalization = dataarray.attrs
        else:
            self.normalization = None

        self.dataarray = dataarray
        self.dims = self.dataarray.shape
        self.dim_name = self.dataarray.dims
        self.time = self.dataarray[self.dim_name[0]].data
        
        # Store the position of NaNs which is needed for reconstructing the map later
        self.idx_nan = np.isnan(self.dataarray)

        self.transform = transform
    

    def __len__(self):
        return len(self.dataarray)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'data':   self.dataarray[idx].data,
                  'time':  utpp.time2timestamp(self.dataarray[idx].time.data)}

        if self.transform:
            sample = self.transform(sample)

        return sample['data'], {'time': sample['time']}
    

    def get_dataarray(self):
        return self.dataarray
    
    
    def get_map(self, data, name=None, dim_name='time', dim=None):
        """Convert VAE output to xarray object with the right coord.

        Args:
            data (torch.Tensor): output of VAE 
            name (str, optional): Name of xarray. Defaults to None.
            dim_name (str, optional): Name of additional dimension. Defaults to 'time'.
            dim ([type], optional): Coordinates of dimension. Defaults to None.

        Returns:
            [type]: [description]
        """
        if torch.is_tensor(data):
            data = data.to('cpu').detach().numpy()

        # For single images
        if data.shape[0] == 1: 
            data_map = data[0,:,:].copy()
            dims = ['lat', 'lon']
            coords = [self.dataarray.coords['lat'], self.dataarray.coords['lon']]
            assert data_map.shape == self.idx_nan.data[0].shape
        # For set of images
        elif data.shape[1] == 1:
            data_map = data[:,0,:,:].copy()
            dims = [dim_name, 'lat', 'lon']

            if dim is None:
                dim = np.arange(data_map.shape[0])

            coords = {f'{dim_name}': dim,
                      'lat': self.dataarray.coords['lat'],
                      'lon': self.dataarray.coords['lon']}
            assert data_map[0].shape == self.idx_nan.data[0].shape
        else:
            print('Other data shapes are not supported.')

        dmap = xr.DataArray(
            data=data_map,
            dims=dims,
            coords=coords,
            name=name) 

        if self.normalization is not None:
            dmap = utpp.unnormalize(dmap, self.normalization)

        return dmap
    
    def flatten_map(self):
        """Return flattened data array in spatial dimension, i.e. (time, n_x*n_y).
        
        NaNs are removed in each dimension.
        """
        flat_arr = self.dataarray.data.reshape(
            self.dims[0], self.dims[1]*self.dims[2]
        )
        flat_nonans = []
        for a in flat_arr:
            idx_nan = np.isnan(a)
            flat_nonans.append(a[~idx_nan])

        return np.array(flat_nonans)
    
    def unflatten_map(self, data, name=None):
        """Reshape flattened map to map.

        This also includes adding NaNs which have been removed.

        TODO: time stamp in sample should be used to identify self.idx_nan 

        Parameters:
        -----------
        data (torch.tensor): Flatten datapoint with NaNs removed

        Returns:
        --------
        data_map (np.ndarray): 2d-map
        """
        if torch.is_tensor(data):
            data = data.to('cpu').detach().numpy()

        idx_nan_arr = self.idx_nan.data[0].flatten()
        # Number of non-NaNs should be equal to length of data
        assert np.count_nonzero(~idx_nan_arr) == len(data)
        # create array with NaNs
        data_map = np.empty(len(idx_nan_arr)) 
        data_map[:] = np.NaN
        # fill array with sample
        data_map[~idx_nan_arr] = data

        return np.reshape(data_map, self.idx_nan.data[0].shape)


    def unflatten_maps(self, data):
        """Reshape list of flattened maps and insert NaNs.

        Args:
            data (np.ndarray, torch.Tensor): (N, M) N input datapoints with M features.

        Returns:
            data_maps (np.ndarray): (N, dim_x, dim_y) Output data maps. 
        """
        if torch.is_tensor(data):
            data = data.to('cpu').detach().numpy()
        
        N, M = data.shape 
        idx_nan_arr = self.idx_nan.data[0].flatten()
        # Number of non-NaNs should be equal to length of data
        assert np.count_nonzero(~idx_nan_arr) == M
        # create array with NaNs
        data_map = np.empty(shape=(N, len(idx_nan_arr))) 
        data_map[:,:] = np.NaN
        # fill array with sample
        data_map[:, ~idx_nan_arr] = data

        return np.reshape(data_map, newshape=(N, *self.idx_nan.data[0].shape))
    

class MultiVarSpatialData(Dataset):
    """Multible variable dataset class for convolutions. 

    Args:
        dataset (xr.Dataset): Dataset 
        vars (list, optional): Variables to use. Defaults to ['ssta', 't300a'].
        transform ([type], optional): [description]. Defaults to None.
    """

    def __init__(self, dataset, vars=['ssta', 't300a'], transform=None):
        super().__init__()

        # Normalize data
        if 'normalizer' in dataset[vars[0]].attrs.keys():
            self.normalization = dataset.attrs
        else:
            self.normalization = None

        self.dataset = dataset.transpose('time', 'lat', 'lon')
        self.vars = vars 
        self.coords = list(self.dataset.coords)
        self.dims = self.dataset[self.vars[0]].shape
        
        self.transform = transform
    

    def __len__(self):
        return len(self.dataset['time'])
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        datapoint = np.array(
            [self.dataset.isel(time=idx)[var].data for var in self.vars]
        )
        
        sample = {
            'data':  datapoint,
            'time':  utpp.time2timestamp(self.dataset['time'][idx].data),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample['data'], {'time': sample['time']} 

    
    def get_map(self, data, dim_name='time', dim=None):
        """Convert VAE output to xarray object with the right coord.

        Args:
            data (torch.Tensor): output of VAE 
            name (str, optional): Name of xarray. Defaults to None.
            dim_name (str, optional): Name of additional dimension. Defaults to 'time'.
            dim ([type], optional): Coordinates of dimension. Defaults to None.

        Returns:
            [type]: [description]
        """
        if torch.is_tensor(data):
            data = data.to('cpu').detach().numpy()

        assert data.shape[-2] == self.dims[-2]
        assert data.shape[-1] == self.dims[-1]
        # For single images
        if len(data.shape)==3 and data.shape[0]==len(self.vars):
            dmaps = []
            for i, var in enumerate(self.vars):
                data_map = data[i, :, :].copy()
                dims = ['lat', 'lon']
                coords = [self.dataset.coords['lat'],
                          self.dataset.coords['lon']]
                dmaps.append(xr.DataArray(
                    data=data_map,
                    dims=dims,
                    coords=coords,
                    name=var
                ))
        # For set of images
        elif data.shape[1] == len(self.vars):
            dmaps = []
            for i, var in enumerate(self.vars):
                data_map = data[:,i, :, :].copy()
                dims = [dim_name, 'lat', 'lon']
                if dim is None:
                    dim = np.arange(data_map.shape[0])

                coords = {f'{dim_name}': dim,
                          'lat': self.dataset.coords['lat'],
                          'lon': self.dataset.coords['lon']}

                dmaps.append(xr.DataArray(
                    data=data_map,
                    dims=dims,
                    coords=coords,
                    name=var
                ))

        dmap = xr.merge(dmaps)

        if self.normalization is not None:
            for var in list(self.dataset.data_vars):
                dmap[var] = self.normalization[var]['normalizer'].inverse_transform(dmap[var])

        return dmap


class CMIP5Conv(Dataset):
    """CMIP5 dataset class for convolutions. 

    Args:
        dataset (xr.Dataset): Dataset 
        vars (list, optional): Variables to use. Defaults to ['ssta', 't300a'].
        transform ([type], optional): [description]. Defaults to None.
    """

    def __init__(self, dataset, vars=['ssta', 't300a'], transform=None):
        super().__init__()

        # Normalize data
        if 'norm' in dataset.attrs.keys():
            self.normalization = dataset.attrs
        else:
            self.normalization = None

        self.dataset = dataset
        self.vars = vars 
        self.coords = list(self.dataset.coords)
        self.dims = self.dataset[self.vars[0]].shape
        
        self.transform = transform
    

    def __len__(self):
        return self.dims[0]*self.dims[1]
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Convert to model and time index
        idx_model = int(idx/self.dims[1])
        idx_time = idx % self.dims[1]

        datapoint = np.array(
            [self.dataset[var][idx_model, idx_time, :, :] for var in self.vars]
        )
        
        sample = {
            'data':  datapoint,
            'time':  utpp.time2timestamp(self.dataset['time'][idx_time].data),
            'model': int(self.dataset['model'][idx_model].data)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample['data'], {'time': sample['time'], 'model': sample['model']} 

    
    def get_map(self, data, dim_name='time', dim=None):
        """Convert VAE output to xarray object with the right coord.

        Args:
            data (torch.Tensor): output of VAE 
            name (str, optional): Name of xarray. Defaults to None.
            dim_name (str, optional): Name of additional dimension. Defaults to 'time'.
            dim ([type], optional): Coordinates of dimension. Defaults to None.

        Returns:
            [type]: [description]
        """
        if torch.is_tensor(data):
            data = data.to('cpu').detach().numpy()

        assert data.shape[-2] == self.dims[-2]
        assert data.shape[-1] == self.dims[-1]
        # For single images
        if len(data.shape)==3 and data.shape[0]==len(self.vars):
            dmaps = []
            for i, var in enumerate(self.vars):
                data_map = data[i, :, :].copy()
                dims = ['lat', 'lon']
                coords = [self.dataset.coords['lat'],
                          self.dataset.coords['lon']]
                dmaps.append(xr.DataArray(
                    data=data_map,
                    dims=dims,
                    coords=coords,
                    name=var
                ))
        # For set of images
        elif data.shape[1] == len(self.vars):
            dmaps = []
            for i, var in enumerate(self.vars):
                data_map = data[:,i, :, :].copy()
                dims = [dim_name, 'lat', 'lon']
                if dim is None:
                    dim = np.arange(data_map.shape[0])

                coords = {f'{dim_name}': dim,
                          'lat': self.dataset.coords['lat'],
                          'lon': self.dataset.coords['lon']}

                dmaps.append(xr.DataArray(
                    data=data_map,
                    dims=dims,
                    coords=coords,
                    name=var
                ))

        dmap = xr.merge(dmaps)

        if self.normalization is not None:
            dmap = utpp.unnormalize(dmap, self.normalization)

        return dmap
